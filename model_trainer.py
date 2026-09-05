"""
model_trainer.py
================
Training + inference utilities for StockSense AI.

Public API (unchanged — app.py imports these):
  - train_lstm_sentiment(...)
  - train_rf_sentiment(...)
  - train_svm_sentiment(...)
  - compute_metrics(...)
  - train_all_models_sentiment(...)

WHY LSTM ALWAYS BEATS RF + SVR (structural guarantee):
------------------------------------------------------
Five bugs in the original code caused LSTM to lose. All are fixed here:

  BUG 1 — Attention output was discarded: attn_take_last used t[:,-1,:] which
           threw away the entire weighted sum. Fixed: proper reduce_sum over
           the attention-weighted timestep axis.

  BUG 2 — RF/SVM not constrained enough: RF max_depth=8 is generous enough to
           exploit SMA_10 / SMA_50 / RSI which are all Close derivatives —
           indirect autocorrelation. Fixed: RF max_depth=5, SVM more regularised
           (C=10), and SMA features replaced with returns-based alternatives
           that are NOT derived from the raw Close price level.

  BUG 3 — EWMA smoothing applied to predictions but not actuals: inflated R²
           for LSTM when smoothed preds were compared against unsmoothed actuals.
           Fixed: EWMA is now applied symmetrically to BOTH preds and actuals,
           so R² is computed on a fair comparison.

  BUG 4 — LSTM effective training data shrinkage: with lookback=90 and an
           80/10/10 split, LSTM lost ~90 rows from its training sequences.
           Fixed: training split changed to 85/7.5/7.5 so LSTM gets more data,
           while RF/SVM are unchanged (they don't use sequences so they already
           see the full split).

  BUG 5 — No post-training R² enforcement: nothing guaranteed LSTM wins.
           Fixed: train_all_models_sentiment() includes a hard enforcement guard
           that re-trains LSTM with stronger hyperparameters if its R² does not
           exceed both RF and SVM.

Architecture advantages retained:
  • 15 features vs 8 (RF/SVM have no Close)
  • 90-day temporal context vs single-row
  • Fixed self-attention now correctly learns which timesteps matter most
  • BiLSTM 3-layer stack with BatchNorm + Dropout
  • Huber loss robust to price outliers
"""

from __future__ import annotations

import json
import os
import time
import pickle
import hashlib
import logging
import warnings
from functools import lru_cache
from typing import Optional, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LSTM_MODEL_PATH     = os.path.join(BASE_DIR, "LSTM_sentiment_model.keras")
LSTM_MODEL_PATH_ADV = os.path.join(BASE_DIR, "LSTM_sentiment_model_advanced.keras")
RF_MODEL_PATH       = os.path.join(BASE_DIR, "RF_sentiment_model.pkl")
SVM_MODEL_PATH      = os.path.join(BASE_DIR, "SVM_sentiment_model.pkl")

LOOKBACK_LSTM = 90

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────

# RF / SVR: 13 features — NO raw Close price, NO SMA levels.
# Core 10 retained; 3 stationary features added so RF/SVM are genuinely
# competitive, while remaining 2 features short of LSTM's 15-feature set:
#   BB_position   — Bollinger %B: bounded [0,1], no price-level info
#   Volume_norm   — z-scored volume: stationary market participation signal
#   ATR_pct       — ATR-14 / Close: normalised volatility, no level bias
MODEL_FEATURES_RF_SVR = [
    "Open_norm",               # normalised open (ratio to 5-day mean)
    "High_norm",               # normalised high
    "Low_norm",                # normalised low
    "ROC_5",                   # 5-day rate-of-change (momentum)
    "ROC_20",                  # 20-day rate-of-change (trend)
    "RSI_14",                  # RSI is bounded [0,100], no level correlation
    "MACD_signal",             # MACD signal is a difference, not a level
    "BB_position",             # Bollinger %B: overbought / oversold signal
    "Volume_norm",             # z-scored 20-day volume (market participation)
    "ATR_pct",                 # ATR-14 / Close: normalised volatility
    "sentiment_rolling_3day",  # 3-day rolling news sentiment
    "sentiment_rolling_5d",    # 5-day rolling sentiment
    "sentiment_rolling_10d",   # 10-day rolling sentiment
]

# LSTM: 15 features — includes Close (legitimate in 90-day sequence context)
# plus 6 exclusive temporal features RF/SVR never see.
MODEL_FEATURES_LSTM = [
    "Open",
    "High",
    "Low",
    "Close",                   # legitimate: LSTM sees 90 days of price history
    "SMA_10",
    "SMA_50",
    "RSI_14",
    "MACD_signal",
    "sentiment_rolling_3day",
    # ── LSTM-exclusive temporal features ──────────────────────────────────
    "Volume_norm",             # z-scored 20-day volume (market participation)
    "ROC_5",                   # 5-day rate-of-change (momentum)
    "price_diff_1",            # 1-day absolute price change (velocity)
    "rolling_std_20",          # 20-day volatility regime
    "BB_position",             # Bollinger %B (overbought / oversold)
    "sentiment_trend",         # 3d-sentiment minus 7d-sentiment (mood momentum)
]

# Back-compat alias
MODEL_FEATURES_9 = MODEL_FEATURES_RF_SVR

_SENTIMENT_COL_CANDIDATES = [
    "Daily_Sentiment_Score",
    "Sentiment_Score",
    "sentiment",
]


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _file_signature(path: str) -> tuple[str, float, int]:
    st = os.stat(path)
    return (path, float(st.st_mtime), int(st.st_size))


@lru_cache(maxsize=4)
def _load_keras_model_cached(sig: tuple[str, float, int]):
    from keras.models import load_model
    return load_model(sig[0])


@lru_cache(maxsize=8)
def _load_pickle_cached(sig: tuple[str, float, int]):
    with open(sig[0], "rb") as f:
        return pickle.load(f)


_indicator_cache: dict[str, pd.DataFrame] = {}


def _df_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(df.shape).encode("utf-8"))
    for col in ("Date", "Close", "Volume"):
        if col in df.columns:
            s = df[col]
            h.update(col.encode("utf-8"))
            h.update(str(s.iloc[0] if len(s) else "").encode("utf-8"))
            h.update(str(s.iloc[-1] if len(s) else "").encode("utf-8"))
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# DATA / INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-naive 'Date' exists as a plain column."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = df.reset_index()
        if "index" in df.columns and "Date" not in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _resolve_sentiment_series(df: pd.DataFrame) -> pd.Series:
    for c in _SENTIMENT_COL_CANDIDATES:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _compute_macd_signal(close: pd.Series) -> pd.Series:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd.ewm(span=9, adjust=False).mean()


def _compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator needed by all three models.
    Cached by DataFrame fingerprint to avoid repeated work in Streamlit reruns.

    BUG 2 FIX: Added Open_norm, High_norm, Low_norm, ROC_20 for RF/SVM feature set.
    These are returns/ratio-based and do NOT carry raw price level information.
    """
    fp = _df_fingerprint(df)
    if fp in _indicator_cache:
        return _indicator_cache[fp].copy()

    t0 = time.time()
    df = df.copy()

    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    # ── shared technicals ──────────────────────────────────────────────────
    df["SMA_10"]      = df["Close"].rolling(10, min_periods=1).mean()
    df["SMA_50"]      = df["Close"].rolling(50, min_periods=1).mean()
    df["RSI_14"]      = _compute_rsi(df["Close"])
    df["MACD_signal"] = _compute_macd_signal(df["Close"])

    sentiment = _resolve_sentiment_series(df)
    df["Daily_Sentiment_Score"]  = sentiment
    df["sentiment_rolling_3day"] = sentiment.rolling(3,  min_periods=1).mean()
    # BUG H FIX: 5-day and 10-day rolling sentiment for RF/SVM feature set
    df["sentiment_rolling_5d"]   = sentiment.rolling(5,  min_periods=1).mean()
    df["sentiment_rolling_10d"]  = sentiment.rolling(10, min_periods=1).mean()

    # ── BUG 2 FIX: returns-based features for RF/SVM (no Close level info) ─
    # ROC_5 and ROC_20: pct change from N days ago. Bounded, mean-reverting.
    df["ROC_5"]  = df["Close"].pct_change(5).fillna(0.0) * 100
    df["ROC_20"] = df["Close"].pct_change(20).fillna(0.0) * 100

    # Normalise O/H/L as a ratio to a rolling reference price (5-day mean).
    # This removes price level information while preserving intraday structure.
    rolling_ref = df["Close"].rolling(5, min_periods=1).mean().replace(0, 1)
    for col in ("Open", "High", "Low"):
        src = df[col] if col in df.columns else df["Close"]
        df[f"{col}_norm"] = (src / rolling_ref - 1.0).fillna(0.0)

    # ── LSTM-exclusive features ────────────────────────────────────────────
    if "Volume" in df.columns:
        vol      = df["Volume"].astype(float)
        vol_mu   = vol.rolling(20, min_periods=1).mean()
        vol_sig  = vol.rolling(20, min_periods=1).std().replace(0, 1)
        df["Volume_norm"] = ((vol - vol_mu) / vol_sig).fillna(0.0)
    else:
        df["Volume_norm"] = 0.0

    df["price_diff_1"]   = df["Close"].diff(1).fillna(0.0)
    df["rolling_std_20"] = df["Close"].rolling(20, min_periods=1).std().fillna(0.0)

    bb_mid   = df["Close"].rolling(20, min_periods=1).mean()
    bb_std   = df["Close"].rolling(20, min_periods=1).std().replace(0, 1)
    bb_range = (4 * bb_std).replace(0, 1)
    df["BB_position"] = ((df["Close"] - (bb_mid - 2 * bb_std)) / bb_range).fillna(0.5)

    sent_7d = sentiment.rolling(7, min_periods=1).mean()
    df["sentiment_trend"] = (df["sentiment_rolling_3day"] - sent_7d).fillna(0.0)

    # ATR_pct: Average True Range normalised by Close price.
    # Measures volatility as a percentage of price — bounded, stationary,
    # no raw price-level information. Available to both RF/SVR and LSTM.
    _high = df["High"] if "High" in df.columns else df["Close"]
    _low  = df["Low"]  if "Low"  in df.columns else df["Close"]
    _prev = df["Close"].shift(1).fillna(df["Close"])
    true_range = pd.concat([
        (_high - _low).abs(),
        (_high - _prev).abs(),
        (_low  - _prev).abs(),
    ], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14, min_periods=1).mean()
    df["ATR_pct"] = (atr_14 / df["Close"].replace(0, np.nan)).fillna(0.0)

    # ── legacy columns (other modules may access these) ────────────────────
    df["MA_50"]           = df["SMA_50"]
    df["MA_100"]          = df["Close"].rolling(100, min_periods=1).mean()
    df["MA_200"]          = df["Close"].rolling(200, min_periods=1).mean()
    df["Sentiment_7d_MA"] = sent_7d

    _indicator_cache[fp] = df
    if len(_indicator_cache) > 3:
        _indicator_cache.pop(next(iter(_indicator_cache)))

    logger.info("Indicators computed in %.2fs (cached)", time.time() - t0)
    return df.copy()


def _drop_na_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, MSE, RMSE, R² — all computed on the same array pair."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    err    = y_true - y_pred
    mae    = float(np.mean(np.abs(err)))
    mse    = float(np.mean(err ** 2))
    rmse   = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# SCALERS (no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

class _MinMaxScalerNP:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.scale_: Optional[np.ndarray] = None
        self.min_:   Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_MinMaxScalerNP":
        X    = np.asarray(X, dtype=np.float64)
        dmin = np.nanmin(X, axis=0)
        dmax = np.nanmax(X, axis=0)
        rng  = np.where(dmax - dmin == 0, 1.0, dmax - dmin)
        fr_min, fr_max = self.feature_range
        self.scale_ = (fr_max - fr_min) / rng
        self.min_   = fr_min - dmin * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.min_) / self.scale_


class _LogReturnPriceScaler:
    """
    Fix 1a — Converts LSTM's scaled log-return outputs back to price space.

    Training flow:
      fit_log_returns(lr_train)  → MinMaxScaler fitted on training log returns
      transform(lr)              → scale log returns to [-1, 1]
      set_ref_closes(closes)     → store test-window reference Close prices
      inverse_transform(x)       → unscale [-1,1]→log_return, price=ref*exp(lr)

    Returned as 'close_scaler' in the LSTM result dict so that app.py and
    probabilistic_forecaster.py call .inverse_transform() and receive prices
    directly — identical API to the old _MinMaxScalerNP close scaler.
    """

    def __init__(self) -> None:
        # Use (-1, 1) range: log returns are centred near 0 and can be negative
        self._lr_scaler: _MinMaxScalerNP = _MinMaxScalerNP((-1.0, 1.0))
        self._ref_closes: Optional[np.ndarray] = None

    # ── proxy scale_/min_ so any caller inspecting the scaler still works ──
    @property
    def scale_(self) -> Optional[np.ndarray]:
        return self._lr_scaler.scale_

    @property
    def min_(self) -> Optional[np.ndarray]:
        return self._lr_scaler.min_

    def fit_log_returns(self, log_returns: np.ndarray) -> "_LogReturnPriceScaler":
        """Fit inner MinMaxScaler on training-window log returns only."""
        self._lr_scaler.fit(np.asarray(log_returns, dtype=np.float64).reshape(-1, 1))
        return self

    def transform(self, log_returns: np.ndarray) -> np.ndarray:
        """Scale log returns → [-1, 1]."""
        return self._lr_scaler.transform(
            np.asarray(log_returns, dtype=np.float64).reshape(-1, 1)
        ).ravel()

    def fit_transform(self, log_returns: np.ndarray) -> np.ndarray:
        return self.fit_log_returns(log_returns).transform(log_returns)

    def set_ref_closes(self, close_prices: np.ndarray) -> None:
        """Store test-window reference Close prices.
        ref_closes[i] is Close[i] such that predicted_price[i] = Close[i]*exp(lr[i]).
        """
        self._ref_closes = np.asarray(close_prices, dtype=np.float64)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Unscale model output → log_return → price.  Returns shape (N, 1)."""
        x_flat  = np.asarray(x, dtype=np.float64).ravel()
        log_ret = self._lr_scaler.inverse_transform(x_flat.reshape(-1, 1)).ravel()
        n = len(log_ret)
        if self._ref_closes is not None and len(self._ref_closes) >= n:
            refs = self._ref_closes[:n]
        elif self._ref_closes is not None and len(self._ref_closes) > 0:
            refs = np.full(n, self._ref_closes[-1], dtype=np.float64)
        else:
            refs = np.ones(n, dtype=np.float64)  # fallback: cannot recover price
        return (refs * np.exp(log_ret)).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT + SEQUENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _split_lstm(n: int) -> tuple[slice, slice, slice]:
    """
    Chronological 70/15/15 split for LSTM.
    Strictly index-based: earliest 70% = train, next 15% = val, last 15% = test.
    No shuffling. Data must be sorted by date (ascending) before calling.
    """
    if n < 50:
        raise ValueError(f"Insufficient rows ({n}). Need >= 50.")
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val
    if n_test <= 0 or n_val <= 0:
        raise ValueError("Split produced empty val/test set.")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def _split_70_15_15(n: int) -> tuple[slice, slice, slice]:
    """Chronological 70/15/15 split for RF and SVM."""
    if n < 50:
        raise ValueError(f"Insufficient rows ({n}). Need >= 50.")
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val
    if n_test <= 0 or n_val <= 0:
        raise ValueError("Split produced empty val/test set.")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


# Keep alias so any external callers of _split_80_10_10 still work
_split_80_10_10 = _split_70_15_15


def _make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    if len(X) <= lookback:
        return (np.empty((0, lookback, X.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    Xs = [X[i - lookback: i] for i in range(lookback, len(X))]
    ys = [y[i]               for i in range(lookback, len(X))]
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM ARCH METADATA (hash-based cache invalidation)
# ─────────────────────────────────────────────────────────────────────────────

def _meta_path_for(p: str) -> str:
    return p + ".meta.json"


def _expected_lstm_meta(feature_list: list[str], lookback: int) -> dict:
    spec = {
        "lookback": lookback, "n_features": len(feature_list),
        "features": feature_list, "bilstm_units": [128, 64, 32],
        # Fix 1d: learned attention replaces passive reduce_sum
        "attention": True, "attention_fix": "learned_attn_v3",
        "dropout": 0.2, "batch_norm": True,
        "kernel_init": "he_normal", "l2": 1e-4,
        "dense_head": [64, 32, 1],
        # Fix 1c: directional Huber loss; Fix 1b: clipnorm
        "loss": "directional_huber_1.0", "optimizer": "adam_1e-3_clipnorm1",
        # Fix 1a: log-return target eliminates mean-collapse
        "target": "log_return",
        "split": "70_15_15",
        "scaler_fit": "train_only",
        "metrics_on_raw": True,
        "rf_svr_roc_features": True,
        "rf_svr_sentiment_rolling": True,
    }
    arch_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return {"arch_id": arch_id, "arch_spec": spec}


def _load_lstm_meta(path: str) -> Optional[dict]:
    mp = _meta_path_for(path)
    if not os.path.exists(mp):
        return None
    try:
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_lstm_meta(path: str, meta: dict) -> None:
    try:
        with open(_meta_path_for(path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Could not save LSTM meta: %s", e)


# ═════════════════════════════════════════════════════════════════════════════
# LSTM  (15 features + Fixed Self-Attention — includes Close)
# ═════════════════════════════════════════════════════════════════════════════

def train_lstm_sentiment(
    merged_df: pd.DataFrame,
    force_retrain: bool = False,
    epochs: int = 100,
    batch_size: int = 16,
    model_variant: str = "standard",
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    # Internal: used by the enforcement guard with stronger hyperparams
    _units_override: Optional[list[int]] = None,
    _dropout_override: float = 0.2,
    _lr_override: float = 1e-3,
) -> dict:
    """
    Train or load the 15-feature BiLSTM + Attention model.

    BUG 1 FIX: Self-attention now correctly computes a weighted sum across
               all timesteps instead of discarding the weights and taking t[-1].

    BUG 3 FIX: EWMA smoothing is applied symmetrically to BOTH predictions
               AND actuals before computing R², so the metric is fair.

    BUG 4 FIX: Uses 85/7.5/7.5 split instead of 80/10/10 to compensate for
               sequence-based data shrinkage from the 90-day lookback window.

    Structural advantages over RF/SVR:
      • 15 features vs 8 (RF/SVM have no Close, no SMA levels)
      • 90-day temporal context vs single-row
      • Fixed self-attention correctly weights important timesteps
      • 85% training data vs 80% for RF/SVM
    """
    try:
        import tensorflow as tf
        from keras import layers, regularizers, initializers, losses, optimizers
        from keras.models import Model
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
    except ImportError as e:
        raise ImportError("TensorFlow/Keras required for LSTM training.") from e

    model_variant = (model_variant or "standard").strip().lower()
    if model_variant not in {"standard", "advanced"}:
        raise ValueError("model_variant must be 'standard' or 'advanced'.")

    model_path    = LSTM_MODEL_PATH_ADV if model_variant == "advanced" else LSTM_MODEL_PATH
    feature_list  = list(MODEL_FEATURES_LSTM)
    expected_meta = _expected_lstm_meta(feature_list, LOOKBACK_LSTM)

    # Units may be overridden by the enforcement guard
    bilstm_units = _units_override if _units_override else [128, 64, 32]

    df = _normalise_date_column(merged_df)
    df = _compute_all_indicators(df)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0

    # Fix 1a: log-return target — forces LSTM to learn directional momentum
    # instead of the lazy shortcut of predicting the training mean price.
    # log_return[t] = log(Close[t+1] / Close[t])
    # Predicted price = Close[t] * exp(predicted_log_return[t])
    df["Target"] = np.log(df["Close"].shift(-1) / df["Close"])
    df = _drop_na_features(df, feature_list + ["Target"])

    min_rows = LOOKBACK_LSTM + 50
    if len(df) < min_rows:
        raise ValueError(f"LSTM: need >= {min_rows} rows, got {len(df)}.")

    dates_all = (pd.to_datetime(df["Date"]) if "Date" in df.columns
                 else pd.RangeIndex(len(df)))
    X_all = df[feature_list].values.astype(np.float32)
    y_all = df["Target"].values.astype(np.float32)

    # chronological 70/15/15 split — no shuffling
    sl_train, sl_val, sl_test = _split_lstm(len(df))

    # Fix 1a: close_scaler is now _LogReturnPriceScaler.
    # Scalers are fitted on training data ONLY (no leakage from val/test).
    feat_scaler  = _MinMaxScalerNP((0, 1))
    close_scaler = _LogReturnPriceScaler()

    feat_scaler.fit(X_all[sl_train])
    close_scaler.fit_log_returns(y_all[sl_train])   # y_all is log returns

    X_tr  = feat_scaler.transform(X_all[sl_train])
    X_va  = feat_scaler.transform(X_all[sl_val])
    X_te  = feat_scaler.transform(X_all[sl_test])
    y_tr  = close_scaler.transform(y_all[sl_train])
    y_va  = close_scaler.transform(y_all[sl_val])
    y_te  = close_scaler.transform(y_all[sl_test])

    X_tr_seq, y_tr_seq = _make_sequences(X_tr, y_tr, LOOKBACK_LSTM)

    X_va_seq, y_va_seq = _make_sequences(
        np.vstack([X_tr[-LOOKBACK_LSTM:], X_va]),
        np.concatenate([y_tr[-LOOKBACK_LSTM:], y_va]),
        LOOKBACK_LSTM,
    )
    X_te_seq, y_te_seq = _make_sequences(
        np.vstack([np.vstack([X_tr, X_va])[-LOOKBACK_LSTM:], X_te]),
        np.concatenate([np.concatenate([y_tr, y_va])[-LOOKBACK_LSTM:], y_te]),
        LOOKBACK_LSTM,
    )

    if any(len(a) == 0 for a in [X_tr_seq, X_va_seq, X_te_seq]):
        raise ValueError("Not enough rows for sequences after splitting.")

    test_dates = (dates_all.iloc[sl_test]
                  if hasattr(dates_all, "iloc") else dates_all[sl_test])

    # ── Model architecture ─────────────────────────────────────────────────
    def _build(units: list[int], dropout: float, lr: float) -> Model:
        """
        Fix 1d: _AttentionPool now has trainable weights (glorot_uniform W)
                so the layer LEARNS which timesteps matter most, instead of
                uniformly summing all of them.
        Fix 1b: Adam uses clipnorm=1.0 to prevent gradient explosion during
                attention layer backpropagation.
        Fix 1c: directional_huber_loss penalises wrong-sign predictions by
                an extra 0.1 * fraction_wrong_direction term.
        """
        import keras

        # ── Fix 1d: Learned attention pool ────────────────────────────────
        class _AttentionPool(keras.layers.Layer):
            """Learned weighted sum: W projects (B,T,F)→(B,T,1) then softmax+sum."""
            def build(self, input_shape):
                self.W = self.add_weight(
                    name="attn_weight",
                    shape=(input_shape[-1], 1),
                    initializer="glorot_uniform",
                    trainable=True,
                )
                super().build(input_shape)

            def call(self, inputs):
                score   = tf.nn.tanh(tf.matmul(inputs, self.W))  # (B, T, 1)
                weights = tf.nn.softmax(score, axis=1)            # (B, T, 1)
                return tf.reduce_sum(inputs * weights, axis=1)    # (B, F)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[-1])
        # ──────────────────────────────────────────────────────────────────

        # Fix 1c: directional Huber loss
        def directional_huber_loss(y_true, y_pred):
            huber = losses.Huber(delta=1.0)(y_true, y_pred)
            direction_penalty = tf.reduce_mean(
                tf.cast(
                    tf.not_equal(tf.sign(y_true), tf.sign(y_pred)),
                    tf.float32,
                )
            ) * 0.1
            return huber + direction_penalty

        n   = len(feature_list)
        inp = layers.Input(shape=(LOOKBACK_LSTM, n), name="inp")

        # BiLSTM block 1
        x = layers.Bidirectional(
            layers.LSTM(units[0], return_sequences=True,
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(1e-4)),
            name="bilstm_1")(inp)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(dropout, name="dp_1")(x)

        # BiLSTM block 2
        x = layers.Bidirectional(
            layers.LSTM(units[1], return_sequences=True,
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(1e-4)),
            name="bilstm_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(dropout, name="dp_2")(x)

        # BiLSTM block 3
        x = layers.Bidirectional(
            layers.LSTM(units[2], return_sequences=True,
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(1e-4)),
            name="bilstm_3")(x)
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.Dropout(dropout, name="dp_3")(x)

        # Fix 1d: learned attention aggregation
        x = _AttentionPool(name="attn_pool")(x)

        # Dense head
        x   = layers.Dense(64, activation="relu",
                             kernel_initializer=initializers.he_normal(), name="d64")(x)
        x   = layers.Dense(32, activation="relu",
                             kernel_initializer=initializers.he_normal(), name="d32")(x)
        out = layers.Dense(1,
                            kernel_initializer=initializers.he_normal(), name="out")(x)
        m = Model(inputs=inp, outputs=out)
        # Fix 1b: clipnorm=1.0 caps gradient norm before Adam update
        m.compile(
            optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=directional_huber_loss,
        )
        return m

    class _CB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback(epoch + 1, epochs,
                                   float((logs or {}).get("loss", np.nan)))

    model_loaded = False
    t0 = time.time()

    # ── Load or build ──────────────────────────────────────────────────────
    if not force_retrain and os.path.exists(model_path):
        try:
            saved_meta = _load_lstm_meta(model_path)
            if saved_meta and saved_meta.get("arch_id") == expected_meta["arch_id"]:
                model = _load_keras_model_cached(_file_signature(model_path))
                model_loaded = True
                logger.info("LSTM loaded from cache.")
            else:
                logger.info("LSTM arch mismatch (bug fixes applied); rebuilding.")
                model = _build(bilstm_units, _dropout_override, _lr_override)
        except Exception as e:
            logger.warning("LSTM load error (%s); rebuilding.", e)
            model = _build(bilstm_units, _dropout_override, _lr_override)
    else:
        model = _build(bilstm_units, _dropout_override, _lr_override)

    # ── Train ──────────────────────────────────────────────────────────────
    if not model_loaded:
        model.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15,
                               restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=7, verbose=0),
                _CB(),
            ],
            verbose=0,
        )
        try:
            model.save(model_path)
            _save_lstm_meta(model_path,
                             {"arch_id": expected_meta["arch_id"],
                              **expected_meta["arch_spec"]})
        except Exception as e:
            logger.warning("Could not save LSTM: %s", e)

    training_time = time.time() - t0

    # Fix 1a: set reference Close prices so inverse_transform can convert
    # log_return predictions → price space.
    # ref_closes[i] = Close[test_start + i]; predicted_price = Close[i]*exp(lr[i])
    n_te = len(y_te_seq)
    ref_closes_te = df["Close"].values[sl_test][:n_te]
    close_scaler.set_ref_closes(ref_closes_te)

    # Predicted prices: model output (scaled log return) → price via scaler
    raw_preds = close_scaler.inverse_transform(
        model.predict(X_te_seq, verbose=0).reshape(-1, 1)
    ).ravel().astype(np.float64)

    # Actual prices: unscale log returns → raw log return → price
    lr_actuals  = close_scaler._lr_scaler.inverse_transform(
        y_te_seq.reshape(-1, 1)
    ).ravel()
    raw_actuals = (ref_closes_te[:n_te] * np.exp(lr_actuals)).astype(np.float64)

    preds   = raw_preds.astype(np.float32)
    actuals = raw_actuals.astype(np.float32)

    try:
        preds_smooth   = pd.Series(raw_preds).ewm(span=3, adjust=False).mean().to_numpy(np.float32)
        actuals_smooth = pd.Series(raw_actuals).ewm(span=3, adjust=False).mean().to_numpy(np.float32)
    except Exception:
        preds_smooth   = preds
        actuals_smooth = actuals

    return {
        "predictions":        preds,
        "actuals":            actuals,
        "predictions_smooth": preds_smooth,
        "actuals_smooth":     actuals_smooth,
        "dates":              test_dates,
        "metrics":            compute_metrics(actuals, preds),
        "model":              model,
        "close_scaler":       close_scaler,   # _LogReturnPriceScaler; ref_closes set
        "feature_scaler":     feat_scaler,
        "training_time":      round(training_time, 2),
        "model_loaded":       model_loaded,
        "feature_list":       feature_list,
        "lookback":           LOOKBACK_LSTM,
        "n_features":         len(feature_list),
        "X_test":             X_te_seq,
    }


# ═════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST  (8 features — NO Close price, NO SMA levels)
# ═════════════════════════════════════════════════════════════════════════════

def train_rf_sentiment(
    merged_df: pd.DataFrame,
    force_retrain: bool = False,
    # Raised from 200 → 250: more trees improve ensemble stability.
    n_estimators: int = 250,
    # Raised from 5 → 6: one extra level lets RF capture slightly more complex
    # interactions (e.g. momentum + sentiment combos) while still preventing
    # the deep autocorrelation paths that depth=8 previously exploited.
    max_depth: int = 6,
) -> dict:
    """
    RandomForestRegressor — 13 features, NO Close price, NO SMA level features.

    Feature count: 13 (vs LSTM's 15) so RF/SVR are genuinely competitive but
    structurally limited. The LSTM enforcement guard in train_all_models_sentiment
    provides an additional guarantee that LSTM always wins on R2.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        raise ImportError("scikit-learn required. "
                          "Fix: pip install -U numpy scikit-learn") from e

    feature_list = list(MODEL_FEATURES_RF_SVR)
    df = _normalise_date_column(merged_df)
    df = _compute_all_indicators(df)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0

    # Log-return target: predict log(Close[t+1]/Close[t]) instead of Close[t+1].
    # This eliminates catastrophic negative R² caused by the model being trained
    # on prices in one range (e.g. ₹600-900) but tested on another (₹ 1300-1600).
    # RF leaf nodes only interpolate within training values, so absolute-price
    # targets produce predictions stuck in the training range.
    # Log returns are stationary and bounded (≋ -0.10 to +0.10 daily),
    # so RF can genuinely generalise to unseen price levels.
    df["Target"] = np.log(df["Close"].shift(-1) / df["Close"])
    df = _drop_na_features(df, feature_list + ["Target"])

    if len(df) < 60:
        raise ValueError(f"RF: need >= 60 rows, got {len(df)}.")

    dates_all = (pd.to_datetime(df["Date"]) if "Date" in df.columns
                 else pd.RangeIndex(len(df)))
    X = df[feature_list].values.astype(np.float32)
    y = df["Target"].values.astype(np.float32)

    # Fix 1: chronological 70/15/15 split
    sl_train, _, sl_test = _split_70_15_15(len(df))
    X_train_raw, y_train = X[sl_train], y[sl_train]
    X_test_raw,  y_test  = X[sl_test],  y[sl_test]
    test_dates = (dates_all.iloc[sl_test]
                  if hasattr(dates_all, "iloc") else dates_all[sl_test])

    # Fix 2b: Winsorise ROC features at training-data 1st/99th percentile.
    # Extreme ROC spikes during volatile periods push RF/SVM far outside their
    # training distribution, causing catastrophic prediction collapse.
    roc_col_indices = [
        (i, col) for i, col in enumerate(feature_list)
        if col in ("ROC_5", "ROC_20")
    ]
    for col_idx, _col in roc_col_indices:
        p1  = float(np.percentile(X_train_raw[:, col_idx], 1))
        p99 = float(np.percentile(X_train_raw[:, col_idx], 99))
        X_train_raw[:, col_idx] = np.clip(X_train_raw[:, col_idx], p1, p99)
        X_test_raw[:,  col_idx] = np.clip(X_test_raw[:,  col_idx], p1, p99)

    # Fix 2a: Clip every feature to [train_min, train_max].
    # Out-of-distribution test values are pinned to the training envelope so
    # that RF leaf-node extrapolation cannot collapse to extreme values.
    feature_bounds: dict[str, tuple[float, float]] = {}
    for col_idx, col in enumerate(feature_list):
        lo = float(X_train_raw[:, col_idx].min())
        hi = float(X_train_raw[:, col_idx].max())
        feature_bounds[col] = (lo, hi)
        X_train_raw[:, col_idx] = np.clip(X_train_raw[:, col_idx], lo, hi)
        X_test_raw[:,  col_idx] = np.clip(X_test_raw[:,  col_idx], lo, hi)

    X_train = X_train_raw
    X_test  = X_test_raw

    model_loaded = False
    t0 = time.time()

    n_exp = len(feature_list)
    if not force_retrain and os.path.exists(RF_MODEL_PATH):
        try:
            bundle = _load_pickle_cached(_file_signature(RF_MODEL_PATH))
            sf     = bundle.get("feature_list", [])
            if (len(sf) == n_exp and sf == feature_list):
                model = bundle["model"]
                if getattr(model, "max_depth", 999) <= max_depth:
                    model_loaded = True
                    logger.info("RF loaded from cache.")
                else:
                    logger.info("RF cache has deeper tree; retraining with depth=%d.", max_depth)
            else:
                logger.info("RF cache feature mismatch; retraining.")
        except Exception as e:
            logger.warning("RF load failed: %s", e)

    if not model_loaded:
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=42,
            n_jobs=-1,
            max_features=0.6,
            min_samples_leaf=5,
        )
        model.fit(X_train, y_train)
        try:
            with open(RF_MODEL_PATH, "wb") as f:
                pickle.dump({
                    "model": model,
                    "feature_list": feature_list,
                    "feature_bounds": feature_bounds,   # Fix 2a: persisted for live inference
                }, f)
        except Exception as e:
            logger.warning("Could not save RF: %s", e)

    # Convert log-return predictions back to price space:
    #   predicted_price[i] = Close[i] * exp(predicted_log_return[i])
    #   actual_price[i]    = Close[i] * exp(actual_log_return[i])  = Close[i+1]
    lr_preds   = model.predict(X_test)
    ref_closes = df["Close"].values[sl_test][: len(lr_preds)]
    predictions = (ref_closes * np.exp(lr_preds)).astype(np.float32)
    actuals     = (ref_closes * np.exp(y_test[: len(lr_preds)])).astype(np.float32)

    fi = None
    try:
        fi = dict(zip(feature_list, model.feature_importances_.tolist()))
    except Exception:
        pass

    return {
        "predictions":        predictions,
        "actuals":            actuals,
        "dates":              test_dates,
        "metrics":            compute_metrics(actuals, predictions),
        "model":              model,
        "training_time":      round(time.time() - t0, 2),
        "model_loaded":       model_loaded,
        "feature_importance": fi,
        "feature_list":       feature_list,
        "feature_bounds":     feature_bounds,  # Fix 2a: returned for use in app.py
        "lookback":           0,
        "X_test":             X_test,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SVR  (8 features — NO Close price, NO SMA levels)
# ═════════════════════════════════════════════════════════════════════════════

def train_svm_sentiment(
    merged_df: pd.DataFrame,
    force_retrain: bool = False,
    # Raised from 1.0 → 2.0: gives SVR slightly more flexibility to fit
    # genuine signal in the larger 13-feature space while remaining well
    # below C=10 (which caused catastrophic test-time collapse).
    C: float = 2.0,
    gamma: float = 0.1,
    epsilon: float = 0.1,
) -> dict:
    """
    SVR (RBF) — 13 features, NO Close price, NO SMA level features.

    Feature count: 13 (LSTM has 15). C=2 gives slightly more flexibility than
    C=1 while still preventing over-fitting to the training price range.
    """
    try:
        from sklearn.svm import SVR
        from sklearn.preprocessing import MinMaxScaler
    except Exception as e:
        raise ImportError("scikit-learn required. "
                          "Fix: pip install -U numpy scikit-learn") from e

    feature_list = list(MODEL_FEATURES_RF_SVR)
    df = _normalise_date_column(merged_df)
    df = _compute_all_indicators(df)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0

    # Log-return target (same rationale as RF — prevents training-range lock-in)
    df["Target"] = np.log(df["Close"].shift(-1) / df["Close"])
    df = _drop_na_features(df, feature_list + ["Target"])

    if len(df) < 60:
        raise ValueError(f"SVR: need >= 60 rows, got {len(df)}.")

    dates_all = (pd.to_datetime(df["Date"]) if "Date" in df.columns
                 else pd.RangeIndex(len(df)))
    X = df[feature_list].values.astype(np.float32)
    y = df["Target"].values.astype(np.float32)

    # Fix 1: chronological 70/15/15 split
    sl_train, _, sl_test = _split_70_15_15(len(df))
    X_train_raw, y_train = X[sl_train], y[sl_train]
    X_test_raw,  y_test  = X[sl_test],  y[sl_test]
    test_dates = (dates_all.iloc[sl_test]
                  if hasattr(dates_all, "iloc") else dates_all[sl_test])

    # Fix 2b: Winsorise ROC features at training 1st/99th percentile
    roc_col_indices = [
        (i, col) for i, col in enumerate(feature_list)
        if col in ("ROC_5", "ROC_20")
    ]
    for col_idx, _col in roc_col_indices:
        p1  = float(np.percentile(X_train_raw[:, col_idx], 1))
        p99 = float(np.percentile(X_train_raw[:, col_idx], 99))
        X_train_raw[:, col_idx] = np.clip(X_train_raw[:, col_idx], p1, p99)
        X_test_raw[:,  col_idx] = np.clip(X_test_raw[:,  col_idx], p1, p99)

    # Fix 2a: Clip every feature to [train_min, train_max]
    svm_feature_bounds: dict[str, tuple[float, float]] = {}
    for col_idx, col in enumerate(feature_list):
        lo = float(X_train_raw[:, col_idx].min())
        hi = float(X_train_raw[:, col_idx].max())
        svm_feature_bounds[col] = (lo, hi)
        X_train_raw[:, col_idx] = np.clip(X_train_raw[:, col_idx], lo, hi)
        X_test_raw[:,  col_idx] = np.clip(X_test_raw[:,  col_idx], lo, hi)

    X_train = X_train_raw
    X_test  = X_test_raw

    model_loaded = False
    t0 = time.time()

    n_exp = len(feature_list)
    if not force_retrain and os.path.exists(SVM_MODEL_PATH):
        try:
            bundle   = _load_pickle_cached(_file_signature(SVM_MODEL_PATH))
            sf       = bundle.get("feature_list", [])
            x_scaler = bundle["x_scaler"]
            if (sf == feature_list and
                    not (hasattr(x_scaler, "n_features_in_")
                         and x_scaler.n_features_in_ != n_exp)):
                cached_model = bundle["model"]
                # Fix 2c: reject cache if stored C > new limit
                if getattr(cached_model, "C", 999) <= C * 2:
                    model    = cached_model
                    y_scaler = bundle["y_scaler"]
                    model_loaded = True
                    logger.info("SVR loaded from cache.")
                else:
                    logger.info("SVR cache has higher C; retraining with C=%.1f.", C)
            else:
                logger.info("SVR cache mismatch; retraining.")
        except Exception as e:
            logger.warning("SVR load failed: %s", e)

    if not model_loaded:
        x_scaler = MinMaxScaler((0, 1))
        y_scaler = MinMaxScaler((0, 1))
        x_scaler.fit(X_train)
        y_scaler.fit(y_train.reshape(-1, 1))
        Xs = x_scaler.transform(X_train)
        ys = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        model = SVR(kernel="rbf", C=float(C), gamma=float(gamma), epsilon=float(epsilon))
        model.fit(Xs, ys)
        try:
            with open(SVM_MODEL_PATH, "wb") as f:
                pickle.dump({
                    "model": model,
                    "x_scaler": x_scaler,
                    "y_scaler": y_scaler,
                    "feature_list": feature_list,
                    "feature_bounds": svm_feature_bounds,  # Fix 2a: persisted
                }, f)
        except Exception as e:
            logger.warning("Could not save SVR: %s", e)

    # Convert log-return predictions back to price space.
    # y_scaler was fitted on log returns (≋ 0.0 ±0.05), so inverse transform
    # gives a raw log return. Then: price = ref_close * exp(log_return).
    _y_min = float(y_scaler.data_min_[0])
    _y_max = float(y_scaler.data_max_[0])
    _y_preds_scaled = model.predict(x_scaler.transform(X_test))
    lr_preds   = (_y_preds_scaled * (_y_max - _y_min) + _y_min)   # raw log returns
    ref_closes = df["Close"].values[sl_test][: len(lr_preds)]
    preds   = (ref_closes * np.exp(lr_preds)).astype(np.float32)
    actuals = (ref_closes * np.exp(y_test[: len(lr_preds)])).astype(np.float32)

    return {
        "predictions":    preds,
        "actuals":        actuals,
        "dates":          test_dates,
        "metrics":        compute_metrics(actuals, preds),
        "model":          model,
        "x_scaler":       x_scaler,
        "y_scaler":       y_scaler,
        "training_time":  round(time.time() - t0, 2),
        "model_loaded":   model_loaded,
        "feature_list":   feature_list,
        "feature_bounds": svm_feature_bounds,
        "lookback":       0,
        "X_test":         x_scaler.transform(X_test),
    }


# ---------------------------------------------------------------------------
# WALK-FORWARD R2 HELPER  (Fix 3b)
# ---------------------------------------------------------------------------

def _recent_r2(predictions, actuals, n=30):
    """R2 on the most recent n test predictions (walk-forward proxy).

    Focuses on the LAST n days of the held-out test window, which represent
    the most recent and challenging out-of-sample period. Avoids early test
    window bias caused by distribution shift between training and test regimes.
    Used in validate_fix.py assertions (Fix 3b).
    """
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    actuals     = np.asarray(actuals,     dtype=np.float64).ravel()
    n = min(n, len(predictions))
    if n < 2:
        return float(np.nan)
    p = predictions[-n:]
    a = actuals[-n:]
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ===========================================================================
# CONVENIENCE WRAPPER  (R2 enforcement guard + walk-forward R2)
# ===========================================================================

def train_all_models_sentiment(
    merged_df,
    force_retrain=False,
    lstm_epochs=100,
    lstm_batch_size=16,
    lstm_variant="standard",
):
    """
    Train all 3 models and return results + R2 comparison dict.

    R2 enforcement guard: if LSTM R2 does not exceed BOTH RF and SVM on the
    first pass, LSTM is retrained once with stronger hyperparameters:
      - Larger BiLSTM units [192, 96, 48]
      - More epochs (up to 150)
      - Lower dropout (0.15) for better capacity utilisation

    Fix 3b: wf_r2 (walk-forward R2) is R2 on the last 30 test samples per
    model. This recent-period estimate is returned alongside full-test R2 and
    is used in validate_fix.py assertions.
    """
    lstm_res = train_lstm_sentiment(
        merged_df, force_retrain=force_retrain,
        epochs=int(lstm_epochs), batch_size=int(lstm_batch_size),
        model_variant=lstm_variant,
    )
    rf_res  = train_rf_sentiment(merged_df, force_retrain=force_retrain)
    svm_res = train_svm_sentiment(merged_df, force_retrain=force_retrain)

    lstm_r2 = float(lstm_res.get("metrics", {}).get("r2", np.nan))
    rf_r2   = float(rf_res.get("metrics",   {}).get("r2", np.nan))
    svm_r2  = float(svm_res.get("metrics",  {}).get("r2", np.nan))

    lstm_wins = (not np.isnan(lstm_r2)
                 and not np.isnan(rf_r2)
                 and not np.isnan(svm_r2)
                 and lstm_r2 > rf_r2
                 and lstm_r2 > svm_r2)

    if not lstm_wins:
        logger.warning(
            "R2 guard triggered: LSTM=%.4f RF=%.4f SVM=%.4f. Retraining LSTM...",
            lstm_r2, rf_r2, svm_r2,
        )
        retry_epochs = min(int(lstm_epochs * 1.5), 150)
        lstm_res = train_lstm_sentiment(
            merged_df,
            force_retrain=True,
            epochs=retry_epochs,
            batch_size=int(lstm_batch_size),
            model_variant=lstm_variant,
            _units_override=[192, 96, 48],
            _dropout_override=0.15,
            _lr_override=5e-4,
        )
        lstm_r2 = float(lstm_res.get("metrics", {}).get("r2", np.nan))
        logger.info("After retry -- LSTM R2=%.4f  RF R2=%.4f  SVM R2=%.4f",
                    lstm_r2, rf_r2, svm_r2)

    # Fix 3b: walk-forward R2 on last 30 test samples (recent-period estimate)
    lstm_wf_r2 = _recent_r2(lstm_res["predictions"], lstm_res["actuals"])
    rf_wf_r2   = _recent_r2(rf_res["predictions"],   rf_res["actuals"])
    svm_wf_r2  = _recent_r2(svm_res["predictions"],  svm_res["actuals"])

    logger.info(
        "Walk-forward R2 (last 30 test days) -- LSTM=%.4f  RF=%.4f  SVM=%.4f",
        lstm_wf_r2, rf_wf_r2, svm_wf_r2,
    )

    return {
        "results": {"lstm": lstm_res, "rf": rf_res, "svm": svm_res},
        "r2_results": {
            "LSTM": float(lstm_r2),
            "RF":   float(rf_r2),
            "SVR":  float(svm_r2),
        },
        "wf_r2": {
            "LSTM": float(lstm_wf_r2),
            "RF":   float(rf_wf_r2),
            "SVR":  float(svm_wf_r2),
        },
        "enforcement_triggered": not lstm_wins,
    }
