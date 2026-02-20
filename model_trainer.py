"""
model_trainer.py
================
Encapsulates all model training and prediction logic for the three hybrid
models (LSTM + Sentiment, Random Forest + Sentiment, SVM + Sentiment).

Each trainer:
  - Accepts a merged stock + sentiment DataFrame
  - Handles feature engineering
  - Trains (or loads a cached) model
  - Returns predictions and evaluation metrics

Author: Upgraded Stock Prediction App
"""

from __future__ import annotations

import os
import time
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Paths for saved models ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "LSTM_sentiment_model.keras")
RF_MODEL_PATH   = os.path.join(BASE_DIR, "RF_sentiment_model.pkl")
SVM_MODEL_PATH  = os.path.join(BASE_DIR, "SVM_sentiment_model.pkl")

# ── Feature lists ─────────────────────────────────────────────────────────────
PRICE_FEATURES  = ["Open", "High", "Low", "Close", "Volume"]
SENTIMENT_ONLY  = ["Daily_Sentiment_Score"]
MA_FEATURES     = ["MA_100", "MA_200"]
ROLLING_SENT    = ["Sentiment_Rolling_7d_Avg"]

LSTM_FEATURES = PRICE_FEATURES + SENTIMENT_ONLY          # 6 features
RF_SVM_FEATURES = (
    PRICE_FEATURES + MA_FEATURES + SENTIMENT_ONLY + ROLLING_SENT
)  # 9 features

LOOKBACK = 100   # Number of past days used as context for LSTM


# ──────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a 'Date' column (plain datetime, tz-naive).

    Handles three incoming shapes:
    1. DatetimeIndex  → reset_index() → 'Date' column
    2. Integer index with 'Date' column already present
    3. Integer index without 'Date' column (adds a placeholder range index)
    """
    df = df.copy()
    # Case 1: DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = df.reset_index()          # index → 'Date' column (or 'index')
        # reset_index names the level by the index.name; normalise to 'Date'
        if "index" in df.columns and "Date" not in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
    # Case 2: already has 'Date'
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _compute_mas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 100-day and 200-day moving average columns to a OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'MA_100' and 'MA_200' columns appended.
    """
    df = df.copy()
    df["MA_100"] = df["Close"].rolling(window=100).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    return df


def _drop_na_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Drop rows with NaN in any of the given feature columns.
    
    Always returns a DataFrame with a plain integer RangeIndex.
    The original 'Date' column (if present) is preserved as a data column.
    """
    # Only drop=True we preserve 'Date' as a regular column (not in index)
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predicted values (original scale, not normalised).

    Returns
    -------
    dict with keys: mae, mse, rmse, r2
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  LSTM + SENTIMENT MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_sentiment(
    merged_df: pd.DataFrame,
    force_retrain: bool = False,
    epochs: int = 30,
    batch_size: int = 32,
    progress_callback=None,
) -> dict:
    """
    Train (or load) a hybrid LSTM model that uses price + sentiment features.

    Architecture
    ────────────
    Input:  (samples, LOOKBACK, 6)   ← 6 features per timestep
    LSTM(128, return_sequences=True)
    Dropout(0.2)
    LSTM(64, return_sequences=False)
    Dropout(0.2)
    Dense(32, relu)
    Dense(1)

    Parameters
    ----------
    merged_df : pd.DataFrame
        Stock + sentiment DataFrame (output of merge_sentiment_with_stock).
    force_retrain : bool
        If True, always retrain even if a saved model exists.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    progress_callback : callable, optional
        Called each epoch with (epoch, total_epochs, loss).

    Returns
    -------
    dict with keys:
        predictions   : np.ndarray (N,)  — inverse-scaled predictions
        actuals       : np.ndarray (N,)  — inverse-scaled actuals
        dates         : pd.DatetimeIndex — corresponding dates
        metrics       : dict             — mae, mse, rmse, r2
        training_time : float            — seconds
        model_loaded  : bool             — True if loaded from file
    """
    try:
        from keras.models import Sequential, load_model
        from keras.layers import LSTM, Dense, Dropout
        from keras.callbacks import Callback
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as e:
        raise ImportError(
            "Keras / TensorFlow not found. Run: pip install tensorflow"
        ) from e

    # ── Feature preparation ──────────────────────────────────────────────────
    df = _normalise_date_column(merged_df)
    df = _compute_mas(df)

    # Ensure sentiment columns exist (default to 0 if missing)
    for col in SENTIMENT_ONLY + ROLLING_SENT:
        if col not in df.columns:
            df[col] = 0.0

    # Validate that all required feature columns are present
    missing_cols = [c for c in LSTM_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"LSTM: Missing required feature columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = _drop_na_features(df, LSTM_FEATURES + ["Close"])
    if len(df) < LOOKBACK + 20:
        raise ValueError(
            f"Not enough data after cleaning ({len(df)} rows). "
            f"Need at least {LOOKBACK + 20} rows. "
            f"Try a longer date range (minimum ~3 years recommended)."
        )

    feature_data = df[LSTM_FEATURES].values.astype(np.float32)
    close_data   = df["Close"].values.reshape(-1, 1).astype(np.float32)
    dates_all    = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.RangeIndex(len(df))

    n_features = feature_data.shape[1]   # should be len(LSTM_FEATURES)

    # ── Scaling ─────────────────────────────────────────────────────────────
    # IMPORTANT: Scalers are fit on the CURRENT dataset.  A saved model was
    # trained on a different stock/date range and uses different scale
    # parameters.  We ALWAYS fit fresh scalers, meaning the saved model is
    # only reusable for PREDICTION (not for accurate inverse-scaling).
    # Therefore we must retrain whenever the data differs significantly.
    feature_scaler = MinMaxScaler()
    close_scaler   = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(feature_data)
    scaled_close    = close_scaler.fit_transform(close_data)

    # ── Sequence building ────────────────────────────────────────────────────
    X, y, seq_dates = [], [], []
    for i in range(LOOKBACK, len(scaled_features)):
        X.append(scaled_features[i - LOOKBACK : i])
        y.append(scaled_close[i, 0])
        seq_dates.append(dates_all.iloc[i] if hasattr(dates_all, "iloc") else dates_all[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    seq_dates = pd.DatetimeIndex(seq_dates)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates      = seq_dates[split_idx:]

    model_loaded = False
    t0 = time.time()

    # ── Load or train ────────────────────────────────────────────────────────
    # Only reuse a saved model if:
    #   (a) force_retrain is False
    #   (b) the file exists
    #   (c) the saved model's input shape matches the current feature count
    if not force_retrain and os.path.exists(LSTM_MODEL_PATH):
        logger.info(f"Trying to load existing LSTM model from {LSTM_MODEL_PATH}")
        try:
            model = load_model(LSTM_MODEL_PATH)
            # Validate input shape matches current dataset
            saved_n_features = model.input_shape[-1]   # (None, LOOKBACK, n_features)
            if saved_n_features != n_features:
                logger.warning(
                    f"Saved LSTM expects {saved_n_features} features but current "
                    f"data has {n_features}. Retraining from scratch."
                )
                model_loaded = False
            else:
                model_loaded = True
                logger.info("Saved LSTM shape OK — reusing model.")
        except Exception as e:
            logger.warning(f"Could not load saved LSTM: {e}. Retraining …")
            model_loaded = False

    if not model_loaded:
        logger.info("Training LSTM from scratch …")
        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(LOOKBACK, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Progress-aware Keras callback (suppress verbose TF output)
        class ProgressCB(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_callback:
                    progress_callback(epoch + 1, epochs, logs.get("loss", 0))

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[ProgressCB()],
            verbose=0,
        )
        model.save(LSTM_MODEL_PATH)
        logger.info(f"LSTM model saved to {LSTM_MODEL_PATH}")

    training_time = time.time() - t0

    # ── Prediction & inverse-transform ───────────────────────────────────────
    preds_scaled = model.predict(X_test, verbose=0)
    preds = close_scaler.inverse_transform(preds_scaled).flatten()
    actuals = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = compute_metrics(actuals, preds)

    return {
        "predictions":   preds,
        "actuals":       actuals,
        "dates":         test_dates,
        "metrics":       metrics,
        "training_time": round(training_time, 2),
        "model_loaded":  model_loaded,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.  RANDOM FOREST + SENTIMENT MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_rf_sentiment(
    merged_df: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 10,
) -> dict:
    """
    Train a RandomForestRegressor on price + technical + sentiment features.

    Input features : Open, High, Low, Close, Volume, MA_100, MA_200,
                     Daily_Sentiment_Score, Sentiment_Rolling_7d_Avg
    Target          : Next-day Close price (shifted by 1)

    Parameters
    ----------
    merged_df : pd.DataFrame
        Stock + sentiment DataFrame.
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    dict with keys: predictions, actuals, dates, metrics, training_time
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler

    df = _normalise_date_column(merged_df)
    df = _compute_mas(df)

    for col in SENTIMENT_ONLY + ROLLING_SENT:
        if col not in df.columns:
            df[col] = 0.0

    # Target: next-day Close
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)
    df = _drop_na_features(df, RF_SVM_FEATURES + ["Target"])

    if len(df) < 50:
        raise ValueError(
            f"Not enough clean data for Random Forest ({len(df)} rows)."
        )

    dates_all = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.RangeIndex(len(df))

    X = df[RF_SVM_FEATURES].values.astype(np.float32)
    y = df["Target"].values.astype(np.float32)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = (
        dates_all.iloc[split_idx:]
        if hasattr(dates_all, "iloc")
        else dates_all[split_idx:]
    )

    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    training_time = time.time() - t0

    preds   = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    # Persist model
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return {
        "predictions":   preds,
        "actuals":       y_test,
        "dates":         pd.DatetimeIndex(test_dates),
        "metrics":       metrics,
        "training_time": round(training_time, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SVM / SVR + SENTIMENT MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_svm_sentiment(
    merged_df: pd.DataFrame,
    C: float = 100.0,
    gamma: float = 0.1,
    epsilon: float = 0.1,
) -> dict:
    """
    Train an SVR on price + technical + sentiment features.

    Input features : same as RF (9 features)
    Target          : Next-day Close price (shifted by 1)
    Scaling         : MinMaxScaler applied to all features AND target.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Stock + sentiment DataFrame.
    C, gamma, epsilon : float
        SVR hyperparameters.

    Returns
    -------
    dict with keys: predictions, actuals, dates, metrics, training_time
    """
    from sklearn.svm import SVR
    from sklearn.preprocessing import MinMaxScaler

    df = _normalise_date_column(merged_df)
    df = _compute_mas(df)

    for col in SENTIMENT_ONLY + ROLLING_SENT:
        if col not in df.columns:
            df[col] = 0.0

    df = df.copy()
    df["Target"] = df["Close"].shift(-1)
    df = _drop_na_features(df, RF_SVM_FEATURES + ["Target"])

    if len(df) < 50:
        raise ValueError(
            f"Not enough clean data for SVM ({len(df)} rows)."
        )

    dates_all = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.RangeIndex(len(df))

    X = df[RF_SVM_FEATURES].values.astype(np.float32)
    y = df["Target"].values.astype(np.float32)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_dates = (
        dates_all.iloc[split_idx:]
        if hasattr(dates_all, "iloc")
        else dates_all[split_idx:]
    )

    # Scale X and y independently
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s  = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    t0 = time.time()
    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train_s, y_train_s)
    training_time = time.time() - t0

    preds_scaled = model.predict(X_test_s)
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_test, preds)

    # Bundle scaler with model for later inference
    bundle = {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler}
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return {
        "predictions":   preds,
        "actuals":       y_test,
        "dates":         pd.DatetimeIndex(test_dates),
        "metrics":       metrics,
        "training_time": round(training_time, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CONVENIENCE: RUN ALL THREE MODELS
# ──────────────────────────────────────────────────────────────────────────────

def run_all_models(
    merged_df: pd.DataFrame,
    force_retrain_lstm: bool = False,
    lstm_epochs: int = 30,
    progress_callbacks: Optional[dict] = None,
) -> dict[str, dict]:
    """
    Train and evaluate all three hybrid models.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Stock + sentiment merged DataFrame.
    force_retrain_lstm : bool
        Whether to force-retrain the LSTM.
    lstm_epochs : int
        Training epochs for LSTM.
    progress_callbacks : dict, optional
        Keys: 'lstm', 'rf', 'svm' — each is a callable(current, total, ...).

    Returns
    -------
    dict with keys 'lstm', 'rf', 'svm' — each containing the trainer output.
    """
    callbacks = progress_callbacks or {}
    results: dict[str, dict] = {}

    # ── LSTM ──────────────────────────────────────────────────────────────────
    try:
        logger.info("Training LSTM + Sentiment model …")
        results["lstm"] = train_lstm_sentiment(
            merged_df,
            force_retrain=force_retrain_lstm,
            epochs=lstm_epochs,
            progress_callback=callbacks.get("lstm"),
        )
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        results["lstm"] = {"error": str(e)}

    # ── RF ────────────────────────────────────────────────────────────────────
    try:
        logger.info("Training Random Forest + Sentiment model …")
        results["rf"] = train_rf_sentiment(merged_df)
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        results["rf"] = {"error": str(e)}

    # ── SVM ───────────────────────────────────────────────────────────────────
    try:
        logger.info("Training SVM + Sentiment model …")
        results["svm"] = train_svm_sentiment(merged_df)
    except Exception as e:
        logger.error(f"SVM training failed: {e}")
        results["svm"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    import yfinance as yf
    from sentiment_utils import merge_sentiment_with_stock

    df = yf.download("TCS.NS", start="2020-01-01", end="2024-01-01")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()

    merged = merge_sentiment_with_stock(df.set_index("Date"), pd.Series(dtype=float))
    merged = merged.reset_index()

    print("Running all models on TCS.NS …")
    res = run_all_models(merged, force_retrain_lstm=True, lstm_epochs=5)

    for name, r in res.items():
        if "error" in r:
            print(f"  {name.upper()} → ERROR: {r['error']}")
        else:
            m = r["metrics"]
            print(
                f"  {name.upper()} → "
                f"MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
                f"R²={m['r2']:.4f}  time={r['training_time']}s"
            )
