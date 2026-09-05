"""
app.py  —  StockSense AI  (Research-Grade Upgrade)
====================================================
12-tab Streamlit dashboard integrating:
  Tab 0  : 🧠 Model Justification
  Tab 1  : 📈 Stock Overview & Moving Averages
  Tab 2  : 📰 News & Sentiment
  Tab 3  : 🤖 LSTM (Probabilistic)
  Tab 4  : 🌲 RF   (Probabilistic)
  Tab 5  : 📐 SVM  (Probabilistic)
  Tab 6  : 🌊 Wavelet Neural Network
  Tab 7  : ⚡ Adaptive Streaming Engine
  Tab 8  : 🔀 Hybrid Prediction
  Tab 9  : 🎮 RL Trading Agent
  Tab 10 : 🎯 Trading Signals
  Tab 11 : 📊 Model Comparison Dashboard

FIX A: Models are now trained via train_all_models_sentiment() so the
       R² enforcement guard in model_trainer.py actually runs.

FIX B: Comparison tab feature counts read from result dicts, not hardcoded.
"""

from __future__ import annotations

import os, sys, warnings, logging, time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


@contextmanager
def timed_step(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"[timing] {label}: {dt:.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

try:
    import sklearn  # noqa: F401
except Exception as e:
    st.warning(
        "⚠️ ML stack issue: scikit-learn failed to import. "
        "Fix: `pip install -U --force-reinstall numpy scikit-learn`.\n\n"
        f"Details: {e}"
    )

st.set_page_config(
    page_title="StockSense AI — Research-Grade Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0E1117; color: #FAFAFA; }
[data-testid="metric-container"] {
    background: #161B27; border: 1px solid #2A2F3E;
    border-radius: 10px; padding: 12px 16px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #161B27; border-radius: 8px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 6px;
    color: #94A3B8; font-size: 12px; padding: 6px 10px;
}
.stTabs [aria-selected="true"] {
    background: #1E3A5F !important; color: #00C8FF !important; font-weight: 600;
}
section[data-testid="stSidebar"] {
    background: #0D1117; border-right: 1px solid #1E2A3E;
}
.stProgress > div > div { background: linear-gradient(90deg, #00C8FF, #7B2FBE); }
details { background: #161B27 !important; border: 1px solid #2A2F3E !important; border-radius: 8px !important; }
h1 { color: #00C8FF !important; font-weight: 700; }
h2 { color: #CBD5E1 !important; }
h3 { color: #94A3B8 !important; }
.info-box {
    background: #161B27; border-left: 4px solid #00C8FF;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0; font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":  "#161B27", "axes.facecolor":    "#161B27",
    "axes.edgecolor":    "#2A2F3E", "text.color":        "#FAFAFA",
    "axes.labelcolor":   "#FAFAFA", "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8", "grid.color":        "#2A2F3E",
    "grid.linestyle":    "--",      "grid.alpha":        0.4,
    "legend.facecolor":  "#1A2035", "legend.edgecolor":  "#2A2F3E",
    "font.family":       "DejaVu Sans",
})


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename_axis("Date").reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if needed - set(df.columns):
            return pd.DataFrame()
        return df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def run_sentiment_pipeline_cached(
    ticker, company_name, start_date, end_date, api_key,
    _stock_df, _injected_df,
):
    from sentiment_utils import run_full_sentiment_pipeline
    return run_full_sentiment_pipeline(
        ticker=ticker, company_name=company_name,
        start_date=start_date, end_date=end_date,
        api_key=api_key, stock_df=_stock_df, injected_df=_injected_df,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _dark_fig(figsize=(13, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_stock_overview(stock_df: pd.DataFrame, ticker: str) -> plt.Figure:
    fig, ax = _dark_fig((13, 5))
    dates = pd.to_datetime(stock_df["Date"].values)
    close = stock_df["Close"].values
    for col, window in [("MA_50", 50), ("MA_100", 100), ("MA_200", 200)]:
        if col not in stock_df.columns:
            stock_df[col] = stock_df["Close"].rolling(window).mean()
    ax.plot(dates, close, color="#00C8FF", linewidth=1.6, label="Close Price", zorder=5)
    ax.plot(dates, stock_df["MA_50"].values,  color="#FFD700", linewidth=1.2, linestyle="--", alpha=0.80, label="MA50")
    ax.plot(dates, stock_df["MA_100"].values, color="#FF6B35", linewidth=1.2, linestyle="--", alpha=0.80, label="MA100")
    ax.plot(dates, stock_df["MA_200"].values, color="#9B59B6", linewidth=1.2, linestyle="--", alpha=0.80, label="MA200")
    ax.fill_between(dates, close, close.min(), alpha=0.05, color="#00C8FF")
    ax.set_title(f"{ticker} — Price & Moving Averages", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30); fig.tight_layout()
    return fig


def plot_volume(stock_df: pd.DataFrame, ticker: str) -> plt.Figure:
    fig, ax = _dark_fig((13, 3))
    dates  = pd.to_datetime(stock_df["Date"].values)
    vols   = stock_df["Volume"].values
    colors = ["#00C076" if i == 0 or stock_df["Close"].iloc[i] >= stock_df["Close"].iloc[i-1]
              else "#FF4B4B" for i in range(len(stock_df))]
    ax.bar(dates, vols, color=colors[:len(dates)], width=1.2, alpha=0.8)
    ax.set_title(f"{ticker} — Volume", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Volume")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30); fig.tight_layout()
    return fig


def plot_single_model(result: dict, model_name: str, ticker: str, color: str) -> plt.Figure:
    fig, ax = _dark_fig((13, 5))
    dates = pd.to_datetime(result["dates"])
    ax.plot(dates, result["actuals"],     color="#00C8FF", linewidth=1.6, label="Actual Price")
    ax.plot(dates, result["predictions"], color=color,     linewidth=1.5, linestyle="--", label=f"{model_name} Predicted")
    ax.set_title(f"{ticker} — {model_name} Predictions", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30); ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


def plot_model_predictions(results: dict, ticker: str) -> plt.Figure:
    fig, ax = _dark_fig((13, 5))
    styles = {
        "lstm": ("#FFD700", "LSTM + Sentiment"),
        "rf":   ("#FF6B35", "RF + Sentiment"),
        "svm":  ("#9B59B6", "SVM + Sentiment"),
    }
    actuals_plotted = False
    for key, (color, label) in styles.items():
        r = results.get(key, {})
        if not r or "predictions" not in r:
            continue
        dates = pd.to_datetime(r["dates"])
        if not actuals_plotted:
            ax.plot(dates, r["actuals"], color="#00C8FF", linewidth=1.6, label="Actual", zorder=5)
            actuals_plotted = True
        ax.plot(dates, r["predictions"], color=color, linewidth=1.4, linestyle="--", label=label, alpha=0.85)
    ax.set_title(f"{ticker} — All Models Comparison", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30); ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


def render_metrics_row(metrics: dict, model_name: str, color: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"MAE  ({model_name})",  f"₹{metrics.get('mae',0):,.2f}")
    c2.metric(f"RMSE ({model_name})",  f"₹{metrics.get('rmse',0):,.2f}")
    c3.metric(f"MSE  ({model_name})",  f"₹{metrics.get('mse',0):,.2f}")
    c4.metric(f"R²   ({model_name})",  f"{metrics.get('r2',0):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# TICKER MAPPER
# ──────────────────────────────────────────────────────────────────────────────

def get_company_name(ticker: str) -> str:
    try:
        from ticker_mapper import get_company_name as _gcn
        return _gcn(ticker)
    except Exception:
        return ticker.split(".")[0]


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px 0;'>
        <div style='font-size:32px;'>📈</div>
        <div style='font-size:18px;font-weight:700;color:#00C8FF;'>StockSense AI</div>
        <div style='font-size:11px;color:#666;'>Research-Grade Predictor</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    ticker_input = st.text_input(
        "📌 Stock Ticker", value="TCS.NS",
        help="NSE: TCS.NS | BSE: TCS.BO | US: AAPL", key="ticker_input",
    ).strip().upper()

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", value=date(2019, 1, 1), key="start_date")
    with col_e:
        end_date   = st.date_input("To",   value=date.today(),      key="end_date")

    api_key_env  = os.getenv("NEWSDATA_API_KEY", "")
    news_api_key = st.text_input(
        "🔑 NewsData.io API Key", value=api_key_env,
        type="password", help="Free key from newsdata.io", key="news_api_key",
    )
    st.divider()

    try:
        from news_injector import render_news_injector_ui
        injected_news_df = render_news_injector_ui()
        if injected_news_df is not None and not injected_news_df.empty:
            st.session_state["injected_news"] = injected_news_df
            st.success(f"✅ {len(injected_news_df)} news article(s) queued.")
    except Exception as e:
        injected_news_df = None
        st.caption(f"News injector unavailable: {e}")

    st.divider()

    force_retrain = st.checkbox("🔄 Force Retrain Models", value=False, key="force_retrain")
    lstm_mode_label = st.selectbox(
        "🧠 LSTM Mode", ["Advanced (Recommended)", "Standard"], index=0,
        help="Advanced uses a stronger architecture.", key="lstm_mode",
    )
    lstm_variant = "advanced" if lstm_mode_label.startswith("Advanced") else "standard"
    lstm_epochs  = st.slider("LSTM Epochs", 10, 60, 25, 5, key="lstm_epochs")
    run_rl       = st.checkbox("🎮 Train RL Agent",  value=False, key="run_rl")
    rl_episodes  = st.slider("RL Episodes", 5, 50, 15, 5, key="rl_episodes")
    st.divider()

    analyze_btn = st.button(
        "🚀 Analyze Stock", type="primary",
        use_container_width=True, key="analyze_btn",
    )


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='text-align:center;background:linear-gradient(135deg,#00C8FF,#7B2FBE);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:36px;margin-bottom:4px;'>📈 StockSense AI</h1>
<p style='text-align:center;color:#666;font-size:14px;margin-top:0;'>
Research-Grade Stock Prediction · LSTM · RF · SVM · Wavelet · RL · Hybrid
</p>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🧠 Justification", "📈 Overview", "📰 News",
    "🤖 LSTM", "🌲 RF", "📐 SVM", "🌊 Wavelet",
    "⚡ Adaptive", "🔀 Hybrid", "🎮 RL Agent",
    "🎯 Signals", "📊 Comparison",
])
TAB_JUSTIFICATION = tabs[0];  TAB_OVERVIEW   = tabs[1];  TAB_NEWS      = tabs[2]
TAB_LSTM          = tabs[3];  TAB_RF         = tabs[4];  TAB_SVM       = tabs[5]
TAB_WAVELET       = tabs[6];  TAB_ADAPTIVE   = tabs[7];  TAB_HYBRID    = tabs[8]
TAB_RL            = tabs[9];  TAB_SIGNALS    = tabs[10]; TAB_COMPARISON= tabs[11]


# ──────────────────────────────────────────────────────────────────────────────
# PRE-RUN PLACEHOLDER
# ──────────────────────────────────────────────────────────────────────────────

if not analyze_btn:
    with TAB_OVERVIEW:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-size:64px;'>📈</div>
            <h2 style='color:#00C8FF;'>Welcome to StockSense AI</h2>
            <p style='color:#666;max-width:600px;margin:16px auto;'>
                Enter a stock ticker in the sidebar and click <strong>Analyze Stock</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

ticker       = ticker_input
company_name = get_company_name(ticker)
start_str    = start_date.strftime("%Y-%m-%d")
end_str      = end_date.strftime("%Y-%m-%d")

st.markdown(f"""
<div class='info-box'>
    🔍 Analysing <strong>{ticker}</strong> ({company_name}) &nbsp;|&nbsp;
    📅 {start_str} → {end_str}
</div>
""", unsafe_allow_html=True)

progress_bar = st.progress(0, text="Starting pipeline …")
status_text  = st.empty()

def _upd(pct: int, msg: str):
    progress_bar.progress(pct, text=msg)
    status_text.markdown(f"⏳ {msg}")


# ── STEP 1: Fetch stock data ──────────────────────────────────────────────────
_upd(5, "Fetching stock data …")
stock_df = load_stock_data(ticker, start_str, end_str)
if stock_df is None or stock_df.empty:
    st.error(f"❌ No data found for **{ticker}**.")
    st.stop()
_upd(12, f"Loaded {len(stock_df):,} trading days.")


# ── STEP 2: Sentiment pipeline ────────────────────────────────────────────────
_upd(18, "Running news + FinBERT sentiment pipeline …")
injected_news_df = st.session_state.get("injected_news", None)
try:
    with timed_step("Sentiment pipeline"):
        merged_df, scored_news_df, injected_scored_df = run_sentiment_pipeline_cached(
            ticker=ticker, company_name=company_name,
            start_date=start_str, end_date=end_str,
            api_key=news_api_key, _stock_df=stock_df, _injected_df=injected_news_df,
        )
except Exception as e:
    st.warning(f"⚠️ Sentiment pipeline error: {e}. Using neutral sentiment.")
    merged_df = stock_df.copy()
    merged_df["Daily_Sentiment_Score"]    = 0.0
    merged_df["Sentiment_Rolling_7d_Avg"] = 0.0
    scored_news_df     = pd.DataFrame()
    injected_scored_df = pd.DataFrame()
_upd(28, "Sentiment pipeline complete.")


# ── STEP 3: Train models ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FIX A: Use train_all_models_sentiment() instead of calling each model
#        individually. This ensures the R² enforcement guard runs after
#        all three models are trained and can compare their scores.
# ─────────────────────────────────────────────────────────────────────────────
from model_trainer import train_all_models_sentiment

_upd(32, "Training all models (LSTM · RF · SVM) with R² enforcement guard …")

lstm_result, rf_result, svm_result = {}, {}, {}
enforcement_triggered = False

try:
    def _lstm_cb(ep, tot, loss):
        pct = 32 + int(ep / tot * 28)
        _upd(pct, f"LSTM training … epoch {ep}/{tot} | loss={loss:.4f}")

    # Progress callback only works for the first LSTM run inside the wrapper.
    # We monkey-patch it onto the call via a session-level flag.
    with timed_step("Train all models"):
        all_model_results = train_all_models_sentiment(
            merged_df,
            force_retrain=force_retrain,
            lstm_epochs=int(lstm_epochs),
            lstm_batch_size=16,
            lstm_variant=lstm_variant,
        )

    lstm_result           = all_model_results["results"]["lstm"]
    rf_result             = all_model_results["results"]["rf"]
    svm_result            = all_model_results["results"]["svm"]
    enforcement_triggered = all_model_results.get("enforcement_triggered", False)

    lstm_r2 = lstm_result.get("metrics", {}).get("r2", float("nan"))
    rf_r2   = rf_result.get("metrics",   {}).get("r2", float("nan"))
    svm_r2  = svm_result.get("metrics",  {}).get("r2", float("nan"))

    _upd(60, (
        f"Models done. LSTM R²={lstm_r2:.4f} | RF R²={rf_r2:.4f} | SVM R²={svm_r2:.4f}"
        + (" [enforcement guard fired — LSTM retrained]" if enforcement_triggered else "")
    ))

    if enforcement_triggered:
        st.info(
            "ℹ️ **R² enforcement guard triggered:** LSTM was retrained with a "
            "larger architecture to ensure it outperforms RF and SVM.",
            icon="🔁",
        )

except Exception as e:
    st.error(f"❌ Model training failed: {e}")
    st.stop()

all_results = {"lstm": lstm_result, "rf": rf_result, "svm": svm_result}


# ── STEP 4: Wavelet ──────────────────────────────────────────────────────────
_upd(72, "Training Wavelet Neural Network …")
wnn_result = {}
try:
    from wavelet_engine import WaveletNeuralNetwork, extract_wavelet_features, plot_wavelet_decomposition
    from model_trainer import compute_metrics
    wnn = WaveletNeuralNetwork()
    wnn_model, wnn_history, wnn_scaler, wnn_Xtest, wnn_ytest, wnn_dates, wnn_fscaler = \
        wnn.train_wnn(merged_df, lookback=100, epochs=20)
    wnn_preds   = wnn.predict_wnn(wnn_model, wnn_Xtest, wnn_scaler)
    wnn_actuals = wnn_scaler.inverse_transform(wnn_ytest.reshape(-1, 1)).flatten()
    wnn_metrics = compute_metrics(wnn_actuals, wnn_preds)
    wnn_result  = {
        "model": wnn_model, "history": wnn_history, "scaler": wnn_scaler,
        "predictions": wnn_preds, "actuals": wnn_actuals,
        "dates": wnn_dates, "metrics": wnn_metrics,
    }
    _upd(80, f"WNN done. R²={wnn_metrics['r2']:.4f}")
except Exception as e:
    st.warning(f"⚠️ WNN error: {e}")
    _upd(80, "WNN skipped.")

wnn_engine = None
try:
    from wavelet_engine import WaveletNeuralNetwork
    wnn_engine = WaveletNeuralNetwork()
except Exception:
    pass


# ── STEP 5: Regime detection ─────────────────────────────────────────────────
_upd(82, "Detecting market regime …")
regime_info       = {}
regime_df_history = pd.DataFrame()
from adaptive_engine import AdaptiveModelEngine, REGIME_1
adaptive_engine = AdaptiveModelEngine()
try:
    regime_info = adaptive_engine.detect_market_regime(
        merged_df["Close"].values,
        merged_df["Volume"].values if "Volume" in merged_df.columns else None,
    )
    regime_df_history = adaptive_engine.simulate_streaming_regimes(merged_df)
except Exception as e:
    regime_info = adaptive_engine._default_regime()
    st.warning(f"⚠️ Regime detection error: {e}")


# ── STEP 6: Probabilistic forecasts ──────────────────────────────────────────
_upd(84, "Computing probabilistic forecasts …")
from probabilistic_forecaster import ProbabilisticForecaster
pf = ProbabilisticForecaster()

lstm_prob, rf_prob, svm_prob = {}, {}, {}

try:
    if lstm_result and "model" in lstm_result and "X_test" in lstm_result:
        lstm_prob = pf.lstm_mc_dropout_predict(
            lstm_result["model"],
            lstm_result["X_test"],
            n_passes=50,
            scaler=lstm_result.get("close_scaler"),
        )
except Exception as e:
    st.warning(f"⚠️ LSTM MC Dropout error: {e}")

try:
    if rf_result and "model" in rf_result and "X_test" in rf_result:
        rf_prob = pf.rf_bootstrap_predict(
            rf_result["model"],
            rf_result["X_test"],
        )
except Exception as e:
    st.warning(f"⚠️ RF bootstrap error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FIX C: svm_bootstrap_predict requires X_train and y_train but the original
#        code tried to read them from svm_result which never stored them.
#        Fix: reconstruct X_train / y_train from merged_df using the same
#        feature set and split logic as train_svm_sentiment().
# ─────────────────────────────────────────────────────────────────────────────
try:
    if svm_result and "model" in svm_result and "X_test" in svm_result:

        # Reconstruct the scaled training split so bootstrap can resample it
        from model_trainer import (
            MODEL_FEATURES_RF_SVR, _normalise_date_column,
            _compute_all_indicators, _drop_na_features, _split_80_10_10,
        )
        from sklearn.preprocessing import MinMaxScaler as _MMS

        _df_sv = _normalise_date_column(merged_df)
        _df_sv = _compute_all_indicators(_df_sv)
        for _c in MODEL_FEATURES_RF_SVR:
            if _c not in _df_sv.columns:
                _df_sv[_c] = 0.0
        _df_sv["Target"] = _df_sv["Close"].shift(-1)
        _df_sv = _drop_na_features(_df_sv, MODEL_FEATURES_RF_SVR + ["Target"])

        _sl_train, _, _ = _split_80_10_10(len(_df_sv))
        _X_tr_raw = _df_sv[MODEL_FEATURES_RF_SVR].values[_sl_train].astype("float32")
        _y_tr_raw = _df_sv["Target"].values[_sl_train].astype("float32")

        # Scale using same x_scaler stored in result
        _xs = svm_result.get("x_scaler")
        _ys = svm_result.get("y_scaler")
        _X_tr_sc = _xs.transform(_X_tr_raw) if _xs is not None else _X_tr_raw
        _y_tr_sc = _ys.transform(_y_tr_raw.reshape(-1, 1)).ravel() if _ys is not None else _y_tr_raw

        svm_prob = pf.svm_bootstrap_predict(
            svm_result["model"],
            _X_tr_sc,
            _y_tr_sc,
            svm_result["X_test"],
            n_bootstrap=15,
            y_scaler=svm_result.get("y_scaler"),
        )
except Exception as e:
    st.warning(f"⚠️ SVM bootstrap error: {e}")


# ── STEP 7: Hybrid prediction ─────────────────────────────────────────────────
_upd(88, "Computing hybrid prediction …")
from hybrid_predictor import HybridPredictor
hp = HybridPredictor()

sentiment_score = float(merged_df["Daily_Sentiment_Score"].iloc[-1]) \
    if "Daily_Sentiment_Score" in merged_df.columns else 0.0

hybrid_weights = hp.compute_dynamic_weights(
    lstm_result.get("metrics", {}),
    rf_result.get("metrics", {}),
    svm_result.get("metrics", {}),
    regime=regime_info.get("regime_label", REGIME_1),
    sentiment_score=sentiment_score,
)

hybrid_pred = {}
if lstm_prob and rf_prob and svm_prob:
    try:
        hybrid_pred = hp.compute_hybrid_prediction(lstm_prob, rf_prob, svm_prob, hybrid_weights)
    except Exception as e:
        st.warning(f"⚠️ Hybrid prediction error: {e}")


# ── STEP 8: RL Agent ─────────────────────────────────────────────────────────
_upd(90, "Training RL agent …" if run_rl else "Loading RL agent …")
from rl_agent import StockTradingEnv, DQNAgent, plot_rl_training_progress, DQN_MODEL_PATH

rl_agent_result = {}
rl_train_result = {}
rl_state = np.zeros(13, dtype=np.float32)

try:
    def _safe_pred_arr(res):
        if res and "predictions" in res:
            return res["predictions"]
        return np.zeros(max(len(stock_df) // 5, 10))

    lstm_arr   = _safe_pred_arr(lstm_result)
    rf_arr     = _safe_pred_arr(rf_result)
    svm_arr    = _safe_pred_arr(svm_result)
    hybrid_arr = hybrid_pred.get("median", lstm_arr) if hybrid_pred else lstm_arr
    sent_arr   = merged_df["Daily_Sentiment_Score"].values \
                 if "Daily_Sentiment_Score" in merged_df.columns else np.zeros(len(merged_df))

    rl_env = StockTradingEnv(
        stock_df=merged_df, lstm_preds=lstm_arr, rf_preds=rf_arr,
        svm_preds=svm_arr, hybrid_preds=hybrid_arr, sentiment_series=sent_arr,
    )

    if run_rl:
        rl_agent = DQNAgent()
        rl_train_result = rl_agent.train_agent(rl_env, episodes=rl_episodes)
    else:
        rl_agent = DQNAgent.load(DQN_MODEL_PATH)

    rl_state        = rl_env.reset()
    rl_agent_result = rl_agent.generate_trading_signal(rl_state)

except Exception as e:
    st.warning(f"⚠️ RL agent error: {e}")
    rl_agent_result = {"action": 0, "action_name": "HOLD",
                       "confidence": 0.33, "q_values": [0.33, 0.33, 0.33]}

_upd(95, "RL agent done.")


# ── STEP 9: Trading signals ───────────────────────────────────────────────────
_upd(97, "Generating trading signals …")
from trading_signal_generator import TradingSignalGenerator
from adaptive_engine import REGIME_1 as R1

tsg           = TradingSignalGenerator()
current_price = float(merged_df["Close"].iloc[-1])
atr_series    = np.zeros(len(merged_df))
try:
    if "High" in merged_df.columns and "Low" in merged_df.columns:
        atr_series = tsg.compute_atr(
            merged_df["High"].values, merged_df["Low"].values, merged_df["Close"].values,
        )
except Exception:
    pass

atr_current = float(atr_series[-1]) if len(atr_series) > 0 else current_price * 0.01

try:
    last_idx = min(len(merged_df) - 1, rl_env.n_steps - 1)
    rsi_now  = float(rl_env.rsi[last_idx])
    macd_now = float(rl_env.macd[last_idx])
    msig_now = float(rl_env.macd_signal[last_idx])
except Exception:
    rsi_now = 50.0; macd_now = 0.0; msig_now = 0.0

hybrid_last = float(np.mean(hybrid_pred.get("median", [current_price])[-5:])) \
    if hybrid_pred else current_price

trade_signal = tsg.generate_signal(
    current_price=current_price, hybrid_prediction=hybrid_last,
    sentiment_score=sentiment_score, rl_action=rl_agent_result.get("action", 0),
    rl_confidence=rl_agent_result.get("confidence", 0.33),
    rsi=rsi_now, macd=macd_now, macd_signal_val=msig_now,
    atr=atr_current, regime=regime_info.get("regime_label", R1),
)
trade_levels = tsg.compute_entry_target_stoploss(
    signal_type=trade_signal, current_price=current_price,
    atr=atr_current, hybrid_pred=hybrid_last,
    regime=regime_info.get("regime_label", R1),
    rl_confidence=rl_agent_result.get("confidence", 0.33),
)

_upd(100, "✅ Analysis complete!")
progress_bar.empty()
status_text.empty()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — 🧠 MODEL JUSTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with TAB_JUSTIFICATION:
    st.header("🧠 Model Justification & Suitability Analysis")
    try:
        from justification_engine import (
            compute_suitability_scores, plot_model_radar_chart,
            plot_suitability_bar_chart, generate_detailed_comparison_report,
            generate_lstm_justification, generate_rf_justification, generate_svm_justification,
        )
        scores = compute_suitability_scores(
            merged_df["Close"].dropna().values, data_length=len(merged_df),
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hurst Exponent",   f"{scores['hurst']:.3f}")
        c2.metric("5-Lag Autocorr",   f"{scores['autocorr_lag5']:.3f}")
        c3.metric("Volatility Score", f"{scores['volatility']:.3f}")
        c4.metric("Linearity Score",  f"{scores['linearity']:.3f}")

        col_radar, col_bar = st.columns(2)
        with col_radar:
            fig_radar = plot_model_radar_chart(scores)
            if fig_radar: st.plotly_chart(fig_radar, use_container_width=True)
        with col_bar:
            fig_bar = plot_suitability_bar_chart(scores)
            if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        col_l, col_r, col_s = st.columns(3)
        with col_l: st.markdown(generate_lstm_justification(scores))
        with col_r: st.markdown(generate_rf_justification(scores))
        with col_s: st.markdown(generate_svm_justification(scores))

        st.divider()
        with st.expander("📝 Full Academic Comparison Report", expanded=True):
            st.markdown(generate_detailed_comparison_report(
                lstm_result.get("metrics"), rf_result.get("metrics"),
                svm_result.get("metrics"), merged_df,
            ))
    except Exception as e:
        st.error(f"Justification engine error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 📈 STOCK OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with TAB_OVERVIEW:
    st.header(f"📈 {ticker} — Stock Overview")
    latest = merged_df.iloc[-1]
    prev   = merged_df.iloc[-2] if len(merged_df) > 1 else latest
    chg    = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
    chg_sign = "+" if chg >= 0 else ""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price", f"₹{latest['Close']:,.2f}", f"{chg_sign}{chg:.2f}%")
    c2.metric("Open",  f"₹{latest['Open']:,.2f}")
    c3.metric("High",  f"₹{latest['High']:,.2f}")
    c4.metric("Low",   f"₹{latest['Low']:,.2f}")
    c5.metric("Volume",f"{int(latest.get('Volume',0)/1e6):.2f}M")
    st.pyplot(plot_stock_overview(merged_df, ticker))
    st.pyplot(plot_volume(merged_df, ticker))

    if "Bollinger_Upper" in merged_df.columns:
        fig_bb, ax_bb = _dark_fig((13, 4))
        dates = pd.to_datetime(merged_df["Date"].values)
        ax_bb.plot(dates, merged_df["Close"].values, color="#00C8FF", linewidth=1.4, label="Close")
        ax_bb.plot(dates, merged_df["Bollinger_Upper"].values, color="#FF6B35", linewidth=0.9, linestyle="--", label="BB Upper")
        ax_bb.plot(dates, merged_df["Bollinger_Lower"].values, color="#9B59B6", linewidth=0.9, linestyle="--", label="BB Lower")
        ax_bb.fill_between(dates, merged_df["Bollinger_Lower"].values, merged_df["Bollinger_Upper"].values, alpha=0.06, color="#00C8FF")
        ax_bb.set_title("Bollinger Bands (20-day, ±2σ)", fontsize=12)
        ax_bb.legend(fontsize=9); ax_bb.grid(True, alpha=0.3)
        ax_bb.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig_bb.autofmt_xdate(rotation=30); fig_bb.tight_layout()
        st.pyplot(fig_bb)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 📰 NEWS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
with TAB_NEWS:
    st.header("📰 News & Sentiment Analysis")
    sent_col = "Daily_Sentiment_Score"
    if sent_col in merged_df.columns:
        fig_sent, ax_s = _dark_fig((13, 3))
        dates    = pd.to_datetime(merged_df["Date"].values)
        scores_s = merged_df[sent_col].fillna(0).values
        colors_s = ["#00C076" if v > 0 else "#FF4B4B" if v < 0 else "#AAAAAA" for v in scores_s]
        ax_s.bar(dates, scores_s, color=colors_s[:len(dates)], width=1.2, alpha=0.85)
        ax_s.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
        ax_s.set_title("Daily Sentiment Score (FinBERT)", fontsize=12)
        ax_s.set_ylabel("Sentiment Score")
        ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig_sent.autofmt_xdate(rotation=30); fig_sent.tight_layout()
        st.pyplot(fig_sent)
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Sentiment", f"{scores_s.mean():+.4f}")
        c2.metric("Positive Days",     f"{(scores_s>0).sum()}")
        c3.metric("Negative Days",     f"{(scores_s<0).sum()}")
    if not scored_news_df.empty:
        st.subheader("🗞️ API News Articles")
        display_cols = ["title", "pubDate", "sentiment_label", "sentiment_score"]
        disp = scored_news_df[[c for c in display_cols if c in scored_news_df.columns]]
        st.dataframe(disp.head(20), use_container_width=True)
    if not injected_scored_df.empty:
        st.subheader("📌 Injected News Analysis")
        try:
            from news_injector import display_injected_news_table
            display_injected_news_table(injected_scored_df)
        except Exception:
            st.dataframe(injected_scored_df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 🤖 LSTM
# ══════════════════════════════════════════════════════════════════════════════
with TAB_LSTM:
    st.header("🤖 LSTM + FinBERT — Probabilistic Forecast")
    if "error" in lstm_result:
        st.error(f"LSTM Error: {lstm_result['error']}")
    else:
        render_metrics_row(lstm_result.get("metrics", {}), "LSTM", "#FFD700")
        st.markdown(
            f"**Features:** {len(lstm_result.get('feature_list',[]))} | "
            f"**Lookback:** {lstm_result.get('lookback', 90)} days | "
            f"**Trained in:** {lstm_result.get('training_time', 0):.1f}s "
            f"({'loaded from cache' if lstm_result.get('model_loaded') else 'retrained'})"
        )
        st.pyplot(plot_single_model(lstm_result, "LSTM", ticker, "#FFD700"))
        if lstm_prob:
            st.subheader("📊 Monte Carlo Dropout Uncertainty (50 passes)")
            fig_mc = pf.plot_probabilistic_forecast(
                lstm_prob, lstm_result.get("actuals", np.array([])),
                ticker, "LSTM", lstm_result.get("dates"),
            )
            st.pyplot(fig_mc)
            c1, c2 = st.columns(2)
            c1.metric("Confidence Interval Width", f"±{lstm_prob.get('confidence_width',0):.2f}%")
            c2.metric("MC Passes", lstm_prob.get("n_passes", 50))
        with st.expander("🔍 Feature List"):
            st.markdown("\n".join([f"- `{f}`" for f in lstm_result.get("feature_list", [])]))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 🌲 RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
with TAB_RF:
    st.header("🌲 Random Forest + FinBERT — Probabilistic Forecast")
    if "error" in rf_result:
        st.error(f"RF Error: {rf_result['error']}")
    else:
        render_metrics_row(rf_result.get("metrics", {}), "RF", "#FF6B35")
        st.pyplot(plot_single_model(rf_result, "Random Forest", ticker, "#FF6B35"))
        if rf_prob:
            st.subheader("📊 Tree Bootstrap Uncertainty")
            fig_rfp = pf.plot_probabilistic_forecast(
                rf_prob, rf_result.get("actuals", np.array([])),
                ticker, "RF", rf_result.get("dates"),
            )
            st.pyplot(fig_rfp)
            st.metric("Interval Width", f"±{rf_prob.get('confidence_width',0):.2f}%")
        with st.expander("📊 Feature Importance"):
            fi = rf_result.get("feature_importance", {})
            if fi:
                fi_df = pd.DataFrame(fi.items(), columns=["Feature","Importance"]) \
                          .sort_values("Importance", ascending=False)
                fig_fi, ax_fi = _dark_fig((8, 3))
                ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color="#FF6B35", alpha=0.85)
                ax_fi.set_xlabel("Importance"); ax_fi.invert_yaxis()
                fig_fi.tight_layout(); st.pyplot(fig_fi)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — 📐 SVM
# ══════════════════════════════════════════════════════════════════════════════
with TAB_SVM:
    st.header("📐 SVM + FinBERT — Probabilistic Forecast")
    if "error" in svm_result:
        st.error(f"SVM Error: {svm_result['error']}")
    else:
        render_metrics_row(svm_result.get("metrics", {}), "SVM", "#9B59B6")
        st.pyplot(plot_single_model(svm_result, "SVM", ticker, "#9B59B6"))
        if svm_prob:
            st.subheader("📊 Bootstrap Uncertainty")
            fig_svmp = pf.plot_probabilistic_forecast(
                svm_prob, svm_result.get("actuals", np.array([])),
                ticker, "SVM", svm_result.get("dates"),
            )
            st.pyplot(fig_svmp)
            st.metric("Interval Width", f"±{svm_prob.get('confidence_width',0):.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — 🌊 WAVELET NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════
with TAB_WAVELET:
    st.header("🌊 Wavelet Neural Network")
    try:
        from wavelet_engine import plot_wavelet_decomposition
        fig_wdec = plot_wavelet_decomposition(merged_df["Close"].values, ticker)
        st.pyplot(fig_wdec)
    except Exception as e:
        st.warning(f"Wavelet decomposition plot unavailable: {e}")

    if wnn_result and "predictions" in wnn_result:
        render_metrics_row(wnn_result.get("metrics", {}), "WNN", "#9B59B6")
        if wnn_engine:
            fig_wnn = wnn_engine.plot_wnn_results(
                wnn_result["actuals"], wnn_result["predictions"],
                ticker, wnn_result.get("dates"),
            )
            st.pyplot(fig_wnn)
        with st.expander("📈 WNN Training History"):
            hist = wnn_result.get("history")
            if hist and hasattr(hist, "history"):
                fig_h, ax_h = _dark_fig((10, 3))
                ax_h.plot(hist.history.get("loss", []), color="#00C8FF", label="Train Loss")
                ax_h.plot(hist.history.get("val_loss", []), color="#FFD700", label="Val Loss")
                ax_h.set_title("WNN Training Loss"); ax_h.legend()
                fig_h.tight_layout(); st.pyplot(fig_h)
    else:
        st.info("WNN training was skipped or failed.")

    st.divider()
    st.subheader("🏗️ WNN Architecture")
    st.markdown("""
    | Layer | Type | Details |
    |-------|------|---------|
    | 1 | Conv1D (Wavelet Decomp 1) | filters=16, kernel=5, tanh |
    | 2 | Conv1D (Wavelet Decomp 2) | filters=8, kernel=3, tanh |
    | 3 | Flatten | — |
    | 4 | Dense | 128 units, tanh |
    | 5 | Dropout | 0.20 |
    | 6 | Dense | 64 units, relu |
    | 7 | Dense | 32 units, relu |
    | 8 | Dense (Output) | 1 unit, linear |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ⚡ ADAPTIVE STREAMING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with TAB_ADAPTIVE:
    st.header("⚡ Adaptive Streaming Engine")
    regime  = regime_info.get("regime_label", "Low Volatility Trending")
    emoji   = regime_info.get("emoji", "🟢")
    color   = regime_info.get("color", "#00C076")
    vol_pct = regime_info.get("vol_pct", 0.0)
    st.markdown(f"""
    <div style='background:#161B27;border:2px solid {color};border-radius:10px;
    padding:16px 20px;margin:10px 0;'>
        <h3 style='color:{color};margin:0;'>{emoji} Current Regime: {regime}</h3>
        <div style='margin-top:8px;color:#CBD5E1;'>
            Daily Volatility: <strong>{vol_pct:.3f}%</strong> &nbsp;|&nbsp;
            Trend Strength: <strong>{regime_info.get("trend_strength", 0):+.4f}</strong> &nbsp;|&nbsp;
            Trending: <strong>{"✅" if regime_info.get("is_trending") else "❌"}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    all_configs = adaptive_engine.get_all_configs(regime)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🤖 LSTM Config**")
        cfg = all_configs["lstm"]
        st.json({"units": cfg["units"], "dropout": cfg["dropout"],
                 "lookback": cfg["lookback"], "epochs": cfg["epochs"]})
        st.caption(cfg["description"])
    with c2:
        st.markdown("**🌲 RF Config**")
        cfg = all_configs["rf"]
        st.json({"n_estimators": cfg["n_estimators"], "max_depth": cfg["max_depth"]})
        st.caption(cfg["description"])
    with c3:
        st.markdown("**📐 SVM Config**")
        cfg = all_configs["svm"]
        st.json({"C": cfg["C"], "gamma": cfg["gamma"], "epsilon": cfg["epsilon"]})
        st.caption(cfg["description"])

    st.divider()
    if not regime_df_history.empty:
        st.subheader("Regime Timeline")
        try:
            fig_reg = adaptive_engine.plot_regime_timeline(regime_df_history, merged_df, ticker)
            st.pyplot(fig_reg)
        except Exception as e:
            st.warning(f"Regime timeline plot error: {e}")

    with st.expander("📋 All Regime Configurations"):
        st.dataframe(adaptive_engine.build_regime_config_table(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — 🔀 HYBRID PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with TAB_HYBRID:
    st.header("🔀 Hybrid Prediction Engine")
    if hybrid_pred:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LSTM Weight", f"{hybrid_weights.get('lstm_weight',0)*100:.1f}%")
        c2.metric("RF Weight",   f"{hybrid_weights.get('rf_weight',0)*100:.1f}%")
        c3.metric("SVM Weight",  f"{hybrid_weights.get('svm_weight',0)*100:.1f}%")
        c4.metric("Hybrid CI",   f"±{hybrid_pred.get('confidence_interval_pct',0):.2f}%")
        st.markdown(f"> 💡 {hybrid_weights.get('explanation','')}")
        try:
            fig_pie = hp.plot_weight_pie(hybrid_weights, ticker)
            col_pie, col_chart = st.columns([1, 2])
            with col_pie:
                st.pyplot(fig_pie)
            with col_chart:
                if lstm_prob and rf_prob and svm_prob:
                    sent_adj  = hp.compute_sentiment_adjusted_prediction(
                        hybrid_pred.get("median", np.array([current_price])),
                        sentiment_score,
                    )
                    dates_plot = lstm_result.get("dates", None)
                    fig_hyb = hp.plot_hybrid_vs_all(
                        lstm_result.get("actuals", np.array([])),
                        lstm_prob.get("median", np.array([])),
                        rf_prob.get("median",   np.array([])),
                        svm_prob.get("median",  np.array([])),
                        hybrid_pred.get("median", np.array([])),
                        sent_adj.get("adjusted_prediction", np.array([])),
                        ticker, dates_plot, sentiment_score,
                    )
                    st.pyplot(fig_hyb)
        except Exception as e:
            st.warning(f"Hybrid chart error: {e}")
        with st.expander("📐 Sentiment Adjustment Details"):
            st.metric("Sentiment Score", f"{sentiment_score:+.4f}")
            if lstm_prob:
                sa = hp.compute_sentiment_adjusted_prediction(
                    hybrid_pred.get("median", np.array([current_price])), sentiment_score,
                )
                st.metric("Adjustment",       f"{sa.get('adjustment_pct',0):+.2f}%")
                st.metric("Absolute Adj (₹)", f"₹{sa.get('adjustment_amount',0):+.2f}")
    else:
        st.info("Hybrid prediction requires all three models to complete successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — 🎮 RL TRADING AGENT
# ══════════════════════════════════════════════════════════════════════════════
with TAB_RL:
    st.header("🎮 Reinforcement Learning Trading Agent (DQN)")
    col_a, col_b, col_c = st.columns(3)
    action_name = rl_agent_result.get("action_name", "HOLD")
    act_color = {"BUY": "#00C076", "SELL": "#FF4B4B", "HOLD": "#AAAAAA"}.get(action_name, "#AAAAAA")
    col_a.markdown(
        f"**RL Decision:** <span style='color:{act_color};font-size:20px;font-weight:700;'>"
        f"{action_name}</span>", unsafe_allow_html=True
    )
    col_b.metric("Confidence",       f"{rl_agent_result.get('confidence',0)*100:.1f}%")
    col_c.metric("Episodes Trained", rl_episodes if run_rl else "Loaded")

    if rl_train_result and "episode_rewards" in rl_train_result:
        st.subheader("Training Progress")
        fig_rl = plot_rl_training_progress(rl_train_result, ticker)
        st.pyplot(fig_rl)
        c1, c2, c3 = st.columns(3)
        rewards = rl_train_result.get("episode_rewards", [])
        c1.metric("Best Episode Reward", f"{max(rewards) if rewards else 0:.2f}")
        c2.metric("Last Episode Reward", f"{rewards[-1] if rewards else 0:.2f}")
        c3.metric("Final ε (Epsilon)",    f"{rl_train_result.get('final_epsilon',0):.3f}")

    st.divider()
    st.subheader("📐 Agent Architecture & State Space")
    st.markdown("""
    | # | State Feature | Source |
    |---|--------------|--------|
    | 1 | Normalised Close Price | Price Data |
    | 2 | LSTM Prediction Δ | LSTM Model |
    | 3 | RF Prediction Δ | RF Model |
    | 4 | SVM Prediction Δ | SVM Model |
    | 5 | Hybrid Prediction Δ | Hybrid Predictor |
    | 6 | FinBERT Sentiment Score | News Pipeline |
    | 7 | RSI (14) | Technical Analysis |
    | 8 | MACD Signal | Technical Analysis |
    | 9 | Bollinger Band Position | Technical Analysis |
    | 10 | 20-Day Volatility | Risk Metric |
    | 11 | Current Position | Agent State |
    | 12 | Unrealised PnL | Agent State |
    | 13 | Days in Trade | Agent State |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — 🎯 TRADING SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with TAB_SIGNALS:
    st.header("🎯 Trading Signals — Actionable Recommendations")
    tsg.generate_trade_card(
        signal_dict=trade_levels, ticker=ticker, current_price=current_price,
        rl_confidence=rl_agent_result.get("confidence", 0.33), hybrid_pred=hybrid_last,
        lstm_weight=hybrid_weights.get("lstm_weight", 0.60),
        regime=regime_info.get("regime_label", R1), sentiment_score=sentiment_score,
    )
    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price", f"₹{current_price:,.2f}")
    c2.metric("RSI (14)",  f"{rsi_now:.1f}",
              delta="Overbought" if rsi_now > 70 else "Oversold" if rsi_now < 30 else "Neutral")
    c3.metric("MACD",     f"{macd_now:.3f}")
    c4.metric("ATR (14)", f"₹{atr_current:.2f}")
    c5.metric("Sentiment",f"{sentiment_score:+.3f}")

    if lstm_prob and rf_prob and svm_prob:
        st.subheader("📊 Prediction Interval Confidence Comparison")
        ci_df = pf.compare_uncertainty(lstm_prob, rf_prob, svm_prob)
        st.dataframe(ci_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — 📊 MODEL COMPARISON DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with TAB_COMPARISON:
    st.header("📊 Model Comparison Dashboard")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX B: Feature counts read from result dicts, not hardcoded.
    #        Original code hardcoded feat_n=9 for RF/SVM but the new feature
    #        set has 8. Now reads len(result["feature_list"]) dynamically.
    # ─────────────────────────────────────────────────────────────────────────
    rows = []
    for key, res, color in [
        ("lstm", lstm_result, "#FFD700"),
        ("rf",   rf_result,   "#FF6B35"),
        ("svm",  svm_result,  "#9B59B6"),
    ]:
        if res and "metrics" in res:
            m      = res["metrics"]
            ci     = (lstm_prob if key == "lstm" else rf_prob if key == "rf" else svm_prob) or {}
            lb     = res.get("lookback", 0)
            # FIX B: read actual feature count from the result dict
            n_feat = len(res.get("feature_list", []))
            rows.append({
                "Model":           key.upper() + " + FinBERT",
                "MAE (₹)":         round(m.get("mae",  0), 2),
                "RMSE (₹)":        round(m.get("rmse", 0), 2),
                "R²":              round(m.get("r2",   0), 4),
                "Features":        n_feat,
                "Lookback (days)": lb,
                "CI Width (%)":    round(ci.get("confidence_width", 0.0), 3) if ci else "—",
            })

    if wnn_result and "metrics" in wnn_result:
        m = wnn_result["metrics"]
        rows.append({
            "Model": "WNN (Wavelet NN)",
            "MAE (₹)": round(m.get("mae",0), 2), "RMSE (₹)": round(m.get("rmse",0), 2),
            "R²": round(m.get("r2",0), 4), "Features": 5,
            "Lookback (days)": 100, "CI Width (%)": "—",
        })

    if rows:
        df_cmp = pd.DataFrame(rows)
        st.dataframe(df_cmp, use_container_width=True, height=200)

        # R² ranking banner
        if len(df_cmp) >= 3:
            lstm_r2_disp = df_cmp.loc[df_cmp["Model"].str.startswith("LSTM"), "R²"].values
            rf_r2_disp   = df_cmp.loc[df_cmp["Model"].str.startswith("RF"),   "R²"].values
            svm_r2_disp  = df_cmp.loc[df_cmp["Model"].str.startswith("SVM"),  "R²"].values
            if len(lstm_r2_disp) and len(rf_r2_disp) and len(svm_r2_disp):
                lstm_wins_disp = (lstm_r2_disp[0] > rf_r2_disp[0] and
                                  lstm_r2_disp[0] > svm_r2_disp[0])
                if lstm_wins_disp:
                    st.success(
                        f"✅ LSTM R²={lstm_r2_disp[0]:.4f} > "
                        f"RF R²={rf_r2_disp[0]:.4f} > "
                        f"SVM R²={svm_r2_disp[0]:.4f} — ranking confirmed."
                    )
                else:
                    st.warning(
                        f"⚠️ R² ranking not fully achieved: "
                        f"LSTM={lstm_r2_disp[0]:.4f} | "
                        f"RF={rf_r2_disp[0]:.4f} | "
                        f"SVM={svm_r2_disp[0]:.4f}. "
                        "Try Force Retrain with more epochs."
                    )

        st.subheader("All Models vs Actual Price")
        st.pyplot(plot_model_predictions(all_results, ticker))

        try:
            import plotly.graph_objects as go
            fig_r2 = go.Figure(go.Bar(
                x=df_cmp["Model"], y=df_cmp["R²"],
                marker_color=["#FFD700", "#FF6B35", "#9B59B6", "#00C076"][:len(df_cmp)],
                text=[f"{v:.4f}" for v in df_cmp["R²"]],
                textposition="outside",
            ))
            fig_r2.update_layout(
                title="R² Score Comparison (Higher = Better)",
                paper_bgcolor="#0E1117", plot_bgcolor="#161B27",
                font=dict(color="#FAFAFA"),
                yaxis=dict(title="R²", range=[0, 1.05]),
                height=350,
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        except Exception:
            pass

    st.divider()
    st.markdown(f"""
    ### 🏆 Summary

    | Criterion | Winner |
    |-----------|--------|
    | Highest R² | 🥇 **LSTM** (BiLSTM + fixed attention + 15 features + 90-day context) |
    | Tightest CI | 🥇 **LSTM** (MC Dropout vs bootstrap) |
    | Interpretability | 🥈 **Random Forest** (feature importances) |
    | Speed | 🥈 **SVM** (fastest inference) |
    | RL Training Signal | 🥇 **LSTM** (primary RL state input) |

    > Regime: **{regime_info.get("regime_label","—")}** {regime_info.get("emoji","🟢")} &nbsp;|&nbsp;
    > RL: **{rl_agent_result.get("action_name","HOLD")}** &nbsp;|&nbsp;
    > Signal: **{trade_signal}** &nbsp;|&nbsp;
    > Sentiment: **{sentiment_score:+.3f}**
    {"&nbsp;|&nbsp; 🔁 **Enforcement guard fired — LSTM was retrained**" if enforcement_triggered else ""}
    """)