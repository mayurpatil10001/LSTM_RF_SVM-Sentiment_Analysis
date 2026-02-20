"""
app.py  —  Stock Price Prediction with FinBERT Sentiment Analysis
=================================================================
Upgraded Streamlit application integrating:
  • Live stock data via yfinance
  • Financial news via newsdata.io API
  • FinBERT sentiment analysis (ProsusAI/finbert)
  • Hybrid LSTM / Random Forest / SVM prediction models
  • Multi-tab comparison dashboard

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import date, timedelta, datetime

import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Local modules ─────────────────────────────────────────────────────────────
from ticker_mapper import get_company_name
from sentiment_utils import (
    fetch_news,
    run_finbert,
    aggregate_daily_sentiment,
    merge_sentiment_with_stock,
    get_sentiment_summary,
    label_color,
    RateLimitError,
)
from model_trainer import run_all_models

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StockSense AI — Sentiment-Powered Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL DARK CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Global dark background ── */
    html, body, [class*="css"] {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #161B27 0%, #0E1117 100%);
        border-right: 1px solid #2A2F3E;
    }
    /* ── Card-like metric boxes ── */
    [data-testid="stMetric"] {
        background: #161B27;
        border: 1px solid #2A2F3E;
        border-radius: 10px;
        padding: 14px 18px;
    }
    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab"] {
        color: #AAAAAA;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #00C8FF !important;
        border-bottom: 2px solid #00C8FF !important;
    }
    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    /* ── Headings ── */
    h1 { color: #00C8FF !important; font-weight: 700; }
    h2 { color: #FFFFFF !important; font-weight: 600; }
    h3 { color: #CBD5E1 !important; }
    /* ── Positive / Negative badge ── */
    .badge-pos { color: #00C076; font-weight: 700; }
    .badge-neg { color: #FF4B4B; font-weight: 700; }
    .badge-neu { color: #AAAAAA; font-weight: 700; }
    /* ── Sidebar section header ── */
    .sidebar-section {
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #556080;
        margin: 18px 0 6px 0;
        border-top: 1px solid #2A2F3E;
        padding-top: 12px;
    }
    /* ── Primary button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0075FF 0%, #00C8FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 28px;
        width: 100%;
        font-size: 14px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    /* ── Info box ── */
    .custom-info {
        background: #1A2035;
        border-left: 4px solid #00C8FF;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: MATPLOTLIB DARK STYLE
# ══════════════════════════════════════════════════════════════════════════════
def _apply_dark_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#161B27",
        "axes.facecolor":   "#161B27",
        "axes.edgecolor":   "#2A2F3E",
        "axes.labelcolor":  "#CBD5E1",
        "xtick.color":      "#CBD5E1",
        "ytick.color":      "#CBD5E1",
        "grid.color":       "#2A2F3E",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "legend.facecolor": "#1A2035",
        "legend.edgecolor": "#2A2F3E",
        "text.color":       "#FAFAFA",
        "font.size":        10,
    })


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and return a clean DataFrame.

    Handles both yfinance < 0.2.36 (flat columns) and yfinance >= 1.0
    (MultiIndex columns with ticker suffix).  Cached for 1 hour.
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    # ── Flatten MultiIndex columns (yfinance 1.x returns (Price, Ticker)) ──
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]   # keep only the price label
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # ── Drop Adj Close if present (auto_adjust=True already adjusts Close) ──
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
    return df


@st.cache_data(show_spinner=False, ttl=1800)
def cached_fetch_news(ticker, company, start, end, api_key):
    """Cached wrapper around fetch_news to avoid redundant API calls."""
    return fetch_news(ticker, company, start, end, api_key)


@st.cache_data(show_spinner=False)
def cached_finbert(news_hash: str, _news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cached FinBERT inference.
    news_hash: str hash of the news data to invalidate cache on new data.
    _news_df: prefixed with _ so Streamlit doesn't try to hash the DataFrame.
    """
    return run_finbert(_news_df)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_price_and_mas(stock_df: pd.DataFrame, ticker: str) -> plt.Figure:
    """Plot Close price with 100-day and 200-day moving averages."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(stock_df["Date"], stock_df["Close"],
            color="#00C8FF", linewidth=1.4, label="Close Price", alpha=0.9)
    if "MA_100" in stock_df.columns:
        ax.plot(stock_df["Date"], stock_df["MA_100"],
                color="#FF9F00", linewidth=1.2, label="100-Day MA", alpha=0.85)
    if "MA_200" in stock_df.columns:
        ax.plot(stock_df["Date"], stock_df["MA_200"],
                color="#FF4B4B", linewidth=1.2, label="200-Day MA", alpha=0.85)
    ax.set_title(f"{ticker} — Stock Price with Moving Averages",
                 fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def plot_sentiment_timeline(
    stock_df: pd.DataFrame,
    daily_sentiment: pd.Series,
    ticker: str,
) -> plt.Figure:
    """
    Two-panel chart:
      Top:    Stock close price + vertical dashed lines for major sentiment spikes
      Bottom: Daily sentiment bar chart (green/red/grey)
    """
    _apply_dark_style()
    fig = plt.figure(figsize=(13, 8))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── Price ──────────────────────────────────────────────────────────────
    ax1.plot(stock_df["Date"], stock_df["Close"],
             color="#00C8FF", linewidth=1.5, label="Close Price")

    # ── Spike annotations ─────────────────────────────────────────────────
    if not daily_sentiment.empty:
        spikes = daily_sentiment[daily_sentiment.abs() > 0.5]
        for dt, val in spikes.items():
            color = "#00C076" if val > 0 else "#FF4B4B"
            ax1.axvline(x=dt, color=color, linestyle="--", alpha=0.5, linewidth=1)

    ax1.set_title(f"{ticker} — News Sentiment Impact on Price", fontsize=14, pad=10)
    ax1.set_ylabel("Price (₹)")
    ax1.legend(loc="upper left")
    ax1.tick_params(labelbottom=False)

    # ── Sentiment bars ────────────────────────────────────────────────────
    if not daily_sentiment.empty:
        sent_dates  = daily_sentiment.index
        sent_values = daily_sentiment.values
        bar_colors  = [
            "#00C076" if v > 0.05 else "#FF4B4B" if v < -0.05 else "#555555"
            for v in sent_values
        ]
        ax2.bar(sent_dates, sent_values, color=bar_colors, width=1.5, alpha=0.85)
        ax2.axhline(0, color="#AAAAAA", linewidth=0.8)
        ax2.set_ylabel("Sentiment")
    else:
        ax2.text(0.5, 0.5, "No sentiment data",
                 ha="center", va="center", transform=ax2.transAxes,
                 color="#AAAAAA", fontsize=11)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return fig


def plot_sentiment_distribution(summary: dict) -> plt.Figure:
    """Horizontal bar chart of positive / neutral / negative article counts."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        summary["positive_pct"],
        summary["neutral_pct"],
        summary["negative_pct"],
    ]
    colors = ["#00C076", "#AAAAAA", "#FF4B4B"]
    bars = ax.barh(labels, values, color=colors, height=0.55, alpha=0.9)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", color="#FAFAFA", fontsize=10
        )
    ax.set_xlim(0, 110)
    ax.set_xlabel("Percentage of Articles (%)")
    ax.set_title("Overall Sentiment Distribution", fontsize=12)
    fig.tight_layout()
    return fig


def plot_model_predictions(results: dict, ticker: str) -> plt.Figure:
    """
    Overlay actual vs all three model predictions on a single chart.
    Each model may have a different test window — we align on dates.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    model_styles = {
        "lstm": ("#FFD700", "LSTM + Sentiment",  1.5),
        "rf":   ("#FF6B35", "RF + Sentiment",    1.5),
        "svm":  ("#9B59B6", "SVM + Sentiment",   1.5),
    }

    plotted_actual = False
    for key, style in model_styles.items():
        r = results.get(key, {})
        if "error" in r or "predictions" not in r:
            continue
        color, label, lw = style
        ax.plot(r["dates"], r["predictions"],
                color=color, linewidth=lw, label=label, alpha=0.8)
        if not plotted_actual and "actuals" in r:
            ax.plot(r["dates"], r["actuals"],
                    color="#00C8FF", linewidth=1.8,
                    label="Actual Price", alpha=0.95)
            plotted_actual = True

    ax.set_title(f"{ticker} — All Model Predictions vs Actual", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def plot_rmse_comparison(metrics_rows: list[dict]) -> plt.Figure:
    """Bar chart comparing RMSE across models."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names  = [r["Model"] for r in metrics_rows]
    rmses  = [r["RMSE"] for r in metrics_rows]
    colors = ["#FFD700", "#FF6B35", "#9B59B6"]
    bars   = ax.bar(names, rmses, color=colors, width=0.45, alpha=0.9)
    for bar, val in zip(bars, rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"₹{val:.2f}", ha="center", fontsize=9, color="#FAFAFA"
        )
    ax.set_ylabel("RMSE (₹)")
    ax.set_title("RMSE Comparison Across Models", fontsize=12)
    fig.tight_layout()
    return fig


def plot_single_model(result: dict, model_name: str, ticker: str,
                      color: str) -> plt.Figure:
    """Plot actual vs predicted for a single model."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(result["dates"], result["actuals"],
            color="#00C8FF", linewidth=1.6, label="Actual Price")
    ax.plot(result["dates"], result["predictions"],
            color=color, linewidth=1.5, linestyle="--",
            label=f"Predicted ({model_name})")
    ax.set_title(f"{ticker} — {model_name} Predictions", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Banner image
    banner_path = os.path.join(
        os.path.dirname(__file__), "Stock_banner", "Stock price banner.png"
    )
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)

    st.markdown("# 📈 StockSense AI")
    st.markdown(
        "<div class='custom-info'>Hybrid ML predictions powered by "
        "FinBERT sentiment analysis.</div>",
        unsafe_allow_html=True,
    )

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>📌 Stock Input</div>",
                unsafe_allow_html=True)

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="TCS.NS",
        placeholder="e.g. TCS.NS, RELIANCE.NS, INFY.NS",
        help="Enter NSE ticker (ends in .NS) or BSE (.BO)",
    ).strip().upper()

    company_name = get_company_name(ticker_input)
    st.caption(f"🏢 **Company:** {company_name}")

    # ── Date range ────────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>📅 Date Range</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=date(2022, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today() - timedelta(days=30),
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=date.today(),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )

    if start_date >= end_date:
        st.error("⚠️ Start date must be before end date.")
        st.stop()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # ── Model settings ────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>⚙️ Model Settings</div>",
                unsafe_allow_html=True)

    force_retrain_lstm = st.checkbox(
        "🔄 Force Retrain LSTM",
        value=False,
        help="If unchecked, existing LSTM_sentiment_model.keras is loaded.",
    )
    lstm_epochs = st.slider("LSTM Epochs", 5, 100, 30, 5)

    # ── API Key info ──────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'>🔑 API Status</div>",
                unsafe_allow_html=True)

    if NEWSDATA_API_KEY:
        st.success("✅ newsdata.io key loaded from .env")
    else:
        st.warning("⚠️ No NEWSDATA_API_KEY in .env — news disabled")
        manual_key = st.text_input("Enter API Key (optional)", type="password")
        if manual_key:
            NEWSDATA_API_KEY = manual_key

    # ── Analyse button ────────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
    run_analysis = st.button("🚀 Fetch & Analyse")

    st.markdown("---")
    st.caption("StockSense AI v2.0 • FinBERT + ML/DL Hybrid")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='text-align:center;'>"
    f"📊 StockSense AI — <span style='color:#00C8FF'>{ticker_input}</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center;color:#888;'>"
    f"{company_name} &nbsp;|&nbsp; {start_str} → {end_str}</p>",
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "stock_df"      not in st.session_state:
    st.session_state.stock_df      = None
if "news_df"       not in st.session_state:
    st.session_state.news_df       = None
if "sentiment_df"  not in st.session_state:
    st.session_state.sentiment_df  = None
if "merged_df"     not in st.session_state:
    st.session_state.merged_df     = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "daily_sent"    not in st.session_state:
    st.session_state.daily_sent    = pd.Series(dtype=float)
if "sent_summary"  not in st.session_state:
    st.session_state.sent_summary  = {}


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE  (triggered by "Fetch & Analyse" button)
# ══════════════════════════════════════════════════════════════════════════════
if run_analysis:
    progress_placeholder = st.empty()

    # ── Step 1: Load stock data ────────────────────────────────────────────
    with st.spinner(f"⬇️ Downloading {ticker_input} from Yahoo Finance …"):
        stock_df = load_stock_data(ticker_input, start_str, end_str)

    if stock_df is None or stock_df.empty:
        st.error(
            f"❌ No data found for **{ticker_input}**. "
            "Check the ticker symbol and date range."
        )
        st.stop()

    stock_df["MA_100"] = stock_df["Close"].rolling(100).mean()
    stock_df["MA_200"] = stock_df["Close"].rolling(200).mean()
    st.session_state.stock_df = stock_df
    st.success(f"✅ Loaded {len(stock_df):,} trading days for {ticker_input}")

    # ── Step 2: Fetch news ────────────────────────────────────────────────
    # NOTE: newsdata.io free plan fetches the MOST RECENT articles only.
    # Historical date-range filtering requires the paid Archive endpoint.
    news_df = pd.DataFrame()
    if NEWSDATA_API_KEY:
        with st.spinner(
            f"📰 Fetching recent news for {ticker_input} from newsdata.io …"
        ):
            try:
                news_df = cached_fetch_news(
                    ticker_input, company_name,
                    start_str, end_str, NEWSDATA_API_KEY,
                )
                if news_df.empty:
                    st.info(
                        f"ℹ️ No recent news articles found for **{ticker_input}**. "
                        "The free newsdata.io plan returns the latest available "
                        "articles — continuing with neutral sentiment (0)."
                    )
                else:
                    st.success(
                        f"✅ Fetched {len(news_df)} recent news articles for "
                        f"{ticker_input} (free plan: latest articles only)"
                    )
            except RateLimitError as e:
                st.warning(f"⚠️ {e}")
            except Exception as e:
                st.warning(f"⚠️ News fetch failed: {e}")
    else:
        st.info("ℹ️ No API key set — running without news sentiment.")

    st.session_state.news_df = news_df

    # ── Step 3: FinBERT inference ─────────────────────────────────────────
    sentiment_df  = pd.DataFrame()
    daily_sent    = pd.Series(dtype=float)
    sent_summary  = {}

    if not news_df.empty:
        with st.spinner("🧠 Running FinBERT sentiment analysis (this may take 1–3 min) …"):
            # NOTE: progress bar inside a cached function cannot be updated
            # mid-execution — we show it as complete once the call returns.
            finbert_bar = st.progress(0, text="Analysing articles with FinBERT …")
            try:
                news_hash = str(len(news_df)) + str(news_df["pubDate"].iloc[0])
                sentiment_df = cached_finbert(news_hash, news_df)
                sentiment_df["sentiment_score"] = pd.to_numeric(
                    sentiment_df["sentiment_score"], errors="coerce"
                ).fillna(0.0)

                daily_sent   = aggregate_daily_sentiment(sentiment_df)
                sent_summary = get_sentiment_summary(sentiment_df)
                finbert_bar.progress(1.0, text="FinBERT analysis complete ✅")
                st.success(f"✅ FinBERT analysed {len(news_df)} articles")
            except Exception as e:
                finbert_bar.empty()
                st.warning(f"⚠️ FinBERT failed: {e}. Using neutral sentiment.")

    st.session_state.sentiment_df = sentiment_df
    st.session_state.daily_sent   = daily_sent
    st.session_state.sent_summary = sent_summary

    # ── Step 4: Merge sentiment with stock data ───────────────────────────
    with st.spinner("🔗 Merging sentiment scores with stock data …"):
        idx_df = stock_df.set_index("Date")
        merged_df = merge_sentiment_with_stock(idx_df, daily_sent)
        merged_df = merged_df.reset_index()  # brings 'Date' back as column
    st.session_state.merged_df = merged_df

    # ── Step 5: Train / load all models ──────────────────────────────────
    with st.spinner("🤖 Training hybrid models (this may take a few minutes) …"):
        lstm_prog = st.progress(0)

        def _lstm_cb(epoch, total, loss):
            lstm_prog.progress(min(epoch / total, 1.0))

        try:
            model_results = run_all_models(
                merged_df,
                force_retrain_lstm=force_retrain_lstm,
                lstm_epochs=lstm_epochs,
                progress_callbacks={"lstm": _lstm_cb},
            )
            st.success("✅ All models trained successfully")
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            model_results = {}

    st.session_state.model_results = model_results
    st.session_state.analysis_done = True


# ══════════════════════════════════════════════════════════════════════════════
# TABS  (always visible; data shown only after analysis)
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Stock Overview",
    "📰 News & Sentiment",
    "🤖 LSTM + FinBERT",
    "🌲 Random Forest + FinBERT",
    "📐 SVM + FinBERT",
    "📊 Model Comparison",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — STOCK OVERVIEW & MOVING AVERAGES
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"📈 {ticker_input} — Price Overview")

    if not st.session_state.analysis_done or st.session_state.stock_df is None:
        st.info("👈 Configure settings in the sidebar and click **Fetch & Analyse**.")
    else:
        sdf = st.session_state.stock_df

        # KPI row
        latest_close  = float(sdf["Close"].iloc[-1])
        prev_close    = float(sdf["Close"].iloc[-2]) if len(sdf) > 1 else latest_close
        change        = latest_close - prev_close
        change_pct    = change / prev_close * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest Close",  f"₹{latest_close:,.2f}",
                  f"{change:+.2f} ({change_pct:+.2f}%)")
        k2.metric("52-Week High",  f"₹{sdf['High'].max():,.2f}")
        k3.metric("52-Week Low",   f"₹{sdf['Low'].min():,.2f}")
        k4.metric("Avg Volume",
                  f"{int(sdf['Volume'].mean()):,}")

        st.pyplot(plot_price_and_mas(sdf, ticker_input))

        with st.expander("📋 Raw OHLCV Data"):
            display = sdf[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display.tail(30).reset_index(drop=True),
                         use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — NEWS & SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📰 Financial News & FinBERT Sentiment")

    if not st.session_state.analysis_done:
        st.info("👈 Run the analysis first.")
    else:
        sdf        = st.session_state.stock_df
        news_df    = st.session_state.news_df
        sent_df    = st.session_state.sentiment_df
        daily_sent = st.session_state.daily_sent
        summary    = st.session_state.sent_summary

        if news_df is None or news_df.empty:
            st.info(
                "ℹ️ No news data available for this run. "
                "This can happen if: (a) no API key is set, (b) the newsdata.io "
                "free plan returned no articles for this ticker, or (c) a network "
                "error occurred. Predictions use neutral sentiment (score = 0)."
            )
        else:
            # ── Free-plan information note ────────────────────────────────
            st.info(
                "📌 **newsdata.io free plan** — articles shown are the most "
                "recently published matching this stock. Historical date-range "
                "filtering requires a paid Archive subscription."
            )

            # ── Summary KPIs ─────────────────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Articles", summary.get("total_articles", 0))
            s2.metric("Positive",       f"{summary.get('positive_pct', 0):.1f}%")
            s3.metric("Negative",       f"{summary.get('negative_pct', 0):.1f}%")
            s4.metric("Avg Score",      f"{summary.get('avg_score', 0):.3f}")

            dom = summary.get("dominant_sentiment", "neutral")
            dom_color = {"positive": "green", "negative": "red", "neutral": "grey"}[dom]
            st.markdown(
                f"**Dominant Sentiment:** "
                f"<span style='color:{dom_color};font-weight:700;'>"
                f"{dom.capitalize()}</span>",
                unsafe_allow_html=True,
            )

            # ── Charts ────────────────────────────────────────────────────
            ch1, ch2 = st.columns([3, 2])
            with ch1:
                st.pyplot(plot_sentiment_timeline(sdf, daily_sent, ticker_input))
            with ch2:
                st.pyplot(plot_sentiment_distribution(summary))

            # ── News table ────────────────────────────────────────────────
            st.subheader("📄 Article-Level Sentiment")
            if not sent_df.empty:
                disp_cols = ["pubDate", "title", "source_id",
                             "sentiment_label", "sentiment_score", "confidence"]
                disp_cols = [c for c in disp_cols if c in sent_df.columns]
                table = sent_df[disp_cols].copy()
                if "pubDate" in table.columns:
                    # Handle both tz-aware and tz-naive datetimes safely
                    pub = pd.to_datetime(table["pubDate"], utc=True, errors="coerce")
                    table["pubDate"] = pub.dt.strftime("%Y-%m-%d")
                if "sentiment_score" in table.columns:
                    table["sentiment_score"] = table["sentiment_score"].round(4)
                if "confidence" in table.columns:
                    table["confidence"] = table["confidence"].round(4)
                st.dataframe(table, use_container_width=True, height=400)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — LSTM + FINBERT
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 LSTM + FinBERT Hybrid Model")

    if not st.session_state.analysis_done:
        st.info("👈 Run the analysis first.")
    else:
        res = (st.session_state.model_results or {}).get("lstm", {})

        if "error" in res:
            st.error(f"❌ LSTM Training Error: {res['error']}")
        elif "predictions" not in res:
            st.warning("LSTM results not available.")
        else:
            m = res["metrics"]

            # ── Status ───────────────────────────────────────────────────
            if res.get("model_loaded"):
                st.success("✅ LSTM loaded from saved checkpoint (LSTM_sentiment_model.keras)")
            else:
                st.success("✅ LSTM trained from scratch with sentiment features")

            # ── Metrics ───────────────────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE",           f"₹{m['mae']:.2f}")
            c2.metric("MSE",           f"{m['mse']:.2f}")
            c3.metric("RMSE",          f"₹{m['rmse']:.2f}")
            c4.metric("R² Score",      f"{m['r2']:.4f}")
            c5.metric("Training Time", f"{res['training_time']}s")

            st.pyplot(
                plot_single_model(res, "LSTM + FinBERT",
                                  ticker_input, "#FFD700")
            )
            st.markdown(
                "<div class='custom-info'>"
                "📌 <strong>Architecture:</strong> "
                "LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → "
                "Dense(32) → Dense(1)<br>"
                "📌 <strong>Input features (6):</strong> "
                "Open, High, Low, Close, Volume, Daily_Sentiment_Score<br>"
                f"📌 <strong>Lookback window:</strong> 100 days"
                "</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — RANDOM FOREST + FINBERT
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🌲 Random Forest + FinBERT Hybrid Model")

    if not st.session_state.analysis_done:
        st.info("👈 Run the analysis first.")
    else:
        res = (st.session_state.model_results or {}).get("rf", {})

        if "error" in res:
            st.error(f"❌ Random Forest Error: {res['error']}")
        elif "predictions" not in res:
            st.warning("RF results not available.")
        else:
            m = res["metrics"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE",           f"₹{m['mae']:.2f}")
            c2.metric("MSE",           f"{m['mse']:.2f}")
            c3.metric("RMSE",          f"₹{m['rmse']:.2f}")
            c4.metric("R² Score",      f"{m['r2']:.4f}")
            c5.metric("Training Time", f"{res['training_time']}s")

            st.pyplot(
                plot_single_model(res, "Random Forest + FinBERT",
                                  ticker_input, "#FF6B35")
            )
            st.markdown(
                "<div class='custom-info'>"
                "📌 <strong>Model:</strong> RandomForestRegressor "
                "(n_estimators=200, max_depth=10)<br>"
                "📌 <strong>Input features (9):</strong> "
                "Open, High, Low, Close, Volume, MA_100, MA_200, "
                "Daily_Sentiment_Score, Sentiment_Rolling_7d_Avg<br>"
                "📌 <strong>Target:</strong> Next-day Close price"
                "</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — SVM + FINBERT
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📐 SVM (SVR) + FinBERT Hybrid Model")

    if not st.session_state.analysis_done:
        st.info("👈 Run the analysis first.")
    else:
        res = (st.session_state.model_results or {}).get("svm", {})

        if "error" in res:
            st.error(f"❌ SVM Error: {res['error']}")
        elif "predictions" not in res:
            st.warning("SVM results not available.")
        else:
            m = res["metrics"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE",           f"₹{m['mae']:.2f}")
            c2.metric("MSE",           f"{m['mse']:.2f}")
            c3.metric("RMSE",          f"₹{m['rmse']:.2f}")
            c4.metric("R² Score",      f"{m['r2']:.4f}")
            c5.metric("Training Time", f"{res['training_time']}s")

            st.pyplot(
                plot_single_model(res, "SVM + FinBERT",
                                  ticker_input, "#9B59B6")
            )
            st.markdown(
                "<div class='custom-info'>"
                "📌 <strong>Model:</strong> SVR (kernel=rbf, C=100, γ=0.1, ε=0.1)<br>"
                "📌 <strong>Input features (9):</strong> "
                "Open, High, Low, Close, Volume, MA_100, MA_200, "
                "Daily_Sentiment_Score, Sentiment_Rolling_7d_Avg<br>"
                "📌 <strong>Scaling:</strong> MinMaxScaler on features AND target"
                "</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — MODEL COMPARISON DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("📊 Model Comparison Dashboard")

    if not st.session_state.analysis_done:
        st.info("👈 Run the analysis first.")
    else:
        results = st.session_state.model_results or {}

        # ── Overlay chart ─────────────────────────────────────────────────
        st.pyplot(plot_model_predictions(results, ticker_input))

        # ── Metrics table ─────────────────────────────────────────────────
        st.subheader("📋 Performance Metrics Summary")

        metrics_rows = []
        model_display = {
            "lstm": "LSTM + Sentiment",
            "rf":   "Random Forest + Sentiment",
            "svm":  "SVM + Sentiment",
        }
        for key, label in model_display.items():
            r = results.get(key, {})
            if "error" in r:
                metrics_rows.append({
                    "Model": label, "MAE": "—", "MSE": "—",
                    "RMSE": "—", "R²": "—", "Training Time": "Error",
                })
            elif "metrics" in r:
                m = r["metrics"]
                metrics_rows.append({
                    "Model":         label,
                    "MAE":           round(m["mae"],  2),
                    "MSE":           round(m["mse"],  2),
                    "RMSE":          round(m["rmse"], 2),
                    "R²":            round(m["r2"],   4),
                    "Training Time": f"{r['training_time']}s",
                })

        if metrics_rows:
            df_metrics = pd.DataFrame(metrics_rows)
            st.dataframe(
                df_metrics.set_index("Model"),
                use_container_width=True,
            )

            # ── RMSE bar chart ────────────────────────────────────────────
            numeric_rows = [
                r for r in metrics_rows
                if isinstance(r.get("RMSE"), (int, float))
            ]
            if numeric_rows:
                cmp1, cmp2 = st.columns([1, 2])
                with cmp1:
                    st.pyplot(plot_rmse_comparison(numeric_rows))

                # ── Auto-conclusion ───────────────────────────────────────
                with cmp2:
                    st.subheader("🏆 Model Verdict")
                    best = min(numeric_rows, key=lambda x: x["RMSE"])
                    worst = max(numeric_rows, key=lambda x: x["RMSE"])
                    st.markdown(
                        f"""
<div style="background:#1A2035;border-left:4px solid #00C076;
            padding:16px 20px;border-radius:6px;line-height:1.7;">
🥇 <strong style="color:#00C076">Best Model: {best['Model']}</strong><br>
&nbsp;&nbsp;RMSE = ₹{best['RMSE']:.2f} &nbsp;|&nbsp; R² = {df_metrics.loc[df_metrics['Model']==best['Model'],'R²'].values[0]:.4f}

<br><br>

📊 <strong>All Models Ranked by RMSE:</strong><br>
""" + "".join(
                            f"&nbsp;&nbsp;{i+1}. {r['Model']} — RMSE ₹{r['RMSE']:.2f}<br>"
                            for i, r in enumerate(
                                sorted(numeric_rows, key=lambda x: x["RMSE"])
                            )
                        ) + f"""
<br>
🔻 <strong style="color:#FF4B4B">Highest RMSE: {worst['Model']}</strong> (₹{worst['RMSE']:.2f})
<br><br>
<em style="color:#888;">Lower RMSE = better price prediction accuracy.
The best model captures both price momentum and news sentiment signals most effectively.</em>
</div>
""",
                        unsafe_allow_html=True,
                    )
