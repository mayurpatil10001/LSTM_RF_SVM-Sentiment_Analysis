"""
sentiment_utils.py
==================
Utilities for:
  1. Fetching financial news from newsdata.io REST API
  2. Running FinBERT sentiment inference (ProsusAI/finbert)
  3. Aggregating daily sentiment scores
  4. Merging sentiment with stock price DataFrames

All heavy model objects are loaded once and cached globally to avoid
re-loading on every Streamlit re-run.

Author: Upgraded Stock Prediction App
"""

from __future__ import annotations

import os
import time
import logging
import requests
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── FinBERT lazy-load globals ────────────────────────────────────────────────
_finbert_tokenizer = None
_finbert_model = None
_FINBERT_MODEL_NAME = "ProsusAI/finbert"

# ─── Sentiment label → numeric score mapping ─────────────────────────────────
LABEL_TO_SCORE: dict[str, float] = {
    "positive": +1.0,
    "negative": -1.0,
    "neutral":   0.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  NEWS FETCHING
# ──────────────────────────────────────────────────────────────────────────────

def _build_search_query(ticker: str, company_name: str) -> str:
    """
    Build an effective search query for a given stock.

    Combines the short ticker name (without .NS/.BO suffix) with the company
    name to maximise recall from newsdata.io free-tier searches.
    """
    # Strip exchange suffix  e.g. 'TCS.NS' → 'TCS'
    short_ticker = ticker.split(".")[0].upper()

    # Use the first two words of the company name for a focused query
    company_words = company_name.split()[:3]
    company_short = " ".join(company_words)

    # E.g.: "TCS Tata Consultancy" — broad enough to catch most articles
    return f"{short_ticker} {company_short}"


def fetch_news(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    api_key: str,
    max_results: int = 50,
) -> pd.DataFrame:
    """
    Fetch financial news articles from newsdata.io for a given company.

    NOTE: The newsdata.io free-tier /1/news endpoint does NOT support
    ``from_date`` / ``to_date`` date filtering (those parameters require the
    Archive endpoint which is a paid feature).  This function therefore
    fetches the most-recent articles matching the company query and returns
    them all — the date range parameters are accepted for API-compatibility
    but are NOT forwarded to the API.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. 'TCS.NS').
    company_name : str
        Full company name used to build the search query.
    start_date : str
        Start date in 'YYYY-MM-DD' format (kept for signature compatibility;
        NOT sent to the API on the free plan).
    end_date : str
        End date in 'YYYY-MM-DD' format (kept for signature compatibility;
        NOT sent to the API on the free plan).
    api_key : str
        newsdata.io API key.
    max_results : int
        Maximum number of articles to return (capped at 50 per API page).

    Returns
    -------
    pd.DataFrame
        Columns: ['title', 'description', 'pubDate', 'source_id', 'link']
        Returns empty DataFrame on failure or if no articles are found.
    """
    if not api_key or api_key.strip() == "":
        logger.warning("newsdata.io API key is missing — skipping news fetch.")
        return _empty_news_df()

    BASE_URL = "https://newsdata.io/api/1/news"
    query = _build_search_query(ticker, company_name)

    # ── Free-tier parameters (NO from_date / to_date / timeframe) ────────────
    params: dict = {
        "apikey":   api_key,
        "q":        query,
        "language": "en",
        "category": "business",
        "size":     min(max_results, 10),   # free plan: max 10 per page
    }

    articles: list[dict] = []
    next_page: Optional[str] = None
    collected = 0
    max_pages = max(1, max_results // 10)   # limit pagination to avoid rate-limit
    pages_fetched = 0

    try:
        while collected < max_results and pages_fetched < max_pages:
            if next_page:
                params["page"] = next_page
            elif pages_fetched > 0:
                break   # no next page token → stop

            resp = requests.get(BASE_URL, params=params, timeout=15)

            # ── Rate limit handling ──────────────────────────────────────────
            if resp.status_code == 429:
                logger.warning("newsdata.io rate-limit hit.")
                raise RateLimitError(
                    "newsdata.io rate limit reached. "
                    "Please wait a minute before retrying."
                )

            if resp.status_code != 200:
                # Log the full error body to help with debugging
                try:
                    err_body = resp.json()
                    msg = err_body.get("results", {}).get("message", resp.text[:200])
                except Exception:
                    msg = resp.text[:200]
                logger.error(
                    f"newsdata.io returned HTTP {resp.status_code}: {msg}"
                )
                break

            data = resp.json()
            if data.get("status") != "success":
                err_msg = (
                    data.get("results", {}).get("message")
                    or data.get("message")
                    or "Unknown API error"
                )
                logger.error(f"newsdata.io API error: {err_msg}")
                break

            results = data.get("results", []) or []
            if not results:
                break

            for art in results:
                articles.append({
                    "title":       art.get("title", "") or "",
                    "description": art.get("description", "") or "",
                    "pubDate":     art.get("pubDate", "") or "",
                    "source_id":   art.get("source_id", "") or "",
                    "link":        art.get("link", "") or "",
                })

            collected += len(results)
            next_page = data.get("nextPage")
            pages_fetched += 1

            if not next_page:
                break

    except RateLimitError:
        raise
    except requests.exceptions.ConnectionError:
        logger.error("No internet connection — cannot fetch news.")
        return _empty_news_df()
    except requests.exceptions.Timeout:
        logger.error("newsdata.io request timed out.")
        return _empty_news_df()
    except Exception as exc:
        logger.exception(f"Unexpected error fetching news: {exc}")
        return _empty_news_df()

    if not articles:
        logger.info(f"No articles found for query: '{query}'")
        return _empty_news_df()

    df = pd.DataFrame(articles)

    # ── Parse & clean publication dates ──────────────────────────────────────
    df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce", utc=True)
    # Convert to IST and keep as tz-aware for downstream compatibility
    df["pubDate"] = df["pubDate"].dt.tz_convert("Asia/Kolkata").dt.normalize()
    df = df.dropna(subset=["pubDate"])
    df = df[df["title"].str.strip().ne("")]   # drop empty-title rows
    df = df.sort_values("pubDate").reset_index(drop=True)

    logger.info(f"Fetched {len(df)} news articles for '{query}'")
    return df


def _empty_news_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected news schema."""
    return pd.DataFrame(columns=["title", "description", "pubDate", "source_id", "link"])


class RateLimitError(Exception):
    """Custom exception raised when the newsdata.io rate limit is exceeded."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FINBERT SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _load_finbert():
    """
    Lazily load the FinBERT tokenizer and model into module-level globals.
    Subsequent calls return immediately without reloading.
    """
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None or _finbert_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch  # noqa: F401 — confirm torch is available

            logger.info(f"Loading FinBERT from HuggingFace: {_FINBERT_MODEL_NAME}")
            _finbert_tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_NAME)
            _finbert_model = AutoModelForSequenceClassification.from_pretrained(
                _FINBERT_MODEL_NAME
            )
            _finbert_model.eval()
            logger.info("FinBERT loaded successfully.")
        except ImportError as e:
            raise ImportError(
                "transformers / torch not installed. "
                "Run: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT model: {e}") from e


def _run_finbert_single(text: str) -> tuple[str, float]:
    """
    Run FinBERT on a single text string.

    Parameters
    ----------
    text : str
        News headline + description (concatenated).

    Returns
    -------
    tuple[str, float]
        (label, confidence_score)  where label ∈ {positive, negative, neutral}
    """
    import torch
    import torch.nn.functional as F

    _load_finbert()

    # Truncate to 512 tokens to respect BERT's limit
    inputs = _finbert_tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = _finbert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

    # FinBERT label order: positive=0, negative=1, neutral=2
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())
    label = label_map[predicted_idx]

    return label, confidence


def run_finbert(
    articles_df: pd.DataFrame,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run FinBERT sentiment inference on each row of the articles DataFrame.

    Parameters
    ----------
    articles_df : pd.DataFrame
        Must contain at least 'title', 'description', 'pubDate' columns.
    progress_callback : callable, optional
        Called with (current_index, total) to report progress.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus columns:
        ['sentiment_label', 'sentiment_score', 'confidence']
    """
    if articles_df.empty:
        df = articles_df.copy()
        df["sentiment_label"] = pd.Series(dtype=str)
        df["sentiment_score"] = pd.Series(dtype=float)
        df["confidence"]      = pd.Series(dtype=float)
        return df

    labels: list[str]  = []
    scores: list[float] = []
    confidences: list[float] = []

    total = len(articles_df)

    for idx, row in articles_df.iterrows():
        # Combine headline + description for richer context
        text = f"{row.get('title', '')} {row.get('description', '')}".strip()

        if not text:
            labels.append("neutral")
            scores.append(0.0)
            confidences.append(1.0)
        else:
            try:
                label, conf = _run_finbert_single(text)
                labels.append(label)
                scores.append(LABEL_TO_SCORE[label] * conf)
                confidences.append(conf)
            except Exception as e:
                logger.warning(f"FinBERT inference failed on row {idx}: {e}")
                labels.append("neutral")
                scores.append(0.0)
                confidences.append(0.0)

        if progress_callback:
            progress_callback(len(labels), total)

    result = articles_df.copy()
    result["sentiment_label"] = labels
    result["sentiment_score"] = scores
    result["confidence"]      = confidences
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DAILY SENTIMENT AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_daily_sentiment(sentiment_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate per-article sentiment scores into a daily sentiment score.

    Aggregation method:
        daily_score = mean( sentiment_score )  where sentiment_score =
        (Positive ↦ +confidence, Negative ↦ −confidence, Neutral ↦ 0)

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Must contain 'pubDate' (datetime) and 'sentiment_score' (float).

    Returns
    -------
    pd.Series
        Index = date (DatetimeIndex, tz-naive), values = daily sentiment score.
        Returns empty Series if input is empty.
    """
    if sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
        return pd.Series(dtype=float)

    df = sentiment_df.copy()

    # Normalize pubDate to tz-naive date-only
    if hasattr(df["pubDate"].dtype, "tz") and df["pubDate"].dt.tz is not None:
        df["date"] = df["pubDate"].dt.tz_localize(None).dt.normalize()
    else:
        df["date"] = pd.to_datetime(df["pubDate"]).dt.normalize()

    daily = (
        df.groupby("date")["sentiment_score"]
          .mean()
          .rename("daily_sentiment")
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MERGE SENTIMENT WITH STOCK DATA
# ──────────────────────────────────────────────────────────────────────────────

def merge_sentiment_with_stock(
    stock_df: pd.DataFrame,
    sentiment_series: pd.Series,
    rolling_window: int = 7,
) -> pd.DataFrame:
    """
    Merge daily sentiment scores into the stock DataFrame.

    Missing sentiment days (weekdays with no news) are forward-filled first,
    then zero-filled (neutral) for any remaining gaps.

    Also computes a rolling average sentiment feature.

    Parameters
    ----------
    stock_df : pd.DataFrame
        DataFrame with DatetimeIndex (stock OHLCV data).
    sentiment_series : pd.Series
        Daily sentiment scores (indexed by date).
    rolling_window : int
        Window size for the rolling average sentiment feature.

    Returns
    -------
    pd.DataFrame
        stock_df with two new columns:
        - 'Daily_Sentiment_Score'    : merged & forward-filled sentiment
        - 'Sentiment_Rolling_7d_Avg': rolling average of the above
    """
    merged = stock_df.copy()

    # Ensure stock index is tz-naive for merging
    if hasattr(merged.index, "tz") and merged.index.tz is not None:
        merged.index = merged.index.tz_convert(None)

    # Align sentiment series index to tz-naive
    if not sentiment_series.empty:
        sent = sentiment_series.copy()
        try:
            if sent.index.tz is not None:
                sent.index = sent.index.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        sent.index = pd.to_datetime(sent.index)

        # Reindex to stock dates → forward fill → zero-fill remaining
        merged["Daily_Sentiment_Score"] = (
            sent.reindex(merged.index)
               .ffill()
               .fillna(0.0)
        )
    else:
        # No news available — all neutral
        merged["Daily_Sentiment_Score"] = 0.0

    # Rolling average sentiment
    merged["Sentiment_Rolling_7d_Avg"] = (
        merged["Daily_Sentiment_Score"]
              .rolling(window=rolling_window, min_periods=1)
              .mean()
    )

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 5.  HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def get_sentiment_summary(sentiment_df: pd.DataFrame) -> dict:
    """
    Compute an overall sentiment summary for the period.

    Returns
    -------
    dict with keys:
        total_articles, positive_pct, negative_pct, neutral_pct,
        avg_score, dominant_sentiment
    """
    if sentiment_df.empty or "sentiment_label" not in sentiment_df.columns:
        return {
            "total_articles":   0,
            "positive_pct":     0.0,
            "negative_pct":     0.0,
            "neutral_pct":      0.0,
            "avg_score":        0.0,
            "dominant_sentiment": "neutral",
        }

    total = len(sentiment_df)
    counts = sentiment_df["sentiment_label"].value_counts()
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    neu = counts.get("neutral",  0)

    avg_score = float(sentiment_df["sentiment_score"].mean())
    dominant = sentiment_df["sentiment_label"].mode()
    dominant_str = dominant.iloc[0] if not dominant.empty else "neutral"

    return {
        "total_articles":     total,
        "positive_pct":       round(pos / total * 100, 1),
        "negative_pct":       round(neg / total * 100, 1),
        "neutral_pct":        round(neu / total * 100, 1),
        "avg_score":          round(avg_score, 4),
        "dominant_sentiment": dominant_str,
    }


def label_color(label: str) -> str:
    """Return a hex color string for a sentiment label."""
    return {"positive": "#00C076", "negative": "#FF4B4B", "neutral": "#AAAAAA"}.get(
        label.lower(), "#AAAAAA"
    )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from ticker_mapper import get_company_name

    load_dotenv()
    api_key = os.getenv("NEWSDATA_API_KEY", "")

    ticker = "TCS.NS"
    company = get_company_name(ticker)
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Fetching news for {company} ({start} → {end}) …")
    news_df = fetch_news(ticker, company, start, end, api_key)
    print(f"  → {len(news_df)} articles fetched")

    if not news_df.empty:
        print("Running FinBERT …")
        sent_df = run_finbert(news_df)
        print(sent_df[["title", "sentiment_label", "sentiment_score"]].head())

        daily = aggregate_daily_sentiment(sent_df)
        print("\nDaily Sentiment:\n", daily)

        summary = get_sentiment_summary(sent_df)
        print("\nSummary:", summary)
