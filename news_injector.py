"""
news_injector.py
================
Manual News Injection handler for StockSense AI.

Provides three modes for entering news:
  1. Manual Entry   — enter up to 10 individual headlines with metadata
  2. Paste Bulk     — paste multiple headlines (one per line)
  3. Upload CSV     — upload a CSV with date, headline, description columns

All injected news is run through FinBERT and merged with any API-fetched news.

Author: StockSense AI — Research-Grade Upgrade
"""

from __future__ import annotations

import io
import hashlib
import logging
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── News categories ────────────────────────────────────────────────────────────
NEWS_CATEGORIES = [
    "Earnings",
    "Merger/Acquisition",
    "Regulatory",
    "Product Launch",
    "Macro/Economy",
    "Management Change",
    "Analyst Rating",
    "Other",
]

# ── Expected CSV columns ───────────────────────────────────────────────────────
REQUIRED_CSV_COLS = ["date", "headline"]
OPTIONAL_CSV_COLS = ["description", "category"]


# ──────────────────────────────────────────────────────────────────────────────
# PARSING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def parse_manual_entries(entries_list: list[dict]) -> pd.DataFrame:
    """
    Parse a list of manually entered news entries into a standardised DataFrame.

    Parameters
    ----------
    entries_list : list[dict]
        Each dict must contain: 'date', 'headline'.
        Optionally: 'description', 'category'.

    Returns
    -------
    pd.DataFrame
        Columns: [date, headline, description, category]
        Returns empty DataFrame if no valid entries.
    """
    records = []
    for entry in entries_list:
        headline = (entry.get("headline") or "").strip()
        if not headline:
            continue  # skip empty headlines
        records.append({
            "date":        pd.to_datetime(entry.get("date", date.today())),
            "headline":    headline,
            "description": (entry.get("description") or "").strip(),
            "category":    (entry.get("category") or "Other").strip(),
            "source":      "manual_entry",
        })

    if not records:
        return _empty_injected_df()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def parse_bulk_text(text: str, bulk_date: date) -> pd.DataFrame:
    """
    Parse a newline-separated block of headlines into a DataFrame.

    Parameters
    ----------
    text : str
        One headline per line.
    bulk_date : date
        Date to apply to all headlines.

    Returns
    -------
    pd.DataFrame
        Columns: [date, headline, description, category, source]
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return _empty_injected_df()

    records = [{
        "date":        pd.to_datetime(bulk_date),
        "headline":    line,
        "description": "",
        "category":    "Other",
        "source":      "bulk_paste",
    } for line in lines]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.reset_index(drop=True)


def parse_csv_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Parse an uploaded CSV file into a news DataFrame.

    Expected columns: date, headline, description (optional)

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit file uploader object.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (parsed_df, error_message)
        error_message is empty string if success.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return _empty_injected_df(), f"Could not read CSV: {e}"

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Check required columns
    missing = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        return (
            _empty_injected_df(),
            f"CSV missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_CSV_COLS}",
        )

    # Add optional columns if missing
    if "description" not in df.columns:
        df["description"] = ""
    if "category" not in df.columns:
        df["category"] = "Other"

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # Filter to non-empty headlines
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].ne("") & df["headline"].ne("nan")]

    if df.empty:
        return _empty_injected_df(), "No valid rows found after parsing."

    df["source"] = "csv_upload"
    return df[["date", "headline", "description", "category", "source"]].reset_index(drop=True), ""


def validate_news_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that a news DataFrame has the required structure and content.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "No news entries found."
    if "headline" not in df.columns:
        return False, "Missing required 'headline' column."
    if "date" not in df.columns:
        return False, "Missing required 'date' column."
    empty_headlines = df["headline"].astype(str).str.strip().eq("").sum()
    if empty_headlines == len(df):
        return False, "All headlines are empty."
    return True, ""


def merge_injected_with_api_news(
    injected_df: pd.DataFrame,
    api_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge manually injected news with API-fetched news, deduplicating by headline.

    Parameters
    ----------
    injected_df : pd.DataFrame
        Manual entries with columns [date, headline, description, category, source].
    api_df : pd.DataFrame
        API-fetched news with columns [title, description, pubDate, source_id, link].

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with unified columns:
        [date, headline, description, category, source]
        Deduplicated on (date, headline).
    """
    combined_parts = []

    # Process API news into unified schema
    if api_df is not None and not api_df.empty:
        api_unified = pd.DataFrame({
            "date":        pd.to_datetime(api_df.get("pubDate", pd.Series()), errors="coerce").dt.tz_localize(None).dt.normalize(),
            "headline":    api_df.get("title", pd.Series()).fillna("").astype(str),
            "description": api_df.get("description", pd.Series()).fillna("").astype(str),
            "category":    "Other",
            "source":      "newsdata_api",
        })
        api_unified = api_unified.dropna(subset=["date"])
        api_unified = api_unified[api_unified["headline"].str.strip().ne("")]
        combined_parts.append(api_unified)

    # Add injected news
    if injected_df is not None and not injected_df.empty:
        inj = injected_df.copy()
        inj["date"] = pd.to_datetime(inj["date"], errors="coerce").dt.normalize()
        combined_parts.append(inj[["date", "headline", "description", "category", "source"]])

    if not combined_parts:
        return _empty_injected_df()

    combined = pd.concat(combined_parts, ignore_index=True)

    # Deduplicate on (date, headline) — keep first occurrence (API takes precedence)
    combined["_headline_lower"] = combined["headline"].str.lower().str.strip()
    combined = combined.drop_duplicates(subset=["date", "_headline_lower"]).drop(columns=["_headline_lower"])
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def _empty_injected_df() -> pd.DataFrame:
    """Return an empty DataFrame with the injected news schema."""
    return pd.DataFrame(columns=["date", "headline", "description", "category", "source"])


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI COMPONENT
# ──────────────────────────────────────────────────────────────────────────────

def render_news_injector_ui() -> Optional[pd.DataFrame]:
    """
    Render the manual news injection UI in the Streamlit sidebar.

    Returns
    -------
    Optional[pd.DataFrame]
        Injected news DataFrame if the user submitted entries, else None.
    """
    st.subheader("📰 Manual News Injection")
    st.caption("Bypass the paid API and enter news directly to influence predictions.")

    mode = st.radio(
        "Input Mode",
        options=["Manual Entry", "Paste Bulk News", "Upload CSV"],
        horizontal=True,
        key="news_injector_mode",
    )

    injected_df: Optional[pd.DataFrame] = None

    # ── MODE 1: MANUAL ENTRY ──────────────────────────────────────────────────
    if mode == "Manual Entry":
        if "news_entry_count" not in st.session_state:
            st.session_state.news_entry_count = 1

        entries = []
        for i in range(st.session_state.news_entry_count):
            with st.expander(f"📄 News Item {i+1}", expanded=(i == 0)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    headline = st.text_input(
                        "Headline *",
                        key=f"headline_{i}",
                        placeholder="e.g. TCS Q3 results beat analyst estimates by 12%",
                    )
                with col2:
                    news_date = st.date_input(
                        "News Date",
                        value=date.today(),
                        key=f"news_date_{i}",
                    )
                description = st.text_area(
                    "Description (optional)",
                    key=f"description_{i}",
                    height=80,
                    placeholder="Additional context about the news...",
                )
                category = st.selectbox(
                    "Category",
                    options=NEWS_CATEGORIES,
                    key=f"category_{i}",
                )
                entries.append({
                    "headline":    headline,
                    "date":        news_date,
                    "description": description,
                    "category":    category,
                })

        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("➕ Add Another News Item", key="add_news_item"):
                if st.session_state.news_entry_count < 10:
                    st.session_state.news_entry_count += 1
                    st.rerun()
        with col_clear:
            if st.button("🗑️ Clear All", key="clear_news_items"):
                st.session_state.news_entry_count = 1
                st.rerun()

        if st.button("✅ Apply Manual News", type="primary", key="apply_manual_news"):
            injected_df = parse_manual_entries(entries)
            if injected_df.empty:
                st.warning("⚠️ No valid headlines entered.")

    # ── MODE 2: PASTE BULK NEWS ───────────────────────────────────────────────
    elif mode == "Paste Bulk News":
        bulk_text = st.text_area(
            "Paste headlines (one per line)",
            height=200,
            placeholder="TCS wins $500M cloud migration deal\nInfosys Q3 revenue up 8% YoY\nRBI raises interest rates by 25bps...",
            key="bulk_news_text",
        )
        bulk_date = st.date_input(
            "Apply all to date",
            value=date.today(),
            key="bulk_news_date",
        )
        if st.button("✅ Apply Bulk News", type="primary", key="apply_bulk_news"):
            if bulk_text.strip():
                injected_df = parse_bulk_text(bulk_text, bulk_date)
                st.success(f"✅ Parsed {len(injected_df)} headlines from bulk text.")
            else:
                st.warning("⚠️ No text entered.")

    # ── MODE 3: UPLOAD CSV ────────────────────────────────────────────────────
    else:
        st.info(
            "📋 **Expected CSV format:**\n"
            "- Required columns: `date`, `headline`\n"
            "- Optional columns: `description`, `category`\n"
            "- Date format: YYYY-MM-DD"
        )
        uploaded_file = st.file_uploader(
            "Upload news CSV",
            type=["csv"],
            key="news_csv_uploader",
        )
        if uploaded_file is not None:
            parsed_df, err = parse_csv_upload(uploaded_file)
            if err:
                st.error(f"❌ {err}")
            else:
                st.success(f"✅ Loaded {len(parsed_df)} rows from CSV.")
                st.dataframe(parsed_df.head(10), use_container_width=True)
                if st.button("✅ Apply CSV News", type="primary", key="apply_csv_news"):
                    injected_df = parsed_df

    return injected_df


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def display_injected_news_table(df: pd.DataFrame) -> None:
    """
    Display a styled sentiment preview table for injected news.

    Shows per-article sentiment badge + score and a daily aggregated summary.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: [date, headline, sentiment_label, sentiment_score]
    """
    if df.empty:
        st.info("No injected news to display.")
        return

    st.markdown("### 📋 Injected News — Sentiment Preview")

    # Build display rows
    display_rows = []
    for _, row in df.iterrows():
        label = row.get("sentiment_label", "neutral")
        score = row.get("sentiment_score", 0.0)
        if label == "positive":
            badge = "🟢 Positive"
        elif label == "negative":
            badge = "🔴 Negative"
        else:
            badge = "⚪ Neutral"

        display_rows.append({
            "Date":      str(row.get("date", ""))[:10],
            "Headline":  str(row.get("headline", ""))[:80] + ("…" if len(str(row.get("headline", ""))) > 80 else ""),
            "Category":  str(row.get("category", "Other")),
            "Sentiment": badge,
            "Score":     round(float(score), 4) if pd.notna(score) else 0.0,
        })

    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        height=min(400, 50 + 35 * len(display_rows)),
    )

    # Daily aggregated sentiment preview
    if "sentiment_score" in df.columns:
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.normalize()
        daily = df_copy.groupby("date")["sentiment_score"].mean()

        overall = float(daily.mean()) if not daily.empty else 0.0
        if overall > 0.2:
            verdict = f"📈 BULLISH (+{overall:.2f})"
            color = "#00C076"
        elif overall < -0.2:
            verdict = f"📉 BEARISH ({overall:.2f})"
            color = "#FF4B4B"
        else:
            verdict = f"➡️ NEUTRAL ({overall:.2f})"
            color = "#AAAAAA"

        date_range = ""
        if not daily.empty:
            d1 = daily.index.min().strftime("%Y-%m-%d")
            d2 = daily.index.max().strftime("%Y-%m-%d")
            date_range = f" from **{d1}** to **{d2}**"

        st.markdown(
            f"""<div style="background:#1A2035;border-left:4px solid {color};
            padding:14px 18px;border-radius:6px;margin:10px 0;font-size:14px;">
            🧠 <strong>Sentiment Impact Preview:</strong><br>
            Based on injected news{date_range}, overall market sentiment is:
            <span style="color:{color};font-weight:700;font-size:16px;"> {verdict}</span>
            </div>""",
            unsafe_allow_html=True,
        )


def compute_injected_news_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the injected news DataFrame for cache invalidation.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        MD5 hash string.
    """
    if df is None or df.empty:
        return "empty"
    key = str(len(df)) + str(df["headline"].iloc[0] if "headline" in df.columns else "")
    return hashlib.md5(key.encode()).hexdigest()
