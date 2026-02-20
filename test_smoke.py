"""
test_smoke.py — Smoke test for RF + SVM trainers.
Downloads a small slice of TCS.NS from yfinance (fallback if CSV broken).
"""
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf

print("Loading stock data via yfinance…")
try:
    df = yf.download("TCS.NS", start="2021-01-01", end="2024-01-01", progress=False)
    if df.empty:
        raise ValueError("Empty dataframe from yfinance")
    # Flatten MultiIndex columns (yfinance v0.2+)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
    print(f"  Loaded {len(df)} rows from yfinance")
except Exception as e:
    print(f"  yfinance failed: {e}")
    sys.exit(1)

# Confirm required columns exist
required = ["Date", "Open", "High", "Low", "Close", "Volume"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    print(f"  Available: {list(df.columns)}")
    sys.exit(1)

print(f"  Columns: {list(df.columns)}")
print(f"  Rows after cleaning: {len(df)}")

# Merge with empty sentiment (all neutral)
from sentiment_utils import merge_sentiment_with_stock

df_indexed = df.set_index("Date")
merged = merge_sentiment_with_stock(df_indexed, pd.Series(dtype=float))
print(f"\nMerged shape : {merged.shape}")
print(f"Columns      : {list(merged.columns)}")
print(f"Index type   : {type(merged.index).__name__}")

# ── Random Forest ──────────────────────────────────────────────────────────────
print("\nTesting Random Forest…")
from model_trainer import train_rf_sentiment
try:
    rf = train_rf_sentiment(merged)
    m  = rf["metrics"]
    print(f"  MAE  = {m['mae']:.2f}")
    print(f"  RMSE = {m['rmse']:.2f}")
    print(f"  R²   = {m['r2']:.4f}")
    print(f"  Time = {rf['training_time']}s")
    print(f"  Predictions shape: {rf['predictions'].shape}")
    print("  ✅ RF PASSED")
except Exception as e:
    print(f"  ❌ RF FAILED: {e}")
    import traceback; traceback.print_exc()

# ── SVM ────────────────────────────────────────────────────────────────────────
print("\nTesting SVM…")
from model_trainer import train_svm_sentiment
try:
    svm = train_svm_sentiment(merged)
    m   = svm["metrics"]
    print(f"  MAE  = {m['mae']:.2f}")
    print(f"  RMSE = {m['rmse']:.2f}")
    print(f"  R²   = {m['r2']:.4f}")
    print(f"  Time = {svm['training_time']}s")
    print("  ✅ SVM PASSED")
except Exception as e:
    print(f"  ❌ SVM FAILED: {e}")
    import traceback; traceback.print_exc()

# ── Ticker mapper ──────────────────────────────────────────────────────────────
print("\nTesting ticker_mapper…")
from ticker_mapper import get_company_name
samples = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "SBIN.NS", "UNKNWN.NS", "AAPL"]
for t in samples:
    print(f"  {t:25s} → {get_company_name(t)}")
print("  ✅ Ticker mapper PASSED")

print("\n✅ Smoke test complete")
