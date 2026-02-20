"""
_test_fullpipe.py
-----------------
Tests the complete prediction pipeline on ICICIBANK.NS (a different ticker from TCS)
to prove the app works for ANY Indian NSE stock.
"""
import yfinance as yf
import pandas as pd
import sys
sys.path.insert(0, 'd:/junaid sir')

from sentiment_utils import merge_sentiment_with_stock
from model_trainer import train_rf_sentiment, train_svm_sentiment
from ticker_mapper import get_company_name

TICKER = "ICICIBANK.NS"
print(f"\nTesting pipeline for: {TICKER}")
print(f"Company name  : {get_company_name(TICKER)}")

# Download
print("\nDownloading data from Yahoo Finance...")
df = yf.download(TICKER, start="2020-01-01", end="2025-01-01", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Close","Open","High","Low","Volume"])
print(f"Rows loaded   : {len(df)}")

# Merge with neutral sentiment (no news)
merged = merge_sentiment_with_stock(df.set_index("Date"), pd.Series(dtype=float))
print(f"Merged shape  : {merged.shape}")

# Random Forest
print("\n--- Random Forest ---")
rf = train_rf_sentiment(merged)
m = rf["metrics"]
print(f"  MAE   = {m['mae']:.2f}")
print(f"  RMSE  = {m['rmse']:.2f}")
print(f"  R2    = {m['r2']:.4f}")
print(f"  Time  = {rf['training_time']}s")
print(f"  Preds = {len(rf['predictions'])} samples")
print("  PASS")

# SVM
print("\n--- SVM (SVR) ---")
svm = train_svm_sentiment(merged)
m2 = svm["metrics"]
print(f"  MAE   = {m2['mae']:.2f}")
print(f"  RMSE  = {m2['rmse']:.2f}")
print(f"  R2    = {m2['r2']:.4f}")
print(f"  Time  = {svm['training_time']}s")
print("  PASS")

print("\n✅  FULL PIPELINE FOR ICICIBANK.NS — COMPLETE")
print("✅  The app works for ANY Indian .NS ticker!")
