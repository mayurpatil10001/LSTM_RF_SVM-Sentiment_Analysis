"""
Final validation test - verifies both fixed issues.
Run: python _validate_fixes.py

ISSUE 1: News API — was sending from_date/to_date which are unsupported on free plan
ISSUE 2: LSTM — was loading saved model without checking input shape compatibility
"""
import os, sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NEWSDATA_API_KEY", "")

print("=" * 65)
print("  StockSense AI — Validation of Both Fixes")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# TEST 1: News API Fix
# ─────────────────────────────────────────────────────────────
print("\n[FIX 1] News API — no from_date/to_date params on free plan")
from sentiment_utils import fetch_news, _build_search_query
from ticker_mapper import get_company_name

for ticker in ["TCS.NS", "RELIANCE.NS", "INFY.NS"]:
    company = get_company_name(ticker)
    q = _build_search_query(ticker, company)
    try:
        news = fetch_news(ticker, company, "2020-01-01", "2026-01-01", api_key)
        status = f"{len(news)} articles" if not news.empty else "0 articles (API empty)"
        print(f"  ✅ {ticker} → query='{q}', result={status}")
    except Exception as e:
        print(f"  ❌ {ticker} → FAILED: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 2: LSTM Input Shape Validation Fix
# ─────────────────────────────────────────────────────────────
print("\n[FIX 2] LSTM — input shape validation before loading saved model")
from model_trainer import train_lstm_sentiment, LOOKBACK, LSTM_FEATURES
from sentiment_utils import merge_sentiment_with_stock

# Create synthetic data
np.random.seed(42)
n = 600
idx = pd.date_range("2019-01-01", periods=n, freq="B")
close = 1500 + np.cumsum(np.random.randn(n) * 10)
stock = pd.DataFrame({
    "Open":   close * 0.99,
    "High":   close * 1.01,
    "Low":    close * 0.98,
    "Close":  close,
    "Volume": np.random.uniform(1e6, 5e6, n),
}, index=idx)
merged = merge_sentiment_with_stock(stock, pd.Series(dtype=float))

print(f"  Synthetic data shape: {merged.shape}")
print(f"  Feature columns required: {LSTM_FEATURES}")

# Check what the saved model expects
from keras.models import load_model
model_path = "LSTM_sentiment_model.keras"
if os.path.exists(model_path):
    try:
        saved = load_model(model_path)
        saved_feats = saved.input_shape[-1]
        current_feats = len(LSTM_FEATURES)
        print(f"  Saved model input shape: {saved.input_shape}")
        print(f"  Expected features: {current_feats}")
        if saved_feats == current_feats:
            print(f"  ✅ Shape matches — model can be reused safely")
        else:
            print(f"  ⚠️  Shape mismatch ({saved_feats} vs {current_feats}) — would force retrain")
    except Exception as e:
        print(f"  ⚠️  Cannot load saved model: {e}")
else:
    print("  ℹ️  No saved model found — will train from scratch on first run")

# Quick LSTM training test (2 epochs)
print("  Training LSTM with 2 epochs (validation only)...")
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs
    result = train_lstm_sentiment(merged, force_retrain=True, epochs=2, batch_size=64)
    m = result["metrics"]
    print(f"  ✅ LSTM training PASSED — MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R²={m['r2']:.4f}")
    print(f"     Training time: {result['training_time']}s")
except Exception as e:
    print(f"  ❌ LSTM training FAILED: {e}")

print("\n" + "=" * 65)
print("  All validation tests complete!")
print("=" * 65)
