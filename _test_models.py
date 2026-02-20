"""Quick model integration test with synthetic data."""
import pandas as pd, numpy as np
from sentiment_utils import merge_sentiment_with_stock
from model_trainer import train_rf_sentiment, train_svm_sentiment

# Synthetic stock data (500 rows)
np.random.seed(42)
n = 500
idx = pd.date_range("2021-01-01", periods=n, freq="B")
close = 3000 + np.cumsum(np.random.randn(n) * 20)
synthetic = pd.DataFrame({
    "Open":   close * np.random.uniform(0.98, 1.00, n),
    "High":   close * np.random.uniform(1.00, 1.02, n),
    "Low":    close * np.random.uniform(0.97, 1.00, n),
    "Close":  close,
    "Volume": np.random.uniform(1e6, 5e6, n),
}, index=idx)

merged = merge_sentiment_with_stock(synthetic, pd.Series(dtype=float))
print(f"Data shape: {merged.shape}")

# RF
print("Training RF...")
rf = train_rf_sentiment(merged)
m = rf["metrics"]
print(f"  RF - MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.4f}, time={rf['training_time']}s")
print("  RF PASSED")

# SVM
print("Training SVM...")
svm = train_svm_sentiment(merged)
m = svm["metrics"]
print(f"  SVM - MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.4f}, time={svm['training_time']}s")
print("  SVM PASSED")

print("\n✅ All model integration tests PASSED")
