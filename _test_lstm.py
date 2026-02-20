"""Quick LSTM integration test with synthetic data (2 epochs)."""
import os, pandas as pd, numpy as np
from sentiment_utils import merge_sentiment_with_stock
from model_trainer import train_lstm_sentiment

# Synthetic stock data (600 rows -- needs >120 after LOOKBACK=100)
np.random.seed(0)
n = 600
idx = pd.date_range("2020-01-01", periods=n, freq="B")
close = 3000 + np.cumsum(np.random.randn(n) * 15)
synthetic = pd.DataFrame({
    "Open":   close * np.random.uniform(0.98, 1.00, n),
    "High":   close * np.random.uniform(1.00, 1.02, n),
    "Low":    close * np.random.uniform(0.97, 1.00, n),
    "Close":  close,
    "Volume": np.random.uniform(1e6, 5e6, n),
}, index=idx)

merged = merge_sentiment_with_stock(synthetic, pd.Series(dtype=float))
print(f"Data shape: {merged.shape}")

# Remove saved model so we always retrain
model_path = "LSTM_sentiment_model.keras"
if os.path.exists(model_path):
    os.remove(model_path)

print("Training LSTM (2 epochs)...")
lstm = train_lstm_sentiment(merged, force_retrain=True, epochs=2, batch_size=64)
m = lstm["metrics"]
print(f"  LSTM - MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.4f}, time={lstm['training_time']}s")
print(f"  Model loaded from file: {lstm['model_loaded']}")
print("  LSTM PASSED")

print("\n✅ LSTM integration test PASSED")
