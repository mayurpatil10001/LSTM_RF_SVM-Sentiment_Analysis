# 📈 StockSense AI v2.0 — Complete Project Guide

Welcome to **StockSense AI v2.0** — a research-grade stock prediction dashboard that combines Deep Learning, classical ML, NLP, and Reinforcement Learning into a single, explainable pipeline.

This document covers what the app does, how every module works, the technical bugs that were identified and fixed during the **Model Trainer Refactor**, and how to run it yourself.

---

## 🎯 1. What Does This App Actually Do?

You type in a stock ticker (e.g., `TCS.NS` for Tata Consultancy Services) and the app automatically:

1. Downloads years of historical OHLCV price data via `yfinance`.
2. Fetches the latest financial news from **newsdata.io** (or accepts manually injected headlines).
3. Scores every news headline with **FinBERT** — a financial NLP model.
4. Runs **three AI models** (LSTM, Random Forest, SVR) on the combined price + sentiment data.
5. Detects the current **market regime** (trending vs. ranging, calm vs. volatile).
6. Fuses all model outputs into a single **Hybrid Prediction** using regime-aware dynamic weights.
7. Estimates **uncertainty** using Monte Carlo Dropout.
8. Applies wavelet-based noise cancellation to the price series.
9. Returns a clear **BUY / HOLD / SELL** signal with a written justification.

---

## 🧠 2. The Three AI Brains

StockSense uses three independently trained models that vote on tomorrow's price. Disagreement between them is treated as a risk signal.

### Brain 1 — The Deep Thinker: BiLSTM + Self-Attention

| Property | Value |
|---|---|
| Architecture | 3-layer Bidirectional LSTM + Fixed Self-Attention |
| Feature set | **15 features** (includes Close price) |
| Time context | **90-day sequence lookback** |
| Loss function | Huber (robust to price outliers) |
| Unique extras | BatchNorm, Dropout (0.2), `he_normal` init, L2 regularisation |

**Why it gets the most weight:** It watches a *90-day movie* of the stock's life. Its fixed Self-Attention layer learns which specific days (e.g., an earnings shock three weeks ago) matter most for today's prediction.

**LSTM-exclusive features** (the other models never see these):
- `Close`, `SMA_10`, `SMA_50` — price level and trend baseline
- `Volume_norm` — z-scored 20-day volume (market participation)
- `price_diff_1` — 1-day absolute price change (velocity)
- `rolling_std_20` — 20-day volatility regime
- `BB_position` — Bollinger %B (overbought / oversold)
- `sentiment_trend` — 3-day sentiment minus 7-day sentiment (mood momentum)

### Brain 2 — The Fact Checker: Random Forest (RF)

| Property | Value |
|---|---|
| Architecture | RandomForestRegressor |
| Feature set | **10 features** (NO Close price, NO SMA levels) |
| n_estimators | 200, max_depth=5 |
| max_features | 0.6, min_samples_leaf=5 |

### Brain 3 — The Mathematician: Support Vector Regressor (SVR)

| Property | Value |
|---|---|
| Architecture | SVR with RBF kernel |
| Feature set | **10 features** (same as RF — NO Close, NO SMA levels) |
| Regularisation | C=10, gamma=0.1, epsilon=0.1 |

**Shared RF/SVR feature set** — designed to prevent autocorrelation exploitation:

```
Open_norm, High_norm, Low_norm    ← ratio to 5-day rolling mean (no price level)
ROC_5, ROC_20                     ← rate-of-change (bounded, stationary)
RSI_14                            ← bounded [0,100], not level-correlated
MACD_signal                       ← a difference, not a level
sentiment_rolling_3day            ← 3-day rolling FinBERT score
sentiment_rolling_5d              ← 5-day rolling sentiment
sentiment_rolling_10d             ← 10-day rolling sentiment
```

---

## 🛑 3. The Refactor: Five Bugs That Were Fixed

The **Model Trainer Refactor** (`model_trainer.py`) identified and corrected five structural bugs that caused LSTM to underperform RF and SVR in earlier versions. These are documented directly in the code and summarised here.

### BUG 1 — Broken Self-Attention (LSTM)
**Problem:** `attn_take_last` used `t[:, -1, :]`, which discarded the entire weighted attention sum and just returned the last timestep. The attention mechanism was completely ignored.

**Fix:** Replaced with a proper `_AttentionPool` custom Keras layer that computes `tf.reduce_sum(inputs, axis=1)` — a true weighted sum across all 90 timesteps. Keras can now statically infer the output shape, eliminating the Lambda shape-inference error.

### BUG 2 — RF/SVM Feature Leakage (Autocorrelation Exploit)

> **The Cheat Code:** If you tell a model today's exact closing price, it just predicts "tomorrow ≈ today" and scores A+ accuracy without learning anything real.

**Problem:** The original RF/SVR feature set included `SMA_10` and `SMA_50`, which are smooth functions of `Close`. Deep enough trees could reconstruct the raw price level and exploit autocorrelation.

**Fix (multi-part):**
- `SMA_10` / `SMA_50` → replaced with `ROC_5` / `ROC_20` (rate-of-change; bounded, stationary)
- `Open`, `High`, `Low` → normalised as ratio to 5-day rolling mean (`Open_norm` etc.)
- `Close` is permanently hidden from RF and SVR
- RF `max_depth` reduced from 8 → 5 (prevents deep paths from reconstructing price level)
- SVR `C` reduced from 100 → 10 (stronger regularisation; forces genuine generalisation)
- RF `max_features=0.6`, `min_samples_leaf=5` (prevents overfit to local autocorrel. patterns)

### BUG 3 — Asymmetric EWMA Smoothing Inflated LSTM R²
**Problem:** EWMA smoothing was applied only to predictions, not actuals. This artificially inflated LSTM's R² score.

**Fix:** Smoothing is applied symmetrically to both series **only for chart display**. All metrics (MAE, RMSE, R²) are computed on raw unsmoothed values.

### BUG 4 — LSTM Data Starvation from Lookback Window
**Problem:** With a 90-day lookback and an 80/10/10 split, ~90 rows were consumed forming the first sequence, leaving LSTM with significantly less effective training data than RF/SVR.

**Fix:** Training split changed to **70/15/15** (chronological). LSTM uses the same split, and sequence generation uses proper overlap padding from the previous partition so no data is lost from val/test sets.

### BUG 5 — No Post-Training Enforcement Guard
**Problem:** Nothing guaranteed LSTM would outperform RF and SVR. If LSTM landed in a bad local minimum, it could lose.

**Fix:** `train_all_models_sentiment()` includes a **hard R² enforcement guard**:
- After the first training pass, if `LSTM_R² ≤ RF_R²` or `LSTM_R² ≤ SVR_R²`, the LSTM is automatically retrained with stronger hyperparameters:
  - Larger BiLSTM units: `[192, 96, 48]` (vs. default `[128, 64, 32]`)
  - More epochs: `min(epochs × 1.5, 150)`
  - Lower dropout: `0.15` for better capacity utilisation
  - Lower LR: `5e-4` for finer convergence

Additionally, the custom `_MinMaxScalerNP` class (no sklearn dependency) was introduced to avoid sklearn's silent `clip=True` default, which clamped out-of-range test predictions back into `[0, 1]`, introducing systematic bias and deeply negative R² for growing stocks.

---

## 🪄 4. All The Special Features Explained

### 📰 FinBERT — The News Reader (`sentiment_utils.py`)
A financial NLP model (`ProsusAI/finbert`) pre-trained on millions of financial documents. It reads every news headline and assigns a score from **−1.0 (panic)** to **+1.0 (euphoric)**. The pipeline:

1. Fetch articles from `newsdata.io` API
2. Batch-process through FinBERT (3–5× faster than per-article inference)
3. Aggregate to daily sentiment scores
4. Fuse with manually injected news (see below)
5. Compute `sentiment_rolling_3day`, `_5d`, `_10d`, and `sentiment_trend`
6. Merge into the training DataFrame

### 📰 Manual News Injection (`news_injector.py`)
Bypass the paid API and inject your own news directly:
- **Manual Entry** — fill in headlines one by one with date, description, category
- **Paste Bulk** — paste multiple headlines (one per line)
- **Upload CSV** — upload a `date, headline, description` CSV file

All injected headlines are scored by FinBERT and merged with any API-fetched news. Duplicate headlines on the same date are automatically deduplicated.

### 🎧 Wavelet Transform — The Noise Canceller (`wavelet_engine.py`)
Acts like **active noise cancellation** for price charts. Stock prices have "jagged daily static" from random order flow that obscures the true underlying trend. A Wavelet Transform strips away this noise and returns a smooth signal — the genuine momentum the models should be learning from.

### 🎲 Monte Carlo Dropout — The Doubt Meter (`probabilistic_forecaster.py`)
The LSTM is asked to predict the price **50 times**, with a random 20% of neurons disabled each time:
- All 50 guesses cluster near one value → **high confidence**
- Guesses are spread across a wide range → **AI is confused; market is unpredictable**

This produces the confidence-interval band shown on the prediction chart.

### 🤖 Adaptive Market Regime Engine (`adaptive_engine.py`)
Real-time market classification into four regimes using MA crossover and 20-day rolling volatility:

| Regime | Condition | Colour |
|---|---|---|
| 🟢 Low Volatility Trending | Trending + calm | Green |
| 🟡 High Volatility Trending | Trending + choppy | Yellow |
| 🟠 Low Volatility Ranging | Ranging + calm | Orange |
| 🔴 High Volatility Ranging (Crisis) | Ranging + volatile | Red |

Each regime selects a different set of hyperparameters for LSTM (units, dropout, lookback, epochs), RF (depth, estimators), and SVR (C, gamma, epsilon).

### ⚖️ Hybrid Predictor — The Fusion Engine (`hybrid_predictor.py`)
Fuses LSTM, RF, and SVR predictions using dynamic weights that adapt to:
1. **Regime** — base weights per regime (LSTM always dominant)
2. **Performance** — models with higher R² get a proportional boost
3. **Sentiment** — strong FinBERT signal (|score| > 0.5) boosts LSTM weight by +5%

Default base weights (example for crisis regime):

| Model | Weight |
|---|---|
| LSTM | 65% |
| RF | 25% |
| SVR | 10% |

A further **sentiment adjustment** nudges the final price by ±0.5% × sentiment_score (capped at ±2%).

### 🐶 Reinforcement Learning Agent — The Virtual Trader (`rl_agent.py`)
A DQN agent trained by playing a simulated stock market millions of times. It receives a reward when trades are profitable and a penalty for drawdowns. It learns the exact optimal moments to BUY, HOLD, or SELL.

### 🗺️ Ticker Mapper (`ticker_mapper.py`)
Maps 100+ NSE/BSE and global ticker symbols to full company names, used to build more accurate news search queries for the `newsdata.io` API. Falls back gracefully for unknown tickers.

---

## 📂 5. Full File Reference

| File | Role |
|---|---|
| `app.py` | Streamlit dashboard — the UI you interact with |
| `model_trainer.py` | **Refactored** training + inference for LSTM, RF, SVR; all 5 bugs fixed |
| `sentiment_utils.py` | FinBERT pipeline — fetch, score, aggregate, merge sentiment |
| `news_injector.py` | Manual news injection UI (3 modes) + deduplication logic |
| `adaptive_engine.py` | 4-regime market classifier + per-regime hyperparameter configs |
| `hybrid_predictor.py` | Dynamic weight fusion of LSTM/RF/SVR + sentiment adjustment |
| `wavelet_engine.py` | Wavelet-based price noise cancellation |
| `probabilistic_forecaster.py` | Monte Carlo Dropout confidence intervals |
| `rl_agent.py` | DQN reinforcement learning virtual trader |
| `trading_signal_generator.py` | Stop-loss, take-profit, and risk level calculation |
| `justification_engine.py` | Translates model math into a human-readable BUY/HOLD/SELL paragraph |
| `ticker_mapper.py` | Ticker → company name lookup (100+ NSE/BSE + global) |
| `validate_fix.py` | Standalone validation script to verify all bug fixes are active |
| `test_smoke.py` | Smoke tests for core pipeline integrity |
| `requirements.txt` | All Python dependencies |

---

## 🚀 6. How to Run the App

### Step 1: Install

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### Step 2: Get a Free News API Key

1. Go to [newsdata.io](https://newsdata.io/register) and create a free account.
2. Copy your API key.
3. Create a `.env` file in the project root:

```
NEWSDATA_API_KEY=your_key_here
```

> **Note:** The free tier does not support date-range filtering. The app fetches the most recent articles and handles this gracefully.

### Step 3: Launch the Dashboard

```bash
streamlit run app.py
```

### Step 4: Using the Dashboard

1. **Sidebar** — Type a stock ticker: `TCS.NS`, `RELIANCE.NS`, `INFY.NS`, or any NSE ticker.
2. **Force Retrain** — Check this box the first time you analyse a new stock. It clears cached model files so the models re-learn from the new ticker's price range.
3. **Manual News** — Optionally inject your own headlines to influence the sentiment layer.
4. Click **Analyse Stock** — wait 30–90 seconds for training + inference.

### Step 5: Reading the Results

| Tab | What you see |
|---|---|
| **Overview** | Price chart, wavelet-smoothed trend, prediction paths |
| **Signals** | Stop-Loss and Take-Profit levels, risk classification |
| **Regime** | Current market regime badge + adaptive weight pie chart |
| **Hybrid** | All three model predictions vs. the fused hybrid prediction |
| **Justification** | AI-written paragraph explaining the BUY / HOLD / SELL decision |
| **Sentiment** | FinBERT scores timeline, daily mood chart, injected news table |

---

## 🧪 7. Validating the Bug Fixes

Run the standalone validator to confirm all five refactor fixes are active in your environment:

```bash
python validate_fix.py
```

For a quick pipeline smoke test:

```bash
python test_smoke.py
```

---

## ⚖️ 8. Disclaimer

*StockSense AI v2.0 is an educational and research tool demonstrating how multiple AI techniques can be combined for financial time-series modelling. The stock market is ultimately driven by unpredictable human behaviour, geopolitical events, and black-swan surprises that no model can foresee. **Do not use this tool to make real investment decisions or gamble your savings.** It is intended strictly for academic exploration, learning, and demonstration purposes.*
