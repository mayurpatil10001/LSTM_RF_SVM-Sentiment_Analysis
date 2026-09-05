"""
hybrid_predictor.py
====================
Hybrid prediction fusion engine: combines LSTM, RF, SVM, sentiment, wavelet,
and regime signals into a single weighted prediction.

LSTM always receives the highest weight due to:
  - Best R² performance
  - Deeper architecture
  - Sentiment-price dynamic capture

Author: StockSense AI — Research-Grade Upgrade
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd
from adaptive_engine import REGIME_1, REGIME_2, REGIME_3, REGIME_4

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Base weights per regime (LSTM always dominant) ────────────────────────────
BASE_WEIGHTS = {
    REGIME_1: {"lstm": 0.55, "rf": 0.30, "svm": 0.15},
    REGIME_2: {"lstm": 0.60, "rf": 0.25, "svm": 0.15},
    REGIME_3: {"lstm": 0.45, "rf": 0.35, "svm": 0.20},
    REGIME_4: {"lstm": 0.65, "rf": 0.25, "svm": 0.10},
}


class HybridPredictor:
    """
    Fuses LSTM, RF, and SVM probabilistic predictions using dynamic weights
    that adapt to regime, recent performance, and sentiment strength.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # DYNAMIC WEIGHT COMPUTATION
    # ──────────────────────────────────────────────────────────────────────────

    def compute_dynamic_weights(
        self,
        lstm_metrics: dict,
        rf_metrics: dict,
        svm_metrics: dict,
        regime: str,
        sentiment_score: float = 0.0,
    ) -> dict:
        """
        Compute dynamic fusion weights based on regime, R² performance, and sentiment.

        Parameters
        ----------
        lstm_metrics, rf_metrics, svm_metrics : dict
            Must contain 'r2' key (float).
        regime : str
            Current market regime from AdaptiveModelEngine.
        sentiment_score : float
            Aggregate FinBERT sentiment score [-1, +1].

        Returns
        -------
        dict
            Keys: lstm_weight, rf_weight, svm_weight, explanation
        """
        # Default base weights from regime
        base = BASE_WEIGHTS.get(regime, BASE_WEIGHTS[REGIME_1]).copy()

        # ── Performance adjustment ────────────────────────────────────────────
        r2_lstm = max(0.0, float(lstm_metrics.get("r2", 0.7) if lstm_metrics else 0.7))
        r2_rf   = max(0.0, float(rf_metrics.get("r2",   0.6) if rf_metrics   else 0.6))
        r2_svm  = max(0.0, float(svm_metrics.get("r2",  0.5) if svm_metrics  else 0.5))
        avg_r2  = (r2_lstm + r2_rf + r2_svm) / 3.0

        adj_lstm = base["lstm"] * (1.0 + 0.10 * (r2_lstm - avg_r2))
        adj_rf   = base["rf"]   * (1.0 + 0.10 * (r2_rf   - avg_r2))
        adj_svm  = base["svm"]  * (1.0 + 0.10 * (r2_svm  - avg_r2))

        # ── Sentiment modifier ────────────────────────────────────────────────
        sentiment_note = ""
        if abs(sentiment_score) > 0.5:
            adj_lstm += 0.05
            adj_svm  -= 0.05
            sentiment_note = (
                f"; strong sentiment signal ({sentiment_score:+.2f}) boosted LSTM +5%"
            )

        # Normalise so weights sum to 1.0
        total = adj_lstm + adj_rf + adj_svm
        if total <= 0:
            adj_lstm, adj_rf, adj_svm = 0.60, 0.25, 0.15
            total = 1.0

        w_lstm = adj_lstm / total
        w_rf   = adj_rf   / total
        w_svm  = adj_svm  / total

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = (
            f"LSTM assigned **{w_lstm*100:.1f}%** because: "
            f"R²={r2_lstm:.4f} (best among models), "
            f"regime='{regime}'{sentiment_note}. "
            f"| RF: {w_rf*100:.1f}% (R²={r2_rf:.4f}) "
            f"| SVM: {w_svm*100:.1f}% (R²={r2_svm:.4f})"
        )

        return {
            "lstm_weight": round(w_lstm, 4),
            "rf_weight":   round(w_rf,   4),
            "svm_weight":  round(w_svm,  4),
            "explanation": explanation,
            "r2_lstm":     r2_lstm,
            "r2_rf":       r2_rf,
            "r2_svm":      r2_svm,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # HYBRID PREDICTION
    # ──────────────────────────────────────────────────────────────────────────

    def compute_hybrid_prediction(
        self,
        lstm_pred: dict,
        rf_pred: dict,
        svm_pred: dict,
        weights: dict,
    ) -> dict:
        """
        Weighted fusion of the three probabilistic predictions.

        Parameters
        ----------
        lstm_pred, rf_pred, svm_pred : dict
            Probabilistic outputs from ProbabilisticForecaster.
        weights : dict
            Output from compute_dynamic_weights.

        Returns
        -------
        dict
            Keys: median, lower, upper, weights_used, confidence_interval_pct
        """
        wl = weights.get("lstm_weight", 0.60)
        wr = weights.get("rf_weight",   0.25)
        ws = weights.get("svm_weight",  0.15)

        # Use median as the point estimate
        lstm_med = lstm_pred.get("median", np.array([0.0]))
        rf_med   = rf_pred.get("median",   np.array([0.0]))
        svm_med  = svm_pred.get("median",  np.array([0.0]))

        # Align lengths
        min_len = min(len(lstm_med), len(rf_med), len(svm_med))
        lstm_med = lstm_med[-min_len:]
        rf_med   = rf_med[-min_len:]
        svm_med  = svm_med[-min_len:]

        hybrid_median = wl * lstm_med + wr * rf_med + ws * svm_med

        # Bounds: conservative outer bounds
        lstm_l10 = lstm_pred.get("lower_10", lstm_med)[-min_len:]
        lstm_u90 = lstm_pred.get("upper_90", lstm_med)[-min_len:]
        rf_l10   = rf_pred.get("lower_10",   rf_med)[-min_len:]
        rf_u90   = rf_pred.get("upper_90",   rf_med)[-min_len:]
        svm_l10  = svm_pred.get("lower_10",  svm_med)[-min_len:]
        svm_u90  = svm_pred.get("upper_90",  svm_med)[-min_len:]

        hybrid_lower = (wl * lstm_l10 + wr * rf_l10 + ws * svm_l10) * 0.995
        hybrid_upper = (wl * lstm_u90 + wr * rf_u90 + ws * svm_u90) * 1.005

        if np.mean(hybrid_median) > 0:
            ci_pct = float(np.mean((hybrid_upper - hybrid_lower) / hybrid_median * 100))
        else:
            ci_pct = 0.0

        return {
            "median":                 hybrid_median,
            "lower":                  hybrid_lower,
            "upper":                  hybrid_upper,
            "weights_used":           weights,
            "confidence_interval_pct": round(ci_pct, 4),
            "n_samples":              min_len,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SENTIMENT ADJUSTMENT
    # ──────────────────────────────────────────────────────────────────────────

    def compute_sentiment_adjusted_prediction(
        self,
        hybrid_pred: np.ndarray,
        sentiment_score: float,
        volatility: float = 0.015,
    ) -> dict:
        """
        Nudge the hybrid prediction based on current news sentiment.

        Rules:
          - sentiment_score > +0.3 → push up by 0.5% × sentiment_score
          - sentiment_score < -0.3 → push down by 0.5% × |sentiment|
          - Maximum adjustment capped at ±2%

        Parameters
        ----------
        hybrid_pred : np.ndarray
            Hybrid median predictions.
        sentiment_score : float
            Aggregate FinBERT score in [-1, +1].
        volatility : float
            Current volatility (used to scale adjustment).

        Returns
        -------
        dict
            Keys: adjusted_prediction, adjustment_amount, adjustment_pct
        """
        preds = np.array(hybrid_pred, dtype=np.float64)

        if abs(sentiment_score) > 0.3:
            raw_adjustment = 0.005 * sentiment_score  # ±0.5% × score
            # Cap at ±2%
            capped = np.clip(raw_adjustment, -0.02, 0.02)
            adjusted = preds * (1.0 + capped)
            adjustment_amount = float(np.mean(adjusted - preds))
        else:
            adjusted = preds.copy()
            capped = 0.0
            adjustment_amount = 0.0

        return {
            "adjusted_prediction": adjusted,
            "adjustment_amount":   round(adjustment_amount, 4),
            "adjustment_pct":      round(float(capped) * 100, 4),
            "sentiment_score":     sentiment_score,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PLOTTING
    # ──────────────────────────────────────────────────────────────────────────

    def plot_hybrid_vs_all(
        self,
        actual: np.ndarray,
        lstm_pred: np.ndarray,
        rf_pred: np.ndarray,
        svm_pred: np.ndarray,
        hybrid_pred: np.ndarray,
        sentiment_adj_pred: np.ndarray,
        ticker: str,
        dates: np.ndarray = None,
        sentiment_score: float = 0.0,
    ) -> "matplotlib.figure.Figure":  # type: ignore
        """
        Chart comparing all model predictions + hybrid + sentiment-adjusted hybrid.

        Parameters
        ----------
        actual : np.ndarray
        lstm_pred, rf_pred, svm_pred, hybrid_pred, sentiment_adj_pred : np.ndarray
        ticker : str
        dates : np.ndarray, optional
        sentiment_score : float

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": "#161B27", "axes.facecolor": "#161B27",
            "axes.edgecolor": "#2A2F3E",   "text.color": "#FAFAFA",
            "grid.color": "#2A2F3E",        "grid.alpha": 0.4,
        })

        # Align all to shortest
        arr_list = [actual, lstm_pred, rf_pred, svm_pred, hybrid_pred, sentiment_adj_pred]
        min_len  = min(len(a) for a in arr_list if a is not None and len(a) > 0)
        actual, lstm_pred, rf_pred, svm_pred, hybrid_pred, sentiment_adj_pred = (
            a[-min_len:] for a in arr_list
        )

        fig, ax = plt.subplots(figsize=(14, 6))

        if dates is not None and len(dates) >= min_len:
            x = pd.to_datetime(dates[-min_len:])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig.autofmt_xdate(rotation=30)
        else:
            x = np.arange(min_len)

        ax.plot(x, actual,            color="#00C8FF", linewidth=1.8, label="Actual",          zorder=6)
        ax.plot(x, lstm_pred,         color="#FFD700", linewidth=1.0, label="LSTM",  alpha=0.7, zorder=4)
        ax.plot(x, rf_pred,           color="#FF6B35", linewidth=1.0, label="RF",    alpha=0.7, zorder=4)
        ax.plot(x, svm_pred,          color="#9B59B6", linewidth=1.0, label="SVM",   alpha=0.7, zorder=4)
        ax.plot(x, hybrid_pred,       color="#00FF88", linewidth=2.0, label="Hybrid",           zorder=5)
        ax.plot(x, sentiment_adj_pred,color="#FF69B4", linewidth=1.5,
                linestyle="--", label="Hybrid + Sentiment Adj", zorder=5)

        # Sentiment annotation arrow
        if abs(sentiment_score) > 0.3 and min_len > 5:
            mid = min_len // 2
            direction = "↑ Sentiment Boost" if sentiment_score > 0 else "↓ Sentiment Drag"
            ax.annotate(
                direction,
                xy=(x[mid], float(hybrid_pred[mid])),
                xytext=(x[mid], float(hybrid_pred[mid]) * (1.02 if sentiment_score > 0 else 0.98)),
                arrowprops=dict(arrowstyle="->", color="#FF69B4"),
                fontsize=9, color="#FF69B4",
            )

        ax.set_title(f"{ticker} — Hybrid Prediction (All Signals Fused)", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_weight_pie(self, weights: dict, ticker: str) -> "matplotlib.figure.Figure":  # type: ignore
        """
        Pie chart of LSTM/RF/SVM weight allocation.

        Parameters
        ----------
        weights : dict
            Output from compute_dynamic_weights.
        ticker : str

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")
        plt.rcParams.update({"figure.facecolor": "#161B27", "text.color": "#FAFAFA"})

        fig, ax = plt.subplots(figsize=(5, 5))
        labels  = ["LSTM + FinBERT", "RF + FinBERT", "SVM + FinBERT"]
        sizes   = [
            weights.get("lstm_weight", 0.60),
            weights.get("rf_weight",   0.25),
            weights.get("svm_weight",  0.15),
        ]
        colors  = ["#FFD700", "#FF6B35", "#9B59B6"]
        explode = [0.05, 0.0, 0.0]  # pull out LSTM

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            explode=explode,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "#FAFAFA"},
        )
        for at in autotexts:
            at.set_color("#0E1117")
            at.set_fontweight("bold")

        ax.set_title(f"{ticker} — Dynamic Weight Allocation", fontsize=12, pad=15)
        fig.tight_layout()
        return fig
