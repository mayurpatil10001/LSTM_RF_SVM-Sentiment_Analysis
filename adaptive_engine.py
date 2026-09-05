"""
adaptive_engine.py
==================
Adaptive Model Complexity Engine for streaming stock data.

Detects market regimes in real-time and adjusts model hyperparameters
to match current market conditions.

Regimes:
  REGIME_1: Low Volatility Trending   (green)
  REGIME_2: High Volatility Trending  (yellow)
  REGIME_3: Low Volatility Ranging    (orange)
  REGIME_4: High Volatility Ranging   (red — crisis)

Author: StockSense AI — Research-Grade Upgrade
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Regime constants ──────────────────────────────────────────────────────────
REGIME_1 = "Low Volatility Trending"
REGIME_2 = "High Volatility Trending"
REGIME_3 = "Low Volatility Ranging"
REGIME_4 = "High Volatility Ranging (Crisis)"

REGIME_COLORS = {
    REGIME_1: "#00C076",   # green
    REGIME_2: "#FFD700",   # yellow
    REGIME_3: "#FF9F00",   # orange
    REGIME_4: "#FF4B4B",   # red
}

REGIME_EMOJIS = {
    REGIME_1: "🟢",
    REGIME_2: "🟡",
    REGIME_3: "🟠",
    REGIME_4: "🔴",
}


class AdaptiveModelEngine:
    """
    Detects the current market regime and returns optimal hyperparameter
    configurations for each model type.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # REGIME DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_market_regime(
        self,
        price_series: pd.Series | np.ndarray,
        volume_series: Optional[pd.Series | np.ndarray] = None,
        vol_window: int = 20,
    ) -> dict:
        """
        Classify current market into one of four regimes.

        Parameters
        ----------
        price_series : Series or ndarray
            Close price series.
        volume_series : Series or ndarray, optional
            Volume series for volume regime confirmation.
        vol_window : int
            Rolling window for volatility computation.

        Returns
        -------
        dict
            {regime_label, volatility_value, trend_strength,
             is_trending, is_high_vol, color, emoji}
        """
        prices = np.array(price_series, dtype=np.float64)
        prices = prices[~np.isnan(prices)]

        if len(prices) < vol_window + 5:
            return self._default_regime()

        try:
            # Daily returns
            returns = np.diff(prices) / prices[:-1]
            returns = returns[~np.isnan(returns)]

            # Annualised rolling volatility (20-day)
            rolling_vol = pd.Series(returns).rolling(vol_window).std().dropna()
            current_vol = float(rolling_vol.iloc[-1]) if len(rolling_vol) > 0 else 0.015

            # Trend detection via MA crossover
            if len(prices) >= 50:
                ma_short = np.mean(prices[-20:])
                ma_long  = np.mean(prices[-50:])
                trend_strength = (ma_short - ma_long) / (ma_long + 1e-8)
                is_trending = abs(trend_strength) > 0.02  # 2% MA divergence
            else:
                trend_strength = 0.0
                is_trending = False

            # Volatility thresholds (daily)
            is_high_vol = current_vol > 0.02  # 2% daily vol threshold

            # Classify regime
            if is_trending and not is_high_vol:
                regime = REGIME_1
            elif is_trending and is_high_vol:
                regime = REGIME_2
            elif not is_trending and not is_high_vol:
                regime = REGIME_3
            else:
                regime = REGIME_4  # crisis

            return {
                "regime_label":     regime,
                "volatility_value": round(current_vol, 6),
                "trend_strength":   round(trend_strength, 4),
                "is_trending":      is_trending,
                "is_high_vol":      is_high_vol,
                "color":            REGIME_COLORS[regime],
                "emoji":            REGIME_EMOJIS[regime],
                "vol_pct":          round(current_vol * 100, 3),
            }

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return self._default_regime()

    def _default_regime(self) -> dict:
        """Return REGIME_1 as safe default."""
        return {
            "regime_label":     REGIME_1,
            "volatility_value": 0.01,
            "trend_strength":   0.0,
            "is_trending":      True,
            "is_high_vol":      False,
            "color":            REGIME_COLORS[REGIME_1],
            "emoji":            REGIME_EMOJIS[REGIME_1],
            "vol_pct":          1.0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # ADAPTIVE CONFIGURATIONS
    # ──────────────────────────────────────────────────────────────────────────

    def get_adaptive_lstm_config(self, regime: str) -> dict:
        """
        Return LSTM hyperparameter config optimised for the detected regime.

        Parameters
        ----------
        regime : str
            One of the REGIME_* constants.

        Returns
        -------
        dict
            Keys: units, dropout, lookback, epochs, description
        """
        configs = {
            REGIME_1: {
                "units":       [128, 128, 64],
                "dropout":     0.10,
                "lookback":    100,
                "epochs":      80,
                "description": "Stable trending: balanced depth, moderate lookback",
            },
            REGIME_2: {
                "units":       [256, 128, 64],
                "dropout":     0.20,
                "lookback":    150,
                "epochs":      100,
                "description": "Volatile trending: deep network, extended lookback for momentum",
            },
            REGIME_3: {
                "units":       [64, 32],
                "dropout":     0.15,
                "lookback":    60,
                "epochs":      60,
                "description": "Stable ranging: shallow network, short lookback for local patterns",
            },
            REGIME_4: {
                "units":       [256, 256, 128, 64],
                "dropout":     0.30,
                "lookback":    200,
                "epochs":      120,
                "description": "Crisis regime: maximum depth, high dropout for uncertainty, long lookback",
            },
        }
        return configs.get(regime, configs[REGIME_1])

    def get_adaptive_rf_config(self, regime: str) -> dict:
        """
        Return Random Forest hyperparameter config for the detected regime.

        Parameters
        ----------
        regime : str

        Returns
        -------
        dict
        """
        configs = {
            REGIME_1: {"n_estimators": 100, "max_depth": 8,  "description": "Stable: lighter ensemble"},
            REGIME_2: {"n_estimators": 200, "max_depth": 12, "description": "Volatile: deeper trees"},
            REGIME_3: {"n_estimators": 80,  "max_depth": 6,  "description": "Ranging: compact ensemble"},
            REGIME_4: {"n_estimators": 300, "max_depth": 15, "description": "Crisis: large ensemble, deep trees"},
        }
        return configs.get(regime, configs[REGIME_1])

    def get_adaptive_svm_config(self, regime: str) -> dict:
        """
        Return SVR hyperparameter config for the detected regime.

        Parameters
        ----------
        regime : str

        Returns
        -------
        dict
        """
        configs = {
            REGIME_1: {"C": 10,   "gamma": 0.01,  "epsilon": 0.10, "description": "Stable: soft margin"},
            REGIME_2: {"C": 100,  "gamma": 0.1,   "epsilon": 0.01, "description": "Volatile: tighter margin"},
            REGIME_3: {"C": 1,    "gamma": 0.001, "epsilon": 0.10, "description": "Ranging: wide epsilon tube"},
            REGIME_4: {"C": 1000, "gamma": 0.5,   "epsilon": 0.001,"description": "Crisis: maximum C, fine gamma"},
        }
        return configs.get(regime, configs[REGIME_1])

    def get_all_configs(self, regime: str) -> dict:
        """
        Return all three model configs for a given regime in one call.

        Parameters
        ----------
        regime : str

        Returns
        -------
        dict
            Keys: lstm, rf, svm
        """
        return {
            "lstm": self.get_adaptive_lstm_config(regime),
            "rf":   self.get_adaptive_rf_config(regime),
            "svm":  self.get_adaptive_svm_config(regime),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # STREAMING SIMULATION
    # ──────────────────────────────────────────────────────────────────────────

    def simulate_streaming_regimes(
        self,
        stock_df: pd.DataFrame,
        window: int = 20,
        step: int = 5,
    ) -> pd.DataFrame:
        """
        Simulate regime detection over time by sliding a window across the data.

        Parameters
        ----------
        stock_df : pd.DataFrame
            Must contain 'Close', 'Volume', 'Date'.
        window : int
            Lookback window for each regime detection step.
        step : int
            Step size between regime detections.

        Returns
        -------
        pd.DataFrame
            Columns: [Date, Close, Regime, Volatility, TrendStrength, Color]
        """
        if "Close" not in stock_df.columns:
            return pd.DataFrame()

        prices  = stock_df["Close"].values
        dates   = stock_df["Date"].values if "Date" in stock_df.columns else np.arange(len(prices))
        volumes = stock_df["Volume"].values if "Volume" in stock_df.columns else None

        records = []
        for i in range(window, len(prices), step):
            segment = prices[max(0, i - window * 3):i]  # 3× window for stability
            v_seg   = volumes[max(0, i - window * 3):i] if volumes is not None else None

            regime_info = self.detect_market_regime(segment, v_seg, vol_window=window)
            records.append({
                "Date":          pd.to_datetime(dates[i - 1]) if i - 1 < len(dates) else i,
                "Close":         float(prices[i - 1]),
                "Regime":        regime_info["regime_label"],
                "Volatility":    regime_info["volatility_value"],
                "TrendStrength": regime_info["trend_strength"],
                "Color":         regime_info["color"],
                "Emoji":         regime_info["emoji"],
            })

        return pd.DataFrame(records)

    def plot_regime_timeline(
        self,
        regime_df: pd.DataFrame,
        stock_df: pd.DataFrame,
        ticker: str,
    ) -> "matplotlib.figure.Figure":  # type: ignore
        """
        Plot price series with regime-coloured background shading.

        Parameters
        ----------
        regime_df : pd.DataFrame
            Output from simulate_streaming_regimes().
        stock_df : pd.DataFrame
            Full price data for plotting.
        ticker : str

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch

        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": "#161B27",
            "axes.facecolor":   "#161B27",
            "axes.edgecolor":   "#2A2F3E",
            "text.color":       "#FAFAFA",
            "grid.color":       "#2A2F3E",
        })

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Price line
        dates  = pd.to_datetime(stock_df["Date"].values)
        prices = stock_df["Close"].values
        ax1.plot(dates, prices, color="#00C8FF", linewidth=1.3, label="Close Price")

        # Shade regime periods
        if not regime_df.empty:
            for i in range(len(regime_df)):
                row = regime_df.iloc[i]
                x_start = pd.to_datetime(row["Date"])
                x_end   = pd.to_datetime(regime_df["Date"].iloc[i + 1]) if i + 1 < len(regime_df) else dates[-1]
                ax1.axvspan(x_start, x_end, alpha=0.12, color=row["Color"])

            # Volatility line
            ax2.plot(
                pd.to_datetime(regime_df["Date"]),
                regime_df["Volatility"] * 100,
                color="#FF9F00", linewidth=1.2, label="Daily Vol %",
            )
            ax2.axhline(2.0, color="#FF4B4B", linestyle="--", alpha=0.6, label="High Vol Threshold (2%)")
            ax2.axhline(1.0, color="#00C076", linestyle="--", alpha=0.6, label="Low Vol Threshold (1%)")
            ax2.set_ylabel("Volatility (%)", fontsize=9)
            ax2.legend(fontsize=7, loc="upper right")

        ax1.set_title(f"{ticker} — Adaptive Regime Detection (Streaming Simulation)", fontsize=13)
        ax1.set_ylabel("Price (₹)")
        ax1.legend(loc="upper left")

        legend_elements = [
            Patch(facecolor=REGIME_COLORS[r], label=r, alpha=0.7)
            for r in [REGIME_1, REGIME_2, REGIME_3, REGIME_4]
        ]
        ax1.legend(handles=legend_elements + [ax1.lines[0]], loc="upper left", fontsize=8)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        return fig

    def build_regime_config_table(self) -> pd.DataFrame:
        """
        Build a summary table showing how model configs change per regime.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for regime in [REGIME_1, REGIME_2, REGIME_3, REGIME_4]:
            lcd = self.get_adaptive_lstm_config(regime)
            rfc = self.get_adaptive_rf_config(regime)
            svmc = self.get_adaptive_svm_config(regime)
            rows.append({
                "Regime":            f"{REGIME_EMOJIS[regime]} {regime}",
                "LSTM Units":        str(lcd["units"]),
                "LSTM Dropout":      lcd["dropout"],
                "LSTM Lookback":     lcd["lookback"],
                "LSTM Epochs":       lcd["epochs"],
                "RF Estimators":     rfc["n_estimators"],
                "RF Max Depth":      rfc["max_depth"],
                "SVM C":             svmc["C"],
                "SVM Gamma":         svmc["gamma"],
            })
        return pd.DataFrame(rows)
