"""
trading_signal_generator.py
============================
Final output layer: combines RL agent decisions + hybrid predictions +
technical analysis to produce actionable BUY/SELL trade recommendations
with entry price, target price, stop loss, and risk/reward ratio.

Author: StockSense AI — Research-Grade Upgrade
"""

from __future__ import annotations

import logging
import warnings
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional
from adaptive_engine import REGIME_1, REGIME_2, REGIME_3, REGIME_4

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Signal labels ─────────────────────────────────────────────────────────────
STRONG_BUY  = "STRONG BUY"
BUY         = "BUY"
HOLD        = "HOLD"
SELL        = "SELL"
STRONG_SELL = "STRONG SELL"

SIGNAL_COLORS = {
    STRONG_BUY:  "#00C076",
    BUY:         "#7BC67E",
    HOLD:        "#AAAAAA",
    SELL:        "#FF9F00",
    STRONG_SELL: "#FF4B4B",
}

SIGNAL_EMOJIS = {
    STRONG_BUY:  "🟢🟢",
    BUY:         "🟢",
    HOLD:        "⚪",
    SELL:        "🔴",
    STRONG_SELL: "🔴🔴",
}


class TradingSignalGenerator:
    """
    Converts model outputs into actionable trade recommendations.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # ATR COMPUTATION
    # ──────────────────────────────────────────────────────────────────────────

    def compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """
        Compute Average True Range (ATR) for dynamic stop-loss placement.

        True Range = max(H-L, |H-Cp|, |L-Cp|)
        ATR = moving average of True Range over [period] bars.

        Parameters
        ----------
        high, low, close : np.ndarray
        period : int

        Returns
        -------
        np.ndarray
            ATR series (same length as input).
        """
        n = len(close)
        high  = np.array(high,  dtype=np.float64)
        low   = np.array(low,   dtype=np.float64)
        close = np.array(close, dtype=np.float64)

        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i]  - low[i],
                abs(high[i]  - close[i - 1]),
                abs(low[i]   - close[i - 1]),
            )

        atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
        return atr

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def generate_signal(
        self,
        current_price: float,
        hybrid_prediction: float,
        sentiment_score: float,
        rl_action: int,
        rl_confidence: float,
        rsi: float,
        macd: float,
        macd_signal_val: float,
        atr: float,
        regime: str = REGIME_1,
    ) -> str:
        """
        Determine the trade signal based on multi-factor confluence.

        Parameters
        ----------
        current_price : float
        hybrid_prediction : float
        sentiment_score : float
        rl_action : int (0=HOLD, 1=BUY, 2=SELL)
        rl_confidence : float [0, 1]
        rsi : float [0, 100]
        macd : float
        macd_signal_val : float
        atr : float
        regime : str

        Returns
        -------
        str
            One of: STRONG BUY, BUY, HOLD, SELL, STRONG SELL
        """
        upside_threshold = current_price * 1.005

        # ── STRONG BUY (all 5 conditions) ─────────────────────────────────────
        if (
            rl_action == 1                          and
            rl_confidence > 0.65                   and
            hybrid_prediction > upside_threshold   and
            sentiment_score > -0.3                 and
            rsi < 70                               and
            macd > 0
        ):
            return STRONG_BUY

        # ── BUY (≥ 3 of 5 conditions) ─────────────────────────────────────────
        buy_conditions = [
            rl_action == 1,
            hybrid_prediction > current_price,
            sentiment_score > 0,
            30 <= rsi <= 60,
            macd > macd_signal_val,
        ]
        if sum(buy_conditions) >= 3:
            return BUY

        # ── STRONG SELL (all 4 conditions) ───────────────────────────────────
        if (
            rl_action == 2                          and
            rl_confidence > 0.65                   and
            hybrid_prediction < current_price * 0.995 and
            sentiment_score < 0.3                  and
            rsi > 65
        ):
            return STRONG_SELL

        # ── SELL (RL says sell + at least one confirming indicator) ───────────
        sell_conditions = [
            rl_action == 2,
            hybrid_prediction < current_price,
            rsi > 65,
            sentiment_score < -0.2,
        ]
        if rl_action == 2 and sum(sell_conditions) >= 2:
            return SELL

        return HOLD

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRY / TARGET / STOP LOSS
    # ──────────────────────────────────────────────────────────────────────────

    def compute_entry_target_stoploss(
        self,
        signal_type: str,
        current_price: float,
        atr: float,
        hybrid_pred: float,
        regime: str = REGIME_1,
        rl_confidence: float = 0.5,
    ) -> dict:
        """
        Compute trade levels: entry, target, stop loss, and risk/reward ratio.

        Parameters
        ----------
        signal_type : str
        current_price : float
        atr : float
        hybrid_pred : float
        regime : str
        rl_confidence : float

        Returns
        -------
        dict
            Keys: signal, entry_price, target_price, stop_loss, rr_ratio,
                  expected_return_pct, max_risk_pct, holding_period_estimate_days,
                  confidence_score, valid
        """
        atr = max(atr, current_price * 0.005)  # floor ATR at 0.5% of price

        # ── Entry price ───────────────────────────────────────────────────────
        if signal_type == STRONG_BUY:
            entry_price = current_price
        elif signal_type == BUY:
            entry_price = current_price * 0.998  # slight dip entry
        else:
            # SELL / HOLD — no long entry
            return {
                "signal":                        signal_type,
                "entry_price":                   None,
                "target_price":                  None,
                "stop_loss":                     None,
                "rr_ratio":                      None,
                "expected_return_pct":           None,
                "max_risk_pct":                  None,
                "holding_period_estimate_days":  None,
                "confidence_score":              rl_confidence,
                "valid":                         False,
            }

        # ── Target price ─────────────────────────────────────────────────────
        if signal_type == STRONG_BUY:
            base_target = max(hybrid_pred, current_price + 2.5 * atr)
        else:  # BUY
            base_target = max(hybrid_pred, current_price + 1.5 * atr)

        # Regime adjustment
        if regime == REGIME_2:
            base_target += 0.5 * atr  # extra upside expected in volatile trends

        target_price = base_target

        # ── Stop loss (ATR-based, dynamic) ────────────────────────────────────
        if signal_type == STRONG_BUY:
            raw_sl = current_price - 1.5 * atr
        else:  # BUY
            raw_sl = current_price - 2.0 * atr

        # Floor constraints
        sl_floor_pct  = current_price * 0.97   # never more than 3% below entry
        sl_tight_pct  = current_price * 0.985  # tighten near support

        stop_loss = max(raw_sl, sl_floor_pct)

        # ── Risk / Reward ratio ───────────────────────────────────────────────
        risk   = entry_price - stop_loss
        reward = target_price - entry_price

        if risk <= 0:
            risk = entry_price * 0.01

        rr_ratio = reward / risk

        # Downgrade if RR < 1.5
        if rr_ratio < 1.5 and signal_type == STRONG_BUY:
            signal_type = BUY
            # Widen target or tighten stop
            target_price = entry_price + 1.5 * risk

        expected_return_pct = (target_price - entry_price) / entry_price * 100
        max_risk_pct        = (entry_price - stop_loss) / entry_price * 100

        # ── Holding period estimate ───────────────────────────────────────────
        if signal_type == STRONG_BUY:
            holding_days = "8–12 trading days"
        else:
            holding_days = "5–8 trading days"

        # Confidence: weighted blend of RL confidence + RR attractiveness
        rr_score = min(rr_ratio / 3.0, 1.0)
        confidence_score = 0.65 * rl_confidence + 0.35 * rr_score

        return {
            "signal":                        signal_type,
            "entry_price":                   round(entry_price,  2),
            "target_price":                  round(target_price, 2),
            "stop_loss":                     round(stop_loss,    2),
            "rr_ratio":                      round(max(rr_ratio, 0.0), 2),
            "expected_return_pct":           round(expected_return_pct, 2),
            "max_risk_pct":                  round(max_risk_pct, 2),
            "holding_period_estimate_days":  holding_days,
            "confidence_score":              round(confidence_score, 4),
            "valid":                         True,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TRADE CARD UI
    # ──────────────────────────────────────────────────────────────────────────

    def generate_trade_card(
        self,
        signal_dict: dict,
        ticker: str,
        current_price: float,
        sentiment_summary: str = "",
        rl_confidence: float = 0.5,
        hybrid_pred: float = 0.0,
        lstm_weight: float = 0.60,
        regime: str = REGIME_1,
        sentiment_score: float = 0.0,
    ) -> None:
        """
        Render an actionable trade card in Streamlit with styled CSS.

        Parameters
        ----------
        signal_dict : dict
            Output from compute_entry_target_stoploss.
        ticker : str
        current_price : float
        sentiment_summary : str
        rl_confidence : float
        hybrid_pred : float
        lstm_weight : float
        regime : str
        sentiment_score : float
        """
        signal    = signal_dict.get("signal", HOLD)
        entry     = signal_dict.get("entry_price")
        target    = signal_dict.get("target_price")
        sl        = signal_dict.get("stop_loss")
        rr        = signal_dict.get("rr_ratio")
        ret_pct   = signal_dict.get("expected_return_pct")
        risk_pct  = signal_dict.get("max_risk_pct")
        hold_d    = signal_dict.get("holding_period_estimate_days", "—")
        conf      = signal_dict.get("confidence_score", 0.5)
        valid     = signal_dict.get("valid", False)

        # Card border color
        border_color = SIGNAL_COLORS.get(signal, "#AAAAAA")
        emoji        = SIGNAL_EMOJIS.get(signal, "⚪")

        # Sentiment label
        if sentiment_score > 0.2:
            sent_label = f"Positive (+{sentiment_score:.2f})"
            sent_color = "#00C076"
        elif sentiment_score < -0.2:
            sent_label = f"Negative ({sentiment_score:.2f})"
            sent_color = "#FF4B4B"
        else:
            sent_label = f"Neutral ({sentiment_score:.2f})"
            sent_color = "#AAAAAA"

        now_str = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

        # ── Card HTML ─────────────────────────────────────────────────────────
        if valid and entry is not None:
            body = f"""
<div style="
    background: #161B27;
    border: 2px solid {border_color};
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    line-height: 1.8;
    box-shadow: 0 4px 20px {border_color}44;
">
    <div style="font-size:22px;font-weight:700;color:{border_color};margin-bottom:4px;">
        {emoji} {signal} — {ticker}
    </div>
    <div style="font-size:11px;color:#666;margin-bottom:16px;">Generated: {now_str}</div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px;">
        <div style="background:#1A2035;border-radius:8px;padding:10px 14px;">
            <div style="font-size:11px;color:#888;">📌 Entry Price</div>
            <div style="font-size:18px;font-weight:700;color:#FAFAFA;">₹{entry:,.2f}</div>
        </div>
        <div style="background:#1A2035;border-radius:8px;padding:10px 14px;">
            <div style="font-size:11px;color:#888;">🎯 Target Price</div>
            <div style="font-size:18px;font-weight:700;color:#00C076;">₹{target:,.2f}
                <span style="font-size:12px;">(+{ret_pct:.2f}%)</span>
            </div>
        </div>
        <div style="background:#1A2035;border-radius:8px;padding:10px 14px;">
            <div style="font-size:11px;color:#888;">🛑 Stop Loss</div>
            <div style="font-size:18px;font-weight:700;color:#FF4B4B;">₹{sl:,.2f}
                <span style="font-size:12px;">(-{risk_pct:.2f}%)</span>
            </div>
        </div>
        <div style="background:#1A2035;border-radius:8px;padding:10px 14px;">
            <div style="font-size:11px;color:#888;">⚖️ Risk / Reward</div>
            <div style="font-size:18px;font-weight:700;color:#00C8FF;">{rr:.2f} : 1</div>
        </div>
    </div>

    <div style="background:#1A2035;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
            <span>🤖 <b>RL Confidence:</b> {rl_confidence*100:.0f}%</span>
            <span>📊 <b>Hybrid Pred:</b> ₹{hybrid_pred:,.0f} (LSTM {lstm_weight*100:.0f}%)</span>
            <span style="color:{sent_color};">😊 <b>Sentiment:</b> {sent_label}</span>
            <span>📅 <b>Hold Period:</b> {hold_d}</span>
            <span>📈 <b>Regime:</b> {regime}</span>
        </div>
    </div>

    <div style="background:#1A0A0A;border-left:3px solid #FF4B4B;border-radius:4px;
                padding:10px 14px;font-size:12px;color:#999;">
        ⚠️ <b>Disclaimer:</b> For educational and research purposes only.
        Not financial advice. Always do your own research. Trade at your own risk.
    </div>
</div>
"""
        else:
            # HOLD / SELL card
            body = f"""
<div style="
    background: #161B27;
    border: 2px solid {border_color};
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
">
    <div style="font-size:22px;font-weight:700;color:{border_color};margin-bottom:4px;">
        {emoji} {signal} — {ticker}
    </div>
    <div style="font-size:11px;color:#666;margin-bottom:16px;">Generated: {now_str}</div>
    <div style="background:#1A2035;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
        <b>Current Price:</b> ₹{current_price:,.2f} &nbsp;|&nbsp;
        <b>Hybrid Pred:</b> ₹{hybrid_pred:,.0f} &nbsp;|&nbsp;
        <span style="color:{sent_color};">Sentiment: {sent_label}</span>
    </div>
    <div style="color:#888;font-size:13px;">
        No active trade entry recommended at this time.
        Monitor for regime change or stronger confirmation signals.
    </div>
    <div style="background:#1A0A0A;border-left:3px solid #FF4B4B;border-radius:4px;
                padding:10px 14px;margin-top:12px;font-size:12px;color:#999;">
        ⚠️ For educational purposes only. Not financial advice.
    </div>
</div>
"""

        st.markdown(body, unsafe_allow_html=True)

    def plot_signal_history(
        self,
        stock_df: pd.DataFrame,
        signal_history: list[dict],
        ticker: str,
    ) -> "matplotlib.figure.Figure":  # type: ignore
        """
        Plot price chart with BUY/SELL signal annotations.

        Parameters
        ----------
        stock_df : pd.DataFrame
        signal_history : list[dict]
            Each dict: {date, signal, price}
        ticker : str

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
        })

        fig, ax = plt.subplots(figsize=(13, 5))
        dates  = pd.to_datetime(stock_df["Date"].values)
        prices = stock_df["Close"].values

        ax.plot(dates, prices, color="#00C8FF", linewidth=1.3, label="Close Price")

        for sig in signal_history:
            sig_date  = pd.to_datetime(sig.get("date", dates[-1]))
            sig_price = sig.get("price", prices[-1])
            sig_type  = sig.get("signal", HOLD)
            color = SIGNAL_COLORS.get(sig_type, "#AAAAAA")
            marker = "^" if "BUY" in sig_type else "v" if "SELL" in sig_type else "o"
            ax.scatter(sig_date, sig_price, c=color, s=120, marker=marker, zorder=5)
            ax.annotate(sig_type[:2], (sig_date, sig_price),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, color=color, ha="center")

        ax.set_title(f"{ticker} — Trading Signal History", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        return fig
