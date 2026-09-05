"""
wavelet_engine.py
=================
Wavelet Neural Network (WNN) engine for StockSense AI.

Two main roles:
  A) Preprocessing — extract multi-resolution wavelet features fed into LSTM
  B) Standalone WNN — trainable wavelet decomposition Keras model shown in Tab 6

Uses PyWavelets (pywt) for DWT decomposition.

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


# ──────────────────────────────────────────────────────────────────────────────
# PART A — WAVELET PREPROCESSING FOR LSTM
# ──────────────────────────────────────────────────────────────────────────────

def extract_wavelet_features(
    price_series: pd.Series | np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> pd.DataFrame:
    """
    Perform multilevel Discrete Wavelet Transform (DWT) and reconstruct each
    component back to the original signal length.

    Level-3 decomposition produces:
      cA3 — approximation (long-term trend)
      cD3 — detail level 3 (medium frequency oscillations)
      cD2 — detail level 2 (short-term fluctuations)
      cD1 — detail level 1 (noise / high-frequency)

    These four reconstructed signals are returned as a DataFrame with the
    same length as the input. They are added EXCLUSIVELY to LSTM's feature
    set — RF and SVM do not use wavelet features.

    Parameters
    ----------
    price_series : pd.Series or np.ndarray
        Raw Close price series.
    wavelet : str
        PyWavelets wavelet name (default: 'db4' — Daubechies 4).
    level : int
        Decomposition level (default: 3).

    Returns
    -------
    pd.DataFrame
        Columns: [Wavelet_A3, Wavelet_D3, Wavelet_D2, Wavelet_D1]
        All columns have the same length as the input series.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("pywt not installed — returning zero wavelet features.")
        n = len(price_series)
        return pd.DataFrame({
            "Wavelet_A3": np.zeros(n),
            "Wavelet_D3": np.zeros(n),
            "Wavelet_D2": np.zeros(n),
            "Wavelet_D1": np.zeros(n),
        })

    series = np.array(price_series, dtype=np.float64)
    n = len(series)

    try:
        # Full multilevel DWT
        coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
        # coeffs = [cA_level, cD_level, ..., cD1]  (level+1 elements)

        def _reconstruct_single(coeffs_all: list, target_idx: int, original_len: int) -> np.ndarray:
            """Zero out all coefficients except target, then reconstruct."""
            zeroed = [np.zeros_like(c) for c in coeffs_all]
            zeroed[target_idx] = coeffs_all[target_idx]
            rec = pywt.waverec(zeroed, wavelet=wavelet)
            # Trim or pad to original length
            if len(rec) > original_len:
                rec = rec[:original_len]
            elif len(rec) < original_len:
                rec = np.pad(rec, (0, original_len - len(rec)), mode="edge")
            return rec.astype(np.float32)

        # Index 0 → cA3 (approximation), 1 → cD3, 2 → cD2, 3 → cD1
        wavelet_a3 = _reconstruct_single(coeffs, 0, n)
        wavelet_d3 = _reconstruct_single(coeffs, 1, n) if len(coeffs) > 1 else np.zeros(n, dtype=np.float32)
        wavelet_d2 = _reconstruct_single(coeffs, 2, n) if len(coeffs) > 2 else np.zeros(n, dtype=np.float32)
        wavelet_d1 = _reconstruct_single(coeffs, 3, n) if len(coeffs) > 3 else np.zeros(n, dtype=np.float32)

        return pd.DataFrame({
            "Wavelet_A3": wavelet_a3,
            "Wavelet_D3": wavelet_d3,
            "Wavelet_D2": wavelet_d2,
            "Wavelet_D1": wavelet_d1,
        })

    except Exception as e:
        logger.warning(f"Wavelet feature extraction failed: {e}. Returning zeros.")
        return pd.DataFrame({
            "Wavelet_A3": np.zeros(n, dtype=np.float32),
            "Wavelet_D3": np.zeros(n, dtype=np.float32),
            "Wavelet_D2": np.zeros(n, dtype=np.float32),
            "Wavelet_D1": np.zeros(n, dtype=np.float32),
        })


def denoise_with_wavelet(
    price_series: pd.Series | np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """
    Apply wavelet soft thresholding to remove high-frequency noise from a
    price series (Donoho-Johnstone universal threshold).

    Threshold = sigma * sqrt(2 * log(n))
    where sigma = median(|cD1|) / 0.6745  (robust noise estimator)

    Parameters
    ----------
    price_series : pd.Series or np.ndarray
        Raw Close price series.
    wavelet : str
        PyWavelets wavelet name.
    level : int
        Decomposition level.
    threshold_mode : str
        'soft' or 'hard' thresholding.

    Returns
    -------
    np.ndarray
        Denoised price series of the same length.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("pywt not installed — returning original series.")
        return np.array(price_series, dtype=np.float32)

    series = np.array(price_series, dtype=np.float64)
    n = len(series)

    try:
        coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)

        # Estimate noise std from the finest detail coefficients (cD1 = last)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(max(n, 2)))

        # Apply thresholding to ALL detail coefficients (skip approximation)
        denoised_coeffs = [coeffs[0]]  # keep approximation unchanged
        for detail in coeffs[1:]:
            denoised_coeffs.append(pywt.threshold(detail, threshold, mode=threshold_mode))

        reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet)

        # Trim / pad to original length
        if len(reconstructed) > n:
            reconstructed = reconstructed[:n]
        elif len(reconstructed) < n:
            reconstructed = np.pad(reconstructed, (0, n - len(reconstructed)), mode="edge")

        return reconstructed.astype(np.float32)

    except Exception as e:
        logger.warning(f"Wavelet denoising failed: {e}. Returning original.")
        return np.array(price_series, dtype=np.float32)


def plot_wavelet_decomposition(
    price_series: pd.Series | np.ndarray,
    ticker: str = "Stock",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """
    Plot 4-panel wavelet decomposition of the price series.

    Panels:
      1. Original Close price
      2. Wavelet_A3 (long-term trend)
      3. Wavelet_D2 (medium oscillations)
      4. Wavelet_D1 (noise component)

    Parameters
    ----------
    price_series : Series or ndarray
        Close price.
    ticker : str
        Ticker symbol for plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#161B27",
        "axes.facecolor":   "#161B27",
        "axes.edgecolor":   "#2A2F3E",
        "text.color":       "#FAFAFA",
        "grid.color":       "#2A2F3E",
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
    })

    wf = extract_wavelet_features(price_series)
    series = np.array(price_series, dtype=np.float64)
    n = len(series)
    x = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"{ticker} — Wavelet Decomposition (db4, Level 3)", fontsize=14, y=1.01)

    plot_data = [
        (series,              "#00C8FF", "Original Close Price",   "Price (₹)"),
        (wf["Wavelet_A3"],    "#FFD700", "A3 — Long-term Trend",   "cA3"),
        (wf["Wavelet_D2"],    "#FF6B35", "D2 — Medium Oscillation","cD2"),
        (wf["Wavelet_D1"],    "#FF4B4B", "D1 — Noise Component",   "cD1"),
    ]

    for ax, (data, color, label, ylabel) in zip(axes, plot_data):
        ax.plot(x, data, color=color, linewidth=1.2, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Steps", fontsize=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PART B — STANDALONE WAVELET NEURAL NETWORK MODEL
# ──────────────────────────────────────────────────────────────────────────────

class WaveletNeuralNetwork:
    """
    Standalone Wavelet Neural Network (WNN) with a custom trainable wavelet
    decomposition layer.

    The WNN learns to optimally decompose the input signal during training,
    unlike traditional fixed-basis DWT. This makes it adaptive to non-stationary
    financial time-series.

    Architecture
    ────────────
    Input → WaveletDecompositionLayer → Flatten → Dense(128, tanh)
          → Dense(64, tanh) → Dense(32, relu) → Dense(1)
    """

    def build_wnn_model(self, input_shape: tuple) -> "tf.keras.Model":  # type: ignore
        """
        Build the WNN Keras model.

        Parameters
        ----------
        input_shape : tuple
            (lookback, n_features)

        Returns
        -------
        tf.keras.Model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, Dense, Flatten, Lambda, Dropout
            )

            inputs = Input(shape=input_shape, name="wnn_input")

            # Trainable wavelet-like decomposition via 1D convolutions with
            # Mexican Hat–inspired initialisation
            x = tf.keras.layers.Conv1D(
                filters=16,
                kernel_size=5,
                padding="same",
                activation="tanh",
                name="wavelet_decomp_1",
                kernel_initializer="glorot_uniform",
            )(inputs)
            x = tf.keras.layers.Conv1D(
                filters=8,
                kernel_size=3,
                padding="same",
                activation="tanh",
                name="wavelet_decomp_2",
            )(x)
            x = Flatten()(x)
            x = Dense(128, activation="tanh", name="wnn_dense1")(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation="tanh", name="wnn_dense2")(x)
            x = Dense(32, activation="relu", name="wnn_dense3")(x)
            output = Dense(1, name="wnn_output")(x)

            model = Model(inputs=inputs, outputs=output, name="WaveletNN")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae"],
            )
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to build WNN model: {e}") from e

    def train_wnn(
        self,
        stock_df: pd.DataFrame,
        lookback: int = 100,
        epochs: int = 80,
        batch_size: int = 32,
    ) -> tuple:
        """
        Train the WNN on stock Close price (denoised).

        Parameters
        ----------
        stock_df : pd.DataFrame
            Must contain 'Close' column.
        lookback : int
            Sequence length for the sliding window.
        epochs : int
            Training epochs.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        tuple
            (trained_model, history, close_scaler, X_test, y_test, test_dates)
        """
        try:
            import tensorflow as tf

            logger.info("Training WNN model …")

            price = stock_df["Close"].values.astype(np.float64)

            # Denoise the price series before feeding to WNN
            denoised = denoise_with_wavelet(price)

            # Extract wavelet features as additional channels
            wf = extract_wavelet_features(price)

            # Stack denoised price + wavelet features as input channels
            feature_matrix = np.column_stack([
                denoised,
                wf["Wavelet_A3"].values,
                wf["Wavelet_D3"].values,
                wf["Wavelet_D2"].values,
                wf["Wavelet_D1"].values,
            ]).astype(np.float32)

            # Scale
            class _MinMaxScalerNP:
                def __init__(self, feature_range=(0.0, 1.0)):
                    self.feature_range = feature_range
                    self.data_min_ = None
                    self.data_max_ = None
                    self.scale_ = None
                    self.min_ = None

                def fit(self, X):
                    X = np.asarray(X, dtype=np.float64)
                    self.data_min_ = np.nanmin(X, axis=0)
                    self.data_max_ = np.nanmax(X, axis=0)
                    data_range = self.data_max_ - self.data_min_
                    data_range = np.where(data_range == 0.0, 1.0, data_range)
                    fr_min, fr_max = self.feature_range
                    self.scale_ = (fr_max - fr_min) / data_range
                    self.min_ = fr_min - self.data_min_ * self.scale_
                    return self

                def transform(self, X):
                    if self.scale_ is None or self.min_ is None:
                        raise ValueError("Scaler not fitted.")
                    X = np.asarray(X, dtype=np.float64)
                    return (X * self.scale_) + self.min_

                def fit_transform(self, X):
                    return self.fit(X).transform(X)

                def inverse_transform(self, X):
                    if self.scale_ is None or self.min_ is None:
                        raise ValueError("Scaler not fitted.")
                    X = np.asarray(X, dtype=np.float64)
                    return (X - self.min_) / self.scale_

            feature_scaler = _MinMaxScalerNP()
            close_scaler = _MinMaxScalerNP()
            scaled_feat = feature_scaler.fit_transform(feature_matrix).astype(np.float32)
            scaled_close = close_scaler.fit_transform(price.reshape(-1, 1)).astype(np.float32)

            # Build sequences
            X, y, seq_dates = [], [], []
            dates = stock_df["Date"].values if "Date" in stock_df.columns else np.arange(len(stock_df))

            for i in range(lookback, len(scaled_feat)):
                X.append(scaled_feat[i - lookback:i])
                y.append(scaled_close[i, 0])
                seq_dates.append(dates[i])

            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            test_dates = seq_dates[split_idx:]

            model = self.build_wnn_model(input_shape=(lookback, feature_matrix.shape[1]))

            es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[es],
                verbose=0,
            )

            return model, history, close_scaler, X_test, y_test, np.array(test_dates), feature_scaler

        except Exception as e:
            raise RuntimeError(f"WNN training failed: {e}") from e

    def predict_wnn(
        self,
        model,
        X_test: np.ndarray,
        scaler,
    ) -> np.ndarray:
        """
        Generate WNN predictions and inverse-scale them.

        Parameters
        ----------
        model : tf.keras.Model
            Trained WNN model.
        X_test : np.ndarray
            Test sequences.
        scaler : MinMaxScaler
            Close price scaler.

        Returns
        -------
        np.ndarray
            Predictions in original price scale.
        """
        preds_scaled = model.predict(X_test, verbose=0)
        return scaler.inverse_transform(preds_scaled).flatten()

    def plot_wnn_results(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        ticker: str,
        dates: Optional[np.ndarray] = None,
    ) -> "matplotlib.figure.Figure":  # type: ignore
        """
        Plot WNN actual vs predicted prices on a dark background chart.

        Parameters
        ----------
        actual : np.ndarray
            Actual Close prices.
        predicted : np.ndarray
            WNN predicted prices.
        ticker : str
            Ticker symbol.
        dates : np.ndarray, optional
            Date array for x-axis.

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
            "figure.facecolor": "#161B27",
            "axes.facecolor":   "#161B27",
            "axes.edgecolor":   "#2A2F3E",
            "text.color":       "#FAFAFA",
            "grid.color":       "#2A2F3E",
        })

        fig, ax = plt.subplots(figsize=(13, 5))

        if dates is not None:
            try:
                x = pd.to_datetime(dates)
                ax.plot(x, actual, color="#00C8FF", linewidth=1.6, label="Actual Price")
                ax.plot(x, predicted, color="#9B59B6", linewidth=1.5,
                        linestyle="--", label="WNN Predicted")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                fig.autofmt_xdate(rotation=30)
            except Exception:
                ax.plot(actual, color="#00C8FF", linewidth=1.6, label="Actual Price")
                ax.plot(predicted, color="#9B59B6", linewidth=1.5,
                        linestyle="--", label="WNN Predicted")
        else:
            ax.plot(actual, color="#00C8FF", linewidth=1.6, label="Actual Price")
            ax.plot(predicted, color="#9B59B6", linewidth=1.5,
                    linestyle="--", label="WNN Predicted")

        ax.set_title(f"{ticker} — Wavelet Neural Network Predictions", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
