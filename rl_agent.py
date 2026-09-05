"""
rl_agent.py
===========
Deep Q-Network (DQN) Reinforcement Learning agent for stock trading decisions.

The RL agent uses LSTM predictions as its primary state signal and learns
BUY / HOLD / SELL policies through reward-shaped experience replay.

Author: StockSense AI — Research-Grade Upgrade
"""

from __future__ import annotations

import os
import logging
import warnings
import pickle
import collections
import numpy as np
import pandas as pd
from typing import Optional, Deque

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DQN_MODEL_PATH = os.path.join(BASE_DIR, "dqn_agent.pkl")

# ── Action constants ───────────────────────────────────────────────────────────
ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# ── State dimension ────────────────────────────────────────────────────────────
STATE_DIM = 13


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class StockTradingEnv:
    """
    Gymnasium-style environment for stock trading.

    State (13-dim):
      [norm_close, lstm_pred, rf_pred, svm_pred, hybrid_pred,
       sentiment, rsi, macd_signal, bollinger_pos, vol_20d,
       current_position, unrealized_pnl, days_in_trade]

    Actions: 0=HOLD, 1=BUY, 2=SELL
    """

    def __init__(
        self,
        stock_df: pd.DataFrame,
        lstm_preds: np.ndarray,
        rf_preds: np.ndarray,
        svm_preds: np.ndarray,
        hybrid_preds: np.ndarray,
        sentiment_series: np.ndarray,
        initial_capital: float = 100_000.0,
    ):
        """
        Parameters
        ----------
        stock_df : pd.DataFrame
            Must contain: Close, Volume, and technical indicators if pre-computed.
        lstm_preds, rf_preds, svm_preds, hybrid_preds : np.ndarray
            Next-step predictions for each model (same length as stock_df test portion).
        sentiment_series : np.ndarray
            Daily sentiment scores.
        initial_capital : float
        """
        self.stock_df     = stock_df.reset_index(drop=True)
        self.lstm_preds   = lstm_preds
        self.rf_preds     = rf_preds
        self.svm_preds    = svm_preds
        self.hybrid_preds = hybrid_preds
        self.sentiment    = sentiment_series
        self.initial_capital = initial_capital

        # Pre-compute technical indicators
        self._compute_indicators()
        self.n_steps = len(self.stock_df)

        # Trading state
        self.reset()

    def _compute_indicators(self) -> None:
        """Compute RSI, MACD, Bollinger Bands, 20-day volatility."""
        prices = self.stock_df["Close"].values.astype(np.float64)
        n = len(prices)

        # ── RSI 14 ────────────────────────────────────────────────────────────
        rsi = np.full(n, 50.0)
        try:
            delta = np.diff(prices)
            gain  = np.where(delta > 0, delta, 0.0)
            loss  = np.where(delta < 0, -delta, 0.0)
            avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
            avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
            avg_loss = np.where(avg_loss == 0, 1e-8, avg_loss)
            rs = avg_gain / avg_loss
            rsi_full = 100 - (100 / (1 + rs))
            rsi[1:] = rsi_full
        except Exception:
            pass
        self.rsi = rsi

        # ── MACD (12-26-9) ────────────────────────────────────────────────────
        macd_line = np.zeros(n)
        signal_line = np.zeros(n)
        try:
            ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
            macd_line = ema12 - ema26
            signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        except Exception:
            pass
        self.macd = macd_line
        self.macd_signal = signal_line

        # ── Bollinger Bands position ─────────────────────────────────────────
        boll_pos = np.full(n, 0.5)
        try:
            mid = pd.Series(prices).rolling(20).mean().values
            std = pd.Series(prices).rolling(20).std().values
            upper = mid + 2 * std
            lower = mid - 2 * std
            band_width = upper - lower
            band_width = np.where(band_width == 0, 1e-8, band_width)
            boll_pos = np.clip((prices - lower) / band_width, 0, 1)
        except Exception:
            pass
        self.bollinger_pos = boll_pos

        # ── 20-day rolling volatility ─────────────────────────────────────────
        vol20 = np.full(n, 0.015)
        try:
            returns = np.diff(prices) / prices[:-1]
            rol_vol = pd.Series(returns).rolling(20).std().values
            vol20[1:] = rol_vol
        except Exception:
            pass
        self.vol20 = vol20

        # Normalise Close for state
        cmin, cmax = prices.min(), prices.max()
        self.norm_close = (prices - cmin) / (cmax - cmin + 1e-8)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_idx       = 20  # start after warm-up
        self.position       = 0   # 0=no position, 1=long
        self.buy_price      = 0.0
        self.days_in_trade  = 0
        self.recent_trades  = []
        self.portfolio_value = self.initial_capital
        self.trade_history  = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build the 13-dimensional state vector at current step."""
        i = min(self.step_idx, self.n_steps - 1)

        # Align prediction arrays
        pred_i = min(i, len(self.lstm_preds) - 1) if len(self.lstm_preds) > 0 else 0
        sent_i  = min(i, len(self.sentiment) - 1) if len(self.sentiment) > 0 else 0

        current_price = float(self.stock_df["Close"].iloc[i])
        price_range   = max(self.stock_df["Close"].max() - self.stock_df["Close"].min(), 1e-8)

        lstm_p = float(self.lstm_preds[pred_i])   if len(self.lstm_preds)   > 0 else current_price
        rf_p   = float(self.rf_preds[pred_i])     if len(self.rf_preds)     > 0 else current_price
        svm_p  = float(self.svm_preds[pred_i])    if len(self.svm_preds)    > 0 else current_price
        hyb_p  = float(self.hybrid_preds[pred_i]) if len(self.hybrid_preds) > 0 else current_price
        sent   = float(self.sentiment[sent_i])    if len(self.sentiment)    > 0 else 0.0

        unrealized_pnl = 0.0
        if self.position == 1 and self.buy_price > 0:
            unrealized_pnl = (current_price - self.buy_price) / self.buy_price

        state = np.array([
            float(self.norm_close[i]),
            np.clip((lstm_p - current_price) / price_range, -1, 1),
            np.clip((rf_p   - current_price) / price_range, -1, 1),
            np.clip((svm_p  - current_price) / price_range, -1, 1),
            np.clip((hyb_p  - current_price) / price_range, -1, 1),
            np.clip(sent, -1, 1),
            float(self.rsi[i]) / 100.0,
            np.clip(float(self.macd[i]) / (price_range + 1e-8), -1, 1),
            float(self.bollinger_pos[i]),
            np.clip(float(self.vol20[i]) * 100, 0, 1),
            float(self.position),
            np.clip(unrealized_pnl, -1, 1),
            np.clip(self.days_in_trade / 60.0, 0, 1),
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> tuple:
        """
        Execute action and return (next_state, reward, done, info).

        Parameters
        ----------
        action : int
            0=HOLD, 1=BUY, 2=SELL

        Returns
        -------
        tuple: (next_state, reward, done, info)
        """
        i = self.step_idx
        current_price = float(self.stock_df["Close"].iloc[i])
        reward = 0.0
        info   = {}

        # ── BUY ──────────────────────────────────────────────────────────────
        if action == ACTION_BUY and self.position == 0:
            self.position      = 1
            self.buy_price     = current_price
            self.days_in_trade = 0
            self.recent_trades.append(i)
            info["trade"] = "BUY"

        # ── SELL ─────────────────────────────────────────────────────────────
        elif action == ACTION_SELL and self.position == 1:
            pnl = (current_price - self.buy_price) / self.buy_price * 100
            reward = pnl  # primary reward: % return

            # Bonus for catching a good trend (>5% rise)
            if pnl > 5.0:
                reward += 1.0

            self.trade_history.append({
                "buy_price":  self.buy_price,
                "sell_price": current_price,
                "pnl_pct":    pnl,
            })
            self.position      = 0
            self.buy_price     = 0.0
            self.days_in_trade = 0
            info["trade"] = f"SELL (PnL={pnl:.2f}%)"

        # ── HOLD ─────────────────────────────────────────────────────────────
        else:
            if self.position == 1:
                unrealized = (current_price - self.buy_price) / self.buy_price * 100
                reward = 0.01 * unrealized  # small +/- hold reward

                # Stop loss penalty: held in loss > 5%
                if unrealized < -5.0 and self.days_in_trade > 10:
                    reward -= 2.0
            self.days_in_trade += 1

        # ── Overtrading penalty ───────────────────────────────────────────────
        recent_5d = [t for t in self.recent_trades if i - t <= 5]
        if len(recent_5d) > 2:
            reward -= 0.1

        # Advance step
        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1
        next_state = self._get_state()

        return next_state, reward, done, info


# ══════════════════════════════════════════════════════════════════════════════
# DQN AGENT
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Network:
      Input(13) → Dense(256, relu) → Dense(128, relu) → Dense(64, relu)
               → Dense(3, linear)  [Q-values for 3 actions]
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10_000,
        batch_size: int = 64,
        update_target_every: int = 100,
    ):
        self.state_dim    = state_dim
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.update_target_every = update_target_every
        self.replay_every = 4  # OPTIMIZATION: replay every N steps, not every step
        self.memory: Deque = collections.deque(maxlen=memory_size)
        self.step_count   = 0

        self.model        = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self._update_target()

    def _build_model(self, lr: float):
        """Build the DQN Q-network."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                Dense(256, activation="relu", input_shape=(self.state_dim,), name="dqn_d1"),
                Dropout(0.1),
                Dense(128, activation="relu", name="dqn_d2"),
                Dense(64,  activation="relu", name="dqn_d3"),
                Dense(self.n_actions, activation="linear", name="dqn_out"),
            ])
            model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
            return model

        except Exception as e:
            logger.error(f"DQN model build failed: {e}")
            return None

    def _update_target(self) -> None:
        """Copy weights from main network to target network."""
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done) -> None:
        """Store transition in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.

        Returns
        -------
        int
            Action: 0=HOLD, 1=BUY, 2=SELL
        """
        if self.model is None or np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        # OPTIMIZATION: Use __call__ instead of predict() for single-sample
        # inference — avoids predict()'s overhead (data conversion, callbacks)
        try:
            import tensorflow as tf
            q_vals = self.model(state[np.newaxis, :], training=False).numpy()[0]
        except Exception:
            q_vals = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q_vals))

    def replay(self) -> Optional[float]:
        """
        Sample a mini-batch from memory and train the Q-network.

        Returns
        -------
        float or None
            Training loss, or None if insufficient memory.
        """
        if len(self.memory) < self.batch_size or self.model is None:
            return None

        batch = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.memory[i] for i in batch]
        )

        states      = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards     = np.array(rewards, dtype=np.float32)
        actions     = np.array(actions, dtype=np.int32)
        dones       = np.array(dones,   dtype=np.float32)

        # Q-targets using Bellman equation with target network
        q_vals      = self.model.predict(states,      verbose=0)
        q_next      = self.target_model.predict(next_states, verbose=0)

        # OPTIMIZATION: Vectorized Bellman update (no Python loop)
        max_q_next = np.max(q_next, axis=1)
        targets = rewards + self.gamma * max_q_next * (1.0 - dones)
        q_vals[np.arange(self.batch_size), actions] = targets

        history = self.model.fit(states, q_vals, epochs=1, verbose=0, batch_size=self.batch_size)
        loss = float(history.history["loss"][0])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self._update_target()

        return loss

    def train_agent(
        self,
        env: StockTradingEnv,
        episodes: int = 20,
    ) -> dict:
        """
        Train the DQN agent over multiple episodes on historical data.

        Parameters
        ----------
        env : StockTradingEnv
        episodes : int
            Number of training episodes.

        Returns
        -------
        dict
            Keys: episode_rewards, episode_losses, final_epsilon
        """
        episode_rewards = []
        episode_losses  = []

        logger.info(f"Training DQN agent for {episodes} episodes …")

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            total_loss   = 0.0
            loss_count   = 0
            done = False

            step_in_episode = 0

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_in_episode += 1

                # OPTIMIZATION: replay only every N steps (was every step!)
                # This is the single biggest speedup — cuts RL time by ~75%
                if step_in_episode % self.replay_every == 0:
                    loss = self.replay()
                    if loss is not None:
                        total_loss += loss
                        loss_count += 1

            episode_rewards.append(total_reward)
            episode_losses.append(total_loss / max(loss_count, 1))

            if (ep + 1) % 10 == 0:
                logger.info(
                    f"Episode {ep+1}/{episodes} — "
                    f"Reward={total_reward:.2f} | ε={self.epsilon:.3f}"
                )

        # Save agent
        self._save(DQN_MODEL_PATH)

        return {
            "episode_rewards": episode_rewards,
            "episode_losses":  episode_losses,
            "final_epsilon":   self.epsilon,
        }

    def generate_trading_signal(
        self,
        state: np.ndarray,
    ) -> dict:
        """
        Generate a trading signal with confidence from Q-values.

        Parameters
        ----------
        state : np.ndarray
            Current state vector (13-dim).

        Returns
        -------
        dict
            Keys: action, action_name, confidence, q_values
        """
        if self.model is None:
            return {
                "action":      ACTION_HOLD,
                "action_name": "HOLD",
                "confidence":  0.33,
                "q_values":    [0.33, 0.33, 0.33],
            }

        q_vals = self.model.predict(state[np.newaxis, :], verbose=0)[0]

        # Softmax for confidence
        q_shifted = q_vals - np.max(q_vals)
        exp_q     = np.exp(q_shifted)
        probs     = exp_q / (exp_q.sum() + 1e-8)

        action = int(np.argmax(q_vals))
        confidence = float(probs[action])

        return {
            "action":      action,
            "action_name": ACTION_NAMES[action],
            "confidence":  round(confidence, 4),
            "q_values":    q_vals.tolist(),
            "probabilities": probs.tolist(),
        }

    def _save(self, path: str) -> None:
        """Pickle-save the agent (weights only, not the Keras model)."""
        try:
            save_data = {
                "epsilon":      self.epsilon,
                "step_count":   self.step_count,
                "gamma":        self.gamma,
                "n_actions":    self.n_actions,
                "state_dim":    self.state_dim,
            }
            if self.model:
                save_data["weights"] = self.model.get_weights()
            with open(path, "wb") as f:
                pickle.dump(save_data, f)
            logger.info(f"DQN agent saved to {path}")
        except Exception as e:
            logger.warning(f"Could not save DQN agent: {e}")

    @classmethod
    def load(cls, path: str = DQN_MODEL_PATH) -> "DQNAgent":
        """
        Load a previously saved DQN agent.

        Parameters
        ----------
        path : str

        Returns
        -------
        DQNAgent
        """
        agent = cls()
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            agent.epsilon    = data.get("epsilon",    0.01)
            agent.step_count = data.get("step_count", 0)
            if "weights" in data and agent.model:
                agent.model.set_weights(data["weights"])
                agent._update_target()
            logger.info(f"DQN agent loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load DQN agent from {path}: {e}")
        return agent


def plot_rl_training_progress(train_result: dict, ticker: str) -> "matplotlib.figure.Figure":  # type: ignore
    """
    Plot episode rewards and losses over RL training.

    Parameters
    ----------
    train_result : dict
        Output from DQNAgent.train_agent().
    ticker : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#161B27", "axes.facecolor": "#161B27",
        "axes.edgecolor": "#2A2F3E",   "text.color": "#FAFAFA",
        "grid.color": "#2A2F3E",
    })

    rewards = train_result.get("episode_rewards", [])
    losses  = train_result.get("episode_losses",  [])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    if rewards:
        episodes = np.arange(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, color="#00C8FF", linewidth=1.2, label="Episode Reward")
        # Smoothed
        if len(rewards) >= 5:
            smooth = pd.Series(rewards).rolling(5, min_periods=1).mean().values
            ax1.plot(episodes, smooth, color="#FFD700", linewidth=2.0, label="5-ep Moving Avg")
        ax1.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Total Reward")
        ax1.legend(fontsize=9)
        ax1.set_title(f"{ticker} — DQN RL Agent Training Progress", fontsize=13)

    if losses:
        episodes = np.arange(1, len(losses) + 1)
        ax2.plot(episodes, losses, color="#FF6B35", linewidth=1.2)
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Episode")

    fig.tight_layout()
    return fig
