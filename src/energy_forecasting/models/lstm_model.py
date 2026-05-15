"""LSTM forecaster — univariate sequence model in PyTorch.

For the brief's "deep learning" category (MLP/LSTM/GRU). Design choices:

- **Univariate input.** At each timestep ``t`` the model sees the past
  ``lookback`` hours of the target (a single channel). Calendar / cyclical /
  holiday features are *not* fed to the LSTM in v1 — they are already partly
  captured by the sequence's periodicity. We can extend to multi-channel
  later if Phase 4F tuning suggests it.
- **Single LSTM layer + linear head.** 64 hidden units, one layer, dropout 0
  inside the LSTM (one layer ⇒ dropout has no effect anyway). A small
  network trains in minutes on CPU and is easier to interpret.
- **Standardized input.** Target is z-scored using training mean/std. The
  prediction is denormalized at the output. Standardization keeps gradients
  in a sensible range without needing batch-norm.
- **Two predict modes** (mirroring SARIMA):
  - :meth:`predict_rolling_one_step` — at each validation hour ``t``, build
    the sequence from actual past ``y`` (combining train tail + validation
    actuals up to ``t-1``). This is the leaderboard protocol.
  - :meth:`predict` — recursive multi-step using own predictions for future
    sequence elements.

Deterministic given the global RNG seed (see ``utils.set_global_seed``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_forecasting.exceptions import ModelError
from energy_forecasting.models.base import BaseForecaster
from energy_forecasting.utils import get_logger

FloatArray = npt.NDArray[np.float64]

CellType = Literal["lstm", "gru"]

_logger = get_logger(__name__)


class LSTMForecaster(BaseForecaster):
    """Univariate LSTM (or GRU) on past-``lookback`` target sequence.

    Set ``cell_type='gru'`` to swap the recurrent cell from LSTM to GRU.
    GRU has fewer parameters (no separate output gate) and trains faster;
    the rest of the model — sequence prep, normalization, head — is identical.
    """

    def __init__(
        self,
        *,
        lookback: int = 168,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 25,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        cell_type: CellType = "lstm",
        name: str | None = None,
    ) -> None:
        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {cell_type!r}")
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.cell_type: CellType = cell_type
        self.name: str = name or (
            f"{cell_type}_h{hidden_size}_lb{lookback}_l{num_layers}_e{epochs}"
        )
        self._net: Any = None
        self._train_mean: float | None = None
        self._train_std: float | None = None
        # Tail of training y, needed to build the first few validation sequences.
        self._train_y_tail: np.ndarray | None = None

    # ─────────────────────────────────────────────────────────────
    #  Internal: torch model class is defined lazily inside fit()
    #  so importing this module doesn't force torch to load.
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_net(
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: CellType,
    ) -> Any:
        import torch.nn as nn

        cell_kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        rnn = nn.LSTM(**cell_kwargs) if cell_type == "lstm" else nn.GRU(**cell_kwargs)

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn
                self.head = nn.Linear(hidden_size, 1)
                self._is_lstm = cell_type == "lstm"

            def forward(self, x):  # x: (batch, lookback, input_size)
                # nn.LSTM returns (output, (h_n, c_n)); nn.GRU returns (output, h_n).
                out = self.rnn(x)
                h_n = out[1][0] if self._is_lstm else out[1]
                last = h_n[-1]  # top layer's last hidden state: (batch, hidden_size)
                return self.head(last).squeeze(-1)  # (batch,)

        return _Net()

    # ─────────────────────────────────────────────────────────────
    #  fit / predict
    # ─────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMForecaster":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if len(X) != len(y):
            raise ModelError(f"X ({len(X)}) and y ({len(y)}) must have matching length")
        if len(y) < self.lookback + 1:
            raise ModelError(
                f"Need at least lookback+1={self.lookback + 1} training rows, got {len(y)}"
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_arr = y.to_numpy(dtype=np.float32)
        self._train_mean = float(y_arr.mean())
        self._train_std = float(y_arr.std() + 1e-8)
        y_norm = (y_arr - self._train_mean) / self._train_std

        # Build sliding-window dataset: features[i] = y_norm[i:i+lookback], target = y_norm[i+lookback]
        seq, tgt = self._build_windows(y_norm, self.lookback)
        seq_t = torch.from_numpy(seq).unsqueeze(-1)  # (N, lookback, 1)
        tgt_t = torch.from_numpy(tgt)  # (N,)

        ds = TensorDataset(seq_t, tgt_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self._net = self._build_net(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        _logger.info(
            f"Fitting {self.name} on {len(seq):,} sequences "
            f"(lookback={self.lookback}, hidden={self.hidden_size}, epochs={self.epochs})"
        )
        self._net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_samples = 0
            for sb, tb in loader:
                optimizer.zero_grad()
                pred = self._net(sb)
                loss = loss_fn(pred, tb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * sb.size(0)
                n_samples += sb.size(0)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                _logger.info(
                    f"  epoch {epoch + 1}/{self.epochs}  train_mse_norm={total_loss / n_samples:.4f}"
                )

        self._train_y_tail = y_arr[-self.lookback:].astype(np.float64)
        _logger.info(f"{self.name} fit complete")
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Recursive multi-step forecast: roll forward using own predictions."""
        self._require_fitted()
        import torch

        history = list(self._train_y_tail)  # type: ignore[arg-type]
        out: list[float] = []
        self._net.eval()
        with torch.no_grad():
            for _ in range(len(X)):
                seq = np.array(history[-self.lookback:], dtype=np.float32)
                seq_norm = (seq - self._train_mean) / self._train_std
                seq_t = torch.from_numpy(seq_norm).reshape(1, self.lookback, 1)
                pred_norm = float(self._net(seq_t).item())
                pred = pred_norm * self._train_std + self._train_mean
                out.append(pred)
                history.append(pred)  # recursive: use own prediction
        return np.asarray(out, dtype=np.float64)

    def predict_rolling_one_step(
        self,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> FloatArray:
        """At each validation hour, use the actual past ``lookback`` ``y`` values."""
        self._require_fitted()
        import torch

        y_v = y_valid.to_numpy(dtype=np.float64)
        if len(X_valid) != len(y_v):
            raise ModelError("X_valid and y_valid must have matching length")

        # Build the full chain: training tail + validation actuals.
        chain = np.concatenate([self._train_y_tail, y_v])  # type: ignore[arg-type]

        out: list[float] = []
        self._net.eval()
        with torch.no_grad():
            for t in range(len(y_v)):
                # Sequence ending at chain[lookback + t - 1] predicts chain[lookback + t].
                start = t
                end = t + self.lookback
                seq = chain[start:end].astype(np.float32)
                seq_norm = (seq - self._train_mean) / self._train_std
                seq_t = torch.from_numpy(seq_norm).reshape(1, self.lookback, 1)
                pred_norm = float(self._net(seq_t).item())
                pred = pred_norm * self._train_std + self._train_mean
                out.append(pred)
        return np.asarray(out, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────
    #  Introspection
    # ─────────────────────────────────────────────────────────────

    def get_hyperparameters(self) -> dict[str, object]:
        return {
            "cell_type": self.cell_type,
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    # ─────────────────────────────────────────────────────────────
    #  Pickle support — convert _net to state_dict, rebuild on load.
    #
    #  Why: the network class is defined inside ``_build_net`` (a closure)
    #  so joblib/pickle can't find it by qualified name. The PyTorch idiom
    #  is to persist only the state_dict (weights as tensors) and rebuild
    #  the architecture from hyperparameters on load. This is also more
    #  robust to architecture-code changes.
    # ─────────────────────────────────────────────────────────────

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("_net") is not None:
            state["_net"] = {"__state_dict__": state["_net"].state_dict()}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        saved_net = state.pop("_net", None)
        self.__dict__.update(state)
        self._net = None
        if isinstance(saved_net, dict) and "__state_dict__" in saved_net:
            self._net = self._build_net(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                cell_type=self.cell_type,
            )
            self._net.load_state_dict(saved_net["__state_dict__"])
            self._net.eval()  # always eval after load — we predict, not retrain

    # ─────────────────────────────────────────────────────────────
    #  Internals
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_windows(y_norm: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        """Build sliding windows: each row is (lookback past values, next value)."""
        n = len(y_norm) - lookback
        seq = np.lib.stride_tricks.sliding_window_view(y_norm[:-1], lookback)[:n]
        tgt = y_norm[lookback : lookback + n]
        return seq.astype(np.float32), tgt.astype(np.float32)

    def _require_fitted(self) -> None:
        if self._net is None or self._train_mean is None:
            raise ModelError(f"{self.name}.predict() called before fit()")
