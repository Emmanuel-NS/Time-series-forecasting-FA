"""Models 2 and 3 - recurrent (LSTM) and convolutional (TCN) sequence models.

Both models solve the same supervised problem: map a window of the last
``lookback`` scaled observations to the next scaled observation. They differ in
*how* they aggregate that window, and that difference is the point of the
comparison.

LSTM
    Processes the window one step at a time, carrying a gated state. Gating lets
    it retain information over long spans without the vanishing gradients of a
    plain RNN (Hochreiter and Schmidhuber, 1997), and it is the architecture most
    commonly reported for cellular traffic prediction (e.g. Trinh et al., 2018).
    Its weakness on this hardware is that the recurrence is inherently
    sequential, so a 144-step window means 144 dependent matrix products per
    sample and no useful parallelism across time.

TCN
    Aggregates the same window with stacked dilated causal convolutions (Bai,
    Kolter and Koltun, 2018). Dilation grows geometrically, so a receptive field
    of 144 steps is reached in a handful of layers, and every time step is
    computed in parallel. Causal padding guarantees no look-ahead. The trade-off
    is a *fixed* receptive field: unlike the LSTM it cannot, even in principle,
    use information older than its receptive field.

Shared training decisions, and why
----------------------------------
* **Loss.** Huber (smooth L1) rather than MSE. The exploratory analysis shows
  isolated spikes of an order of magnitude; under MSE those few samples would
  dominate the gradient and the model would systematically over-predict normal
  traffic to hedge against them. Huber is quadratic near zero and linear in the
  tail, which keeps the fit honest for the bulk of the data.
* **Optimiser.** Adam, with a plateau scheduler. On a series this smooth the
  loss surface is benign; the scheduler mainly helps squeeze out the last few
  percent once the daily cycle has been learnt.
* **Early stopping on validation MAE in original units,** not on the scaled
  training loss. The scaled loss keeps improving after the model has stopped
  getting better at the thing we actually report.
* **Deterministic seeding and single-threaded BLAS** so that the timing
  measurements are reproducible on a 2-core machine.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .. import config as C
from ..metrics import mae as mae_metric
from .base import Forecaster

DEVICE = torch.device("cpu")


def seed_everything(seed: int = C.RANDOM_SEED) -> None:
    """Make a run reproducible, including DataLoader shuffling order."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_threads(n: int = 2) -> None:
    """Pin the thread count so timing comparisons are meaningful.

    The host has 2 physical cores. Letting PyTorch spawn 4 threads on 2 cores
    adds contention and makes repeated timings noisy, which would undermine the
    training-time comparison the assignment asks for.
    """
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None


# ===========================================================================
# Architectures
# ===========================================================================


class LSTMNet(nn.Module):
    """Stacked LSTM whose final hidden state is projected to one scalar."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_size: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L) -> (B,)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


class _CausalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection.

    Causality is enforced by left-padding the input by ``(k-1)*d`` and then
    discarding the same number of trailing activations, so output ``t`` can only
    ever depend on inputs ``<= t``.
    """

    def __init__(self, channels_in: int, channels_out: int, kernel: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(channels_in, channels_out, 1) if channels_in != channels_out else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(self.act(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        skip = x if self.downsample is None else self.downsample(x)
        return self.act(y + skip)


class TCNNet(nn.Module):
    """Temporal convolutional network for one-step-ahead regression."""

    def __init__(
        self, channels: int = 32, levels: int = 5, kernel: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        blocks = []
        for level in range(levels):
            blocks.append(
                _CausalBlock(
                    channels_in=1 if level == 0 else channels,
                    channels_out=channels,
                    kernel=kernel,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L) -> (B,)
        h = self.blocks(x.unsqueeze(1))
        return self.head(h[:, :, -1]).squeeze(-1)


# ===========================================================================
# Shared training loop
# ===========================================================================


@dataclass
class TrainConfig:
    """Training hyperparameters shared by both neural models.

    ``time_budget_seconds`` exists because this study runs on a 2-core laptop.
    A dilated TCN over a 144-step window costs roughly an order of magnitude
    more per epoch than an LSTM of comparable parameter count, since every
    dilation level convolves the whole padded sequence. Without a wall-clock cap
    a single unlucky configuration can consume the time available for the entire
    search. The budget is checked between epochs, so a run is always stopped at
    a valid checkpoint with its best weights retained, and any configuration
    that hit the cap is flagged in the experiment log rather than silently
    reported as converged.
    """

    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    huber_delta: float = 1.0
    patience: int = 8
    grad_clip: float = 1.0
    scheduler_patience: int = 3
    time_budget_seconds: float | None = None
    verbose: bool = True


class _NeuralForecaster(Forecaster):
    """Training/evaluation machinery common to the LSTM and the TCN."""

    needs_windows = True

    def __init__(self, train_config: TrainConfig | None = None) -> None:
        super().__init__()
        self.cfg = train_config or TrainConfig()
        self.net: nn.Module | None = None
        self._scaler = None
        self._budget_exhausted = False

    def _build(self) -> nn.Module:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    # -- fitting ---------------------------------------------------------
    def fit(self, data) -> "_NeuralForecaster":
        seed_everything()
        configure_threads()
        self._scaler = data.scaler
        self.net = self._build().to(DEVICE)
        self.report.n_parameters = sum(p.numel() for p in self.net.parameters())

        # ``make_windows`` returns a read-only strided view, so the tensors are
        # built with an explicit copy; sharing that buffer with torch would be
        # undefined behaviour. The copy is ~4 MB for a 144-step window.
        loader = DataLoader(
            TensorDataset(
                torch.tensor(data.x_train, dtype=torch.float32),
                torch.tensor(data.y_train, dtype=torch.float32),
            ),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(C.RANDOM_SEED),
        )
        x_val = torch.tensor(data.x_val, dtype=torch.float32)

        criterion = nn.HuberLoss(delta=self.cfg.huber_delta)
        optimiser = torch.optim.Adam(
            self.net.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=self.cfg.scheduler_patience
        )

        best_state, best_mae, best_epoch, stale = None, math.inf, 0, 0
        self._budget_exhausted = False

        def _run() -> None:
            nonlocal best_state, best_mae, best_epoch, stale
            started = time.perf_counter()
            for epoch in range(1, self.cfg.epochs + 1):
                self.net.train()
                total = 0.0
                for xb, yb in loader:
                    optimiser.zero_grad(set_to_none=True)
                    loss = criterion(self.net(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                    optimiser.step()
                    total += loss.item() * len(xb)
                train_loss = total / len(loader.dataset)

                # Validation MAE is computed in original activity units, which
                # is the quantity the report compares models on.
                self.net.eval()
                with torch.no_grad():
                    pred_scaled = self.net(x_val).numpy()
                val_mae = mae_metric(
                    data.y_val_raw, self._scaler.inverse_transform(pred_scaled)
                )
                scheduler.step(val_mae)

                self.report.history.append(
                    {
                        "epoch": epoch,
                        "train_huber": round(train_loss, 6),
                        "val_MAE": round(val_mae, 3),
                        "lr": optimiser.param_groups[0]["lr"],
                    }
                )
                if self.cfg.verbose and (epoch == 1 or epoch % 5 == 0):
                    print(f"    epoch {epoch:3d}  huber={train_loss:.5f}  val MAE={val_mae:8.2f}")

                if val_mae < best_mae - 1e-9:
                    best_mae, best_epoch, stale = val_mae, epoch, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    stale += 1
                    if stale >= self.cfg.patience:
                        if self.cfg.verbose:
                            print(f"    early stop at epoch {epoch} (best {best_epoch})")
                        break
                self.report.epochs_run = epoch

                budget = self.cfg.time_budget_seconds
                if budget is not None and time.perf_counter() - started > budget:
                    self._budget_exhausted = True
                    print(
                        f"    stopped at epoch {epoch}: {budget:.0f}s training budget "
                        f"reached (best epoch {best_epoch})"
                    )
                    break

        self._timed_fit(_run)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.report.best_epoch = best_epoch
        self.report.notes = (
            f"{self.report.n_parameters:,} params, best epoch {best_epoch} of "
            f"{self.report.epochs_run} run, val MAE {best_mae:.2f}"
            + (" [TIME BUDGET REACHED]" if self._budget_exhausted else "")
        )
        return self

    # -- prediction ------------------------------------------------------
    def _forward(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), 512):  # bounded memory during inference
                batch = torch.tensor(x[start : start + 512], dtype=torch.float32)
                outputs.append(self.net(batch).numpy())
        return self._scaler.inverse_transform(np.concatenate(outputs))

    def predict_test(self, data) -> np.ndarray:
        return self._forward(data.x_test)

    def predict_val(self, data) -> np.ndarray:
        return self._forward(data.x_val)


class LSTMForecaster(_NeuralForecaster):
    """LSTM one-step-ahead forecaster."""

    name = "LSTM"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        train_config: TrainConfig | None = None,
    ) -> None:
        super().__init__(train_config)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def _build(self) -> nn.Module:
        return LSTMNet(self.hidden_size, self.num_layers, self.dropout)

    def hyperparameters(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "batch_size": self.cfg.batch_size,
            "lr": self.cfg.learning_rate,
        }


class TCNForecaster(_NeuralForecaster):
    """Dilated causal convolutional one-step-ahead forecaster."""

    name = "TCN"

    def __init__(
        self,
        channels: int = 32,
        levels: int = 5,
        kernel: int = 3,
        dropout: float = 0.1,
        train_config: TrainConfig | None = None,
    ) -> None:
        super().__init__(train_config)
        self.channels = channels
        self.levels = levels
        self.kernel = kernel
        self.dropout = dropout

    def _build(self) -> nn.Module:
        net = TCNNet(self.channels, self.levels, self.kernel, self.dropout)
        self.receptive_field = net.receptive_field
        return net

    def hyperparameters(self) -> dict:
        rf = 1 + 2 * (self.kernel - 1) * (2 ** self.levels - 1)
        return {
            "channels": self.channels,
            "levels": self.levels,
            "kernel": self.kernel,
            "dropout": self.dropout,
            "receptive_field": rf,
            "batch_size": self.cfg.batch_size,
            "lr": self.cfg.learning_rate,
        }
