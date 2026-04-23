"""
Train the MLP surrogate on an LHS dataset.

Pipeline:
  1. 80/20 train/val split (seeded)
  2. input standardization (z-score using train stats)
  3. output log1p + standardization
  4. Adam + cosine LR schedule, MSE loss
  5. early-stop if val_R2 > 0.998 on all three outputs (cheap on 10-bar)

Returns a `TrainedSurrogate` bundle with:
  * the fitted model
  * input/output normalization stats
  * val R^2 per output

Intentionally minimal: one file, no framework. We don't need Lightning
for a 64-wide MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .dataset import SurrogateDataset
from .model import MLP, ModelConfig


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class TrainedSurrogate:
    model: nn.Module
    x_stats: NormStats
    y_stats: NormStats             # on log1p(y) space
    output_names: tuple[str, ...]  # e.g. ("weight", "max_stress", "max_disp")
    val_r2: dict[str, float]


def _standardize_fit(a: np.ndarray) -> NormStats:
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return NormStats(mean=mean, std=std)


def _apply(a: np.ndarray, s: NormStats) -> np.ndarray:
    return (a - s.mean) / s.std


def _invert(a: np.ndarray, s: NormStats) -> np.ndarray:
    return a * s.std + s.mean


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train_surrogate(
    dataset: SurrogateDataset,
    seed: int = 0,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    n_hidden_layers: int = 2,
    val_frac: float = 0.2,
    device: str = "cpu",
    verbose: bool = False,
) -> TrainedSurrogate:
    """Fit an MLP to (X -> [weight, max_stress, max_disp])."""
    rng = np.random.default_rng(seed)

    X = dataset.X.astype(np.float32)
    y_raw = np.stack(
        [dataset.weight, dataset.max_stress, dataset.max_disp],
        axis=1,
    ).astype(np.float32)
    y_log = np.log1p(y_raw)

    N = X.shape[0]
    perm = rng.permutation(N)
    n_val = int(round(N * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    x_stats = _standardize_fit(X[tr_idx])
    y_stats = _standardize_fit(y_log[tr_idx])

    X_tr = _apply(X[tr_idx], x_stats).astype(np.float32)
    y_tr = _apply(y_log[tr_idx], y_stats).astype(np.float32)
    X_val = _apply(X[val_idx], x_stats).astype(np.float32)
    y_val = _apply(y_log[val_idx], y_stats).astype(np.float32)

    torch.manual_seed(seed)
    model = MLP(
        ModelConfig(
            n_inputs=X.shape[1],
            n_outputs=3,
            hidden=hidden,
            n_hidden_layers=n_hidden_layers,
        )
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                model.eval()
                val_loss = loss_fn(model(X_val_t), y_val_t).item()
            print(f"epoch {epoch+1:4d}  val_loss={val_loss:.5f}")

    # Final R^2 on original units.
    model.eval()
    with torch.no_grad():
        pred_std = model(X_val_t).cpu().numpy()
    pred_log = _invert(pred_std, y_stats)
    pred_raw = np.expm1(pred_log)
    target_raw = y_raw[val_idx]

    names = ("weight", "max_stress", "max_disp")
    val_r2 = {n: _r2(pred_raw[:, i], target_raw[:, i]) for i, n in enumerate(names)}

    return TrainedSurrogate(
        model=model,
        x_stats=x_stats,
        y_stats=y_stats,
        output_names=names,
        val_r2=val_r2,
    )


def save_surrogate(s: TrainedSurrogate, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": s.model.state_dict(),
            "cfg": asdict(s.model.cfg),
            "x_mean": s.x_stats.mean,
            "x_std": s.x_stats.std,
            "y_mean": s.y_stats.mean,
            "y_std": s.y_stats.std,
            "output_names": list(s.output_names),
            "val_r2": s.val_r2,
        },
        path,
    )


def load_surrogate(path: str | Path, device: str = "cpu") -> TrainedSurrogate:
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = MLP(ModelConfig(**bundle["cfg"])).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return TrainedSurrogate(
        model=model,
        x_stats=NormStats(mean=bundle["x_mean"], std=bundle["x_std"]),
        y_stats=NormStats(mean=bundle["y_mean"], std=bundle["y_std"]),
        output_names=tuple(bundle["output_names"]),
        val_r2=bundle["val_r2"],
    )
