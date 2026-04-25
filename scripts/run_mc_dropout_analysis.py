"""Monte-Carlo dropout surrogate uncertainty study (thesis §4.14).

Trains a dropout-enabled MLP on the 10-bar LHS dataset (same as the
deterministic surrogate in :mod:`src.ml.train`), then does T stochastic
forward passes per test sample to obtain predictive mean + std. Writes:

  * results/mc_dropout_metrics.csv             (per-output summary)
  * results/mc_dropout_per_sample.csv          (per-test-sample detail)
  * figures/fig_4_14_mc_dropout_calibration.{png,svg}

We train a standalone MLP here rather than reusing
``results/10bar_surrogate.pt`` because the deterministic model has no
dropout; the GA/PSO production pipeline keeps using the determinstic
surrogate, so behaviour elsewhere is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; avoids SIGABRT under PyTorch+macOS

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting.style import FIGSIZE_WIDE, save, setup  # noqa: E402


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

N_MC_SAMPLES = 100          # stochastic forward passes per test point
DROPOUT_P = 0.10
HIDDEN = 64
EPOCHS = 400
BATCH_SIZE = 256
LR = 1e-3
SEED = 0


class MLPDropout(nn.Module):
    """MLP identical in width/depth to src.ml.model.MLP but with dropout
    after each ReLU. Dropout stays active at inference for MC sampling."""

    def __init__(self, n_in: int, n_out: int = 3, hidden: int = HIDDEN, p: float = DROPOUT_P) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize(a: np.ndarray):
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    setup()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train = np.load(RESULTS / "10bar_dataset_train.npz")
    test = np.load(RESULTS / "10bar_dataset_test.npz")

    X_tr = train["X"].astype(np.float32)
    y_tr_raw = np.stack([train["weight"], train["max_stress"], train["max_disp"]], axis=1).astype(np.float32)
    y_tr_log = np.log1p(y_tr_raw)

    X_te = test["X"].astype(np.float32)
    y_te_raw = np.stack([test["weight"], test["max_stress"], test["max_disp"]], axis=1).astype(np.float32)

    x_mu, x_sd = _standardize(X_tr)
    y_mu, y_sd = _standardize(y_tr_log)

    X_tr_n = (X_tr - x_mu) / x_sd
    y_tr_n = (y_tr_log - y_mu) / y_sd
    X_te_n = ((X_te - x_mu) / x_sd).astype(np.float32)

    model = MLPDropout(n_in=X_tr.shape[1])
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_n.astype(np.float32)),
                      torch.from_numpy(y_tr_n.astype(np.float32))),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    print(f"[mc-dropout] training {EPOCHS} epochs, p={DROPOUT_P}")
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}")

    # Critical: KEEP dropout active at inference (model.train() not .eval()).
    model.train()
    X_te_t = torch.from_numpy(X_te_n)

    preds_std = np.zeros((N_MC_SAMPLES, X_te_n.shape[0], 3), dtype=np.float32)
    with torch.no_grad():
        for t in range(N_MC_SAMPLES):
            preds_std[t] = model(X_te_t).numpy()

    # Invert normalisation + log1p.
    preds_log = preds_std * y_sd + y_mu                       # (T, N, 3)
    preds_raw = np.expm1(preds_log)                           # (T, N, 3)

    mean_raw = preds_raw.mean(axis=0)                         # (N, 3)
    std_raw = preds_raw.std(axis=0, ddof=1)                   # (N, 3)
    abs_err = np.abs(mean_raw - y_te_raw)                     # (N, 3)

    names = ("weight", "max_stress", "max_disp")
    metrics = []
    for i, n in enumerate(names):
        r2 = _r2(mean_raw[:, i], y_te_raw[:, i])
        rmse = float(np.sqrt(np.mean((mean_raw[:, i] - y_te_raw[:, i]) ** 2)))
        mean_std = float(std_raw[:, i].mean())
        corr = float(np.corrcoef(std_raw[:, i], abs_err[:, i])[0, 1])
        # 90 percent predictive interval coverage using +/- 1.645 sigma.
        lo = mean_raw[:, i] - 1.645 * std_raw[:, i]
        hi = mean_raw[:, i] + 1.645 * std_raw[:, i]
        coverage_90 = float(((y_te_raw[:, i] >= lo) & (y_te_raw[:, i] <= hi)).mean())
        metrics.append({
            "output": n,
            "r2_mean_pred": r2,
            "rmse_mean_pred": rmse,
            "mean_predictive_std": mean_std,
            "corr_std_abs_err": corr,
            "coverage_90pi": coverage_90,
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULTS / "mc_dropout_metrics.csv", index=False)
    print("[mc-dropout] metrics:")
    print(metrics_df.to_string(index=False))

    # Per-sample dump (small test set so fine to keep).
    per_sample = pd.DataFrame({
        f"{n}_true": y_te_raw[:, i] for i, n in enumerate(names)
    })
    for i, n in enumerate(names):
        per_sample[f"{n}_mean"] = mean_raw[:, i]
        per_sample[f"{n}_std"] = std_raw[:, i]
    per_sample.to_csv(RESULTS / "mc_dropout_per_sample.csv", index=False)

    # --- figure: 1x3 grid, predictive std vs |error| per output, with
    # a rolling-mean trend line to show calibration. ---
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_WIDE[0] * 1.4, FIGSIZE_WIDE[1]))
    for i, (n, ax) in enumerate(zip(names, axes)):
        s = std_raw[:, i]
        e = abs_err[:, i]
        ax.scatter(s, e, s=6, alpha=0.35, color="#1f77b4", edgecolor="none")
        order = np.argsort(s)
        window = max(50, len(s) // 20)
        rolling = pd.Series(e[order]).rolling(window, min_periods=10).mean().to_numpy()
        ax.plot(s[order], rolling, color="#d62728", lw=1.6, label=f"rolling mean (w={window})")
        mxy = max(float(s.max()), float(e.max()))
        ax.plot([0, mxy], [0, mxy], ls="--", color="0.6", lw=1.0, label=r"$|err|=\sigma$")
        ax.set_xlabel("MC-dropout predictive std $\\sigma$")
        ax.set_ylabel("absolute error $|y-\\hat\\mu|$")
        unit = {"weight": "lb", "max_stress": "psi", "max_disp": "in"}[n]
        ax.set_title(f"{n} ({unit})")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("MC-dropout calibration on 10-bar held-out set "
                 f"(T={N_MC_SAMPLES}, p={DROPOUT_P})", y=1.02)
    save(fig, FIGURES, "fig_4_14_mc_dropout_calibration")
    print(f"[mc-dropout] wrote figures/fig_4_14_mc_dropout_calibration.(png|svg)")


if __name__ == "__main__":
    main()
