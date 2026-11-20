"""Behaviour cloning (MLP) for the state-based placement task.

This is the imitation-learning baseline for the PPO placement policy: same observation and action
spaces, same environment, trained on scripted-expert demonstrations. It makes the RL-vs-IL
comparison in Phase 2 apples-to-apples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim), nn.Tanh()]  # actions live in [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BCConfig:
    hidden: tuple[int, ...] = (256, 256)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 100
    val_fraction: float = 0.1
    seed: int = 0


def train_bc(obs: np.ndarray, act: np.ndarray, cfg: BCConfig, device: str = "cpu", log=print) -> dict:
    """Train an MLP on (obs, act) pairs. Returns a checkpoint dict (weights + normalization stats)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n = len(obs)
    perm = rng.permutation(n)
    n_val = max(1, int(cfg.val_fraction * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mean = obs[tr_idx].mean(0)
    std = obs[tr_idx].std(0) + 1e-6
    norm = lambda x: (x - mean) / std  # noqa: E731

    x_tr = torch.tensor(norm(obs[tr_idx]), dtype=torch.float32, device=device)
    y_tr = torch.tensor(act[tr_idx], dtype=torch.float32, device=device)
    x_va = torch.tensor(norm(obs[val_idx]), dtype=torch.float32, device=device)
    y_va = torch.tensor(act[val_idx], dtype=torch.float32, device=device)

    model = MLP(obs.shape[1], act.shape[1], cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(x_tr), device=device)
        for i in range(0, len(order), cfg.batch_size):
            b = order[i : i + cfg.batch_size]
            loss = loss_fn(model(x_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        if val < best_val:
            best_val, best_state = val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            log(f"epoch {epoch:4d}  train_loss {loss.item():.5f}  val_loss {val:.5f}")
    return {
        "state_dict": best_state,
        "obs_mean": mean,
        "obs_std": std,
        "obs_dim": obs.shape[1],
        "act_dim": act.shape[1],
        "config": asdict(cfg),
        "best_val_loss": best_val,
    }


class BCPolicy:
    """Loaded BC policy with the common ``reset() / act(obs, info)`` interface."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model = MLP(ckpt["obs_dim"], ckpt["act_dim"], tuple(ckpt["config"]["hidden"])).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mean, self.std = ckpt["obs_mean"], ckpt["obs_std"]
        self.device = device

    def reset(self, obs=None, info=None) -> None:
        pass

    @torch.no_grad()
    def act(self, obs, info=None) -> np.ndarray:
        x = torch.tensor((np.asarray(obs) - self.mean) / self.std, dtype=torch.float32, device=self.device)
        return self.model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
