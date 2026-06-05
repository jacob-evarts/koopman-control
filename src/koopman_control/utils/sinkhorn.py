"""Entropic optimal-transport (Sinkhorn) loss on 2D occupancy grids.

We use the entropic (Sinkhorn) approximation of the 2-Wasserstein distance,
computed in the log domain for numerical stability.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _grid_cost_matrix(grid_size: int) -> torch.Tensor:
    """Squared-Euclidean cost between cells of a ``grid_size`` x ``grid_size`` grid.

    Coordinates are normalized to ``[0, 1]`` and the matrix is rescaled so its
    maximum entry is 1, which decouples the choice of ``epsilon`` from the grid
    resolution. Returns ``(N, N)`` with ``N = grid_size ** 2``.
    """
    coords = torch.linspace(0.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (N, 2)
    cost = torch.cdist(pts, pts, p=2).pow(2)  # (N, N)
    max_cost = torch.clamp(cost.max(), min=1e-12)
    return cost / max_cost


def sinkhorn_distance_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    """Fast (matmul/kernel-form) entropic OT cost between ``a`` and ``b``.

    Iterates the classic scaling recursion ``u = a / (K v)``, ``v = b / (Kᵀ u)``
    with ``K = exp(-C / eps)``. Each step is two ``(B, N) · (N, N)`` matmuls
    (BLAS), so it avoids the ``(B, N, N)`` intermediates of the log-domain
    version and is roughly an order of magnitude faster.

    ``a``, ``b``: ``(B, N)`` non-negative and sum-to-one along ``N``.
    ``cost``: ``(N, N)`` ground cost. Returns ``(B,)`` transport cost ``<P, C>``.
    """
    k = torch.exp(-cost / epsilon)  # (N, N)
    kc = k * cost  # (N, N), reused for the final transport cost
    tiny = 1e-30

    u = torch.full_like(a, 1.0 / a.shape[-1])
    v = torch.full_like(b, 1.0 / b.shape[-1])
    for _ in range(n_iters):
        u = a / (v @ k.t() + tiny)  # (B, N)
        v = b / (u @ k + tiny)  # (B, N)

    # <P, C> = sum_i u_i [ (K∘C) v ]_i with P_ij = u_i K_ij v_j
    return (u * (v @ kc.t())).sum(dim=1)


def sinkhorn_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    n_iters: int,
) -> torch.Tensor:
    """Numerically-stable log-domain entropic OT cost (fallback solver).

    Slower than :func:`sinkhorn_distance_kernel` (builds ``(B, N, N)`` tensors)
    but robust at very small ``epsilon`` where the kernel can underflow.

    ``a``, ``b``: ``(B, N)`` non-negative and sum-to-one along ``N``.
    ``cost``: ``(N, N)`` ground cost. Returns ``(B,)`` transport cost ``<P, C>``.
    """
    log_a = torch.log(a.clamp_min(1e-30))
    log_b = torch.log(b.clamp_min(1e-30))

    c = cost.unsqueeze(0)  # (1, N, N)
    f = torch.zeros_like(a)  # potentials for rows i (a)
    g = torch.zeros_like(b)  # potentials for cols j (b)

    for _ in range(n_iters):
        # f_i = eps * (log a_i - logsumexp_j[(-C_ij + g_j) / eps])
        m_f = (-c + g.unsqueeze(1)) / epsilon  # (B, N_i, N_j)
        f = epsilon * (log_a - torch.logsumexp(m_f, dim=2))
        # g_j = eps * (log b_j - logsumexp_i[(-C_ij + f_i) / eps])
        m_g = (-c + f.unsqueeze(2)) / epsilon  # (B, N_i, N_j)
        g = epsilon * (log_b - torch.logsumexp(m_g, dim=1))

    plan = torch.exp((-c + f.unsqueeze(2) + g.unsqueeze(1)) / epsilon)  # (B, N, N)
    return (plan * c).sum(dim=(1, 2))


class SinkhornGridLoss(nn.Module):
    """Per-channel Sinkhorn OT loss between predicted and target occupancy maps.

    Each ``(C, H, W)`` map is average-pooled to ``grid_size`` x ``grid_size``,
    flattened per channel, and turned into a probability distribution. The
    Wasserstein (shape) term ignores absolute counts, so a separate squared
    mass-mismatch term preserves how many agents are present.
    """

    def __init__(
        self,
        grid_size: int = 16,
        epsilon: float = 0.05,
        n_iters: int = 30,
        mass_weight: float = 1.0,
        uniform_floor: float = 1e-4,
        log_domain: bool = False,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.epsilon = float(epsilon)
        self.n_iters = int(n_iters)
        self.mass_weight = float(mass_weight)
        self.uniform_floor = float(uniform_floor)
        self.log_domain = bool(log_domain)
        self.register_buffer("cost", _grid_cost_matrix(self.grid_size), persistent=False)

    def _pool_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample ``(B, C, H, W)`` to ``(B, C, grid_size, grid_size)``.

        Uses area interpolation instead of ``adaptive_avg_pool2d`` so arbitrary
        ``H/W`` and ``grid_size`` work on MPS (Apple GPU requires exact divisibility
        for adaptive pooling).
        """
        return F.interpolate(
            x,
            size=(self.grid_size, self.grid_size),
            mode="area",
        )

    def _to_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``(B, C, H, W)`` >= 0 -> (dist ``(B, C, N)``, mass ``(B, C)``)."""
        b, c, _, _ = x.shape
        x = x.clamp_min(0.0)
        mass = x.sum(dim=(-1, -2))  # (B, C) full-resolution count proxy
        pooled = self._pool_to_grid(x)  # (B, C, g, g)
        flat = pooled.reshape(b, c, -1)  # (B, C, N)
        flat = flat + self.uniform_floor
        dist = flat / flat.sum(dim=-1, keepdim=True)
        return dist, mass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return per-channel loss ``(B, C)`` = OT(shape) + mass_weight * mass MSE."""
        b, c = pred.shape[0], pred.shape[1]
        p_dist, p_mass = self._to_distribution(pred)
        q_dist, q_mass = self._to_distribution(target)

        cost = self.cost.to(dtype=p_dist.dtype)
        a = p_dist.reshape(b * c, -1)
        q = q_dist.reshape(b * c, -1)
        if self.log_domain:
            ot = sinkhorn_distance(a, q, cost, self.epsilon, self.n_iters)
        else:
            ot = sinkhorn_distance_kernel(a, q, cost, self.epsilon, self.n_iters)
            if not torch.isfinite(ot).all():
                # Kernel underflowed (very small epsilon): fall back to log-domain.
                ot = sinkhorn_distance(a, q, cost, self.epsilon, self.n_iters)
        ot = ot.reshape(b, c)

        n_cells = float(pred.shape[-1] * pred.shape[-2])
        mass_term = ((p_mass - q_mass) / n_cells).pow(2)
        return ot + self.mass_weight * mass_term
