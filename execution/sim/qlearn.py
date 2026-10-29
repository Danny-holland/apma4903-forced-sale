"""Finite-horizon Q-learning by a backward sweep of sampled transitions.

Watkins's update estimates the action value from the observed stage cost and
the next state's value. On a finite horizon the last trade has no continuation,
so its action value is the observed liquidation cost. Each earlier bar is then
sampled until its target, which depends only on the next bar, has been
averaged. When the refill shock is zero, one sample of each state-action is
the conditional expectation, and the sweep reproduces the dynamic program.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.control import Solution, rollout_policy
from sim.market import snap_indices


@dataclass
class QLearningResult:
    """Greedy policy read off the sampled action values."""

    policy: np.ndarray
    q_values: np.ndarray
    visits: np.ndarray
    updates: np.ndarray
    greedy_objective: np.ndarray
    optimal_objective: float
    visit_fraction: float
    samples: int


def train_q(
    solution: Solution,
    n_samples: int = 1,
    seed: int = 0,
    y0: float = 0.0,
) -> QLearningResult:
    """Fill the action-value table from the environment, bar by bar, from the end.

    ``n_samples`` is the number of independent transitions averaged at each
    state-action. Use ``1`` when ``sigma_y`` is zero. The dynamic program's
    value function is consulted only to record the gap in the learning curve;
    it never enters the update.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    market = solution.market
    grid = solution.grid
    n = market.n_periods
    lots = grid.n_lots
    n_y = grid.n_y
    lot = solution.lot
    y_nodes = solution.y_nodes
    q_depth = market.q
    beta = market.beta
    sigma_y = market.sigma_y
    risk_per_share2 = solution.lam * market.sigma**2 * market.tau
    rng = np.random.default_rng(seed)

    def transition(ix: int, iy: int, sold: int) -> tuple[float, int, int]:
        """One environment step. Returns cost, next inventory index, next displacement index."""
        y = float(y_nodes[iy])
        shares = sold * lot
        remaining_ix = ix - sold
        cost = shares * (y + shares / (2.0 * q_depth))
        cost += risk_per_share2 * (remaining_ix * lot) ** 2
        shock = 0.0 if sigma_y == 0.0 else float(rng.normal(0.0, sigma_y))
        iy_next = int(snap_indices(np.array(beta * (y + shares / q_depth) + shock), y_nodes))
        return cost, remaining_ix, iy_next

    q_values = np.full((n, lots + 1, n_y, lots + 1), np.inf)
    visits = np.zeros((n, lots + 1, n_y, lots + 1), dtype=np.int32)

    for ix in range(lots + 1):
        for iy in range(n_y):
            # Average the observed liquidation cost. There is no continuation.
            total = 0.0
            for _ in range(n_samples):
                cost, _, _ = transition(ix, iy, ix)
                total += cost
            q_values[n - 1, ix, iy, ix] = total / n_samples
            visits[n - 1, ix, iy, ix] = n_samples

    start_iy = int(snap_indices(np.array(y0), y_nodes))
    optimal_objective = float(solution.value[0, lots, start_iy])
    update_marks: list[int] = []
    greedy_costs: list[float] = []
    updates = 0

    def record() -> None:
        policy = np.argmin(q_values, axis=-1).astype(int)
        rolled = rollout_policy(solution, "Q-learning", y0, policy)
        update_marks.append(updates)
        greedy_costs.append(rolled.objective)

    record()
    for k in range(n - 2, -1, -1):
        for ix in range(lots + 1):
            for iy in range(n_y):
                for sold in range(ix + 1):
                    total = 0.0
                    for _ in range(n_samples):
                        cost, remaining_ix, iy_next = transition(ix, iy, sold)
                        future = float(np.min(q_values[k + 1, remaining_ix, iy_next, : remaining_ix + 1]))
                        total += cost + future
                    q_values[k, ix, iy, sold] = total / n_samples
                    visits[k, ix, iy, sold] = n_samples
                    updates += n_samples
        record()

    policy = np.argmin(q_values, axis=-1).astype(int)
    # At the deadline the only admissible action is to sell what remains.
    legal = 0
    seen = 0
    for ix in range(lots + 1):
        legal += (ix + 1) * (n - 1) * n_y + n_y
        seen += int(np.sum(visits[: n - 1, ix, :, : ix + 1] > 0))
        seen += int(np.sum(visits[n - 1, ix, :, ix] > 0))
    return QLearningResult(
        policy=policy,
        q_values=q_values,
        visits=visits,
        updates=np.asarray(update_marks, dtype=int),
        greedy_objective=np.asarray(greedy_costs, dtype=float),
        optimal_objective=optimal_objective,
        visit_fraction=seen / legal if legal else 1.0,
        samples=n_samples,
    )
