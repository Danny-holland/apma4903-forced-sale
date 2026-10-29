"""Exact dynamic program, and the Almgren–Chriss and Obizhaeva–Wang schedules.

The dynamic program is the definition of the optimal cost on the grid. The
other two schedules are the closed forms the grid solution has to recover:
Almgren–Chriss when ``beta = 0``, and the Obizhaeva–Wang blocks-and-rate
schedule when risk aversion is zero and the grid is fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss

from sim.market import Grid, Market, snap_indices


@dataclass
class Solution:
    """Result of backward induction on the grid MDP."""

    market: Market
    grid: Grid
    lam: float
    y_nodes: np.ndarray
    lot: float
    value: np.ndarray  # (N, n_lots+1, n_y), cost-to-go before the trade
    policy: np.ndarray  # (N, n_lots+1, n_y), lots sold, integer


@dataclass
class Rollout:
    """One deterministic sale on the grid, from a flat or displaced book."""

    name: str
    trades: np.ndarray  # shares, length N
    inventory: np.ndarray  # shares, length N+1, after each trade; inventory[0] = X
    displacement: np.ndarray  # dollars, length N, before each trade
    impact: float
    risk: float

    @property
    def objective(self) -> float:
        """``J = E[IS] + lambda * (price variance of IS)`` along this inventory path."""
        return self.impact + self.risk


def _hermite_standard(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for ``E[f(Z)]``, ``Z`` standard normal."""
    nodes, weights = hermgauss(n_nodes)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def solve_dp(market: Market, grid: Grid, lam: float, n_hermite: int = 9) -> Solution:
    """Backward induction for the finite-horizon grid MDP.

    Stage cost of selling ``n`` shares in state ``(x, y)`` is

        ``n (y + n / (2 q)) + lambda sigma^2 tau (x - n)^2``.

    The next displacement is ``beta (y + n / q)`` plus a normal refill shock,
    rounded to the displacement grid. The last trade liquidates whatever is left.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    n = market.n_periods
    lots = grid.n_lots
    y_nodes = grid.y_nodes(market)
    n_y = grid.n_y
    lot = grid.lot_size(market)
    beta = market.beta
    q = market.q
    risk_per_share2 = lam * market.sigma**2 * market.tau

    if market.sigma_y == 0.0:
        shock_z = np.array([0.0])
        shock_w = np.array([1.0])
    else:
        shock_z, shock_w = _hermite_standard(n_hermite)

    value = np.full((n, lots + 1, n_y), np.inf)
    policy = np.zeros((n, lots + 1, n_y), dtype=int)

    for ix in range(lots + 1):
        shares = ix * lot
        impact = shares * (y_nodes + shares / (2.0 * q))
        value[n - 1, ix, :] = impact
        policy[n - 1, ix, :] = ix

    for k in range(n - 2, -1, -1):
        for ix in range(lots + 1):
            best = np.full(n_y, np.inf)
            best_n = np.zeros(n_y, dtype=int)
            for sold in range(ix + 1):
                shares = sold * lot
                remaining = (ix - sold) * lot
                impact = shares * (y_nodes + shares / (2.0 * q))
                risk = risk_per_share2 * remaining**2
                y_mean = beta * (y_nodes + shares / q)
                expected = np.zeros(n_y)
                for z, w in zip(shock_z, shock_w):
                    nxt = snap_indices(y_mean + market.sigma_y * z, y_nodes)
                    expected += w * value[k + 1, ix - sold, nxt]
                stage = impact + risk + expected
                better = stage < best
                best = np.where(better, stage, best)
                best_n = np.where(better, sold, best_n)
            value[k, ix] = best
            policy[k, ix] = best_n

    return Solution(
        market=market,
        grid=grid,
        lam=lam,
        y_nodes=y_nodes,
        lot=lot,
        value=value,
        policy=policy,
    )


def rollout_policy(
    solution: Solution,
    name: str,
    y0: float = 0.0,
    policy: np.ndarray | None = None,
) -> Rollout:
    """Apply a lot-valued policy on the deterministic grid (``sigma_y`` is ignored).

    ``policy`` defaults to the optimal one. It has shape ``(N, n_lots+1, n_y)``
    and stores lots sold.
    """
    market = solution.market
    rule = solution.policy if policy is None else policy
    n = market.n_periods
    y_nodes = solution.y_nodes
    lot = solution.lot
    q = market.q
    beta = market.beta
    risk_per_share2 = solution.lam * market.sigma**2 * market.tau

    trades = np.zeros(n)
    inventory = np.zeros(n + 1)
    displacement = np.zeros(n)
    inventory[0] = market.shares
    ix = solution.grid.n_lots
    iy = int(snap_indices(np.array(y0), y_nodes))
    impact = 0.0
    risk = 0.0
    for k in range(n):
        y = float(y_nodes[iy])
        displacement[k] = y
        sold = int(rule[k, ix, iy])
        sold = min(max(sold, 0), ix)
        if k == n - 1:
            sold = ix
        shares = sold * lot
        trades[k] = shares
        impact += shares * (y + shares / (2.0 * q))
        ix -= sold
        inventory[k + 1] = ix * lot
        risk += risk_per_share2 * inventory[k + 1] ** 2
        y_next = beta * (y + shares / q)
        iy = int(snap_indices(np.array(y_next), y_nodes))
    return Rollout(name, trades, inventory, displacement, impact, risk)


def price_variance(inventory: np.ndarray, market: Market) -> float:
    """``sigma^2 tau sum_{k=1}^{N} x_k^2``. ``inventory[0]`` is the pre-trade position."""
    return float(market.sigma**2 * market.tau * np.sum(inventory[1:] ** 2))


def kappa_bps(cost: float, market: Market) -> float:
    """A dollar cost as basis points of arrival value ``X S_0``.

    The reserve divides by the dollar shortfall. Under the slide's
    approximation ``Q = X S_0``, that ratio is the same number.
    """
    return 1.0e4 * cost / (market.shares * market.s0)


def permanent_and_fee(market: Market) -> float:
    """``ε X + (γ/2) X²``, the schedule-independent part of their expected cost.

    This is the continuous-time accounting on the reserve slide: a completed
    one-way sale pays the fee once and pays permanent impact ``γ X²/2``
    whatever the path. The daily-grid shift of temporary impact by ``γ τ/2``
    is the term their continuous-time limit drops.
    """
    return market.epsilon * market.shares + 0.5 * market.gamma * market.shares**2


def refill_temporary_cost(market: Market) -> float:
    """Risk-neutral temporary cost when the book refills before the next trade.

    Their temporary coefficient satisfies ``η = τ/(2q)`` and the risk-neutral
    schedule sells ``X/N`` each bar, so ``(η/T) X² = X²/(2 q N)``.
    """
    return market.shares**2 / (2.0 * market.q * market.n_periods)


def allison_reserve_cost(market: Market) -> float:
    """Their formula: ``ε X + (γ/2 + η/T) X²`` with ``η = τ/(2q)``."""
    return refill_temporary_cost(market) + permanent_and_fee(market)


def reserve_bill(walk: float, market: Market) -> float:
    """Expected implementation shortfall handed to the reserve.

    ``walk`` is the expected cost of walking the book. The fee and the
    permanent impact are then added exactly as in their continuous-time
    formula. Linear permanent impact does not change which schedule is best.
    """
    return walk + permanent_and_fee(market)


def almgren_chriss_holdings(market: Market, lam: float) -> np.ndarray:
    """Holdings ``x_0, ..., x_N`` from Almgren–Chriss with ``gamma = 0`` and ``eta = tau / (2 q)``.

    This is the optimal trajectory of the ``beta = 0`` problem. It does not
    look at residual displacement. ``cosh(kappa tau) = 1 + lambda sigma^2 tau q``.
    """
    n = market.n_periods
    j = np.arange(n + 1, dtype=float)
    if lam == 0.0 or market.sigma == 0.0:
        return market.shares * (1.0 - j / n)
    arg = 1.0 + lam * market.sigma**2 * market.tau * market.q
    theta = float(np.arccosh(arg))
    return market.shares * _sinh_ratio((n - j) * theta, n * theta)


def _sinh_ratio(numer: np.ndarray, denom: float) -> np.ndarray:
    """``sinh(numer) / sinh(denom)`` without overflow. ``denom > 0``."""
    numer = np.asarray(numer, dtype=float)
    if denom < 1e-8:
        return numer / denom
    return np.exp(numer - denom) * np.expm1(-2.0 * numer) / np.expm1(-2.0 * denom)


def allocate_lots(trades: np.ndarray, n_lots: int) -> np.ndarray:
    """Turn a positive share-schedule into integer lots that sum to ``n_lots``."""
    trades = np.asarray(trades, dtype=float)
    if trades.ndim != 1:
        raise ValueError("trades must be a one-dimensional schedule")
    if np.any(trades < 0) or float(trades.sum()) <= 0:
        raise ValueError("trades must be non-negative and not all zero")
    raw = trades / trades.sum() * n_lots
    lots = np.floor(raw).astype(int)
    leftover = int(n_lots - int(lots.sum()))
    order = np.argsort(-(raw - lots))
    for i in range(leftover):
        lots[int(order[i])] += 1
    return lots


def lot_trades_to_policy(trades_lots: np.ndarray, solution: Solution) -> np.ndarray:
    """Open-loop lot list, copied into every displacement node.

    The last trade is replaced by "sell whatever is left", which agrees with
    the list whenever the earlier trades have been followed exactly.
    """
    n = solution.market.n_periods
    trades_lots = np.asarray(trades_lots)
    if len(trades_lots) != n:
        raise ValueError("trade list length must equal n_periods")
    policy = np.zeros_like(solution.policy)
    for k in range(n - 1):
        policy[k, :, :] = int(trades_lots[k])
    policy[n - 1, :, :] = np.arange(solution.grid.n_lots + 1)[:, None]
    return policy


def holdings_to_lot_policy(holdings: np.ndarray, solution: Solution) -> np.ndarray:
    """Open-loop holdings, rounded to a monotone lot path, as a policy that ignores ``y``."""
    grid = solution.grid
    lot = solution.lot
    n = solution.market.n_periods
    lots = np.rint(np.asarray(holdings, dtype=float) / lot).astype(int)
    lots[0] = grid.n_lots
    lots[-1] = 0
    for i in range(1, n + 1):
        lots[i] = int(min(max(int(lots[i]), 0), int(lots[i - 1])))
    lots[-1] = 0
    return lot_trades_to_policy(lots[:-1] - lots[1:], solution)


def flat_holdings(market: Market) -> np.ndarray:
    """Equal shares each bar. This is Almgren–Chriss at ``lambda = 0``."""
    n = market.n_periods
    j = np.arange(n + 1, dtype=float)
    return market.shares * (1.0 - j / n)


def obizhaeva_wang_trades(market: Market) -> np.ndarray:
    """Risk-neutral continuous-time schedule, binned onto the ``N`` trade dates.

    Proposition 3 of Obizhaeva and Wang (2013): a block ``X / (rho T + 2)`` at
    the start, the same block at the deadline, and a constant rate
    ``rho X / (rho T + 2)`` in between. Each bin receives the rate integrated
    over one bar; the endpoint bins also receive the blocks.
    """
    block = market.shares / (market.rho * market.horizon + 2.0)
    rate = market.rho * market.shares / (market.rho * market.horizon + 2.0)
    trades = np.full(market.n_periods, rate * market.tau)
    trades[0] += block
    trades[-1] += block
    return trades


def bellman_residual(solution: Solution) -> float:
    """Max absolute one-step residual of ``value`` on states before the last trade."""
    market = solution.market
    y_nodes = solution.y_nodes
    lot = solution.lot
    q = market.q
    beta = market.beta
    risk_per_share2 = solution.lam * market.sigma**2 * market.tau
    worst = 0.0
    n = market.n_periods
    if market.sigma_y != 0.0:
        raise ValueError("residual check is written for sigma_y = 0")
    for k in range(n - 1):
        for ix in range(solution.grid.n_lots + 1):
            for iy in range(solution.grid.n_y):
                sold = int(solution.policy[k, ix, iy])
                shares = sold * lot
                y = float(y_nodes[iy])
                remaining_ix = ix - sold
                impact = shares * (y + shares / (2.0 * q))
                risk = risk_per_share2 * (remaining_ix * lot) ** 2
                iy_next = int(snap_indices(np.array(beta * (y + shares / q)), y_nodes))
                backup = impact + risk + float(solution.value[k + 1, remaining_ix, iy_next])
                worst = max(worst, abs(backup - float(solution.value[k, ix, iy])))
    return worst
