"""Market primitives for a forced sale into an exponentially resilient book.

The displacement rule is the discrete-time Obizhaeva–Wang book: selling ``n``
shares walks a block-shaped book of density ``q`` and the displacement then
decays at rate ``rho`` over the next bar. Almgren–Chriss is the special case
in which that displacement is gone before the next trade (``beta = 0``) and a
quadratic inventory penalty is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    """One liquidation window.

    ``shares`` is the shortfall ``X``. ``sigma`` is the daily dollar volatility
    of the unaffected price. ``q`` is the book depth in shares per dollar.
    ``half_life`` is the half-life, in bars, of a displacement the sale itself
    created. ``sigma_y`` is the dollar standard deviation of a refill shock
    over one bar.     ``tau`` is the bar length in the same time unit as the
    half-life. ``n_periods`` is Almgren–Chriss's ``N``: the number of trades,
    and the number of intervals over which inventory is exposed to ``sigma``.
    ``epsilon`` is their per-share fee. ``gamma`` is their permanent impact
    in dollars per share. Both are added to the reserve bill and neither
    enters the choice of schedule.
    """

    shares: float = 100_000.0
    s0: float = 50.0
    sigma: float = 1.0
    q: float = 200_000.0
    half_life: float = 1.0
    sigma_y: float = 0.0
    tau: float = 1.0
    n_periods: int = 10
    epsilon: float = 0.0
    gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.shares <= 0:
            raise ValueError("shares must be positive")
        if self.s0 <= 0:
            raise ValueError("s0 must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.q <= 0:
            raise ValueError("q must be positive")
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        if self.sigma_y < 0:
            raise ValueError("sigma_y must be non-negative")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.n_periods < 1:
            raise ValueError("n_periods must be at least 1")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")

    @property
    def horizon(self) -> float:
        """Calendar length ``T = N tau`` of the window."""
        return self.n_periods * self.tau

    @property
    def rho(self) -> float:
        """Resilience rate. A displacement halves in ``half_life`` bars."""
        return float(np.log(2.0) / self.half_life)

    @property
    def beta(self) -> float:
        """Fraction of a displacement still present one bar later, ``e^{-rho tau}``."""
        return float(np.exp(-self.rho * self.tau))


@dataclass(frozen=True)
class Grid:
    """Discretization of inventory and of the book displacement.

    Inventory is an integer number of lots, from ``0`` to ``n_lots``.
    Displacement is rounded to ``n_y`` nodes from ``0`` to ``X / q``, which is
    the displacement created by selling the entire shortfall at once.
    """

    n_lots: int = 20
    n_y: int = 31

    def __post_init__(self) -> None:
        if self.n_lots < 1:
            raise ValueError("n_lots must be at least 1")
        if self.n_y < 2:
            raise ValueError("n_y must be at least 2")

    def lot_size(self, market: Market) -> float:
        return market.shares / self.n_lots

    def y_nodes(self, market: Market) -> np.ndarray:
        y_max = market.shares / market.q
        return np.linspace(0.0, y_max, self.n_y)


def normal_market(**overrides: float) -> Market:
    """Illustrative liquid book. Half of a displacement is gone in one day."""
    base: dict[str, float] = dict(
        shares=100_000.0,
        s0=50.0,
        sigma=1.0,
        q=200_000.0,
        half_life=1.0,
        sigma_y=0.0,
        tau=1.0,
        n_periods=10,
    )
    base.update(overrides)
    return Market(**base)


def downturn_market(**overrides: float) -> Market:
    """Same shortfall, slower refill, higher volatility, thinner book.

    These are scenario values, not an estimated crisis model. Half-life five
    days, dollar volatility doubled, depth halved.
    """
    base: dict[str, float] = dict(
        shares=100_000.0,
        s0=50.0,
        sigma=2.0,
        q=100_000.0,
        half_life=5.0,
        sigma_y=0.0,
        tau=1.0,
        n_periods=10,
    )
    base.update(overrides)
    return Market(**base)


def reference_lambda(market: Market) -> float:
    """Risk aversion, in 1/dollar, that balances one day-count of holding against impact.

    ``lambda_ref * sigma^2 * X^2 * T = X^2 / (2 q)``. The left side is the
    Almgren–Chriss penalty for carrying the whole shortfall for the whole
    window. The right side is the impact cost of selling it in one print.
    """
    return 1.0 / (2.0 * market.q * market.sigma**2 * market.horizon)


def snap_indices(values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Index of the nearest node. Values outside the grid go to the endpoint."""
    values = np.asarray(values, dtype=float)
    scalar = values.ndim == 0
    values = np.atleast_1d(values)
    if np.any(nodes[1:] < nodes[:-1]):
        raise ValueError("nodes must be sorted")
    clipped = np.clip(values, nodes[0], nodes[-1])
    right = np.searchsorted(nodes, clipped, side="left")
    right = np.clip(right, 1, len(nodes) - 1)
    left = right - 1
    choose_right = np.abs(clipped - nodes[right]) < np.abs(clipped - nodes[left])
    out = np.where(choose_right, right, left).astype(int)
    return int(out[0]) if scalar else out
