"""One random price path for a trade list that has already been chosen.

The trade list does not look at the unaffected price, so the average
implementation shortfall across paths equals the impact cost. A single path
still moves with the random walk. That is the picture on the screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.market import Market


@dataclass
class PricePath:
    unaffected: np.ndarray
    mid: np.ndarray
    execution: np.ndarray
    displacement: np.ndarray
    trades: np.ndarray
    inventory: np.ndarray
    cash: float
    implementation_shortfall: float


def simulate_price_path(
    trades: np.ndarray,
    market: Market,
    rng: np.random.Generator,
    y0: float = 0.0,
) -> PricePath:
    """Execute ``trades`` against the continuous displacement, then a price shock.

    Timing matches Almgren–Chriss equation (5): the sale in a bar is priced off
    the unaffected price at the start of the bar, and the Gaussian shock in
    that bar hits the inventory that remains after the sale.
    """
    trades = np.asarray(trades, dtype=float)
    n = len(trades)
    unaffected = np.zeros(n)
    mid = np.zeros(n)
    execution = np.zeros(n)
    displacement = np.zeros(n)
    inventory = np.zeros(n + 1)
    inventory[0] = float(np.sum(trades))
    price = market.s0
    disp = float(y0)
    cash = 0.0
    for k in range(n):
        sold = float(trades[k])
        unaffected[k] = price
        displacement[k] = disp
        mid[k] = price - disp
        execution[k] = price - disp - sold / (2.0 * market.q)
        cash += sold * execution[k]
        inventory[k + 1] = inventory[k] - sold
        price = price + market.sigma * np.sqrt(market.tau) * float(rng.normal())
        disp = market.beta * (disp + sold / market.q)
        if market.sigma_y > 0.0:
            disp = disp + market.sigma_y * float(rng.normal())
    shortfall = inventory[0] * market.s0 - cash
    return PricePath(
        unaffected=unaffected,
        mid=mid,
        execution=execution,
        displacement=displacement,
        trades=trades.copy(),
        inventory=inventory,
        cash=cash,
        implementation_shortfall=float(shortfall),
    )
