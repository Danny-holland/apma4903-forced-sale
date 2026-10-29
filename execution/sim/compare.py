"""Score the flat schedule, Almgren–Chriss, and the dynamic program on one market."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sim.control import (
    Rollout,
    Solution,
    almgren_chriss_holdings,
    allocate_lots,
    flat_holdings,
    holdings_to_lot_policy,
    kappa_bps,
    lot_trades_to_policy,
    obizhaeva_wang_trades,
    price_variance,
    reserve_bill,
    rollout_policy,
    solve_dp,
)
from sim.market import Grid, Market


@dataclass
class ScheduleScore:
    name: str
    impact: float
    risk: float
    objective: float
    kappa_bps: float
    std_bps: float
    bill: float
    trades: np.ndarray
    inventory: np.ndarray
    displacement: np.ndarray

    def as_row(self) -> dict[str, float | str]:
        return {
            "schedule": self.name,
            "kappa_bps": self.kappa_bps,
            "std_bps": self.std_bps,
            "impact_dollars": self.impact,
            "reserve_dollars": self.bill,
            "objective": self.objective,
        }


def score_rollout(rollout: Rollout, market: Market) -> ScheduleScore:
    std = np.sqrt(price_variance(rollout.inventory, market))
    bill = reserve_bill(rollout.impact, market)
    return ScheduleScore(
        name=rollout.name,
        impact=rollout.impact,
        risk=rollout.risk,
        objective=rollout.objective,
        kappa_bps=kappa_bps(bill, market),
        std_bps=1.0e4 * std / (market.shares * market.s0),
        bill=bill,
        trades=rollout.trades,
        inventory=rollout.inventory,
        displacement=rollout.displacement,
    )


def compare_schedules(
    market: Market,
    grid: Grid,
    lam: float,
    y0: float = 0.0,
    extra_policies: list[tuple[str, np.ndarray]] | None = None,
    include_obizhaeva_wang: bool = False,
) -> tuple[Solution, list[ScheduleScore]]:
    """Solve the grid once and score each schedule in that same grid world.

    Almgren–Chriss is computed from the closed form that assumes the book
    refills inside the bar, then its trades are replayed in the true book.
    """
    solution = solve_dp(market, grid, lam)
    named: list[tuple[str, np.ndarray]] = [
        ("Equal trades", holdings_to_lot_policy(flat_holdings(market), solution)),
    ]
    # At zero risk aversion their closed form is equal trades, so the two
    # rows would be the same sale.
    if lam > 0.0:
        named.append(
            (
                "Almgren-Chriss",
                holdings_to_lot_policy(almgren_chriss_holdings(market, lam), solution),
            )
        )
    named.append(("Optimal", solution.policy))
    if include_obizhaeva_wang:
        lots = allocate_lots(obizhaeva_wang_trades(market), grid.n_lots)
        named.append(("Obizhaeva-Wang", lot_trades_to_policy(lots, solution)))
    if extra_policies:
        named.extend(extra_policies)
    scores = [
        score_rollout(rollout_policy(solution, name, y0, policy), market)
        for name, policy in named
    ]
    return solution, scores


def format_table(scores: list[ScheduleScore]) -> str:
    header = (
        f"{'schedule':<18}{'kappa (bp)':>12}{'std (bp)':>12}"
        f"{'walk $':>14}{'C $':>14}{'J':>14}"
    )
    lines = [header, "-" * len(header)]
    for row in scores:
        lines.append(
            f"{row.name:<18}{row.kappa_bps:12.2f}{row.std_bps:12.2f}"
            f"{row.impact:14.1f}{row.bill:14.1f}{row.objective:14.1f}"
        )
    return "\n".join(lines)


def scores_to_dicts(scores: list[ScheduleScore]) -> list[dict[str, float | str]]:
    return [row.as_row() | {"trades": row.trades.tolist()} for row in scores]


def market_brief(market: Market, lam: float) -> dict[str, float]:
    data = asdict(market)
    data["beta"] = market.beta
    data["rho"] = market.rho
    data["lambda"] = lam
    return data
