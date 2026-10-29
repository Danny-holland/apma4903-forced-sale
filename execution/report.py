"""Print the cost tables cited in the research note.

The reserve number is Allison's risk-neutral cost. Risk aversion is zero.
The fee and the permanent impact are zero in the tables, because the slides
leave both symbolic; the program adds them when they are set on the market.

Run from the repository root:

    python execution/report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim.compare import compare_schedules, format_table, score_rollout  # noqa: E402
from sim.control import (  # noqa: E402
    allison_reserve_cost,
    kappa_bps,
    permanent_and_fee,
    rollout_policy,
)
from sim.market import Grid, Market, downturn_market, normal_market, reference_lambda  # noqa: E402
from sim.qlearn import train_q  # noqa: E402


FINE = Grid(n_lots=40, n_y=41)
COARSE = Grid(n_lots=20, n_y=31)
# Slide 14. Not the reference value that equates a variance penalty to impact.
LAM = 0.0


def _row(scores, name: str):
    return next(row for row in scores if row.name == name)


def _quote(market: Market) -> str:
    cost = allison_reserve_cost(market)
    return (
        f"  her refill formula C = {cost:.1f} dollars, "
        f"kappa = {kappa_bps(cost, market):.2f} bp "
        f"({cost / (market.shares * market.s0):.6f} per dollar of shortfall)"
    )


def main() -> None:
    print("risk aversion lambda = 0 (Allison, risk-neutral equal trades)")
    print("fee epsilon = 0, permanent impact gamma = 0 in the tables below")
    print(f"grid for the cost table: {FINE.n_lots} lots, {FINE.n_y} displacement nodes")
    print(
        "kappa = C / (X S0), with Q = X S0. "
        "C = expected walk + epsilon X + (gamma/2) X^2."
    )
    print()

    worlds = [
        ("refill within the day", normal_market(half_life=1e-6)),
        ("half-life 0.5 day", normal_market(half_life=0.5)),
        ("half-life 1 day", normal_market(half_life=1.0)),
        ("half-life 2 days", normal_market(half_life=2.0)),
        ("half-life 5 days", normal_market(half_life=5.0)),
        ("half-life 10 days", normal_market(half_life=10.0)),
        ("downturn book, refill assumed", downturn_market(half_life=1e-6)),
        ("downturn book, half-life 5 days", downturn_market()),
    ]
    for label, market in worlds:
        solution, scores = compare_schedules(market, FINE, LAM)
        learned = train_q(solution, n_samples=1, seed=0)
        scores.append(
            score_rollout(
                rollout_policy(solution, "Q-learning", 0.0, learned.policy),
                market,
            )
        )
        print(label)
        print(
            f"  beta = {market.beta:.4f}, sigma = {market.sigma:g}, "
            f"q = {market.q:.0f}"
        )
        print(_quote(market))
        print(format_table(scores))
        gap = learned.greedy_objective[-1] - learned.optimal_objective
        print(f"  J gap (Q-learning minus dynamic program) = {gap:.6f}")
        print()

    print("kappa against the size of the shortfall, risk aversion 0, fee 0, gamma 0")
    print("kappa is basis points. It would be constant if the sale cost were linear in Q.")
    sizes = (25_000.0, 50_000.0, 100_000.0, 200_000.0)
    books = (
        ("normal, half-life 1 day", dict(half_life=1.0)),
        ("downturn book, half-life 5 days", dict(sigma=2.0, q=100_000.0, half_life=5.0)),
    )
    for book_name, overrides in books:
        print(book_name)
        print(
            f"  {'Q $':>12}{'her kappa':>12}{'equal':>12}{'optimal':>12}{'C optimal $':>14}"
        )
        for shares in sizes:
            market = normal_market(shares=shares, **overrides)
            _solution, scores = compare_schedules(market, FINE, LAM)
            her = kappa_bps(allison_reserve_cost(market), market)
            equal = _row(scores, "Equal trades")
            optimal = _row(scores, "Optimal")
            print(
                f"  {shares * market.s0:12.0f}{her:12.2f}{equal.kappa_bps:12.2f}"
                f"{optimal.kappa_bps:12.2f}{optimal.bill:14.1f}"
            )
        print()

    example = normal_market(epsilon=0.01, gamma=1e-6)
    fee = example.epsilon * example.shares
    permanent = 0.5 * example.gamma * example.shares**2
    print(
        "additive example, not a calibration: fee 1 cent per share, "
        "permanent move of the whole block 10 cents"
    )
    print(
        f"  epsilon X = {fee:.1f} dollars ({kappa_bps(fee, example):.2f} bp), "
        f"(gamma/2) X^2 = {permanent:.1f} dollars "
        f"({kappa_bps(permanent, example):.2f} bp)"
    )
    print(
        "  both are added to every schedule. "
        f"Together {kappa_bps(permanent_and_fee(example), example):.2f} bp "
        "on top of the temporary cost."
    )
    print()

    lam_ref = reference_lambda(normal_market())
    print(
        f"slide 13, for contrast: reference risk aversion {lam_ref:.6e} per dollar, "
        "normal book, refill within the day"
    )
    solution, scores = compare_schedules(normal_market(half_life=1e-6), FINE, lam_ref)
    print(format_table(scores))
    print()

    print(
        f"Q-learning check on {COARSE.n_lots} lots, "
        f"{COARSE.n_y} nodes, half-life 1 day, lambda = 0"
    )
    solution, _scores = compare_schedules(normal_market(), COARSE, LAM)
    result = train_q(solution, n_samples=1, seed=0)
    print("updates", result.updates.tolist())
    print("greedy J", [round(float(v), 1) for v in result.greedy_objective])
    print("optimal J", round(result.optimal_objective, 1))
    print("visit fraction", result.visit_fraction)

    print()
    print(
        f"Q-learning check, downturn book, {COARSE.n_lots} lots, "
        f"{COARSE.n_y} nodes, lambda = 0"
    )
    solution, _scores = compare_schedules(downturn_market(), COARSE, LAM)
    result = train_q(solution, n_samples=1, seed=0)
    print("greedy J", [round(float(v), 1) for v in result.greedy_objective])
    print("optimal J", round(result.optimal_objective, 1))


if __name__ == "__main__":
    main()
