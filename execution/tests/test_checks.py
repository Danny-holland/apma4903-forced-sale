"""Closed-form checks for the liquidation dynamic program."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.control import (  # noqa: E402
    allison_reserve_cost,
    almgren_chriss_holdings,
    bellman_residual,
    holdings_to_lot_policy,
    kappa_bps,
    obizhaeva_wang_trades,
    permanent_and_fee,
    refill_temporary_cost,
    rollout_policy,
    solve_dp,
)
from sim.market import Grid, Market, normal_market, reference_lambda, snap_indices  # noqa: E402
from sim.paths import simulate_price_path  # noqa: E402
from sim.qlearn import train_q  # noqa: E402


class SnapTests(unittest.TestCase):
    def test_endpoints_and_midpoints(self) -> None:
        nodes = np.array([0.0, 1.0, 2.0])
        self.assertEqual(snap_indices(np.array(-1.0), nodes), 0)
        self.assertEqual(snap_indices(np.array(4.0), nodes), 2)
        self.assertEqual(snap_indices(np.array(0.4), nodes), 0)
        self.assertEqual(snap_indices(np.array(0.6), nodes), 1)
        got = snap_indices(np.array([0.0, 1.0, 2.0]), nodes)
        np.testing.assert_array_equal(got, [0, 1, 2])


class DynamicProgramTests(unittest.TestCase):
    def test_bellman_residual_is_zero(self) -> None:
        market = normal_market(n_periods=6)
        solution = solve_dp(market, Grid(n_lots=8, n_y=11), reference_lambda(market))
        self.assertLess(bellman_residual(solution), 1e-8)

    def test_one_period_is_the_block_cost(self) -> None:
        market = normal_market(n_periods=1, shares=1_000.0, q=2_000.0)
        lam = reference_lambda(market)
        solution = solve_dp(market, Grid(n_lots=4, n_y=5), lam)
        # Selling 1000 into depth 2000 costs 1000^2 / (2*2000) = 250.
        self.assertAlmostEqual(float(solution.value[0, 4, 0]), 250.0, places=6)

    def test_risk_neutral_refill_matches_allisons_formula(self) -> None:
        market = normal_market(n_periods=10, half_life=1e-9, epsilon=0.01, gamma=1e-6)
        solution = solve_dp(market, Grid(n_lots=20, n_y=5), lam=0.0)
        rolled = rollout_policy(solution, "optimal")
        self.assertAlmostEqual(rolled.impact, refill_temporary_cost(market), places=6)
        self.assertAlmostEqual(
            rolled.impact + permanent_and_fee(market),
            allison_reserve_cost(market),
            places=6,
        )
        # Q = X S0. Her formula is linear in the fee and quadratic in the shortfall.
        small = normal_market(shares=20_000.0, n_periods=8, half_life=1e-9)
        large = normal_market(shares=40_000.0, n_periods=8, half_life=1e-9)
        self.assertAlmostEqual(
            kappa_bps(allison_reserve_cost(large), large),
            2.0 * kappa_bps(allison_reserve_cost(small), small),
            places=6,
        )

    def test_fee_and_permanent_impact_do_not_change_the_trades(self) -> None:
        grid = Grid(n_lots=8, n_y=9)
        plain = normal_market(n_periods=6, half_life=2.0)
        priced = normal_market(n_periods=6, half_life=2.0, epsilon=0.02, gamma=2e-6)
        without_fee = rollout_policy(solve_dp(plain, grid, 0.0), "plain")
        with_fee = rollout_policy(solve_dp(priced, grid, 0.0), "priced")
        np.testing.assert_allclose(without_fee.trades, with_fee.trades)
        self.assertGreater(permanent_and_fee(priced), 0.0)

    def test_risk_neutral_full_resilience_splits_evenly(self) -> None:
        market = normal_market(n_periods=4, half_life=1e-9)
        self.assertLess(market.beta, 1e-6)
        solution = solve_dp(market, Grid(n_lots=8, n_y=5), lam=0.0)
        rolled = rollout_policy(solution, "optimal")
        # Eight lots over four bars, two lots per bar.
        np.testing.assert_allclose(rolled.trades, np.full(4, market.shares / 4), rtol=0, atol=1e-8)

    def test_matches_almgren_chriss_when_the_book_refills(self) -> None:
        market = normal_market(n_periods=10, half_life=1e-9)
        lam = reference_lambda(normal_market())
        grid = Grid(n_lots=40, n_y=5)
        solution = solve_dp(market, grid, lam)
        self.assertLess(bellman_residual(solution), 1e-6)
        optimal = rollout_policy(solution, "optimal")
        ac = rollout_policy(
            solution,
            "ac",
            policy=holdings_to_lot_policy(almgren_chriss_holdings(market, lam), solution),
        )
        lot = grid.lot_size(market)
        np.testing.assert_allclose(optimal.inventory, ac.inventory, rtol=0, atol=lot + 1e-6)
        # The grid optimum cannot be worse than the rounded closed form.
        self.assertLessEqual(optimal.objective, ac.objective + 1e-6)

    def test_no_resilience_makes_the_impact_cost_path_independent(self) -> None:
        # rho = 0 is beta = 1. Obizhaeva–Wang: with no refill, every fully
        # liquidating schedule pays X^2 / (2q) in impact.
        market = Market(
            shares=10_000.0,
            s0=50.0,
            sigma=1.0,
            q=20_000.0,
            half_life=1e12,
            sigma_y=0.0,
            tau=1.0,
            n_periods=5,
        )
        self.assertAlmostEqual(market.beta, 1.0, places=6)
        solution = solve_dp(market, Grid(n_lots=10, n_y=21), lam=0.0)
        optimal = rollout_policy(solution, "optimal")
        flat = rollout_policy(
            solution,
            "flat",
            policy=holdings_to_lot_policy(
                np.linspace(market.shares, 0.0, market.n_periods + 1), solution
            ),
        )
        block = market.shares**2 / (2.0 * market.q)
        self.assertAlmostEqual(optimal.impact, block, places=4)
        self.assertAlmostEqual(flat.impact, block, places=4)

    def test_obizhaeva_wang_bins_sum_to_the_order(self) -> None:
        market = normal_market(n_periods=20, half_life=10.0)
        trades = obizhaeva_wang_trades(market)
        self.assertAlmostEqual(float(trades.sum()), market.shares, places=6)
        self.assertGreater(trades[0], trades[1])
        self.assertGreater(trades[-1], trades[1])

    def test_risk_neutral_grid_puts_size_on_the_endpoints(self) -> None:
        market = normal_market(n_periods=15, half_life=8.0, shares=15_000.0, q=30_000.0)
        solution = solve_dp(market, Grid(n_lots=30, n_y=41), lam=0.0)
        rolled = rollout_policy(solution, "optimal")
        middle = float(np.mean(rolled.trades[1:-1]))
        self.assertGreater(rolled.trades[0], 2.0 * middle)
        self.assertGreater(rolled.trades[-1], 2.0 * middle)

    def test_a_displaced_book_reduces_the_opening_trade(self) -> None:
        market = normal_market(n_periods=8, half_life=4.0)
        lam = 0.25 * reference_lambda(market)
        solution = solve_dp(market, Grid(n_lots=16, n_y=31), lam)
        calm = rollout_policy(solution, "calm", y0=0.0)
        hot = rollout_policy(solution, "hot", y0=0.35 * market.shares / market.q)
        self.assertLess(hot.trades[0], calm.trades[0])

    def test_implementation_shortfall_averages_to_the_impact_cost(self) -> None:
        market = normal_market(n_periods=6, shares=20_000.0, q=40_000.0, sigma=1.0)
        solution = solve_dp(market, Grid(n_lots=10, n_y=11), reference_lambda(market))
        rolled = rollout_policy(solution, "optimal")
        rng = np.random.default_rng(7)
        draws = [
            simulate_price_path(rolled.trades, market, rng).implementation_shortfall
            for _ in range(4_000)
        ]
        mean = float(np.mean(draws))
        stderr = float(np.std(draws, ddof=1) / np.sqrt(len(draws)))
        self.assertLess(abs(mean - rolled.impact), 4.0 * stderr)


class QLearningTests(unittest.TestCase):
    def test_backward_sweep_matches_the_dynamic_program(self) -> None:
        market = normal_market(n_periods=5, half_life=2.0, shares=8_000.0, q=16_000.0)
        lam = reference_lambda(market)
        solution = solve_dp(market, Grid(n_lots=6, n_y=7), lam)
        result = train_q(solution, n_samples=1, seed=3)
        gap = abs(result.greedy_objective[-1] - result.optimal_objective)
        self.assertLess(gap, 1e-6)
        self.assertGreater(result.greedy_objective[0], result.optimal_objective + 1.0)
        self.assertEqual(result.visit_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
