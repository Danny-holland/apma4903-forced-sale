"""Screen for the forced-sale calculation.

Run from the repository root:

    streamlit run execution/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.compare import ScheduleScore, score_rollout
from sim.control import (
    allison_reserve_cost,
    almgren_chriss_holdings,
    flat_holdings,
    holdings_to_lot_policy,
    kappa_bps,
    rollout_policy,
    solve_dp,
)
from sim.market import Grid, Market, downturn_market, normal_market, reference_lambda
from sim.paths import simulate_price_path
from sim.qlearn import train_q

NAVY = "#1B3A4B"
RUST = "#8C5A2E"
MUTED = "#5C6670"
GREEN = "#2F6F4E"
COLORS = {
    "Equal trades": MUTED,
    "Almgren-Chriss": RUST,
    "Optimal": NAVY,
    "Q-learning": GREEN,
}


@st.cache_data(show_spinner=False)
def cached_solve(
    shares: float,
    s0: float,
    sigma: float,
    q: float,
    half_life: float,
    sigma_y: float,
    n_periods: int,
    n_lots: int,
    n_y: int,
    lam: float,
    epsilon: float,
    gamma: float,
):
    market = Market(
        shares=shares,
        s0=s0,
        sigma=sigma,
        q=q,
        half_life=half_life,
        sigma_y=sigma_y,
        tau=1.0,
        n_periods=n_periods,
        epsilon=epsilon,
        gamma=gamma,
    )
    return solve_dp(market, Grid(n_lots=n_lots, n_y=n_y), lam)


def preference_lambda(n_periods: int, multiple: float) -> float:
    """Risk aversion in 1/dollar. ``multiple`` 1 is the normal-market reference."""
    return multiple * reference_lambda(normal_market(n_periods=n_periods))


def schedule_scores(solution, y0: float, q_policy=None) -> list[ScheduleScore]:
    market = solution.market
    named = [
        ("Equal trades", holdings_to_lot_policy(flat_holdings(market), solution)),
    ]
    if solution.lam > 0.0:
        named.append(
            (
                "Almgren-Chriss",
                holdings_to_lot_policy(
                    almgren_chriss_holdings(market, solution.lam), solution
                ),
            )
        )
    named.append(("Optimal", solution.policy))
    if q_policy is not None:
        named.append(("Q-learning", q_policy))
    return [
        score_rollout(rollout_policy(solution, name, y0, policy), market)
        for name, policy in named
    ]


def style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#333333")
    ax.grid(axis="y", color="#E4DDD2", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_inventory(scores: list[ScheduleScore], n_periods: int):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    days = np.arange(n_periods + 1)
    for score in scores:
        if score.name == "Q-learning":
            continue
        ax.plot(
            days,
            score.inventory / score.inventory[0],
            color=COLORS[score.name],
            linewidth=2.0,
            linestyle="--" if score.name == "Q-learning" else "-",
            label=score.name,
        )
    ax.set_xlabel("Day")
    ax.set_xlim(0, n_periods)
    ax.set_ylim(0, 1.02)
    style_axes(ax, "Shares left, as a fraction of the shortfall")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def figure_costs(scores: list[ScheduleScore]):
    shown = list(scores)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(shown))
    ax.bar(
        x,
        [s.kappa_bps for s in shown],
        color=[COLORS[s.name] for s in shown],
        width=0.72,
    )
    ax.set_xticks(x, [s.name for s in shown])
    style_axes(ax, "Expected shortfall (basis points)")
    fig.tight_layout()
    return fig


def figure_path(path, market: Market):
    days = np.arange(market.n_periods)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    axes[0].plot(days, path.unaffected, color=MUTED, linewidth=1.6, label="Unaffected price")
    axes[0].plot(days, path.mid, color=NAVY, linewidth=1.8, label="Mid, after our impact")
    axes[0].plot(
        days,
        path.execution,
        color=RUST,
        linewidth=1.4,
        linestyle="--",
        label="Price received",
    )
    style_axes(axes[0], "Dollars per share")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].bar(days, path.trades, color=NAVY, width=0.72)
    style_axes(axes[1], "Shares sold")
    axes[2].plot(days, path.displacement, color=RUST, linewidth=1.8)
    style_axes(axes[2], "Book displacement (dollars)")
    axes[2].set_xlabel("Day")
    fig.tight_layout()
    return fig


def figure_learning(updates, costs, optimal: float):
    """Q-learning's cost after each bar, and the dynamic program as a labeled line.

    The reference is drawn on the line itself. A legend swatch for a horizontal
    line sits in the empty part of the axes and reads as a second dashed series.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(
        updates,
        costs,
        color=NAVY,
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        zorder=3,
    )
    ax.axhline(optimal, color=RUST, linewidth=1.5, linestyle="--", zorder=2)
    low = min(float(np.min(costs)), float(optimal))
    high = max(float(np.max(costs)), float(optimal))
    span = max(high - low, 1.0)
    ax.set_ylim(low - 0.14 * span, high + 0.12 * span)
    ax.set_xlim(float(updates[0]), float(updates[-1]))
    ax.text(
        float(updates[0]),
        optimal,
        f"  dynamic program, ${optimal:,.0f}",
        color=RUST,
        va="bottom",
        ha="left",
        fontsize=8,
        zorder=4,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.set_xlabel("Sampled transitions")
    style_axes(ax, "Cost J (dollars)")
    fig.tight_layout()
    return fig


def render(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Forced sale", layout="wide")
    st.title("Forced sale into a book that refills")
    st.markdown(
        "A dollar shortfall Q is sold as X = Q/S0 shares over a fixed window. "
        "The reserve uses the expected bill C, at Almgren–Chriss risk aversion zero. "
        "Each sale walks the book, and whatever displacement is left decays "
        "from one day to the next. Sampled Q-learning is the check that this "
        "schedule is optimal when the refill speed is not handed to the solver."
    )

    with st.sidebar:
        st.header("Market")
        scenario = st.radio("Scenario", ["Normal", "Downturn", "Custom"], index=0)
        preset = normal_market() if scenario != "Downturn" else downturn_market()
        if scenario == "Custom":
            half_life = st.slider("Half-life of a displacement (days)", 0.25, 15.0, 1.0, 0.25)
            sigma = st.slider("Daily volatility (dollars)", 0.25, 4.0, 1.0, 0.25)
            depth = st.slider("Book depth, relative to normal", 0.25, 2.0, 1.0, 0.05)
            q = 200_000.0 * depth
            sigma_y = st.slider("Refill shock (dollars)", 0.0, 0.5, 0.0, 0.01)
        else:
            half_life = preset.half_life
            sigma = preset.sigma
            q = preset.q
            sigma_y = preset.sigma_y
            unit = "day" if half_life == 1 else "days"
            st.caption(
                f"Half-life {half_life:g} {unit}, volatility {sigma:g} dollars, "
                f"depth {q:,.0f} shares per dollar."
            )
        n_periods = st.slider("Trading days", 4, 15, 10)
        risk_multiple = st.slider(
            "Almgren–Chriss risk aversion λ, relative to the reference",
            0.0,
            5.0,
            0.0,
            0.1,
        )
        fee_cents = st.slider("Fee ε (cents per share)", 0.0, 5.0, 0.0, 0.5)
        permanent_cents = st.slider(
            "Permanent move of the whole sale (cents)", 0.0, 50.0, 0.0, 1.0
        )
        y_frac = st.slider("Opening displacement, as a fraction of a one-shot walk", 0.0, 0.8, 0.0, 0.05)
        fine = st.checkbox("Finer grid (40 lots, 41 nodes), as in the note", value=True)
        seed = st.number_input("Seed for the price path", min_value=0, max_value=10_000, value=1, step=1)
        st.caption(
            "The reserve slide sets λ = 0, so the sale is split evenly when the book "
            "refills within the day. Raising λ is the variance penalty on their earlier "
            "slide: the schedule sells sooner, and that penalty is not inside C. "
            "One times the reference makes the penalty for carrying the whole shortfall "
            "over the window equal the one-shot walk in the normal book. "
            "This λ is not the Poisson rate in the reserve slides, and the half-life "
            "is not the cost of holding cash."
        )

    shares = 100_000.0
    s0 = 50.0
    epsilon = fee_cents / 100.0
    gamma = 0.0 if permanent_cents == 0.0 else (permanent_cents / 100.0) / shares
    n_lots, n_y = (40, 41) if fine else (20, 31)
    lam = preference_lambda(n_periods, risk_multiple)
    y0 = y_frac * shares / q

    with st.spinner("Solving the dynamic program..."):
        solution = cached_solve(
            shares, s0, sigma, q, half_life, sigma_y, n_periods, n_lots, n_y, lam, epsilon, gamma
        )
    scores = schedule_scores(solution, y0)
    by_name = {row.name: row for row in scores}

    q_key = (
        shares, s0, sigma, q, half_life, sigma_y, n_periods, n_lots, n_y, lam, round(y0, 8)
    )
    q_result = st.session_state.get("q_result")
    if st.session_state.get("q_key") != q_key:
        q_result = None
    if q_result is not None:
        scores = schedule_scores(solution, y0, q_result.policy)
        by_name = {row.name: row for row in scores}

    her = kappa_bps(allison_reserve_cost(solution.market), solution.market)
    optimal = by_name["Optimal"].kappa_bps
    equal = by_name["Equal trades"].kappa_bps
    beta = solution.market.beta
    if beta < 0.01:
        persists = (
            "This book refills before the next trade. "
            "That is the case Almgren–Chriss prices in closed form."
        )
    else:
        persists = (
            f"This book does not refill before the next trade. "
            f"A fraction {beta:.2f} of each sale is still there the next day. "
            f"Almgren–Chriss assumes that fraction is zero."
        )
    if solution.lam == 0.0:
        st.info(
            f"{persists} The refill formula prices this sale at **{her:.2f} bp**. "
            f"Equal slices in the book on screen cost **{equal:.2f} bp**. "
            f"The optimal schedule costs **{optimal:.2f} bp**."
        )
    else:
        followed = by_name["Almgren-Chriss"].kappa_bps
        st.info(
            f"{persists} The refill formula prices this sale at **{her:.2f} bp**. "
            f"With the variance penalty on, the Almgren–Chriss schedule costs "
            f"**{followed:.2f} bp** in this book, and the optimal schedule costs "
            f"**{optimal:.2f} bp**."
        )

    schedules, path_tab, learner = st.tabs(["Schedules", "One price path", "Q-learning"])

    with schedules:
        left, right = st.columns(2)
        with left:
            render(figure_inventory(scores, n_periods))
        with right:
            render(figure_costs(scores))
        table = []
        for row in scores:
            table.append(
                {
                    "Schedule": row.name,
                    "κ = C/Q (bp)": round(row.kappa_bps, 2),
                    "Reserve cost C (dollars)": round(row.bill, 1),
                    "Walk (dollars)": round(row.impact, 1),
                    "Price-risk std. (bp)": round(row.std_bps, 2),
                    "Objective J (dollars)": round(row.objective, 1),
                }
            )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            f"Grid: {n_lots} lots, {n_y} displacement nodes. "
            f"Half-life {half_life:g} {'day' if half_life == 1 else 'days'}, decay factor β = {solution.market.beta:.3f}. "
            f"Almgren–Chriss λ = {lam:.3e} per dollar. "
            "Q = X S₀. C is the expected walk plus the fee and the permanent impact. "
            "κ = C/Q is the cost per dollar of shortfall. "
            "The standard deviation is the price risk of the inventory path, "
            "σ √(τ Σ x²). It does not enter C when λ = 0."
        )

    with path_tab:
        names = [row.name for row in scores if row.name != "Q-learning"]
        choice = st.selectbox("Schedule", names)
        chosen = by_name[choice]
        path = simulate_price_path(
            chosen.trades, solution.market, np.random.default_rng(int(seed)), y0
        )
        arrival = shares * s0
        realized = 1.0e4 * path.implementation_shortfall / arrival
        realized += chosen.kappa_bps - kappa_bps(chosen.impact, solution.market)
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected cost κ", f"{chosen.kappa_bps:.2f} bp")
        c2.metric("This path", f"{realized:.2f} bp")
        c3.metric("Price-risk std.", f"{chosen.std_bps:.2f} bp")
        render(figure_path(path, solution.market))
        st.caption(
            "The trade list ignores the unaffected price, so the average of the "
            "pathwise shortfall equals the walk. The fee and the permanent impact "
            "are the same on every path. A single path still rises or falls with "
            "the random walk, and the shortfall on that path can be negative."
        )

    with learner:
        st.markdown(
            "The state is the day, the lots still held, and the displacement of "
            "the book. After the grid is chosen, that is a table, and the action "
            "is a whole number of lots. Q-learning fills the table from the "
            "sampled cost of each sale, starting at the last day, because the "
            "horizon is finite and the dollars are not discounted. The greedy "
            "sale is then the same sale as the dynamic program."
        )
        st.markdown(
            "A neural Q-network, PPO, or SAC is built for a state that will not "
            "fit in a table, or for a continuous action. SAC also adds an entropy "
            "bonus, so its optimum is a different objective from the expected "
            "shortfall. On this grid those methods would approximate a sale the "
            "table already returns."
        )
        st.markdown(
            "The action value of the last bar is the observed cost of selling "
            "whatever is left. Each earlier bar is then filled in by sampling "
            "the transition and adding the value of the next bar. With no refill "
            "shock, one sample of every state and action is the expectation, so "
            "the sweep ends on the dynamic program."
        )
        if sigma_y > 0:
            st.caption(
                "A refill shock is on, so each state-action is an average of samples "
                "and the last point is a Monte Carlo estimate of the dynamic program."
            )
            samples = st.slider("Samples per state-action", 1, 20, 5)
        else:
            samples = 1
        if st.button("Run the sampled sweep", type="primary"):
            with st.spinner("Sampling transitions from the last bar backward..."):
                q_result = train_q(solution, n_samples=samples, seed=int(seed), y0=y0)
            st.session_state["q_result"] = q_result
            st.session_state["q_key"] = q_key
        if q_result is None:
            st.caption("No sweep yet for these parameters.")
        else:
            gap = q_result.greedy_objective[-1] - q_result.optimal_objective
            render(
                figure_learning(
                    q_result.updates,
                    q_result.greedy_objective,
                    q_result.optimal_objective,
                )
            )
            st.markdown(
                "Each dot is the cost of the schedule Q-learning can play "
                "after that bar has been sampled. The dashed line is the dynamic "
                "program. It is one number, so it is horizontal. The curve starts "
                "at the cost of selling nothing until the deadline."
            )
            start, learned, optimum, gap_box = st.columns(4)
            start.metric("Start", f"{q_result.greedy_objective[0]:,.0f} dollars")
            learned.metric("After the sweep", f"{q_result.greedy_objective[-1]:,.0f} dollars")
            optimum.metric("Dynamic program", f"{q_result.optimal_objective:,.0f} dollars")
            gap_box.metric("Gap", f"{gap:,.2f} dollars")
            st.caption(
                "Start is the cost of selling nothing until the deadline. "
                "The gap is the learned schedule minus the dynamic program."
            )


if __name__ == "__main__":
    main()
