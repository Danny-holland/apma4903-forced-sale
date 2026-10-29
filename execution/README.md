# Forced sale with decaying impact

The liquidation cost of a public-equity shortfall when each sale leaves a
displacement that decays from day to day. The math is in
`notes/execution-math.tex`. This folder is the numerical solution of those
equations.

## Run

From the repository root, with the packages in `requirements.txt`:

```bash
python execution/tests/test_checks.py
python execution/report.py
streamlit run execution/app.py
```

`report.py` prints the cost tables in the note. Those tables use risk
aversion zero, which is the reserve slide: the bill is the expected walk
plus a per-share fee and a permanent impact. The screen solves the same
dynamic program. The default is that risk-neutral bill. The risk-aversion
slider is the variance penalty from the earlier Almgren–Chriss slide.

## What is computed

- `sim/market.py` — the book. Depth `q`, half-life, and the decay factor β.
- `sim/control.py` — backward induction, the Almgren–Chriss holdings, and the
  Obizhaeva–Wang blocks-and-rate schedule.
- `sim/qlearn.py` — finite-horizon Q-learning. The last bar is the observed
  liquidation cost; each earlier bar is a sample average of Watkins's update.
- `sim/paths.py` — one random unaffected-price path for a trade list that has
  already been chosen.
- `sim/compare.py` — the reserve bill C, the cost per dollar of shortfall,
  the price-risk standard deviation, and the objective J.

The checks in `tests/test_checks.py` are the closed forms: Almgren–Chriss when
the book refills inside the day, a path-independent impact cost when it never
refills, endpoint trades in the risk-neutral Obizhaeva–Wang case, and
Q-learning ending on the dynamic program.
