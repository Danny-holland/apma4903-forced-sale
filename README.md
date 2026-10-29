# The cost of a forced sale

APMA 4903, November 2026. Daniel Holland, with Allison Duffy and Jasmine Li.

The note is [notes/execution-math.pdf](notes/execution-math.pdf). It prices the sale of a cash shortfall when yesterday's trade is still in the book, and it checks that schedule with tabular Q-learning. The papers cited in that work are listed in [references/README.md](references/README.md).

## Run

From this directory, after installing the packages in `execution/requirements.txt`:

```
python execution/tests/test_checks.py
python execution/report.py
streamlit run execution/app.py
```

`report.py` prints the tables in the note. The screen opens on the same bill, with Almgren–Chriss risk aversion at zero.
