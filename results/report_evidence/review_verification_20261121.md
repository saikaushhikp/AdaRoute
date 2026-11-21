# Independent review verification — 21 November 2026

We ran these commands against the current uncommitted working tree; commit `3b081c7ff8cb3c6fb2d95561ae82657081883db1` alone does not contain all reviewed changes.

| Command / check | Observed |
|---|---|
| `just test` | 194 passed, 1 skipped in 24.90 s; Docker Python 3.11.18, pytest 9.2.0. Some tests use mocks; the Redis live test can return early without marking a skip. |
| `just ui-test` | 18 passed in 4 files. |
| `just lint` | Ruff: all checks passed. |
| `just report-eval` | Exit 0; live gateway, 5 requests each for adaptive, cheapest and strongest; 10 cache cases matched their expected decisions. See `runs/20261120T222129.550621Z/`. UTC timestamp corresponds to 21 November in Asia/Kolkata. |
| Grader counterexample | `check_code("def is_even(n):\n    return False")` returned `True`; `check_math("It is not 391")` returned `True`. Reported checker rates are not verified task accuracy. |

The evaluation ran while the test suite was also running, so latency measurements may be affected by concurrent activity. It makes local-only requests and appends normal gateway records. The saved selection results retain final answer and a preview of raw output, not complete raw responses. The manifest records a dirty tree and does not fully pin model, container and dependency versions. Model scores and local costs are not independently measured or invoiced. These limitations prevent causal or general performance claims.
