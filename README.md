# LLM Serving Lab

A small, CPU-only experimental simulator for comparing LLM request-routing policies under heterogeneous demand and changing serving conditions. It models decisions; it does not run or benchmark actual language models.

## Quick start

Requires Python 3.11 or newer. The simulator has no runtime dependencies.

On Linux or macOS:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
llm-serving-lab run --config configs/default.json --output results
python -m unittest discover -s tests -p 'test_simulator.py' -v
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
llm-serving-lab run --config configs/default.json --output results
python -m unittest discover -s tests -p 'test_simulator.py' -v
```

Run without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m llm_serving_lab.cli run --config configs/default.json --output results
```

On Linux or macOS, use `PYTHONPATH=src` instead:

```bash
PYTHONPATH=src python -m llm_serving_lab.cli run --config configs/default.json --output results
```

The repository also contains older `adaroute` code and tests. Those tests are separate from this simulator and are not included in the command above.

The run writes `summary.json`, `requests.csv`, `summary.csv`, and a dependency-free `comparison.svg` into the output directory. Use `--scenario bursty`, `--scenario heterogeneous`, or `--scenario fault_injected` to select a workload. The seed and request count can be overridden on the command line.

## Model and replica decisions

Each request carries task, prompt and expected output token counts, minimum quality, latency target, privacy class, and maximum estimated cost. A model policy first selects a feasible logical model class. A separate replica policy then chooses an instance using its queue estimate, utilization, latency history, and health. This separation allows model-routing choices and instance scheduling choices to be compared independently.

The built-in comparison includes strongest feasible model, cheapest feasible model, static rules, adaptive multi-objective routing, round-robin replicas, and least-loaded replicas. Candidate models that violate a request's quality, privacy, context, or budget constraints are excluded before ranking.

## Metrics and interpretation

Results include TTFT, TPOT, end-to-end latency, p95/p99 latency, completed throughput, quality, estimated token cost, SLO attainment, failures, retries, and replica utilization. Timing and cost are analytical estimates defined by the scenario profiles, not hardware measurements. Fault injection makes replicas temporarily unavailable on a reproducible schedule; the simulator retries on another healthy replica when possible.

Compare policies on the same seed and scenario. Vary workload sizes, scenario fault schedules, model profiles, and policy weights in JSON configs before drawing conclusions. This is an experimental prototype, not a production gateway or a substitute for live serving benchmarks.