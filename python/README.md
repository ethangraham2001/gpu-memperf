# Python Scripts

Contains some utilities for plotting results.

## Install dependencies

```sh
python -m venv .venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## `plot_simple.py`

Generates an X-Y scatter plot from some `.csv` data.

## Orchestrator (`orchestrator.py`)

This is a centralized program for bulk-running `gpu-memperf` benchmarks, with
some additional guardrails in place.

### Usage

```
srun -A dphpc --gpus 5060ti:1 python3 python/orchestrator.py

options:
  -h, --help         show this help message and exit
  --verbose          Enable verbose output
  --program PROGRAM  Run this benchmark when enabled. If none are specified, all enabled by default
  --out OUT
```

### Design

Each benchmark describes itself by implementing base class `Benchmark` - this
tells the orchestrator how to run it, and how to plot its results / interpret
its output.

Currently, benchmark default configurations are described directly in the Python
code, e.g.

```python
class SharedToRegisterBenchmark:
    # ...
    @classmethod
    def default(cls):
        return cls(
            sizes=[4096 * (2**i) for i in range(4)],
            threads=[32 * (2**i) for i in range(4)],
            strides=[2**i for i in range(6)],
            num_iters=int(1e5),
            reps=3,
        )
```

All plots and metrics are bundled together in the specified `--out` directory,
alongside an `info` file that currently contains information on the active
Git repository where the tests were run. Namely

- The commit hash.
- The branch.
- Whether or not there are any unstaged diffs.

This doesn't check whether or not the `gpu-memperf` binary is up to date - so
please make sure to rebuild if any changes were made.
