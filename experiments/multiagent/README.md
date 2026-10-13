# Multi-agent experiments

Code to reproduce the two optimality-gap figures in *Strategic Multi-Agent
Resource Allocation via Statistical Contracts*. All data are synthetic.
After installing the package (`pip install -e ".[dev]"` from the repository
root), run from the repository root:

```bash
python experiments/multiagent/optimality_gaps.py                  # both comparisons
python experiments/multiagent/optimality_gaps.py --weight-sweep   # weight dispersions only
python experiments/multiagent/optimality_gaps.py --gap-scaling    # agent counts only
python experiments/multiagent/optimality_gaps.py --quick          # small smoke test
```

The defaults use seed 20260915, 150 active instances per comparison,
budget 30, and risk level 0.1. The weight sweep uses 30 agents and weight
dispersions 0, 0.2, 0.4, 0.6, 0.8, 1.0, and 1.2 on the same instances.
The agent-count comparison uses 10, 20, 40, 80, and 100 agents with weight
dispersion 0.8. Full parameters and seed constructions are recorded in
`results/optimality_gap_weight_sweep.json` and
`results/optimality_gap_scaling.json`. The concave objective is the
logarithmic objective in the paper, evaluated directly.

The figures use `text.usetex`, so a LaTeX installation is required.

Outputs:
- `figures/optimality_gap_weight_sweep.png`, `figures/optimality_gap_scaling.png`
- `results/optimality_gap_weight_sweep.csv`: trial-level results across weight dispersions.
- `results/optimality_gap_weight_sweep_summary.csv`: aggregated gaps across weight dispersions.
- `results/optimality_gap_scaling.csv`: trial-level results across agent counts.
- `results/optimality_gap_scaling_summary.csv`: aggregated gaps across agent counts.
- `results/*.json`: parameters and seed constructions.

The solvers live in `statcontracts.multiagent`. Their checks against
brute-force enumeration and analytic solutions are in `tests/test_multiagent.py`.
