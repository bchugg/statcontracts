"""Statistical contracts with many competing agents."""

from .mechanisms import (
    mechanism_aon,
    mechanism_cumulative,
    mechanism_ebh,
    null_allocation_proportion,
    rescale_m1,
)
from .optimize import (
    evaluate_log_utility,
    most_violated_subset,
    solve_linear_compact,
    solve_linear_full,
    solve_log,
)

__all__ = [
    "mechanism_aon",
    "mechanism_cumulative",
    "mechanism_ebh",
    "null_allocation_proportion",
    "rescale_m1",
    "evaluate_log_utility",
    "most_violated_subset",
    "solve_linear_compact",
    "solve_linear_full",
    "solve_log",
]
