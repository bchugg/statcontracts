"""The single-shot statistical contract: one agent, one dataset."""

from .contracts import (
    best_response,
    expected_profit,
    is_incentive_aligned,
    max_aligned_level,
    null_profit,
    optimal_license,
    principal_utility,
)
from .models import Binomial, GaussianMean, ThresholdLicense

__all__ = [
    "best_response",
    "expected_profit",
    "is_incentive_aligned",
    "max_aligned_level",
    "null_profit",
    "optimal_license",
    "principal_utility",
    "Binomial",
    "GaussianMean",
    "ThresholdLicense",
]
