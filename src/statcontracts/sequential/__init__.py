"""Statistical contracts with a single agent experimenting adaptively."""

from . import adjusters, incremental, multiround
from .rewards import (
    BettingReward,
    GaussianReward,
    HoeffdingReward,
    RewardProcess,
    kl_bernoulli,
    numeric_cgf_optimal,
)
from .simulate import (
    DEFAULT_SEED,
    SimulationResult,
    sim_bellman,
    sim_cgf_aware,
    sim_cgf_optimal,
    sim_drift_aware,
    sim_plug_in,
    sim_threshold,
    simulate,
)
from .stopping import (
    bellman_benchmark,
    beta_threshold,
    solve_bellman_pi,
    solve_bellman_vi,
    threshold_value_grid,
)

__all__ = [
    "adjusters",
    "incremental",
    "multiround",
    "BettingReward",
    "GaussianReward",
    "HoeffdingReward",
    "RewardProcess",
    "kl_bernoulli",
    "numeric_cgf_optimal",
    "DEFAULT_SEED",
    "SimulationResult",
    "sim_bellman",
    "sim_cgf_aware",
    "sim_cgf_optimal",
    "sim_drift_aware",
    "sim_plug_in",
    "sim_threshold",
    "simulate",
    "bellman_benchmark",
    "beta_threshold",
    "solve_bellman_pi",
    "solve_bellman_vi",
    "threshold_value_grid",
]
