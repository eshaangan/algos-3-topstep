"""
Sample weighting modules for V3 pipeline.

Exports:
- map_event_intervals_to_index, compute_concurrency, compute_uniqueness_weights
- compute_magnitude_weights
- compute_hmm_regime_weights, compute_regime_weights_by_policy, combine_weights
- write_weight_schema, compute_weight_schema_hash
"""

from .uniqueness import (
    map_event_intervals_to_index,
    compute_concurrency,
    compute_uniqueness_weights,
)
from .magnitude import compute_magnitude_weights
from .hmm_weights import (
    compute_hmm_regime_weights,
    compute_regime_weights_by_policy,
    combine_weights,
    analyze_regime_weight_distribution,
)
from .schema import write_weight_schema, compute_weight_schema_hash

__all__ = [
    "map_event_intervals_to_index",
    "compute_concurrency",
    "compute_uniqueness_weights",
    "compute_magnitude_weights",
    "compute_hmm_regime_weights",
    "compute_regime_weights_by_policy",
    "combine_weights",
    "analyze_regime_weight_distribution",
    "write_weight_schema",
    "compute_weight_schema_hash",
]
