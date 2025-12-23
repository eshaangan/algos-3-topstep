"""
Labeling modules for V3 pipeline.

Exports:
- generate_events: Event generation for triple-barrier labeling
- apply_triplebarrier: Apply triple-barrier logic to events
- write_label_schema: Write label schema artifacts
"""

from .events import generate_events
from .triple_barrier import apply_triplebarrier
from .schema import write_label_schema, compute_label_schema_hash

__all__ = [
    "generate_events",
    "apply_triplebarrier",
    "write_label_schema",
    "compute_label_schema_hash",
]
