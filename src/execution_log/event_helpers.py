from __future__ import annotations

from src.star.config import CNOT_TIME, SE_TIME
from src.t_cultivation.config import SE_STAGE_1, SE_STAGE_2

CANONICAL_EVENT_KEYS = (
    "start_time",
    "end_time",
    "factories",
    "operation",
    "aod_assignment",
    "targets",
    "move_vecs",
)


def normalize_factories(factory_field) -> list[int]:
    """Normalize log factory field to a list."""
    if isinstance(factory_field, int):
        return [factory_field]
    if factory_field is None:
        return []
    return list(factory_field)


def resolve_indexed(values, idx: int):
    """Resolve per-factory value from scalar/list payload."""
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return values


def operation_end_time(
    operation: str, start_time: float, movement_time: float = 0.0
) -> float:
    if operation == "SE":
        return start_time + SE_TIME
    if operation == "CNOT":
        return start_time + CNOT_TIME
    if operation == "Rz":
        return start_time + 1
    if operation in ("S", "H"):
        return start_time + SE_TIME
    if operation in ("move", "return_move"):
        return start_time + movement_time
    if operation == "SE_stage_1":
        return start_time + SE_STAGE_1
    if operation == "SE_stage_2":
        return start_time + (movement_time if movement_time else SE_STAGE_2)
    return start_time


def build_event(
    *,
    start_time: float,
    end_time: float,
    factories,
    operation: str,
    aod_assignment,
    targets,
    move_vecs,
) -> dict:
    """Build a canonical execution event dict."""
    return {
        "start_time": start_time,
        "end_time": end_time,
        "factories": factories,
        "operation": operation,
        "aod_assignment": aod_assignment,
        "targets": targets,
        "move_vecs": move_vecs,
    }
