from __future__ import annotations


def normalize_factories(factory_field) -> list[int]:
    if isinstance(factory_field, int):
        return [factory_field]
    if factory_field is None:
        return []
    return list(factory_field)


def resolve_indexed(values, idx: int):
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return values


def _is_move_coord_pair(item) -> bool:
    return (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and all(isinstance(x, str) for x in item[:2])
    )


def resolve_move_vecs(move_vecs, idx: int):
    """Resolve movement endpoints for factory *idx* from a log move_vecs field.

    Supports per-factory lists, a single shared ``[from, to]`` pair, and the
    common case of one pair wrapped as ``[[from, to]]`` for multi-factory moves.
    """
    if move_vecs is None:
        return None
    if _is_move_coord_pair(move_vecs):
        return move_vecs
    if isinstance(move_vecs, list):
        if idx < len(move_vecs) and _is_move_coord_pair(move_vecs[idx]):
            return move_vecs[idx]
        if len(move_vecs) == 1 and _is_move_coord_pair(move_vecs[0]):
            pair = move_vecs[0]
            if idx == 0:
                return pair
            # Receiver in donor/receiver redistribution: reverse endpoints.
            return [pair[1], pair[0]]
    return resolve_indexed(move_vecs, idx)


def entry_start(entry: dict) -> float:
    return entry.get("start_time", 0.0)


def entry_end(entry: dict) -> float:
    start = entry_start(entry)
    return entry.get("end_time", start)
