from math import hypot


def _orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def _on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[
        1
    ] <= max(a[1], b[1])


def _segments_conflict(
    s1: tuple[tuple[int, int], tuple[int, int]],
    s2: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    """Return True when two movement paths intersect/overlap."""
    p1, q1 = s1
    p2, q2 = s2

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q1, q2):
        return True
    if o3 == 0 and _on_segment(p2, q2, p1):
        return True
    if o4 == 0 and _on_segment(p2, q2, q1):
        return True
    return False


def _decompose_non_crossing_batches(
    pairs: list[tuple[int, int]], logic_qubit_locations: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    """Greedily partition CNOT pairs into batches with no crossing move vectors."""
    batches: list[list[tuple[int, int]]] = []
    batch_segments: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []

    for pair in pairs:
        mobile, pivot = pair
        segment = (logic_qubit_locations[mobile], logic_qubit_locations[pivot])
        placed = False
        for i, segs in enumerate(batch_segments):
            if all(not _segments_conflict(segment, s) for s in segs):
                batches[i].append(pair)
                segs.append(segment)
                placed = True
                break
        if not placed:
            batches.append([pair])
            batch_segments.append([segment])
    return batches


def build_clifford_layer_log(
    instruction: dict, logic_qubit_locations: list[tuple[int, int]]
) -> list[dict]:
    """Create a lightweight movement-aware log for Clifford layers.

    This log is intended for animation preparation. Analysis should continue to
    consume dedicated Rz execution logs.
    """
    gate = instruction["gate"]
    if gate in ("H", "S", "Sdg", "X", "Y", "Z"):
        return [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "factories": [],
                "operation": gate if gate != "Sdg" else "S",
                "aod_assignment": None,
                "targets": list(instruction["targets"]),
                "move_vecs": None,
            }
        ]

    if gate == "I":
        return []

    if gate != "CNOT":
        return []

    pairs = list(instruction["targets"])
    cnot_duration = 1.0
    now = 0.0
    layer_log: list[dict] = []
    batches = _decompose_non_crossing_batches(pairs, logic_qubit_locations)

    batch_data: list[
        tuple[
            list[tuple[int, int]], float, list[tuple[tuple[int, int], tuple[int, int]]]
        ]
    ] = []
    for batch_pairs in batches:
        move_duration = 0.0
        move_vecs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for mobile, pivot in batch_pairs:
            x_q, y_q = logic_qubit_locations[mobile]
            x_p, y_p = logic_qubit_locations[pivot]
            move_duration = max(move_duration, hypot(x_p - x_q, y_p - y_q))
            move_vecs.append(((x_q, y_q), (x_p, y_p)))
        batch_data.append((batch_pairs, move_duration, move_vecs))

    for batch_pairs, move_duration, move_vecs in batch_data:
        layer_log.append(
            {
                "start_time": now,
                "end_time": now + move_duration,
                "factories": [],
                "operation": "move",
                "aod_assignment": None,
                "targets": batch_pairs,
                "move_vecs": move_vecs,
            }
        )
        now += move_duration

    layer_log.append(
        {
            "start_time": now,
            "end_time": now + cnot_duration,
            "factories": [],
            "operation": "CNOT",
            "aod_assignment": None,
            "targets": pairs,
            "move_vecs": None,
        }
    )
    now += cnot_duration

    for batch_pairs, move_duration, move_vecs in batch_data:
        layer_log.append(
            {
                "start_time": now,
                "end_time": now + move_duration,
                "factories": [],
                "operation": "return_move",
                "aod_assignment": None,
                "targets": batch_pairs,
                "move_vecs": [vec[::-1] for vec in move_vecs],
            }
        )
        now += move_duration
    return layer_log
