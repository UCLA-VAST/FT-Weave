from math import hypot


def build_clifford_layer_log(
    instruction: dict, logic_qubit_locations: list[tuple[int, int]]
) -> list[dict]:
    """Create a lightweight movement-aware log for Clifford layers.

    This log is intended for animation preparation. Analysis should continue to
    consume dedicated Rz execution logs.
    """
    gate = instruction["gate"]
    if gate == "H":
        return [
            {
                "operation": "H",
                "begin_time": 0.0,
                "end_time": 1.0,
                "qubits": list(instruction["targets"]),
            }
        ]

    if gate != "CNOT":
        return []

    pairs = list(instruction["targets"])
    move_duration = 0.0
    move_vecs = []
    for mobile, pivot in pairs:
        x_q, y_q = logic_qubit_locations[mobile]
        x_p, y_p = logic_qubit_locations[pivot]
        move_duration = max(move_duration, hypot(x_p - x_q, y_p - y_q))
        move_vecs.append(((x_q, y_q), (x_p, y_p)))

    cnot_duration = 1.0
    return [
        {
            "operation": "move",
            "begin_time": 0.0,
            "end_time": move_duration,
            "qubit_pairs": pairs,
            "move_vecs": move_vecs,
        },
        {
            "operation": "CNOT",
            "begin_time": move_duration,
            "end_time": move_duration + cnot_duration,
            "qubit_pairs": pairs,
        },
        {
            "operation": "return_move",
            "begin_time": move_duration + cnot_duration,
            "end_time": 2 * move_duration + cnot_duration,
            "qubit_pairs": pairs,
            "move_vecs": [vec[::-1] for vec in move_vecs],
        },
    ]
