import numpy as np

from src.config import (
    CNOT_TIME,
    SE_TIME,
    TMR_PREPARATION_TIME,
)

from src.ds import QubitAngleTracker, move_duration
from src.analog_rotation import insert_s_gate


def update_qubit_state_per_teleportation(
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    qubit_trackers: dict[int, QubitAngleTracker],
    execution_log: list,
    start_time: float,
    aod_id: int,
):
    """
    Update qubit states
    """
    qubit_with_s_gate = []
    for (qubit, factory_id), success in zip(qubit_factory_pairs, rus_simulation):
        tracker = qubit_trackers[qubit]
        teleportation_angle = tracker.target_angle
        # update device state as the angle is consumed by teleportation

        success, s_gate_inserted = tracker.update_rus_state(success)
        if success:
            tracker.clear_all()
            qubit_trackers.pop(qubit)
            print(
                f"[update_qubit_state_per_teleportation] Qubit {qubit} successfully teleported with angle {teleportation_angle:.4f} at time {start_time:.2f}"
            )
        if s_gate_inserted:
            qubit_with_s_gate.append(qubit)
    if qubit_with_s_gate:
        insert_s_gate(execution_log, start_time, qubit_with_s_gate, aod_id)
        start_time += SE_TIME

    return start_time


def check_teleportation_worthiness(
    routing_batches: list,
    n_aods: int,
    qubit_factory_pairs: list[tuple[int, int]],
    n_tmr_next_run: int,
    total_qubits: int,
    parital_skip: bool = False,
) -> list:
    """
    Check if performing injection is worth it based on movement time and qubit counts.
    """
    THRESHOLD_HIGH_TMR = np.sqrt(total_qubits)
    THRESHOLD_HIGH_RUS = THRESHOLD_HIGH_TMR

    new_routing_batches = []
    # check if executing teleporation is worth it.
    aod_earliest_available_time = [0.0] * n_aods
    move_time_idx_pairs = []
    # print("Checking teleportation worthiness...")
    for i, batches in enumerate(routing_batches):
        max_movement_time = 0.0
        for _, x_q, y_q, factory_id, x_f, y_f in batches:
            max_movement_time = max(
                max_movement_time, move_duration(x_q, y_q, x_f, y_f)
            )
        if not parital_skip or (
            2 * max_movement_time // len(batches) + CNOT_TIME < TMR_PREPARATION_TIME
            or len(qubit_factory_pairs) > THRESHOLD_HIGH_RUS
            or n_tmr_next_run < THRESHOLD_HIGH_TMR
        ):
            move_time_idx_pairs.append((max_movement_time, i))
            new_routing_batches.append(batches)
        else:
            # print("Skipping teleportation for this batch due to high movement time:")
            for qubit, _, _, factory_id, _, _ in batches:
                qubit_factory_pairs.remove((qubit, factory_id))
            # print(batches)
    # Sort by movement time descending
    move_time_idx_pairs.sort(reverse=True)
    for movement_time, idx in move_time_idx_pairs:
        # Assign to the earliest available AOD
        aod_idx = aod_earliest_available_time.index(min(aod_earliest_available_time))
        aod_earliest_available_time[aod_idx] += movement_time
    total_movement_time = max(aod_earliest_available_time) * 2
    # input()
    if (
        total_movement_time + CNOT_TIME > TMR_PREPARATION_TIME
        and len(qubit_factory_pairs) < THRESHOLD_HIGH_RUS
        and n_tmr_next_run > THRESHOLD_HIGH_TMR
    ):
        # print("Breaking teleportation loop to do another TMR")
        return []
    return new_routing_batches
