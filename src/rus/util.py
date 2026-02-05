import numpy as np

from src.config import (
    CNOT_TIME,
    SE_TIME,
    TMR_P,
    TMR_Q,
    ANGLE_S,
    THRESHOLD_HIGH_TMR,
    THRESHOLD_HIGH_RUS,
)

from src.ds import FactoryPool, QubitAngleTracker, move_duration
from src.rus import AngleFactoryIndex
from src.analog_rotation import insert_s_gate


def update_qubit_state_per_teleportation(
    successful_qubits: set[int],
    successful_teleportation_qubits: set[int],
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    angle_factory_index: AngleFactoryIndex,
    execution_log: list,
    start_time: float,
):
    """
    Update qubit states
    """
    is_s_gate_inserted = False
    for (qubit, factory_id), success in zip(qubit_factory_pairs, rus_simulation):
        tracker = qubit_trackers[qubit]
        angle_factory_index.remove_qubit_for_angle(tracker.target_angle)
        factory_pool.free_factory(factory_id)
        angle_factory_index.remove_factory(factory_id)
        if success:
            successful_qubits.add(qubit)
            successful_teleportation_qubits.add(qubit)
            tracker.clear_all()
        else:
            # Free factories for removed angles
            teleportation_angle = tracker.target_angle
            _ = tracker.remove_angle(teleportation_angle)
            tracker.double_target_angle()
            # ! insert S gate
            if np.isclose(ANGLE_S - tracker.target_angle, 0.0, atol=1e-8):
                insert_s_gate(execution_log, start_time, -1, qubit)
                is_s_gate_inserted = True
                successful_qubits.add(qubit)
                successful_teleportation_qubits.add(qubit)
                tracker.clear_all()
            else:
                if tracker.target_angle > ANGLE_S:
                    is_s_gate_inserted = True
                    insert_s_gate(execution_log, start_time, -1, qubit)
                    tracker.set_target_angle(tracker.target_angle - ANGLE_S)
                angle_factory_index.add_qubit_for_angle(tracker.target_angle)

    # for qubit in qubit_trackers.keys():
    #     tracker = qubit_trackers[qubit]
    #     if qubit in successful_teleportation_qubits:
    #         continue
    #     if np.isclose(ANGLE_S - tracker.target_angle, 0.0, atol=1e-8):
    #         insert_s_gate(execution_log, start_time, -1, qubit)
    #         is_s_gate_inserted = True

    return start_time + (SE_TIME if is_s_gate_inserted else 0)


def update_qubit_state_post_teleportation(
    successful_teleportation_qubits: set[int],
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    angle_factory_index: AngleFactoryIndex,
):
    """
    Update qubit states
    """
    for qubit in successful_teleportation_qubits:
        tracker = qubit_trackers[qubit]
        for factory_id, _ in tracker.factories:
            factory_pool.free_factory(factory_id)

    for angle, count in angle_factory_index.angle_to_qubits.items():
        if count == 0:
            # free all factories preparing this angles as no one needs it
            factories = angle_factory_index.get_factories_for_angle(angle)
            for factory_id, qubit_id in factories:
                factory_pool.free_factory(factory_id)


def check_teleportation_worthiness(
    routing_batches: list,
    n_aods: int,
    qubit_factory_pairs: list[tuple[int, int]],
    target_qubits_angles: dict[int, float],
    successful_qubits: set[int],
    parital_skip: bool = False,
) -> list:
    """
    Check if performing injection is worth it based on movement time and qubit counts.
    """

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
            2 * max_movement_time // len(batches) + CNOT_TIME
            < SE_TIME * (TMR_P + TMR_Q)
            or len(qubit_factory_pairs) > THRESHOLD_HIGH_RUS
            or (len(target_qubits_angles) - len(successful_qubits)) < THRESHOLD_HIGH_TMR
        ):
            move_time_idx_pairs.append((max_movement_time, i))
            new_routing_batches.append(batches)
        else:
            # print("Skipping teleportation for this batch due to high movement time:")
            for qubit, _, _, factory_id, _, _ in batches:
                qubit_factory_pairs.remove((qubit, factory_id))
            print(batches)
    # Sort by movement time descending
    move_time_idx_pairs.sort(reverse=True)
    for movement_time, idx in move_time_idx_pairs:
        # Assign to the earliest available AOD
        aod_idx = aod_earliest_available_time.index(min(aod_earliest_available_time))
        aod_earliest_available_time[aod_idx] += movement_time
    total_movement_time = max(aod_earliest_available_time) * 2
    # input()
    if (
        total_movement_time + CNOT_TIME > SE_TIME * (TMR_P + TMR_Q)
        and len(qubit_factory_pairs) < THRESHOLD_HIGH_RUS
        and (len(target_qubits_angles) - len(successful_qubits)) > THRESHOLD_HIGH_TMR
    ):
        # print("Breaking teleportation loop to do another TMR")
        return []
    return new_routing_batches
