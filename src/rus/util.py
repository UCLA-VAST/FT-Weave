from src.config import (
    CNOT_TIME,
    SE_TIME,
    TMR_P,
    TMR_Q,
)

from src.ds.device_state import FactoryPool, QubitAngleTracker
from src.ds.architecture import move_duration
from src.rus.angle_factory_index import AngleFactoryIndex


threshold_high_tmr = 5
threshold_high_rus = 5


def update_qubit_state_per_teleportation(
    successful_qubits: set[int],
    successful_teleportation_qubits: set[int],
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    angle_factory_index: AngleFactoryIndex,
):
    """
    Update qubit states
    """
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
            angle_factory_index.add_qubit_for_angle(tracker.target_angle)


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
            or len(qubit_factory_pairs) > threshold_high_rus
            or (len(target_qubits_angles) - len(successful_qubits)) < threshold_high_tmr
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
        and len(qubit_factory_pairs) < threshold_high_rus
        and (len(target_qubits_angles) - len(successful_qubits)) > threshold_high_tmr
    ):
        # print("Breaking teleportation loop to do another TMR")
        return []
    return new_routing_batches
