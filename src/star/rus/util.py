import numpy as np

from src.star.config import (
    CNOT_TIME,
    SE_TIME,
    TMR_PREPARATION_TIME,
)

from ...ds import FactoryPool, QubitAngleTracker
from src.ds import move_duration
from src.execution_log import insert_s_gate, LogicalSEScheduler


def update_qubit_state_per_teleportation(
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    qubit_trackers: dict[int, QubitAngleTracker],
    execution_log: list,
    start_time: float,
    aod_id: int,
    logical_se_scheduler: LogicalSEScheduler | None = None,
):
    """
    Update qubit states
    """
    qubit_with_s_gate = []
    for (qubit, factory_id), success in zip(qubit_factory_pairs, rus_simulation):
        tracker = qubit_trackers[qubit]
        success, s_gate_inserted = tracker.update_rus_state(success)
        if success:
            tracker.clear_all()
            qubit_trackers.pop(qubit)
        if s_gate_inserted:
            qubit_with_s_gate.append(qubit)
    if qubit_with_s_gate:
        insert_s_gate(
            execution_log,
            start_time,
            qubit_with_s_gate,
            aod_id,
            logical_se_scheduler=logical_se_scheduler,
        )
        start_time += SE_TIME

    return start_time


def check_teleportation_worthiness(
    routing_batches: list,
    n_aods: int,
    qubit_factory_pairs: list[tuple[int, int]],
    n_tmr_next_run: int,
    total_qubits: int,
    parital_skip: bool = False,
    factory_pool: FactoryPool | None = None,
) -> list[list[tuple[int, int, int, int, int, int]]]:
    """
    Check if performing injection is worth it based on movement time and qubit counts.
    """
    THRESHOLD_HIGH_TMR = np.sqrt(total_qubits)
    THRESHOLD_HIGH_RUS = THRESHOLD_HIGH_TMR

    def _has_better_idle_factory_for_batch(batch) -> bool:
        """Return True if every qubit in batch has at least one better idle factory nearby.

        If False, we should keep teleportation for this batch instead of skipping,
        because re-preparing likely cannot improve routing quality.
        """
        if factory_pool is None:
            return False

        idle_factories = factory_pool.get_idle_factories()
        if len(idle_factories) == 0:
            return False

        for _, x_q, y_q, _, x_f, y_f in batch:
            current_move = move_duration(x_q, y_q, x_f, y_f)
            best_idle_move = min(
                move_duration(x_q, y_q, idle.location[0], idle.location[1])
                for idle in idle_factories
            )
            if best_idle_move >= current_move:
                return False

        return True

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
            if not _has_better_idle_factory_for_batch(batches):
                return routing_batches
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
