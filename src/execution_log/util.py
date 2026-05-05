from src.ds import move_duration
from src.star.config import CNOT_TIME
from src.execution_log.event_helpers import (
    build_event,
    normalize_factories,
    operation_end_time,
)


def write_execution_log(
    execution_log: list,
    start_time: float,
    factories: list[int],
    # factory_id: int,
    operation: str,
    aod_assignment: int | None,
    targets: list[int] | list[int | None] | list[float | None] | None,
    movement_time: float = 0,
    move_vecs: list[list[str]] | None = None,
):
    end_time = operation_end_time(
        operation=operation,
        start_time=start_time,
        movement_time=movement_time,
    )
    execution_log.append(
        build_event(
            start_time=start_time,
            end_time=end_time,
            factories=factories,
            operation=operation,
            aod_assignment=aod_assignment,
            targets=targets,
            move_vecs=move_vecs,
        )
    )


def write_rus_result_log(
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    circuit_moment: float,
    execution_log: list,
):
    factories_success = []
    targets_success = []
    factories_fail = []
    targets_fail = []
    for (qubit, factory_id), injection_result in zip(
        qubit_factory_pairs, rus_simulation
    ):
        if injection_result:
            factories_success.append(factory_id)
            targets_success.append(qubit)
        else:
            factories_fail.append(factory_id)
            targets_fail.append(qubit)
    if factories_success:
        write_execution_log(
            execution_log,
            circuit_moment,
            factories_success,
            "RUS_success",
            None,
            targets_success,
        )
    if factories_fail:
        write_execution_log(
            execution_log,
            circuit_moment,
            factories_fail,
            "RUS_fail",
            None,
            targets_fail,
        )

    return execution_log


def execute_rus_teleportation(
    qubit_factory_pairs: list[tuple[int, int]],
    circuit_moment: float,
    execution_log: list,
    aod_id: int,
):
    factories = [factory_id for _, factory_id in qubit_factory_pairs]
    qubits = [qubit for qubit, _ in qubit_factory_pairs]
    write_execution_log(
        execution_log,
        circuit_moment,
        factories,
        "CNOT",
        aod_assignment=aod_id,
        targets=qubits,
    )
    return circuit_moment + CNOT_TIME


def execute_movement(
    routing_batches: list,
    execution_log: list,
    circuit_moment: float,
    aod_earliest_available_time: list[float],
    move_type: str = "move",
) -> float:
    move_time_idx_pairs = []
    for i, batches in enumerate(routing_batches):
        max_movement_time = 0.0
        for _, x_q, y_q, factory_id, x_f, y_f in batches:
            max_movement_time = max(
                max_movement_time, move_duration(x_q, y_q, x_f, y_f)
            )
        move_time_idx_pairs.append((max_movement_time, i))
    # Sort by movement time descending
    move_time_idx_pairs.sort(reverse=True)
    end_time = 0
    for movement_time, idx in move_time_idx_pairs:
        # Assign to the earliest available AOD
        aod_idx = aod_earliest_available_time.index(min(aod_earliest_available_time))
        start_time = max(circuit_moment, aod_earliest_available_time[aod_idx])
        aod_earliest_available_time[aod_idx] = start_time + movement_time
        end_time = max(end_time, aod_earliest_available_time[aod_idx])
        move_vecs = []
        factories = []
        for _, x_q, y_q, factory_id, x_f, y_f in routing_batches[idx]:
            factories.append(factory_id)
            move_vecs.append([f"({x_f},{y_f})", f"({x_q},{y_q})"])

        write_execution_log(
            execution_log,
            start_time,
            factories,
            move_type,
            targets=None,
            movement_time=movement_time,
            move_vecs=move_vecs,
            aod_assignment=aod_idx,
        )
    return end_time


def insert_s_gate(
    execution_log: list,
    start_time: float,
    qubit: list[int],
    aod_id: int,
):
    """
    Update qubit states
    """
    write_execution_log(
        execution_log,
        start_time,
        [],
        "S",
        aod_assignment=aod_id,
        targets=qubit,
    )


def clean_up_execution_log(
    execution_log: list[dict],
    circuit_moment: float,
) -> list[dict]:
    """
    Clean up execution log by removing completed qubits and inserting S gates if needed.

    Args:
        execution_log: List of (start, end, factory_id, operation
        circuit_moment: current time in the circuit execution
    Returns:
        Updated execution_log with S gates inserted and completed qubits removed
    """  # This function is now integrated into update_qubit_state_per_teleportation
    new_log = []
    for entry in execution_log:
        start = entry["start_time"]
        end = entry["end_time"]
        if end <= circuit_moment:
            # This teleportation has completed; we will handle state updates separately
            new_log.append(entry)
    return new_log


def validate_execution_log(
    execution_log: list[dict],
    magic_state_locations: list[tuple[int, int]],
    n_aods: int,
):
    """
    Validate the execution log to ensure all target qubits have been teleported with correct angles.

    Args:
        execution_log: List of execution event tuples with format:
            (start_time, end_time, factory_id, operation, target, ...)
        target_qubits_angles: List of target angles for each qubit
        logic_qubit_locations: List of locations for logic qubits
        magic_state_locations: List of locations for magic state factories

    Raises:
        AssertionError: If validation fails
    """
    # Check 1: At any time, one factory is involved in at most one operation
    factory_time_intervals = [
        [(-1, "init", None)] for _ in range(len(magic_state_locations))
    ]

    aod_time_intervals = [0 for _ in range(n_aods)]

    for entry in execution_log:
        start_time = entry["start_time"]
        end_time = entry["end_time"]
        factories = entry["factories"]
        operation = entry["operation"]
        aod_assignment = entry["aod_assignment"]
        targets = entry["targets"]

        factories = normalize_factories(factories)

        # Skip barrier and special operations
        if operation in ["Barrier", "RUS_success", "RUS_fail"]:
            continue

        # Check for overlapping time intervals for the same factory
        for factory_id in factories:
            if start_time < factory_time_intervals[factory_id][-1][0]:
                raise AssertionError(
                    f"Factory {factory_id} is involved in overlapping operations {operation} at time {start_time}"
                    f" and {factory_time_intervals[factory_id][-1][1]} at time {factory_time_intervals[factory_id][-1][0]}"
                )

        # check for overlapping time intervals for the same AOD
        if aod_assignment is not None:
            if start_time < aod_time_intervals[aod_assignment]:
                raise AssertionError(
                    f"AOD {aod_assignment} is involved in overlapping operations {operation} at time {start_time}"
                )
            aod_time_intervals[aod_assignment] = end_time

        if operation == "Rz":
            for factory_id in factories:
                for _, fac_op, fac_val in factory_time_intervals[factory_id][-3:]:
                    assert (
                        fac_op == "SE" and fac_val is None
                    ), f"Expected SE before Rz for factory {factory_id}, but got {fac_op} with value {fac_val}"

        if operation == "tmr_fail":
            for factory_id in factories:
                for _, fac_op, fac_val in factory_time_intervals[factory_id][-2:]:
                    assert (
                        fac_op == "SE" and fac_val is not None
                    ), f"Expected SE before TMR failure for factory {factory_id}, but got {fac_op} with value {fac_val}"

        if targets is not None and operation != "S":
            assert len(factories) == len(
                targets
            ), f"Number of factories and targets must match for operation {operation} at time {start_time}"
        else:
            targets = [None] * len(factories)
        for factory_id, target in zip(factories, targets):
            factory_time_intervals[factory_id].append((end_time, operation, target))
