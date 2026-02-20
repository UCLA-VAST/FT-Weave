from src.config import (
    CNOT_TIME,
    TMR_PREPARATION_TIME,
    SE_TIME,
    TMR_P,
    TMR_Q,
)
from src.ds import move_duration
from src.ds.device_state import FactoryPool
from collections import defaultdict
import heapq
from itertools import count


def write_execution_log(
    execution_log: list,
    start_time: float,
    factory_id: int,
    operation: str,
    target: int | float | None = None,
    movement_time: float = 0,
    move_vecs: list[str] | None = None,
    aod_assignment: int = 0,
):
    if operation == "SE":
        end_time = start_time + SE_TIME
    elif operation == "CNOT":
        end_time = start_time + CNOT_TIME
    elif operation == "Rz":
        end_time = start_time + 1
    elif operation == "S":
        end_time = start_time + SE_TIME
    elif operation in ["move", "return_move"]:
        end_time = start_time + movement_time
    else:
        end_time = start_time
    if move_vecs:
        execution_log.append(
            (
                start_time,
                end_time,
                factory_id,
                operation,
                target,
                move_vecs,
                aod_assignment,
            )
        )
    else:
        execution_log.append(
            (
                start_time,
                end_time,
                factory_id,
                operation,
                target,
            )
        )


def execute_tmr_preparation_pre_rz(
    factories: list,
    circuit_moment,
    execution_log,
    events: list | None = None,
    event_counter: count | None = None,
    aod_id: int = 0,
):
    """
    PHASE 2: Execute TMR preparation (2 SE + Rz + 3 SE).

    Args:
        factories: List of Factory objects undergoing TMR preparation
        circuit_moment: Current circuit execution time
        execution_log: List of execution events

    Returns:
        Updated circuit_moment and execution_log
    """
    # Log TMR preparation for all active factories
    for fac in factories:
        for p in range(TMR_P):
            fac.set_pre_tmr_state()
            write_execution_log(
                execution_log,
                circuit_moment + p,
                fac.id,
                "SE",
                fac.qubit,
            )

    circuit_moment += TMR_P

    if events is not None:
        assert circuit_moment is not None and event_counter is not None
        task = {
            "type": "TMR_pre_RZ_completion",
            "factory_list": [fac.id for fac in factories],
            "aod_id": aod_id,
        }
        heapq.heappush(events, (circuit_moment, next(event_counter), task))

    return circuit_moment, execution_log


def execute_tmr_preparation_rz(
    factories: list,
    circuit_moment,
    execution_log,
    events: list | None = None,
    event_counter: count | None = None,
    aod_id: int = 0,
):
    """
    PHASE 2: Execute TMR preparation (2 SE + Rz + 3 SE).

    Args:
        factories: List of Factory objects undergoing TMR preparation
        circuit_moment: Current circuit execution time
        execution_log: List of execution events

    Returns:
        Updated circuit_moment and execution_log
    """
    # Log TMR preparation for all active factories
    for fac in factories:
        if fac.angle is not None:
            write_execution_log(
                execution_log,
                circuit_moment,
                fac.id,
                "Rz",
                fac.angle,
            )
            for q in range(TMR_Q):
                write_execution_log(
                    execution_log,
                    circuit_moment + q + 1,
                    fac.id,
                    "SE",
                    fac.qubit,
                )

    circuit_moment += TMR_Q + 1

    write_execution_log(
        execution_log,
        circuit_moment,
        -1,
        "Barrier",
        None,
    )

    # Schedule completion event
    if events is not None:
        assert circuit_moment is not None and event_counter is not None
        task = {
            "type": "TMR_completion",
            "factory_list": [fac.id for fac in factories],
            "aod_id": aod_id,
        }
        heapq.heappush(events, (circuit_moment, next(event_counter), task))

    return circuit_moment, execution_log


def write_tmr_result_log(
    factory_pool: FactoryPool,
    circuit_moment,
    execution_log,
    factory_id_list: list[int] | None = None,
):
    # Log TMR preparation for all active factories
    if factory_id_list:
        for factory_id in factory_id_list:
            factory = factory_pool.get_factory_by_id(factory_id)
            if not factory.tmr_state:
                write_execution_log(
                    execution_log, circuit_moment, factory.id, "TMR_fail"
                )
    else:
        for factory in factory_pool.factories:
            if not factory.tmr_state:
                write_execution_log(
                    execution_log, circuit_moment, factory.id, "TMR_fail"
                )

    circuit_moment += TMR_P

    return execution_log


def write_rus_result_log(
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    circuit_moment: float,
    execution_log: list,
):
    for (qubit, factory_id), injection_result in zip(
        qubit_factory_pairs, rus_simulation
    ):
        if injection_result:
            result = "RUS_success"
        else:
            result = "RUS_fail"
        write_execution_log(
            execution_log,
            circuit_moment,
            factory_id,
            result,
            qubit,
        )

    return execution_log


def execute_rus_teleportation(
    qubit_factory_pairs: list[tuple[int, int]],
    circuit_moment: float,
    execution_log: list,
):
    for qubit, factory_id in qubit_factory_pairs:
        write_execution_log(
            execution_log,
            circuit_moment,
            factory_id,
            "CNOT",
            qubit,
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
    # if move_type == "return_move":
    #     print("circuit_moment: ", circuit_moment)
    #     print("move_time_idx_pairs")
    #     print(move_time_idx_pairs)
    #     input()
    # Sort by movement time descending
    move_time_idx_pairs.sort(reverse=True)
    end_time = 0
    for movement_time, idx in move_time_idx_pairs:
        # Assign to the earliest available AOD
        aod_idx = aod_earliest_available_time.index(min(aod_earliest_available_time))
        start_time = max(circuit_moment, aod_earliest_available_time[aod_idx])
        aod_earliest_available_time[aod_idx] = start_time + movement_time
        end_time = max(end_time, aod_earliest_available_time[aod_idx])
        for _, x_q, y_q, factory_id, x_f, y_f in routing_batches[idx]:

            movement_strs = [f"({x_f},{y_f})", f"({x_q},{y_q})"]
            write_execution_log(
                execution_log,
                start_time,
                factory_id,
                move_type,
                target=None,
                movement_time=movement_time,
                move_vecs=movement_strs,
                aod_assignment=aod_idx,
            )
    return end_time


def insert_s_gate(
    execution_log: list,
    start_time: float,
    factory_id: int,
    qubit: int,
):
    """
    Update qubit states
    """
    write_execution_log(
        execution_log,
        start_time,
        factory_id,
        "S",
        qubit,
    )


def clean_up_execution_log(
    execution_log: list[tuple],
    circuit_moment: float,
) -> list[tuple]:
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
        start, end, factory_id, operation, value = entry[:5]
        if end <= circuit_moment:
            # This teleportation has completed; we will handle state updates separately
            new_log.append(entry)
    return new_log


def validate_execution_log(
    execution_log: list[tuple],
    magic_state_locations: list[tuple[int, int]],
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

    for entry in execution_log:
        start_time = entry[0]
        end_time = entry[1]
        factory_id = entry[2]
        operation = entry[3]
        value = entry[4]

        # Skip barrier and special operations
        if operation in ["Barrier", "RUS_success", "RUS_fail"]:
            continue

        # Check for overlapping time intervals for the same factory
        if start_time < factory_time_intervals[factory_id][-1][0]:
            raise AssertionError(
                f"Factory {factory_id} is involved in overlapping operations {operation} at time {start_time}"
                f" and {factory_time_intervals[factory_id][-1][1]} at time {factory_time_intervals[factory_id][-1][0]}"
            )

        if operation == "Rz":
            for _, fac_op, fac_val in factory_time_intervals[factory_id][-3:]:
                assert (
                    fac_op == "SE" and fac_val is None
                ), f"Expected SE before Rz for factory {factory_id}, but got {fac_op} with value {fac_val}"

        if operation == "tmr_fail" or operation == "move":
            for _, fac_op, fac_val in factory_time_intervals[factory_id][-2:]:
                assert (
                    fac_op == "SE" and fac_val is not None
                ), f"Expected SE before TMR failure for factory {factory_id}, but got {fac_op} with value {fac_val}"

        factory_time_intervals[factory_id].append((end_time, operation, value))
