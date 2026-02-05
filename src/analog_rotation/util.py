from src.config import (
    CNOT_TIME,
    TMR_PREPARATION_TIME,
    SE_TIME,
    TMR_P,
    TMR_Q,
)
from src.ds import move_duration


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


def execute_tmr_preparation(
    factory_pool,
    circuit_moment,
    execution_log,
):
    """
    PHASE 2: Execute TMR preparation (2 SE + Rz + 3 SE).

    Args:
        factory_pool: FactoryPool object containing all factories
        circuit_moment: Current circuit execution time
        execution_log: List of execution events

    Returns:
        Updated circuit_moment and execution_log
    """
    # Log TMR preparation for all active factories
    for fac in factory_pool.get_tmr_factories():
        if fac.angle is not None:
            for p in range(TMR_P):
                write_execution_log(
                    execution_log,
                    circuit_moment + p,
                    fac.id,
                    "SE",
                    fac.qubit,
                )
            write_execution_log(
                execution_log,
                circuit_moment + TMR_P,
                fac.id,
                "Rz",
                fac.angle,
            )
            for q in range(TMR_Q):
                write_execution_log(
                    execution_log,
                    circuit_moment + TMR_P + q + 1,
                    fac.id,
                    "SE",
                    fac.qubit,
                )

    circuit_moment += TMR_PREPARATION_TIME

    write_execution_log(
        execution_log,
        circuit_moment,
        -1,
        "Barrier",
        None,
    )

    return circuit_moment, execution_log


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


def execute_movement(
    routing_batches: list,
    execution_log: list,
    circuit_moment: float,
    n_aods: int = 1,
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
    aod_earliest_available_time = [circuit_moment] * n_aods
    for movement_time, idx in move_time_idx_pairs:
        # Assign to the earliest available AOD
        aod_idx = aod_earliest_available_time.index(min(aod_earliest_available_time))
        start_time = aod_earliest_available_time[aod_idx]
        aod_earliest_available_time[aod_idx] += movement_time
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
    return max(aod_earliest_available_time)


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
