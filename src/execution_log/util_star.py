from src.star.config import (
    TMR_P,
    TMR_Q,
)
from src.ds.device_state_star import FactoryPool, Factory
import heapq
from itertools import count
from .util import (
    write_execution_log,
)
from .logical_se import LogicalSEScheduler


def execute_tmr_preparation_pre_rz(
    factories: list,
    circuit_moment,
    execution_log,
    events: list | None = None,
    event_counter: count | None = None,
    aod_id: int = 0,
    logical_se_scheduler: LogicalSEScheduler | None = None,
) -> tuple[float, list]:
    """
    PHASE 2: Execute TMR preparation (2 SE + Rz + 3 SE).

    Args:
        factories: List of Factory objects undergoing TMR preparation
        circuit_moment: Current circuit execution time
        execution_log: List of execution events
        logical_se_scheduler: optional scheduler that piggy-backs logical-qubit
            SE rounds onto each per-cycle factory SE write.

    Returns:
        Updated circuit_moment and execution_log
    """
    if not factories:
        return circuit_moment, execution_log
    # Log TMR preparation for all active factories
    facs = [fac.id for fac in factories]
    targets = [fac.qubit for fac in factories]
    for p in range(TMR_P):
        for fac in factories:
            fac.set_pre_tmr_state()
        write_execution_log(
            execution_log,
            circuit_moment + p,
            facs,
            "SE",
            aod_id,
            targets,
        )
        if logical_se_scheduler is not None:
            logical_se_scheduler.piggyback(circuit_moment + p, aod_id, execution_log)

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
    factories: list[Factory],
    circuit_moment,
    execution_log,
    events: list | None = None,
    event_count: int | None = None,
    aod_id: int = 0,
    logical_se_scheduler: LogicalSEScheduler | None = None,
) -> tuple[float, list]:
    """
    PHASE 2: Execute TMR preparation (2 SE + Rz + 3 SE).

    Args:
        factories: List of Factory objects undergoing TMR preparation
        circuit_moment: Current circuit execution time
        execution_log: List of execution events
        logical_se_scheduler: optional scheduler that piggy-backs logical-qubit
            SE rounds onto each post-Rz factory SE write.

    Returns:
        Updated circuit_moment and execution_log
    """
    # Log TMR preparation for all active factories
    facs = [fac.id for fac in factories]
    targets = [fac.angle for fac in factories]
    write_execution_log(
        execution_log,
        circuit_moment,
        facs,
        "Rz",
        aod_id,
        targets,
    )
    targets = [fac.qubit for fac in factories]
    for q in range(TMR_Q):
        write_execution_log(
            execution_log,
            circuit_moment + q + 1,
            facs,
            "SE",
            aod_id,
            targets,
        )
        if logical_se_scheduler is not None:
            logical_se_scheduler.piggyback(
                circuit_moment + q + 1, aod_id, execution_log
            )

    circuit_moment += TMR_Q + 1

    write_execution_log(
        execution_log,
        circuit_moment,
        [],
        "Barrier",
        aod_id,
        None,
    )

    # Schedule completion event
    if events is not None:
        assert circuit_moment is not None and event_count is not None
        task = {
            "type": "TMR_completion",
            "factory_list": [fac.id for fac in factories],
            "aod_id": aod_id,
        }
        heapq.heappush(events, (circuit_moment, event_count, task))

    return circuit_moment, execution_log


def write_tmr_result_log(
    factory_pool: FactoryPool,
    circuit_moment,
    execution_log,
    factory_id_list: list[int] | None = None,
):
    # Log TMR preparation for all active factories
    if factory_id_list:
        factories = [
            factory_id
            for factory_id in factory_id_list
            if not factory_pool.get_factory_by_id(factory_id).tmr_state
        ]
    else:
        factories = [
            factory.id for factory in factory_pool.factories if not factory.tmr_state
        ]
    write_execution_log(
        execution_log, circuit_moment, factories, "TMR_fail", None, None
    )

    circuit_moment += TMR_P

    return execution_log
