import logging

import numpy as np
import heapq
from itertools import count

logger = logging.getLogger(__name__)
from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
)
from .config import TMR_PREPARATION_TIME
from .tmr.tmr_scheduling import schedule_tmr_round
from .tmr.tmr_assignment import release_useless_factories
from .tmr.util import (
    update_factory_states_post_tmr,
)

from ..ds import FactoryPool, QubitAngleTracker
from .rus import (
    update_qubit_state_per_teleportation,
)

from ..execution_log import (
    execute_movement,
    execute_rus_teleportation,
    execute_tmr_preparation_pre_rz,
    execute_tmr_preparation_rz,
    write_tmr_result_log,
    write_rus_result_log,
    clean_up_execution_log,
    validate_execution_log,
    LogicalSEScheduler,
)

from .rus.rus_teleportaion import rus_teleportation
from .rus.rus_post_teleportation import rus_post_teleportation
from . import config as star_cfg


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
def factory_angle_execution_parallel(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    code_distance: int,
    column_based_placement: bool = True,
    n_aods: int = 1,
    n_aods_se: int = 1,
    consider_skip_rus: int = 0,  # 0: no skip, 1: partial skip, 2: aggressive skip
    tmr_assignment_method: str = "matching",
    trivial_return: bool = True,
    decompose_move: bool = False,
    rng: np.random.Generator | None = None,
    save_log: bool = False,
    log_path: str | None = None,
    logical_se_interval: int | None = None,
    prepare_lookahead_angles: bool = True,
    allocation_level_range: range | None = None,
):
    """
    Execute angle preparation with parallel TMR and RUS operations using time-stepped simulation.

    This version allows TMR and RUS to run concurrently on different AODs, reducing idle time.

    Args:
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory
        column_based_placement: Whether to use column-based placement
        n_aods: Number of AODs available for parallel operations
        consider_skip_rus: RUS skipping strategy (0: no skip, 1: partial, 2: aggressive)
        tmr_assignment_method: Method for TMR assignment
        trivial_return: Whether to use trivial return routing
        decompose_move: Whether to decompose return moves
        rng: Random number generator

    Returns:
        circuit_moment: Total circuit execution time
        execution_log: List of (time, factory_id, operation, qubit) for visualization
    """
    assert (
        not decompose_move
    ), "Decomposing moves in parallel execution is not supported yet"
    if rng is None:
        rng = np.random.default_rng()
    if n_aods_se > n_aods:
        raise ValueError("Number of AODs for SE cannot exceed total number of AODs")
    logger.info(
        f"Starting with {len(target_qubits_angles)} qubits and {len(magic_state_locations)} factories"
    )

    # Initialize qubit trackers
    qubit_trackers = {
        qubit: QubitAngleTracker(
            qubit_id=qubit,
            target_angle=round(theta, 7),
            factory_limit=len(magic_state_locations),
            code_distance=code_distance,
        )
        for qubit, theta in target_qubits_angles.items()
    }

    factory_pool.set_locations(magic_state_locations)

    execution_log = []
    aod_in_use = [False] * n_aods_se
    aod_earliest_available_time = [0.0] * n_aods
    circuit_moment = 0.0

    se_interval = (
        logical_se_interval
        if logical_se_interval is not None
        else star_cfg.LOGICAL_SE_INTERVAL
    )
    logical_se_scheduler: LogicalSEScheduler | None = None
    if se_interval is not None:
        logical_se_scheduler = LogicalSEScheduler(
            qubit_ids=range(len(logic_qubit_locations)),
            interval=se_interval,
            start_time=circuit_moment,
        )

    def aod_se_avaliable():
        max_time = float("inf")
        aod_id = -1
        for i in range(n_aods_se):
            if not aod_in_use[i] and aod_earliest_available_time[i] < max_time:
                max_time = aod_earliest_available_time[i]
                aod_id = i
        return aod_id

    # ========================================================================
    # MAIN EXECUTION LOOP (TIME-STEPPED)
    # ========================================================================

    # Initial TMR scheduling
    counter = count()  # Unique counter to break ties in heapq

    # Event queue: priority queue with (time, event) tuples
    events = []

    tmr_start_time = set()

    def add_event(time, count, task_type, **kwargs):
        if task_type == "TMR_start" and time in tmr_start_time:
            return  # Avoid scheduling duplicate TMR_start events at the same time
        task = {"type": task_type, **kwargs}
        heapq.heappush(events, (time, count, task))
        logger.debug(
            f"Added {task_type} event at t={time}, total events in queue: {len(events)}"
        )

    def handle_rus_start(local_moment: float, event: dict) -> float:
        """Handle scheduling and dispatch for RUS start events."""
        logger.debug(f"Event: RUS_start at t={local_moment}")
        qubit_factory_pairs, routing_batches = rus_teleportation(
            qubit_trackers,
            factory_pool,
            logic_qubit_locations,
            consider_skip_rus=consider_skip_rus,
            n_aods=n_aods,
            total_qubits=len(target_qubits_angles),
        )
        logger.debug(
            f"RUS teleportation scheduled for {qubit_factory_pairs} qubits at t={local_moment}"
        )
        if not (qubit_factory_pairs and routing_batches):
            aod_in_use[event["aod_id"]] = False
            add_event(local_moment, next(counter), "TMR_start")
            return local_moment

        local_moment = execute_movement(
            routing_batches,
            execution_log,
            local_moment,
            aod_earliest_available_time,
        )
        aod_idx = aod_earliest_available_time.index(min(aod_earliest_available_time))
        local_moment = execute_rus_teleportation(
            qubit_factory_pairs,
            local_moment,
            execution_log,
            aod_idx,
            logical_se_scheduler=logical_se_scheduler,
        )
        aod_earliest_available_time[aod_idx] = local_moment
        logger.debug(
            f"RUS teleportation executed, update AOD available time: {aod_earliest_available_time}"
        )
        add_event(
            local_moment,
            next(counter) - 20,
            "RUS_teleportation",
            qubit_factory_pairs=qubit_factory_pairs,
            routing_batches=routing_batches,
            aod_id=event["aod_id"],
        )
        return local_moment

    def handle_rus_teleportation(local_moment: float, event: dict) -> float:
        """Handle RUS result simulation and post-teleportation routing."""
        logger.debug(f"Event: RUS_teleportation at t={local_moment}")
        qubit_factory_pairs = event["qubit_factory_pairs"]
        rus_simulation = simulate_RUS_injection(
            event["qubit_factory_pairs"], factory_pool, rng
        )
        write_rus_result_log(
            event["qubit_factory_pairs"],
            rus_simulation,
            local_moment,
            execution_log,
        )
        local_moment = update_qubit_state_per_teleportation(
            qubit_factory_pairs,
            rus_simulation,
            qubit_trackers,
            execution_log,
            local_moment,
            aod_id=event["aod_id"],
            logical_se_scheduler=logical_se_scheduler,
        )
        return_routing_batches = rus_post_teleportation(
            event["routing_batches"],
            factory_pool,
            trivial_return=trivial_return,
            decompose_move=False,
        )
        local_moment = execute_movement(
            return_routing_batches,
            execution_log,
            local_moment,
            aod_earliest_available_time,
            move_type="return_move",
        )
        add_event(
            local_moment,
            next(counter) - 60,
            "RUS_finish",
            aod_id=event["aod_id"],
            factory_list=[f for q, f in event["qubit_factory_pairs"]],
        )
        return local_moment

    add_event(0, next(counter), "TMR_start")

    while len(qubit_trackers) > 0:

        if not events:
            for log in execution_log:
                print(log)
            for factory in factory_pool.factories:
                print(factory)
            for qubit, tracker in qubit_trackers.items():
                print(f"Qubit {qubit}: {tracker}")
            assert (
                False
            ), f"Event queue is empty but {len(qubit_trackers)} qubits remain unprocessed"

        circuit_moment, _, event = heapq.heappop(events)
        logger.debug(
            f"Main loop: t={circuit_moment}, remaining qubits={len(qubit_trackers)}, total pending={len(events)}"
        )

        # Process events at current time
        local_moment = circuit_moment
        if event["type"] == "TMR_start":
            logger.debug(f"Event: TMR_start at t={local_moment}")

            aod_id = aod_se_avaliable()
            if aod_id == -1:
                logger.debug("No AOD available for TMR, waiting for next event")
                continue  # No AOD available, wait for next event

            local_moment = max(local_moment, aod_earliest_available_time[aod_id])
            factory_list = factory_pool.get_idle_factories()
            logger.debug(
                f"Before TMR: {len(factory_list)} idle factories available for TMR"
            )
            for factory in factory_pool.factories:
                logger.debug(
                    f"Factory {factory.id}: state={factory.state}, qubit={factory.qubit}"
                )
            if not factory_list:
                # for factory in factory_pool.factories:
                #     print(factory)
                # for qubit, tracker in qubit_trackers.items():
                #     print(f"Qubit {qubit}: {tracker}")
                logger.debug(
                    "No idle factories available for TMR, waiting for next event"
                )
                continue  # No idle factories, wait for next event
            # Execute next round of TMR preparation

            if len(factory_list):
                aod_earliest_available_time[aod_id] = (
                    local_moment + TMR_PREPARATION_TIME
                )  # TMR preparation takes fixed time. Block first to prevent scheduling conflicts.
                logger.debug(
                    "AOD %d assigned for TMR, will be available at t=%.1f",
                    aod_id,
                    aod_earliest_available_time[aod_id],
                )
                local_moment, execution_log = execute_tmr_preparation_pre_rz(
                    factory_list,
                    local_moment,
                    execution_log,
                    events=events,
                    event_counter=counter,
                    aod_id=aod_id,
                    logical_se_scheduler=logical_se_scheduler,
                )
                aod_in_use[aod_id] = True

        elif event["type"] == "TMR_pre_RZ_completion":
            logger.debug(f"Event: TMR_pre_RZ_completion at t={local_moment}")
            factory_list = schedule_tmr_round(
                factory_pool,
                qubit_trackers,
                logic_qubit_locations,
                code_distance=code_distance,
                column_based_placement=column_based_placement,
                tmr_assignment_method=tmr_assignment_method,
                factories_list=event["factory_list"],
                prepare_lookahead_angles=prepare_lookahead_angles,
                allocation_level_range=allocation_level_range,
            )
            logger.debug(f"Scheduled {len(factory_list)} factories for TMR RZ")
            local_moment, execution_log = execute_tmr_preparation_rz(
                factory_list,
                local_moment,
                execution_log,
                events=events,
                event_count=next(counter),
                aod_id=event["aod_id"],
                logical_se_scheduler=logical_se_scheduler,
            )
            # aod_earliest_available_time[event["aod_id"]] = local_moment
            logger.debug(
                f"Added TMR_completion event at t={local_moment}, total events in queue: {len(events)}"
            )

        elif event["type"] == "TMR_completion":
            factory_id_list = event["factory_list"]
            logger.debug(
                f"Event: TMR_completion with {len(factory_id_list)} factories at t={local_moment}"
            )
            simulate_TMR_preparation(factory_pool, rng, factory_id_list)
            execution_log = write_tmr_result_log(
                factory_pool, circuit_moment, execution_log, factory_id_list
            )
            # Update factory state
            factory_list = [
                factory_pool.get_factory_by_id(fid) for fid in factory_id_list
            ]
            update_factory_states_post_tmr(factory_list, factory_pool, qubit_trackers)
            logger.debug("Updated factory states post TMR completion")
            for factory in factory_list:
                if not factory.tmr_state:
                    logger.debug(f"Factory {factory.id} failed TMR. will be reset")
            add_event(local_moment, next(counter), "TMR_start")

            add_event(
                local_moment, next(counter) - 40, "RUS_start", aod_id=event["aod_id"]
            )

        elif event["type"] == "RUS_start":
            local_moment = handle_rus_start(local_moment, event)
        elif event["type"] == "RUS_teleportation":
            local_moment = handle_rus_teleportation(local_moment, event)

        elif event["type"] == "RUS_finish":
            logger.debug(
                f"Event: RUS_finish at t={local_moment}, {len(qubit_trackers)} qubits remaining"
            )
            factory_pool.free_factories(event["factory_list"])

            release_useless_factories(
                factory_pool,
                qubit_trackers,
            )
            if logical_se_scheduler is not None:
                logical_se_scheduler.force_due(
                    local_moment, event["aod_id"], execution_log
                )

            add_event(local_moment, next(counter), "RUS_start", aod_id=event["aod_id"])
            add_event(local_moment, next(counter), "TMR_start")
        else:
            raise ValueError(f"Unknown event type: {event['type']}")
        # input()
        circuit_moment = local_moment

    # Final cleanup: reassign any remaining factories
    assert not qubit_trackers, "All qubits should be completed at this point"
    execution_log = clean_up_execution_log(execution_log, circuit_moment)
    logger.debug(
        f"Execution finished in {circuit_moment} time units, {len(execution_log)} events logged"
    )
    execution_log = sorted(
        execution_log,
        key=lambda x: (
            x.get("start_time", 0),
            x.get("end_time", 0),
            str(x.get("factories")),
            str(x.get("operation")),
        ),
    )

    validate_execution_log(execution_log, magic_state_locations, n_aods)
    if save_log and log_path is not None:
        with open(log_path, "w") as f:
            for entry in execution_log:
                f.write(str(entry) + "\n")
    return circuit_moment, execution_log
