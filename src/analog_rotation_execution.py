from itertools import product
import numpy as np
import heapq
from itertools import count
from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
)
from .config import TMR_PREPARATION_TIME
from .tmr.tmr_scheduling import schedule_tmr_round
from .tmr.tmr_assignment import reassign_factories
from .tmr.util import (
    update_factory_states_post_tmr,
)

from .ds import FactoryPool, QubitAngleTracker
from .rus import (
    update_qubit_state_per_teleportation,
)

from .analog_rotation import (
    execute_movement,
    execute_rus_teleportation,
    execute_tmr_preparation_pre_rz,
    execute_tmr_preparation_rz,
    write_tmr_result_log,
    write_rus_result_log,
    clean_up_execution_log,
    validate_execution_log,
)

from .rus.rus_teleportaion import rus_teleportation
from .rus.rus_post_teleportation import rus_post_teleportation


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================


def get_microarchitecture(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple[int, int],
    placement: str = "seperate_region",
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Generate microarchitecture layout for logic qubits and magic state factories.
    Args:
        n_qubits: Total number of logical qubits
        n_factories: Total number of magic state factories
        qubit_layout: (n_columns, n_rows) layout of the qubit grid
        placement: Placement method for factories ("seperate_region", "col_based", "checkerboard")
    Returns:
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory
    """
    n_columns, n_rows = qubit_layout
    if placement == "seperate_region_row":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x, n_rows + y) for x, y in product(range(n_columns), range(n_rows))
        ]
    elif placement == "seperate_region_col":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x + n_columns, y) for x, y in product(range(n_columns), range(n_rows))
        ]
    elif placement == "row_based":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(0, 2 * n_rows, 2))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(n_columns), range(1, 2 * n_rows + 1, 2))
        ]
    elif placement == "col_based":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(0, 2 * n_columns, 2), range(n_rows))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(1, 2 * n_columns + 1, 2), range(n_rows))
        ]
    elif placement == "checkerboard":
        logic_qubit_locations = []
        magic_state_locations = []
        for x, y in product(range(n_columns * 2), range(n_rows)):
            if (x + y) % 2 == 0:
                logic_qubit_locations.append((x, y))
            else:
                magic_state_locations.append((x, y))
    else:
        raise ValueError(f"Unknown placement strategy: {placement}")
    return logic_qubit_locations[:n_qubits], magic_state_locations[:n_factories]


def factory_angle_execution(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    column_based_placement: bool = True,
    n_aods: int = 1,
    consider_skip_rus: int = 0,  # 0: no skip, 1: partial skip, 2: aggressive skip
    tmr_assignment_method: str = "matching",
    trivial_return: bool = True,
    decompose_move: bool = True,
    rng: np.random.Generator | None = None,
):
    """
    Execute angle preparation on magic state factories with lookahead optimization.

    Args:
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory
        facoty_qubit_map: give (x,y) return 0 if no qubit and factory, 1 if qubit, 2 if factory
        n_aods: Number of AODs available for parallel operations

    Returns:
        circuit_moment: Total circuit execution time
        execution_log: List of (time, factory_id, operation, qubit) for visualization
    """
    if rng is None:
        rng = np.random.default_rng()
    # Map qubit_id to QubitAngleTracker
    qubit_trackers = {
        qubit: QubitAngleTracker(qubit_id=qubit, target_angle=round(theta, 7))
        for qubit, theta in target_qubits_angles.items()
    }

    # Create FactoryPool with n_factories

    factory_pool.set_locations(magic_state_locations)

    # Log for visualization: (start_time, end_time, factory_id, operation, qubit)
    execution_log = []
    aod_earliest_available_time = [0.0] * n_aods

    circuit_moment = 0

    # ========================================================================
    # MAIN EXECUTION LOOP
    # ========================================================================

    while len(qubit_trackers):
        # PHASE 1: Assign idle factories to prepare angles
        factory_list = factory_pool.get_idle_factories()

        # PHASE 2: Execute TMR preparation
        circuit_moment, execution_log = execute_tmr_preparation_pre_rz(
            factory_list,
            circuit_moment,
            execution_log,
        )

        schedule_tmr_round(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
            column_based_placement,
            tmr_assignment_method=tmr_assignment_method,
        )
        circuit_moment, execution_log = execute_tmr_preparation_rz(
            factory_list,
            circuit_moment,
            execution_log,
        )
        simulate_TMR_preparation(factory_pool, rng)

        update_factory_states_post_tmr(factory_list, factory_pool, qubit_trackers)

        execution_log = write_tmr_result_log(
            factory_pool,
            circuit_moment,
            execution_log,
        )
        while True:
            # qubit_factory_pairs: list[tuple(qubit, factory_id)]
            # routing batch: list of tuple (qubit, x_q, y_q, factory_id, x_f, y_f)
            qubit_factory_pairs, routing_batches = rus_teleportation(
                qubit_trackers,
                factory_pool,
                logic_qubit_locations,
                consider_skip_rus=consider_skip_rus,
                n_aods=n_aods,
                total_qubits=len(target_qubits_angles),
            )

            if not qubit_factory_pairs or not routing_batches:
                break

            circuit_moment = execute_movement(
                routing_batches,
                execution_log,
                circuit_moment,
                aod_earliest_available_time,
            )

            circuit_moment = execute_rus_teleportation(
                qubit_factory_pairs, circuit_moment, execution_log
            )

            # return factories qubit to empty spot
            return_routing_batches = rus_post_teleportation(
                routing_batches,
                factory_pool,
                trivial_return=trivial_return,
                decompose_move=decompose_move,
            )

            circuit_moment = execute_movement(
                return_routing_batches,
                execution_log,
                circuit_moment,
                aod_earliest_available_time,
                move_type="return_move",
            )
            rus_simulation = simulate_RUS_injection(
                qubit_factory_pairs, factory_pool, rng
            )

            execution_log = write_rus_result_log(
                qubit_factory_pairs, rus_simulation, circuit_moment, execution_log
            )

            circuit_moment = update_qubit_state_per_teleportation(
                qubit_factory_pairs,
                rus_simulation,
                qubit_trackers,
                factory_pool,
                execution_log,
                circuit_moment,
            )

            if len(qubit_trackers) == 0:
                break

        reassign_factories(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
        )

        # Add time for injection attempts (CNOT + SE per injection)
        execution_log.append(
            (
                circuit_moment,
                circuit_moment,
                -1,
                "Barrier",
                None,
            )
        )
        # input()
    return circuit_moment, execution_log


def factory_angle_execution_parallel(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    column_based_placement: bool = True,
    n_aods: int = 1,
    n_aods_se: int = 1,
    consider_skip_rus: int = 0,  # 0: no skip, 1: partial skip, 2: aggressive skip
    tmr_assignment_method: str = "matching",
    trivial_return: bool = True,
    decompose_move: bool = False,
    rng: np.random.Generator | None = None,
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
    print(
        f"[factory_angle_execution_parallel] Starting with {len(target_qubits_angles)} qubits and {len(magic_state_locations)} factories"
    )

    # Initialize qubit trackers
    qubit_trackers = {
        qubit: QubitAngleTracker(qubit_id=qubit, target_angle=round(theta, 7))
        for qubit, theta in target_qubits_angles.items()
    }

    factory_pool.set_locations(magic_state_locations)

    execution_log = []
    aod_in_use = [False] * n_aods_se
    aod_earliest_available_time = [0.0] * n_aods
    circuit_moment = 0.0

    # Event queue: priority queue with (time, event) tuples
    events = []

    # ========================================================================
    # MAIN EXECUTION LOOP (TIME-STEPPED)
    # ========================================================================

    # Initial TMR scheduling
    print("[factory_angle_execution_parallel] Initial TMR scheduling started")
    aod_in_use[0] = True
    counter = count()  # Unique counter to break ties in heapq
    circuit_moment, execution_log = execute_tmr_preparation_pre_rz(
        factory_pool.get_idle_factories(),
        circuit_moment,
        execution_log,
        events=events,
        event_counter=counter,
        aod_id=0,
    )
    print(
        f"[factory_angle_execution_parallel] Initial TMR scheduling completed at t={circuit_moment}, events scheduled: {len(events)}"
    )
    while len(qubit_trackers) > 0:

        if not events:
            assert False, "Event queue is empty but qubits remain unprocessed"

        circuit_moment, _, event = heapq.heappop(events)
        print(
            f"[factory_angle_execution_parallel] Main loop: t={circuit_moment}, remaining qubits={len(qubit_trackers)}, total pending={len(events)}"
        )

        # Process events at current time
        local_moment = circuit_moment
        if event["type"] == "TMR_start":
            print(
                f"[factory_angle_execution_parallel] Event: TMR_start at t={local_moment}"
            )

            reassign_factories(
                factory_pool,
                qubit_trackers,
                logic_qubit_locations,
            )

            max_time = float("inf")
            aod_id = -1
            for i in range(n_aods_se):
                if not aod_in_use[i] and aod_earliest_available_time[i] < max_time:
                    max_time = aod_earliest_available_time[i]
                    aod_id = i
            if aod_id != -1:
                print("aod_in_use: ", aod_in_use)
                print("aod_earliest_available_time: ", aod_earliest_available_time)
                print("aod_id: ", aod_id)
                print("local_moment: ", local_moment)
                factory_list = factory_pool.get_idle_factories()
                if factory_list:
                    # Execute next round of TMR preparation
                    aod_earliest_available_time[aod_id] = (
                        local_moment + TMR_PREPARATION_TIME
                    )
                    local_moment, execution_log = execute_tmr_preparation_pre_rz(
                        factory_list,
                        local_moment,
                        execution_log,
                        events=events,
                        event_counter=counter,
                        aod_id=aod_id,
                    )
                    aod_in_use[aod_id] = True
                    print(
                        f"[factory_angle_execution_parallel] Added TMR_pre_RZ event at t={local_moment}, total events in queue: {len(events)}"
                    )
                    print("idle_factories: ", [f.id for f in factory_list])

            # Schedule TMR completion
            # Schedule next TMR round for factories that need preparation
            # aod_id = event["aod_id"]
            # factory_list = factory_pool.get_idle_factories()
            # if factory_list:
            #     aod_in_use[aod_id] = True
            #     local_moment, execution_log = execute_tmr_preparation_pre_rz(
            #         factory_list,
            #         local_moment,
            #         execution_log,
            #         events=events,
            #         event_counter=counter,
            #         aod_id=aod_id,
            #     )
            #     aod_earliest_available_time[aod_id] = local_moment
            #     print(
            #         f"[factory_angle_execution_parallel] Added TMR_pre_RZ event, total events in queue: {len(events)}"
            #     )
            #     print("idle_factories: ", [f.id for f in factory_list])

        elif event["type"] == "TMR_completion":
            print(
                f"[factory_angle_execution_parallel] Event: TMR_completion with {len(event["factory_list"])} factories  at t={local_moment}"
            )
            simulate_TMR_preparation(factory_pool, rng, event["factory_list"])
            execution_log = write_tmr_result_log(
                factory_pool, circuit_moment, execution_log, event["factory_list"]
            )
            # Update factory state
            factory_list = [
                factory_pool.get_factory_by_id(fid) for fid in event["factory_list"]
            ]
            update_factory_states_post_tmr(factory_list, factory_pool, qubit_trackers)
            # execute new run of TMR if extra AODs are available.
            # The AOD working on the current TMR completion can not
            # immediately start the next TMR round as it is responsible
            # for the RUS movement
            max_time = float("inf")
            aod_id = -1
            for i in range(n_aods_se):
                if not aod_in_use[i] and aod_earliest_available_time[i] < max_time:
                    max_time = aod_earliest_available_time[i]
                    aod_id = i
            if aod_id != -1:
                print("aod_in_use: ", aod_in_use)
                print("aod_earliest_available_time: ", aod_earliest_available_time)
                print("aod_id: ", aod_id)
                print("local_moment: ", local_moment)
                # if max_time <= local_moment:
                if False:
                    factory_list = factory_pool.get_idle_factories()
                    if factory_list:
                        # Execute next round of TMR preparation
                        local_moment_after_tmr, execution_log = (
                            execute_tmr_preparation_pre_rz(
                                factory_list,
                                local_moment,
                                execution_log,
                                events=events,
                                event_counter=counter,
                                aod_id=aod_id,
                            )
                        )
                        aod_earliest_available_time[aod_id] = local_moment_after_tmr
                        aod_in_use[aod_id] = True
                        print(
                            f"[factory_angle_execution_parallel] Added TMR_pre_RZ event at t={local_moment_after_tmr}, total events in queue: {len(events)}"
                        )
                else:
                    heapq.heappush(
                        events,
                        (
                            max(aod_earliest_available_time[aod_id], local_moment),
                            next(counter),
                            {
                                "type": "TMR_start",
                                "aod_id": aod_id,
                            },
                        ),
                    )
                    print(
                        f"[factory_angle_execution_parallel] Added TMR_start event at t={aod_earliest_available_time[aod_id]}, total events in queue: {len(events)} "
                    )

            # todo: invoke RUS teleportation
            # Execute RUS teleportation with routing and
            qubit_factory_pairs, routing_batches = rus_teleportation(
                qubit_trackers,
                factory_pool,
                logic_qubit_locations,
                consider_skip_rus=consider_skip_rus,
                n_aods=n_aods,
                total_qubits=len(target_qubits_angles),
            )
            print(
                f"[factory_angle_execution_parallel] RUS teleportation: {len(qubit_factory_pairs)} pairs, {len(routing_batches)} batches"
            )

            if not qubit_factory_pairs or not routing_batches:
                print(
                    "[factory_angle_execution_parallel] No RUS pairs or routing batches, skipping"
                )
                continue

            # Execute movement to factories
            print(
                "aod_earliest_available_time before movement: ",
                aod_earliest_available_time,
            )
            local_moment = execute_movement(
                routing_batches,
                execution_log,
                local_moment,
                aod_earliest_available_time,
            )

            local_moment = execute_rus_teleportation(
                qubit_factory_pairs, local_moment, execution_log
            )
            # Schedule finish event
            heapq.heappush(
                events,
                (
                    local_moment,
                    next(counter),
                    {
                        "type": "RUS_teleportation",
                        "qubit_factory_pairs": qubit_factory_pairs,
                        "routing_batches": routing_batches,
                        "aod_id": event["aod_id"],
                    },
                ),
            )
            print(
                f"[factory_angle_execution_parallel] Added RUS_teleportation event at t={local_moment}, total events in queue: {len(events)} (routing batches: {qubit_factory_pairs})"
            )

        elif event["type"] == "TMR_pre_RZ_completion":
            print(
                f"[factory_angle_execution_parallel] Event: TMR_pre_RZ_completion at t={local_moment}"
            )
            factory_list = schedule_tmr_round(
                factory_pool,
                qubit_trackers,
                logic_qubit_locations,
                column_based_placement=column_based_placement,
                tmr_assignment_method=tmr_assignment_method,
                factories_lists=event["factory_list"],
            )
            print(
                f"[factory_angle_execution_parallel] Scheduled {len(factory_list)} factories for TMR RZ"
            )
            local_moment, execution_log = execute_tmr_preparation_rz(
                factory_list,
                local_moment,
                execution_log,
                events=events,
                event_counter=counter,
                aod_id=event["aod_id"],
            )
            # aod_earliest_available_time[event["aod_id"]] = local_moment
            print(
                f"[factory_angle_execution_parallel] Added TMR_completion event at t={local_moment}, total events in queue: {len(events)}"
            )
        elif event["type"] == "RUS_teleportation":
            print(
                f"[factory_angle_execution_parallel] Event: RUS_teleportation at t={local_moment}"
            )
            qubit_factory_pairs = event["qubit_factory_pairs"]

            print(
                f"[factory_angle_execution_parallel] RUS teleportation executed, circuit_moment={local_moment}"
            )

            # Simulate RUS injection results
            rus_simulation = simulate_RUS_injection(
                event["qubit_factory_pairs"], factory_pool, rng
            )

            execution_log = write_rus_result_log(
                event["qubit_factory_pairs"],
                rus_simulation,
                local_moment,
                execution_log,
            )

            # Update qubit states based on teleportation results
            local_moment = update_qubit_state_per_teleportation(
                qubit_factory_pairs,
                rus_simulation,
                qubit_trackers,
                factory_pool,
                execution_log,
                local_moment,
            )

            # Schedule return movement
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

            # Schedule finish event
            heapq.heappush(
                events,
                (
                    local_moment,
                    next(counter),
                    {
                        "type": "RUS_finish",
                        "aod_id": event["aod_id"],
                        "factory_list": [f for q, f in event["qubit_factory_pairs"]],
                    },
                ),
            )
            print(
                f"[factory_angle_execution_parallel] Added RUS_finish event at t={local_moment}, total events in queue: {len(events)}"
            )

        elif event["type"] == "RUS_finish":
            print(
                f"[factory_angle_execution_parallel] Event: RUS_finish at t={local_moment}, {len(qubit_trackers)} qubits remaining"
            )
            for factory_id in event["factory_list"]:
                factory_pool.free_factory(factory_id)
            # Check if there are more qubits to process
            if len(qubit_trackers) == 0:
                continue
            # Execute RUS teleportation
            qubit_factory_pairs, routing_batches = rus_teleportation(
                qubit_trackers,
                factory_pool,
                logic_qubit_locations,
                consider_skip_rus=consider_skip_rus,
                n_aods=n_aods,
                total_qubits=len(target_qubits_angles),
            )

            if qubit_factory_pairs and routing_batches:
                # Execute movement to factories
                local_moment = execute_movement(
                    routing_batches,
                    execution_log,
                    local_moment,
                    aod_earliest_available_time,
                )

                # Schedule finish event
                heapq.heappush(
                    events,
                    (
                        local_moment,
                        next(counter),
                        {
                            "type": "RUS_teleportation",
                            "qubit_factory_pairs": qubit_factory_pairs,
                            "routing_batches": routing_batches,
                            "aod_id": event["aod_id"],
                        },
                    ),
                )
                print(
                    f"[factory_angle_execution_parallel] Added RUS_teleportation event at t={local_moment}, total events in queue: {len(events)}. (routing batches: {qubit_factory_pairs})"
                )
            # Schedule next round of TMR preparation
            # ! Reassign factories if no idle factories
            else:
                aod_in_use[event["aod_id"]] = False

            max_time = float("inf")
            aod_id = -1
            print("aod_earliest_available_time: ", aod_earliest_available_time)
            print("aod_in_use: ", aod_in_use)
            for i in range(n_aods_se):
                if not aod_in_use[i] and aod_earliest_available_time[i] < max_time:
                    max_time = aod_earliest_available_time[i]
                    aod_id = i
            if aod_id != -1:
                start_time = max(local_moment, max_time)
                print("TMR_start aod_id: ", aod_id)
                print(events)
                print(start_time)
                task = {
                    "type": "TMR_start",
                    "aod_id": aod_id,
                }
                heapq.heappush(
                    events,
                    (start_time, next(counter), task),
                )
                print(
                    f"[factory_angle_execution_parallel] Added TMR_start event at t={start_time}, total events in queue: {len(events)}"
                )
                input()

        else:
            raise ValueError(f"Unknown event type: {event['type']}")
        # input()

    # Final cleanup: reassign any remaining factories
    print(
        f"[factory_angle_execution_parallel] Main loop completed at t={circuit_moment}"
    )
    assert not qubit_trackers, "All qubits should be completed at this point"
    execution_log = clean_up_execution_log(execution_log, circuit_moment)
    print(
        f"[factory_angle_execution_parallel] Execution finished in {circuit_moment} time units, {len(execution_log)} events logged"
    )
    execution_log = sorted(execution_log, key=lambda x: x[:4])

    validate_execution_log(execution_log, magic_state_locations)
    return circuit_moment, execution_log
