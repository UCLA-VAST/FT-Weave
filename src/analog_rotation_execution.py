from itertools import product
import numpy as np
from collections import defaultdict

from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
)

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
    consider_skip_rus: int = 0,  # 0: no skip, 1: partial skip, 2: aggressive skip
    tmr_assignment_method: str = "matching",
    trivial_return: bool = True,
    decompose_move: bool = True,
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
    if rng is None:
        rng = np.random.default_rng()

    # Initialize qubit trackers
    qubit_trackers = {
        qubit: QubitAngleTracker(qubit_id=qubit, target_angle=round(theta, 7))
        for qubit, theta in target_qubits_angles.items()
    }

    factory_pool.set_locations(magic_state_locations)

    execution_log = []
    aod_earliest_available_time = [0.0] * n_aods
    circuit_moment = 0.0

    # Event queue: time -> list of events
    events = defaultdict(list)

    # ========================================================================
    # MAIN EXECUTION LOOP (TIME-STEPPED)
    # ========================================================================

    # Initial TMR scheduling
    circuit_moment, execution_log = execute_tmr_preparation_pre_rz(
        factory_pool.get_idle_factories(),
        circuit_moment,
        execution_log,
    )

    while True:

        # Find next event time
        if circuit_moment in events:
            # Process events at current time
            current_events = events[circuit_moment]
            for event in current_events:
                if event["type"] == "TMR_completion":
                    simulate_TMR_preparation(factory_pool, rng, event["factory_list"])

                    # Update factory state
                    update_factory_states_post_tmr(
                        event["factory_id"], factory_pool, qubit_trackers
                    )

                    # Schedule next TMR round for factories that need preparation
                    factory_list = factory_pool.get_idle_factories()
                    if factory_list:
                        # Execute next round of TMR preparation
                        execute_tmr_preparation_pre_rz(
                            factory_list, circuit_moment, execution_log, events=events
                        )

                    # invoke RUS teleportation

                elif event["type"] == "TMR_pre_RZ_completion":
                    factory_list = schedule_tmr_round(
                        factory_pool,
                        qubit_trackers,
                        logic_qubit_locations,
                        column_based_placement=column_based_placement,
                        tmr_assignment_method=tmr_assignment_method,
                    )
                    execute_tmr_preparation_rz(
                        factory_list, circuit_moment, execution_log, events=events
                    )

                elif event["type"] == "RUS_teleportation":
                    # Execute RUS teleportation with routing and return movement
                    angle_factory_index = event.get("angle_factory_index")
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
                        circuit_moment = execute_movement(
                            routing_batches,
                            execution_log,
                            circuit_moment,
                            aod_earliest_available_time,
                        )

                        # Execute RUS injection
                        circuit_moment = execute_rus_teleportation(
                            qubit_factory_pairs, circuit_moment, execution_log
                        )

                        # Schedule return movement
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

                        # Simulate RUS injection results
                        rus_simulation = simulate_RUS_injection(
                            qubit_factory_pairs, factory_pool, rng
                        )

                        execution_log = write_rus_result_log(
                            qubit_factory_pairs,
                            rus_simulation,
                            circuit_moment,
                            execution_log,
                        )

                        # Update qubit states based on teleportation results
                        circuit_moment = update_qubit_state_per_teleportation(
                            qubit_factory_pairs,
                            rus_simulation,
                            qubit_trackers,
                            factory_pool,
                            angle_factory_index,
                            execution_log,
                            circuit_moment,
                        )

                        # Schedule finish event
                        events[circuit_moment + 0.5].append({"type": "RUS_finish"})

                elif event["type"] == "RUS_finish":
                    # Check if there are more qubits to process
                    if qubit_trackers:
                        # Schedule next round of TMR preparation
                        factory_list = factory_pool.get_idle_factories()
                        if factory_list:
                            schedule_tmr_round(
                                factory_pool,
                                qubit_trackers,
                                logic_qubit_locations,
                                column_based_placement=column_based_placement,
                                tmr_assignment_method=tmr_assignment_method,
                            )
                            execute_tmr_preparation_rz(
                                factory_list,
                                circuit_moment,
                                execution_log,
                                events=events,
                            )
                        else:
                            # Reassign factories if no idle factories
                            reassign_factories(
                                factory_pool,
                                qubit_trackers,
                                logic_qubit_locations,
                            )

        if not qubit_trackers:
            break  # All qubits completed
        circuit_moment += 0.5

    # Final cleanup: reassign any remaining factories
    assert not qubit_trackers, "All qubits should be completed at this point"

    circuit_moment = circuit_moment

    return circuit_moment, execution_log
