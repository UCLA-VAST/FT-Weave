import random

from .config import (
    CNOT_TIME,
    SE_TIME,
)
from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
)

from .tmr import (
    get_angles_for_preparation,
    assign_factories_for_batch,
    reassign_factories,
    build_angle_factory_index_from_tmr_results,
)

from .ds import FactoryPool, QubitAngleTracker
from .rus import (
    solve_return_move,
    check_teleportation_worthiness,
    update_qubit_state_per_teleportation,
    update_qubit_state_post_teleportation,
    two_layer_routing,
    assign_teleportation_with_sharing,
)

from .analog_rotation.util import (
    write_execution_log,
    execute_movement,
    execute_rus_teleportation,
    execute_tmr_preparation,
)


random.seed(42)

threshold_high_tmr = 5
threshold_high_rus = 5


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================


def factory_angle_execution(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    column_based_placement: bool = True,
    n_aods: int = 1,
    consider_skip_rus: bool = True,
    tmr_assignment_method: str = "matching",
    trivial_return: bool = True,
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
    # Map qubit_id to QubitAngleTracker
    qubit_trackers = {
        qubit: QubitAngleTracker(qubit_id=qubit, target_angle=round(theta, 7))
        for qubit, theta in target_qubits_angles.items()
    }

    # Create FactoryPool with n_factories

    factory_pool.set_locations(magic_state_locations)

    # Track successfully completed qubits
    successful_qubits = set()

    # Log for visualization: (start_time, end_time, factory_id, operation, qubit)
    execution_log = []

    circuit_moment = 0

    # ========================================================================
    # MAIN EXECUTION LOOP
    # ========================================================================

    while len(target_qubits_angles) > len(successful_qubits):
        # print(f"circuit_moment: {circuit_moment}")
        # print(f"idle factories: {factory_pool.get_num_idle_factories()}")
        # print("successful_qubits")
        # print(successful_qubits)
        # PHASE 1: Assign idle factories to prepare angles
        batch_angles = get_angles_for_preparation(
            successful_qubits, qubit_trackers, factory_pool.get_num_idle_factories()
        )
        # print("batch_angles")
        # print(batch_angles)
        assign_factories_for_batch(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
            batch_angles,
            column=column_based_placement,
            method=tmr_assignment_method,
        )

        # PHASE 2: Execute TMR preparation
        circuit_moment, execution_log = execute_tmr_preparation(
            factory_pool,
            circuit_moment,
            execution_log,
        )

        simulate_TMR_preparation(factory_pool)
        for factory in factory_pool.factories:
            if not factory.tmr_state:
                write_execution_log(
                    execution_log, circuit_moment, factory.id, "TMR_fail"
                )

        # the current imeplementation iterates based on qubits to find the injection path.
        # In reality, we should to a round of injection on all qubits and get the RUS results.
        # Based on the RUS results, we do the next runs of injections. Therefore, we want to
        # change phase_3_simulate_and_inject to the following psuedo-code:
        #
        # collect the maximum number of injections among all qubits
        # qubit_factories_pair: tuple[int,list[int]:
        # injection_sequence: list[list[qubit_factories_pair]]. A element is a list of qubits that can be
        # injected at this injection round with the possible factories.
        # Build angle-factory index from TMR results
        angle_factory_index = build_angle_factory_index_from_tmr_results(
            qubit_trackers, successful_qubits, factory_pool
        )
        successful_teleportation_qubits = set()
        while True:
            # qubit_factory_pairs: list[tuple(qubit, factory_id)]
            # Use enhanced assignment with factory sharing
            qubit_factory_pairs = assign_teleportation_with_sharing(
                successful_qubits,
                qubit_trackers,
                factory_pool,
                logic_qubit_locations,
                angle_factory_index,
            )
            if not qubit_factory_pairs:
                break
            # routing
            # batch: list of tuple (qubit, x_q, y_q, factory_id, x_f, y_f, reverse)
            routing_batches = two_layer_routing(
                factory_pool, logic_qubit_locations, qubit_factory_pairs
            )
            # check if executing teleporation is worth it.
            if consider_skip_rus:
                routing_batches = check_teleportation_worthiness(
                    routing_batches,
                    n_aods,
                    qubit_factory_pairs,
                    target_qubits_angles,
                    successful_qubits,
                )
                if not routing_batches:
                    break
            # print("logic_qubit_locations")
            # print(logic_qubit_locations)
            # print("magic_state_locations")
            # print(magic_state_locations)
            # print("qubit_factory_pairs")
            # print(qubit_factory_pairs)
            # print("routing_batches")
            # print(routing_batches)
            circuit_moment = execute_movement(
                routing_batches,
                execution_log,
                circuit_moment,
                n_aods=n_aods,
            )

            execute_rus_teleportation(
                qubit_factory_pairs, circuit_moment, execution_log
            )
            circuit_moment += CNOT_TIME

            # return factories qubit to empty spot
            # trivial_return = True
            if trivial_return:
                return_routing_batches = []
                for batches in routing_batches:
                    return_routing_batches.append([])
                    for _, x_q, y_q, factory_id, x_f, y_f in batches:
                        return_routing_batches[-1].append(
                            (-1, x_f, y_f, factory_id, x_q, y_q)
                        )
            else:
                return_routing_batches = solve_return_move(
                    routing_batches, factory_pool
                )
            circuit_moment = execute_movement(
                return_routing_batches,
                execution_log,
                circuit_moment,
                n_aods=n_aods,
                move_type="return_move",
            )
            rus_simulation = simulate_RUS_injection(qubit_factory_pairs, factory_pool)
            for (qubit, factory_id), injection_result in zip(
                qubit_factory_pairs, rus_simulation
            ):
                write_execution_log(
                    execution_log,
                    circuit_moment,
                    factory_id,
                    "SE",
                    qubit,
                )
                if injection_result:
                    result = "RUS_success"
                else:
                    result = "RUS_fail"
                write_execution_log(
                    execution_log,
                    circuit_moment + SE_TIME,
                    factory_id,
                    result,
                    qubit,
                )
            circuit_moment += SE_TIME

            update_qubit_state_per_teleportation(
                successful_qubits,
                successful_teleportation_qubits,
                qubit_factory_pairs,
                rus_simulation,
                qubit_trackers,
                factory_pool,
                angle_factory_index,
            )

            if len(target_qubits_angles) == len(successful_qubits):
                break

        update_qubit_state_post_teleportation(
            successful_teleportation_qubits,
            qubit_trackers,
            factory_pool,
            angle_factory_index,
        )
        reassign_factories(
            factory_pool,
            qubit_trackers,
            successful_qubits,
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

    assert len(target_qubits_angles) == len(successful_qubits)
    return circuit_moment, execution_log
