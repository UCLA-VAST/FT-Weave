import random

from .config import (
    TMR_P,
    TMR_Q,
    TMR_PREPARATION_TIME,
    CNOT_TIME,
    SE_TIME,
)
from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
)
from .rus.two_layer_routing import two_layer_routing
from .rus.rus_assignment import assign_teleportation_with_sharing

from .tmr.angle_collection import get_angles_for_preparation
from .tmr.tmr_assignment import assign_factories_for_batch

from .ds.device_state import FactoryPool, QubitAngleTracker
from .ds.architecture import move_duration
from .rus.angle_factory_index import AngleFactoryIndex
from .rus.solve_return_move import solve_return_move

from .util import write_execution_log, write_injection_log

random.seed(42)


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
def build_angle_factory_index_from_tmr_results(
    qubit_trackers: dict[int, QubitAngleTracker],
    successful_qubits: set[int],
    factory_pool: FactoryPool,
) -> AngleFactoryIndex:
    """
    Build an AngleFactoryIndex directly from TMR results.

    This index maps angles to all factories that successfully completed TMR,
    enabling cross-qubit factory sharing for RUS injection.

    Args:
        qubit_trackers: Dict mapping qubit_id to QubitAngleTracker
        factory_pool: FactoryPool containing Factory objects with TMR results

    Returns:
        AngleFactoryIndex with all angle-factory mappings for TMR-successful factories
    """
    angle_factory_index = AngleFactoryIndex()

    # Iterate through all qubits and their factories
    for qubit, tracker in qubit_trackers.items():
        if qubit not in successful_qubits:
            removed_factory = []
            for factory_id, angle in tracker.factories:
                # Only include factories that successfully passed TMR
                factory = factory_pool.get_factory_by_id(factory_id)
                if factory is not None:
                    if factory.tmr_state:
                        angle_factory_index.add_factory(factory_id, angle, qubit)
                    else:
                        factory_pool.free_factory(factory_id)
                        removed_factory.append((factory_id, angle))
            for factory_id, angle in removed_factory:
                tracker.remove_factory(factory_id, angle)

            angle_factory_index.add_qubit_for_angle(tracker.target_angle)

    return angle_factory_index


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


def execute_rus_teleportation(
    qubit_factory_pairs: list[tuple[int, int]],
    factory_pool: FactoryPool,
    circuit_moment: float,
    execution_log: list,
) -> list[bool]:
    rus_simulation = simulate_RUS_injection(qubit_factory_pairs, factory_pool)
    for (qubit, factory_id), injection_result in zip(
        qubit_factory_pairs, rus_simulation
    ):
        write_injection_log(
            execution_log,
            circuit_moment,
            factory_id,
            "RUS_success" if injection_result else "RUS_fail",
            qubit,
        )

    return rus_simulation


def execute_movement(
    routing_batches: list,
    execution_log: list,
    circuit_moment: float,
    move_type: str = "move",
) -> float:
    for batches in routing_batches:
        max_movement_time = 0.0
        for _, x_q, y_q, factory_id, x_f, y_f in batches:
            max_movement_time = max(
                max_movement_time, move_duration(x_q, y_q, x_f, y_f)
            )

        for _, x_q, y_q, factory_id, x_f, y_f in batches:
            movement_strs = [f"({x_f},{y_f})", f"({x_q},{y_q})"]
            write_execution_log(
                execution_log,
                circuit_moment,
                factory_id,
                move_type,
                qubit=None,
                movement_time=max_movement_time,
                move_vecs=movement_strs,
            )
        circuit_moment += max_movement_time
    return circuit_moment


def factory_angle_execution(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    column_based_placement: bool = True,
):
    """
    Execute angle preparation on magic state factories with lookahead optimization.

    Args:
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle
        logic_qubit_locations: (x, y) location for each logical qubit
        magic_state_locations: (x, y) location for each magic state factory

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
        factory_return_move = []
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
            # print("logic_qubit_locations")
            # print(logic_qubit_locations)
            # print("magic_state_locations")
            # print(magic_state_locations)
            # print("qubit_factory_pairs")
            # print(qubit_factory_pairs)
            # print("routing_batches")
            # print(routing_batches)
            if factory_return_move:
                # ! merge movement
                raise NotImplementedError
            circuit_moment = execute_movement(
                routing_batches,
                execution_log,
                circuit_moment,
            )

            rus_simulation = execute_rus_teleportation(
                qubit_factory_pairs, factory_pool, circuit_moment, execution_log
            )
            circuit_moment += CNOT_TIME + SE_TIME
            update_qubit_state_per_teleportation(
                successful_qubits,
                successful_teleportation_qubits,
                qubit_factory_pairs,
                rus_simulation,
                qubit_trackers,
                factory_pool,
                angle_factory_index,
            )

            # ! return factories qubit to empty spot
            trivial_return = False
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
                move_type="return_move",
            )
            if len(target_qubits_angles) == len(successful_qubits):
                break

        update_qubit_state_post_teleportation(
            successful_teleportation_qubits,
            qubit_trackers,
            factory_pool,
            angle_factory_index,
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
