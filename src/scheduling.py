import heapq
import random

from collections import defaultdict

from .config import (
    TMR_P,
    TMR_Q,
    TMR_PREPARATION_TIME,
    CNOT_TIME,
    SE_TIME,
    LOOKAHEAD_THRESHOLD,
    LOOKAHEAD_LEVEL,
)
from .simulation import (
    simulate_TMR_preparation,
    simulate_RUS_injection,
    calculate_success_rate,
)

from .device_state import FactoryPool, QubitAngleTracker

from .util import write_execution_log, write_injection_log

random.seed(42)


def assign_factory(
    qubit_trackers: dict[int, QubitAngleTracker], factory_pool: FactoryPool, queue: list
):
    idle_factories = factory_pool.get_idle_factories()
    for factory in idle_factories:
        assert factory.angle is None

        # Pop angle from primary queue, or secondary if primary is empty
        if not queue:
            return
        generation, success_rate, theta, qubit = heapq.heappop(queue)

        # Assign this angle to the factory object
        factory_pool.assign_factory(factory.id, theta, qubit, success_rate)
        qubit_trackers[qubit].add_factory(factory.id, theta)


def construct_secondary_queue(
    qubit_trackers: dict[int, QubitAngleTracker],
    successful_qubits: set,
    lookahead_level: int,
    lookahead_threshold: int,
) -> list:
    secondary_queue = []
    for qubit, tracker in qubit_trackers.items():
        if qubit in successful_qubits:
            continue

        target_theta = tracker.target_angle
        current_angle = target_theta

        # Greedily add lookahead angles: target, 2*target, 4*target, etc.
        # until we have LOOKAHEAD_THRESHOLD factories per angle
        for _ in range(lookahead_level):
            current_count = tracker.get_angle_count(current_angle)

            if current_count < lookahead_threshold:
                success_rate = calculate_success_rate(current_angle)
                generation = tracker.get_generation(current_angle)
                # Add (LOOKAHEAD_THRESHOLD - current_count) copies to queue
                for _ in range(lookahead_threshold - current_count):
                    heapq.heappush(
                        secondary_queue,
                        (generation, success_rate, current_angle, qubit),
                    )

            # Move to next doubling level
            current_angle *= 2
    return secondary_queue


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================


def phase_1_assign_factories(
    qubit_trackers,
    factory_pool,
    primary_queue,
    successful_qubits,
) -> None:
    """
    PHASE 1: Assign idle factories to prepare angles.

    Args:
        qubit_trackers: Dict mapping qubit_id -> QubitAngleTracker
        factory_pool: FactoryPool object containing Factory objects
        primary_queue: Priority queue of primary angles to prepare
        successful_qubits: Set of successfully completed qubit IDs

    Returns:
        Updated factory_pool
    """
    assign_factory(qubit_trackers, factory_pool, primary_queue)

    # Build lookahead queue based on current state
    # Priority: (generation, success_rate, angle, qubit)
    # Process all gen 0 first, then all gen 1, then all gen 2, etc.
    # Within same generation, larger angles first
    lookahead_level = LOOKAHEAD_LEVEL
    lookahead_threshold = LOOKAHEAD_THRESHOLD
    while factory_pool.get_num_idle_factories() > 0:
        secondary_queue = construct_secondary_queue(
            qubit_trackers, successful_qubits, lookahead_level, lookahead_threshold
        )
        assign_factory(qubit_trackers, factory_pool, secondary_queue)
        lookahead_level += 1
        lookahead_threshold += 1


# TODO1
def get_angles_for_preparation() -> list[tuple[int, float, float]]:
    """
    Generate K angles to prepare in one batch based on the level, where K is the number
    of available factories.

    Args:


    Returns:
        batch_angles: List of (target_qubit, angle, success_rate)
    """
    batch_angles = []
    raise NotImplementedError("get_angles_for_preparation is not implemented yet.")
    return batch_angles


def assign_factories_for_batch(
    batch_angles: list[tuple[int, float, float]],
) -> list[tuple[float, int, float]]:
    """
    Assign factories to prepare the given batch of angles.

    Args:
        batch_angles: List of (target_qubit, angle, success_rate)

    Returns:
        factory_assignments: List of (angle, qubit, success_rate) for each factory
    """
    # TODO1: now you can do a trivial assignment based on the indices. I will update this function later
    factory_assignments = []
    raise NotImplementedError("assign_factories is not implemented yet.")
    assert len(batch_angles) == len(factory_assignments)
    return factory_assignments


def collect_injection_sequence(
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
):
    """
    Collect injection sequence organized by injection rounds.

    Groups qubits that can be injected in parallel based on which factories
    successfully prepared their target angles (TMR phase) and are ready for injection,
    organized by generation level.

    Args:
        qubit_trackers: Dict mapping qubit_id -> QubitAngleTracker
        factory_pool: FactoryPool object containing Factory objects

    Returns:
        injection_sequence: List of lists, where each element represents an injection round.
                           Each injection round is a list of (qubit, factory_ids) tuples.
                           factory_ids is a list of factory IDs that can inject for this qubit.
    """
    injection_sequence = []

    for qubit, tracker in qubit_trackers.items():

        # Get factories with target angle that successfully passed TMR
        generation_to_factories = {}
        max_generation = 0
        removed_factory = []
        for factory_id, angle in tracker.factories:
            factory = factory_pool.get_factory_by_id(factory_id)
            if factory is not None:
                if factory.tmr_state:
                    generation = tracker.get_generation(angle)
                    if generation not in generation_to_factories:
                        generation_to_factories[generation] = []
                    generation_to_factories[generation].append(factory_id)
                    max_generation = max(max_generation, generation)
                else:
                    factory_pool.free_factory(factory_id)
                    removed_factory.append((factory_id, angle))
        for factory_id, angle in removed_factory:
            tracker.remove_factory(factory_id, angle)

        for i in range(max_generation + 1):
            if i in generation_to_factories:
                assert (
                    len(injection_sequence) >= i
                ), "Injection sequence not long enough"
                if len(injection_sequence) == i:
                    injection_sequence.append([])
                injection_sequence[i].append((qubit, generation_to_factories[i]))
            else:
                # Free up factories for this qubit that are working on higher generations
                # If a lower generation is missing, higher-generation preparations
                # cannot be used for injection; mark those factories idle so they
                # can be reassigned in the next scheduling phase.
                # Collect factories to free to avoid modifying the list while iterating.
                factories_to_free = defaultdict(list)
                for factory_id, angle in list(tracker.factories):
                    factories_to_free[angle].append(factory_id)

                for angle, factory_ids in factories_to_free.items():
                    if len(factory_ids) > 1:
                        for factory_id in factory_ids[1:]:
                            # Mark factory assignment as free
                            factory_pool.free_factory(factory_id)
                            # Remove the factory from the tracker
                            tracker.remove_factory(factory_id, angle)

    return injection_sequence


def assign_injection(
    successful_qubits: set[int],
    batch_injection: list[tuple[int, list[int]]],
) -> list[tuple[int, int]]:
    qubit_factory_pairs = []
    for qubit, factory_ids in batch_injection:
        if qubit in successful_qubits:
            continue
        # todo(f): may optimize assignment based on the location to enhance cnot parallelism
        qubit_factory_pairs.append((qubit, factory_ids[0]))

    return qubit_factory_pairs


def update_qubit_states(
    successful_qubits: set[int],
    qubit_factory_pairs: list[tuple[int, int]],
    rus_simulation: list[bool],
    qubit_trackers: dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    primary_queue,
):
    """
    Update the list of qubits to inject based on RUS simulation results.
    Args:
        qubits_to_inject: List of qubit IDs that need injections
        rus_simulation: List of bool indicating RUS injection success for each qubit
    Returns:
        Updated list of qubit IDs that still need injections
    """
    qubits_to_inject_updated = []
    for (qubit, factory_id), success in zip(qubit_factory_pairs, rus_simulation):
        tracker = qubit_trackers[qubit]
        if success:
            for factory_id, _ in tracker.factories:
                factory_pool.free_factory(factory_id)
            successful_qubits.add(qubit)
            tracker.clear_all()
            # Injection succeeded, remove from list
        else:
            # Free factories for removed angles
            injected_angle = tracker.target_angle
            removed_factories = tracker.remove_angle(injected_angle)
            for factory_id, _ in removed_factories:
                factory_pool.free_factory(factory_id)
            tracker.double_target_angle()
            new_target_angle = tracker.target_angle
            target_success_rate = calculate_success_rate(new_target_angle)
            target_count = tracker.get_angle_count(new_target_angle)

            if target_count == 0:
                heapq.heappush(
                    primary_queue, (0, target_success_rate, new_target_angle, qubit)
                )
            qubits_to_inject_updated.append(qubit)
    return qubits_to_inject_updated


def construct_primary_queue(
    successful_qubits: set[int], qubit_trackers: dict[int, QubitAngleTracker]
) -> list[int]:
    primary_queue = []
    for qubit, tracker in qubit_trackers.items():
        if qubit in successful_qubits:
            continue
        else:
            angle = tracker.target_angle
            success_rate = calculate_success_rate(angle)
            heapq.heappush(primary_queue, (0, success_rate, angle, qubit))

    return primary_queue


def phase_2_execute_tmr_preparation(
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


def phase_3_execute_rus_injection(
    qubit_factory_pairs: list[tuple[int, int]],
    factory_pool: FactoryPool,
    circuit_moment: int,
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


def factory_angle_execution(
    factory_pool: FactoryPool,
    target_qubits_angles: dict[int, float],
    logic_qubit_locations: list[tuple[int, int]] = [],
    magic_state_locations: list[tuple[int, int]] = [],
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
    # Initialize queues and tracking structures
    primary_queue = []  # High-priority angles (not yet started)

    # Map qubit_id to QubitAngleTracker
    qubit_trackers = {
        qubit: QubitAngleTracker(qubit_id=qubit, target_angle=theta)
        for qubit, theta in target_qubits_angles.items()
    }

    # Create FactoryPool with n_factories

    factory_pool.set_locations(magic_state_locations)

    # Track successfully completed qubits
    successful_qubits = set()

    # Log for visualization: (start_time, end_time, factory_id, operation, qubit)
    execution_log = []

    circuit_moment = 0

    # Initialize primary queue with all target angles
    for qubit, theta in target_qubits_angles.items():
        success_rate = calculate_success_rate(theta)
        heapq.heappush(primary_queue, (0, success_rate, theta, qubit))

    # ========================================================================
    # MAIN EXECUTION LOOP
    # ========================================================================

    while len(target_qubits_angles) > len(successful_qubits):
        # PHASE 1: Assign idle factories to prepare angles
        # print("primary_queue")
        # print(primary_queue)
        phase_1_assign_factories(
            qubit_trackers,
            factory_pool,
            primary_queue,
            successful_qubits,
        )
        # TODO1: change phase_1_assign_factories to
        # batch_angles = get_angles_for_preparation()
        # factory_assignments = assign_factories_for_batch(batch_angles)

        # PHASE 2: Execute TMR preparation
        circuit_moment, execution_log = phase_2_execute_tmr_preparation(
            factory_pool,
            circuit_moment,
            execution_log,
        )

        simulate_TMR_preparation(factory_pool)

        # the current imeplementation iterates based on qubits to find the injection path.
        # In reality, we should to a round of injection on all qubits and get the RUS results.
        # Based on the RUS results, we do the next runs of injections. Therefore, we want to
        # change phase_3_simulate_and_inject to the following psuedo-code:
        #
        # collect the maximum number of injections among all qubits
        # qubit_factories_pair: tuple[int,list[int]:
        # injection_sequence: list[list[qubit_factories_pair]]. A element is a list of qubits that can be
        # injected at this injection round with the possible factories.
        injection_sequence = collect_injection_sequence(qubit_trackers, factory_pool)
        if injection_sequence:
            # print("injection_sequence")
            # print(injection_sequence)
            # input()
            for batch_injection in injection_sequence:
                # qubit_factory_pairs: list[tuple(qubit, factory_ids)]
                qubit_factory_pairs = assign_injection(
                    successful_qubits, batch_injection
                )
                rus_simulation = phase_3_execute_rus_injection(
                    qubit_factory_pairs, factory_pool, circuit_moment, execution_log
                )
                circuit_moment += CNOT_TIME + SE_TIME
                update_qubit_states(
                    successful_qubits,
                    qubit_factory_pairs,
                    rus_simulation,
                    qubit_trackers,
                    factory_pool,
                    primary_queue,
                )

        primary_queue = construct_primary_queue(successful_qubits, qubit_trackers)

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

    assert len(target_qubits_angles) == len(successful_qubits)

    return circuit_moment, execution_log
