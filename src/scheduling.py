import heapq
import random
from collections import Counter

from .config import (
    TMR_P,
    TMR_Q,
    TMR_PREPARATION_TIME,
    CNOT_TIME,
    SE_TIME,
    LOOKAHEAD_THRESHOLD,
    LOOKAHEAD_LEVEL,
    PRECISION,
)
from .simulation import simulate_angle_preparation, simulate_injection

random.seed(42)

# ============================================================================
# DATA STRUCTURES
# ============================================================================


class QubitAngleTracker:
    """
    Tracks angle preparation state for a single qubit.

    Attributes:
        qubit_id: Identifier for the qubit
        target_angle: Current target angle to prepare
        factories: List of (factory_id, angle) tuples working on this qubit
        angle_counts: Counter tracking how many factories are preparing each angle
    """

    def __init__(self, qubit_id, target_angle):
        self.qubit_id = round(qubit_id, PRECISION)
        self.target_angle = target_angle
        self.factories = []  # List of (factory_id, angle)
        self.angle_counts = Counter()  # angle -> count

    def get_generation(self, angle):
        """Calculate generation level: 0 for original, 1 for 2x, 2 for 4x, etc."""
        if angle == self.target_angle:
            return 0
        # Calculate how many times we've doubled: log2(angle/original)
        import math

        ratio = angle // self.target_angle
        if ratio >= 1:  # Check if power of 2
            return int(math.log2(ratio))
        return 999  # Large number for non-power-of-2 angles

    def add_factory(self, factory_id, angle):
        """Add a factory working on a specific angle."""
        self.factories.append((factory_id, angle))
        self.angle_counts[angle] += 1

    def remove_factory(self, factory_id, angle):
        """Remove a factory from tracking."""
        self.factories = [(fid, a) for fid, a in self.factories if fid != factory_id]
        self.angle_counts[angle] -= 1
        if self.angle_counts[angle] == 0:
            del self.angle_counts[angle]

    def remove_angle(self, angle):
        """Remove all factories working on a specific angle."""
        removed_factories = [(fid, a) for fid, a in self.factories if a == angle]
        self.factories = [(fid, a) for fid, a in self.factories if a != angle]
        if angle in self.angle_counts:
            del self.angle_counts[angle]
        return removed_factories

    def clear_all(self):
        """Clear all factory assignments."""
        self.factories = []
        self.angle_counts.clear()

    def get_factories_for_angle(self, angle):
        """Get list of factory IDs working on a specific angle."""
        return [fid for fid, a in self.factories if a == angle]

    def get_angle_count(self, angle):
        """Get number of factories working on a specific angle."""
        return self.angle_counts.get(angle, 0)

    def get_sorted_factories(self):
        """Get factories sorted by angle (ascending)."""
        return sorted(self.factories, key=lambda x: x[1])

    def has_active_factories(self):
        """Check if any factories are working on this qubit."""
        return len(self.factories) > 0

    def __repr__(self):
        return f"QubitTracker(qubit={self.qubit_id}, target={self.target_angle}, factories={len(self.factories)})"


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================


def calculate_success_rate(angle):
    """Calculate success rate for angle preparation (decreases with angle)."""
    assert angle < 0.6
    return 0.6 - angle


def assign_factory(qubit_trackers, factory_assignments, queue):
    n_factories = len(factory_assignments)
    for factory_id in range(n_factories):
        if factory_assignments[factory_id] != (None, None, None):
            continue  # Factory is busy

        # Pop angle from primary queue, or secondary if primary is empty
        if not queue:
            return
        generation, success_rate, theta, qubit = heapq.heappop(queue)

        # Assign this angle to the factory
        factory_assignments[factory_id] = (theta, qubit, success_rate)
        qubit_trackers[qubit].add_factory(factory_id, theta)


def construct_secondary_queue(
    qubit_trackers, successful_qubits, lookahead_level, lookahead_threshold
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
    factory_assignments,
    primary_queue,
    successful_qubits,
):
    """
    PHASE 1: Assign idle factories to prepare angles.

    Args:
        qubit_trackers: Dict mapping qubit_id -> QubitAngleTracker
        factory_assignments: List of (angle, qubit, success_rate) tuples for each factory
        primary_queue: Priority queue of primary angles to prepare
        successful_qubits: Set of successfully completed qubit IDs

    Returns:
        Updated factory_assignments
    """
    assign_factory(qubit_trackers, factory_assignments, primary_queue)

    # Build lookahead queue based on current state
    # Priority: (generation, success_rate, angle, qubit)
    # Process all gen 0 first, then all gen 1, then all gen 2, etc.
    # Within same generation, larger angles first
    lookahead_level = LOOKAHEAD_LEVEL
    lookahead_threshold = LOOKAHEAD_THRESHOLD
    while any(a == (None, None, None) for a in factory_assignments):
        secondary_queue = construct_secondary_queue(
            qubit_trackers, successful_qubits, lookahead_level, lookahead_threshold
        )
        assign_factory(qubit_trackers, factory_assignments, secondary_queue)
        lookahead_level += 1
        lookahead_threshold += 1

    return factory_assignments


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


# TODO2: I have finished this function. Please check it.
def collect_injection_sequence(tmr_simulation, qubit_trackers):
    """
    Collect injection sequence organized by injection rounds.

    Groups qubits that can be injected in parallel based on which factories
    successfully prepared their target angles (TMR phase) and are ready for injection,
    organized by generation level.

    Args:
        tmr_simulation: List of bool indicating TMR preparation success for each factory
        qubit_trackers: Dict mapping qubit_id -> QubitAngleTracker

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
        for factory_id, angle in tracker.factories:
            if tmr_simulation[factory_id]:
                generation = tracker.get_generation(angle)
                if generation not in generation_to_factories:
                    generation_to_factories[generation] = []
                generation_to_factories[generation].append(factory_id)
                max_generation = max(max_generation, generation)

        for i in range(max_generation + 1):
            if i in generation_to_factories:
                assert (
                    len(injection_sequence) >= i
                ), "Injection sequence not long enough"
                if len(injection_sequence) == i:
                    injection_sequence.append([])
                injection_sequence[i].append((qubit, generation_to_factories[i]))
            else:
                break

    return injection_sequence


def phase_2_execute_tmr_preparation(
    factory_assignments,
    circuit_moment,
    execution_log,
):
    """
    PHASE 2: Execute TMR preparation (2 SE + Rz + 3 SE).

    Args:
        factory_assignments: List of (angle, qubit, success_rate) tuples for each factory
        circuit_moment: Current circuit execution time
        execution_log: List of execution events

    Returns:
        Updated circuit_moment and execution_log
    """
    # Log TMR preparation for all active factories
    for factory_id, (theta, qubit, _) in enumerate(factory_assignments):
        if theta is not None:
            for p in range(TMR_P):
                execution_log.append(
                    (
                        circuit_moment + p,
                        circuit_moment + p + 1,
                        factory_id,
                        "SE",
                        qubit,
                    )
                )
            execution_log.append(
                (
                    circuit_moment + TMR_P,
                    circuit_moment + TMR_P + 1,
                    factory_id,
                    "Rz",
                    theta,
                )
            )
            for q in range(TMR_Q):
                execution_log.append(
                    (
                        circuit_moment + TMR_P + q + 1,
                        circuit_moment + TMR_P + q + 2,
                        factory_id,
                        "SE",
                        qubit,
                    )
                )

    circuit_moment += TMR_PREPARATION_TIME

    execution_log.append(
        (
            circuit_moment,
            circuit_moment,
            -1,
            "Barrier",
            None,
        )
    )

    return circuit_moment, execution_log


def simulate_TMR_preparation(
    factory_assignments: list[tuple[float, int, float]],
) -> list[bool]:
    """
    Simulate TMR preparation success for each factory.

    Args:
        factory_assignments: List of (angle, qubit, success_rate) tuples for each factory

    Returns:
        List of boolean values indicating whether each factory's TMR preparation was successful.
    """
    return [
        simulate_angle_preparation(success_rate)
        for _, _, success_rate in factory_assignments
    ]


def phase_3_simulate_and_inject(
    qubit_trackers,
    factory_assignments,
    target_qubits_angles,
    successful_qubits,
    circuit_moment,
    execution_log,
    primary_queue,
    tmr_simulation,
):
    """
    PHASE 3: Simulate preparation success and attempt injections.

    Uses TMR simulation results to check which factories successfully prepared angles.

    Args:
        qubit_trackers: Dict mapping qubit_id -> QubitAngleTracker
        factory_assignments: List of (angle, qubit, success_rate) tuples for each factory
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle
        successful_qubits: Set of successfully completed qubit IDs
        circuit_moment: Current circuit execution time
        execution_log: List of execution events
        primary_queue: Priority queue to re-queue angles if needed
        tmr_simulation: List of bool indicating TMR preparation success for each factory

    Returns:
        Tuple of (max_injections_this_round, target_qubits_angles)
    """
    max_injections_this_round = 0

    for qubit, tracker in qubit_trackers.items():
        if qubit in successful_qubits:
            assert (
                not tracker.has_active_factories()
            ), f"qubit {qubit} has successfully injected, but there are active factories working for it."
            continue

        target_theta = tracker.target_angle

        # Get factories sorted by angle (smallest first for injection order)
        sorted_factories = tracker.get_sorted_factories()

        qubit_injections = 0
        success = False
        removed_angles = set()
        qubit_begin_time = circuit_moment
        # Try injecting angles in order
        for factory_id, theta in sorted_factories:
            if theta != target_theta:
                continue  # Not the target angle yet

            _, _, success_rate = factory_assignments[factory_id]

            # Check TMR preparation result
            if not tmr_simulation[factory_id]:
                print(f"✗ Qubit {qubit}: angle {theta:.0f}° TMR preparation failed")
                execution_log.append(
                    (
                        qubit_begin_time,
                        qubit_begin_time,
                        factory_id,
                        "TMR_fail",
                        qubit,
                    )
                )

            # Simulate preparation
            else:
                qubit_injections += 1
                execution_log.append(
                    (
                        qubit_begin_time,
                        qubit_begin_time + CNOT_TIME,
                        factory_id,
                        "CNOT",
                        qubit,
                    )
                )
                execution_log.append(
                    (
                        qubit_begin_time + CNOT_TIME,
                        qubit_begin_time + CNOT_TIME + SE_TIME,
                        factory_id,
                        "SE",
                        qubit,
                    )
                )
                qubit_begin_time += CNOT_TIME + SE_TIME

                # Simulate injection
                if simulate_injection():
                    print(
                        f"✓ Qubit {qubit}: angle {theta:.0f}° prepared & injected (p={success_rate:.2f})"
                    )
                    execution_log.append(
                        (
                            qubit_begin_time,
                            qubit_begin_time,
                            factory_id,
                            "RUS_succsss",
                            qubit,
                        )
                    )
                    success = True
                    successful_qubits.add(qubit)
                    break
                else:
                    print(
                        f"? Qubit {qubit}: angle {theta:.0f}° prepared but injected -θ (p={success_rate:.2f})"
                    )
                    execution_log.append(
                        (
                            qubit_begin_time,
                            qubit_begin_time,
                            factory_id,
                            "RUS_fail",
                            qubit,
                        )
                    )
                    removed_angles.add(theta)
                    target_theta *= 2  # Need to prepare 2θ next

        # Update target angle for this qubit
        tracker.target_angle = target_theta
        target_qubits_angles[qubit] = target_theta

        # PHASE 4: Update factory assignments based on results
        if success:
            # Free all factories working on this qubit
            for factory_id, _ in tracker.factories:
                factory_assignments[factory_id] = (None, None, None)
            successful_qubits.add(qubit)
            tracker.clear_all()
        else:
            # Free factories for removed angles
            for angle in removed_angles:
                removed_factories = tracker.remove_angle(angle)
                for factory_id, _ in removed_factories:
                    factory_assignments[factory_id] = (None, None, None)

            # Re-queue target angle if needed
            target_success_rate = calculate_success_rate(target_theta)
            target_count = tracker.get_angle_count(target_theta)

            if target_count == 0:
                heapq.heappush(
                    primary_queue, (0, target_success_rate, target_theta, qubit)
                )

        max_injections_this_round = max(max_injections_this_round, qubit_injections)

    return max_injections_this_round, target_qubits_angles


def factory_angle_execution(n_factories, target_qubits_angles):
    """
    Execute angle preparation on magic state factories with lookahead optimization.

    Args:
        n_factories: Number of available factories
        target_qubits_angles: Dict mapping qubit_id -> target_rotation_angle

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

    # Current assignment for each factory: (angle, qubit, success_rate) or (None, None, None)
    factory_assignments = [(None, None, None)] * n_factories

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
        factory_assignments = phase_1_assign_factories(
            qubit_trackers,
            factory_assignments,
            primary_queue,
            successful_qubits,
        )
        # TODO1: change phase_1_assign_factories to
        # batch_angles = get_angles_for_preparation()
        # factory_assignments = assign_factories_for_batch(batch_angles)

        # PHASE 2: Execute TMR preparation
        circuit_moment, execution_log = phase_2_execute_tmr_preparation(
            factory_assignments,
            circuit_moment,
            execution_log,
        )

        tmr_simulation = simulate_TMR_preparation(factory_assignments)

        # TODO2: pass TMR_simulation to phase_3_inject. In phase_3_simulate_and_inject,
        # the current imeplementation iterates based on qubits to find the injection path.
        # In reality, we should to a round of injection on all qubits and get the RUS results.
        # Based on the RUS results, we do the next runs of injections. Therefore, we want to
        # change phase_3_simulate_and_inject to the following psuedo-code:
        #
        # collect the maximum number of injections among all qubits
        # qubit_factories_pair: tuple[int,list[int]:
        # injection_sequence: list[list[qubit_factories_pair]]. A element is a list of qubits that can be
        # injected at this injection round with the possible factories.
        # injection_sequence = collect_injection_sequence(tmr_simulation, qubit_trackers)
        # initialize qubits_to_inject with all qubits that need injections
        # for batch_injection in injection_sequence:
        #     assign_injection(qubits_to_inject, batch_injection, circuit_moment, execution_log)
        #     rus_simulation = simulate_TMR_preparation(factory_assignments)
        #     update_qubits_to_inject(qubits_to_inject, rus_simulation)

        # PHASE 3: Simulate preparation success and attempt injections
        max_injections_this_round, target_qubits_angles = phase_3_simulate_and_inject(
            qubit_trackers,
            factory_assignments,
            target_qubits_angles,
            successful_qubits,
            circuit_moment,
            execution_log,
            primary_queue,
            tmr_simulation,
        )

        # Add time for injection attempts (CNOT + SE per injection)
        circuit_moment += max_injections_this_round * (CNOT_TIME + SE_TIME)
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
