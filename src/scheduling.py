import heapq
import random
from collections import Counter
import numpy as np

random.seed(42)
# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
TUM_P = 3
TUM_Q = 2
TUM_PREPARATION_TIME = TUM_P + TUM_Q + 1  # 2 SE + Rz + 3 SE
CNOT_TIME = 1
SE_TIME = 1
INJECTION_SUCCESS_RATE = 0.5
LOOKAHEAD_THRESHOLD = 2  # Max factories working on same angle
LOOKAHEAD_LEVEL = 2
PRECISION = 8
LARGE_ANGLE_LIMIT = round(np.pi / 2, PRECISION)  # time to inject S gate

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


def simulate_angle_preparation(success_rate):
    """Simulate angle preparation success based on success rate."""
    return random.random() < success_rate


def simulate_injection():
    """Simulate injection success (50% probability)."""
    return random.random() < INJECTION_SUCCESS_RATE


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
    secondary_queue = []  # Low-priority angles (lookahead/retry)

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

    while primary_queue:

        # --------------------------------------------------------------------
        # PHASE 1: Assign idle factories to prepare angles
        # --------------------------------------------------------------------
        assign_factory(qubit_trackers, factory_assignments, primary_queue)
        # --------------------------------------------------------------------
        # Build lookahead queue based on current state
        # Priority: (generation, success_rate, angle, qubit)
        # Process all gen 0 first, then all gen 1, then all gen 2, etc.
        # Within same generation, larger angles first
        # --------------------------------------------------------------------
        lookahead_level = LOOKAHEAD_LEVEL
        lookahead_threshold = LOOKAHEAD_THRESHOLD
        while any(a == (None, None, None) for a in factory_assignments):
            secondary_queue = construct_secondary_queue(
                qubit_trackers, successful_qubits, lookahead_level, lookahead_threshold
            )
            assign_factory(qubit_trackers, factory_assignments, secondary_queue)
            lookahead_level += 1
            lookahead_threshold += 1
        # --------------------------------------------------------------------
        # PHASE 2: Execute TUM preparation (2 SE + Rz + 3 SE)
        # --------------------------------------------------------------------

        # Log TUM preparation for all active factories
        for factory_id, (theta, qubit, _) in enumerate(factory_assignments):
            if theta is not None:
                for p in range(TUM_P):
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
                        circuit_moment + TUM_P,
                        circuit_moment + TUM_P + 1,
                        factory_id,
                        "Rz",
                        theta,
                    )
                )
                for q in range(TUM_Q):
                    execution_log.append(
                        (
                            circuit_moment + TUM_P + q + 1,
                            circuit_moment + TUM_P + q + 2,
                            factory_id,
                            "SE",
                            qubit,
                        )
                    )

        circuit_moment += TUM_PREPARATION_TIME

        execution_log.append(
            (
                circuit_moment,
                circuit_moment,
                -1,
                "Barrier",
                None,
            )
        )
        # --------------------------------------------------------------------
        # PHASE 3: Simulate preparation success and attempt injections
        # --------------------------------------------------------------------

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

                # Simulate preparation
                if simulate_angle_preparation(success_rate):
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
                        success = True
                        execution_log.append(
                            (
                                qubit_begin_time,
                                qubit_begin_time,
                                factory_id,
                                "RUS_succsss",
                                qubit,
                            )
                        )
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

                else:
                    print(
                        f"✗ Qubit {qubit}: angle {theta:.0f}° preparation failed (p={success_rate:.2f})"
                    )
                    execution_log.append(
                        (
                            qubit_begin_time,
                            qubit_begin_time,
                            factory_id,
                            "TUM_fail",
                            qubit,
                        )
                    )

            # Update target angle for this qubit
            tracker.target_angle = target_theta
            target_qubits_angles[qubit] = target_theta

            # --------------------------------------------------------------------
            # PHASE 4: Update factory assignments based on results
            # --------------------------------------------------------------------

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
