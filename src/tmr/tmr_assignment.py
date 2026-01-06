from collections import defaultdict
import heapq
from itertools import count

from src.ds.device_state import FactoryPool, QubitAngleTracker

from src.simulation import (
    calculate_success_rate,
)


def phase1_exact_match(
    required_angles: list[int],
    available_factories: list[int],
    assignments: list[list[tuple[int, int, str]]],
) -> None:
    """
    Phase 1: Exact one-to-one match
    If required_angles[i] = available_factories[j], assign required_angles[i] to available_factories[j]

    Args:
        required_angles: working copy of required angles per row (modified in place)
        available_factories: working copy of available factories per row (modified in place)
        assignments: list of assignments for each angle row (modified in place)
    Returns:
    """

    # Build a map from available factory capacity -> list of factory indices
    # Only include factories with positive capacity. This lets us find an exact
    # match for a required angle in O(1) average time per angle, making the
    # whole phase O(n).
    value_to_indices = defaultdict(list)
    for j, cap in enumerate(available_factories):
        if cap > 0:
            value_to_indices[cap].append(j)

    # For each angle row, check whether there's a factory with exactly the
    # same remaining capacity. If so, assign and remove that factory from
    # the map (its capacity becomes zero).
    for i in range(len(required_angles)):
        req = required_angles[i]
        if req <= 0:
            continue

        indices = value_to_indices.get(req)
        if not indices:
            continue

        # Pop one matching factory index (use pop for O(1) removal).
        j = indices.pop()

        match_amount = req

        # Record assignment
        assignments[i].append((j, match_amount, "exact"))

        # Update remaining capacity (factory becomes zero since exact match)
        required_angles[i] = 0
        available_factories[j] = 0

        # Clean up map entry if no more factories with that capacity
        if not indices:
            del value_to_indices[req]


def phase2_best_fit(
    required_angles: list[int],
    available_factories: list[int],
    assignments: list[list[tuple[int, int, str]]],
) -> None:
    """
    Phase 2: Best-fit with sorted assignment
    Always work on the largest remaining required angles first.
    If angles don't fit completely in one factory, split and re-insert into sorted list.
    Continue until all angles are assigned.

    Args:
        required_angles: working copy of required angles per row (modified in place)
        available_factories: working copy of available factories per row (modified in place)
        assignments: list of assignments for each angle row (modified in place)
    Returns:
    """

    # Create a max-heap of (-amount, counter, angle_row) for unassigned angles
    # Using a heap gives O(log n) insertion and removal compared to O(n)
    # list insert/pop(0) used previously.
    n_angle_rows = len(required_angles)
    n_factory_rows = len(available_factories)
    heap = []
    unique_counter = count()
    for i in range(n_angle_rows):
        if required_angles[i] > 0:
            # push negative amount to simulate max-heap
            heapq.heappush(heap, (-required_angles[i], next(unique_counter), i))

    # Keep working until all angles are assigned
    while heap:
        # Pop the largest remaining angle requirement
        neg_amount, _, angle_row = heapq.heappop(heap)
        angle_amount = -neg_amount

        # Find best factory (least leftover space)
        best_j = -1
        min_leftover = float("inf")

        for j in range(n_factory_rows):
            if available_factories[j] >= angle_amount:
                leftover = available_factories[j] - angle_amount
                if leftover < min_leftover:
                    min_leftover = leftover
                    best_j = j

        # If no complete fit, pick largest capacity
        if best_j == -1:
            max_capacity = 0
            for j in range(n_factory_rows):
                if available_factories[j] > max_capacity:
                    max_capacity = available_factories[j]
                    best_j = j

        # Make assignment if valid factory found
        if best_j != -1 and available_factories[best_j] > 0:
            amount = min(angle_amount, available_factories[best_j])

            # Record assignment
            assignments[angle_row].append((best_j, amount, "best-fit"))

            # Update remaining capacity
            remaining_after = angle_amount - amount
            available_factories[best_j] -= amount

            # If there's remaining angle requirement, re-insert into heap
            if remaining_after > 0:
                heapq.heappush(
                    heap, (-remaining_after, next(unique_counter), angle_row)
                )
        else:
            # No more capacity available
            print(
                "WARNING: Cannot assign remaining {} angles from row {}".format(
                    angle_amount, angle_row
                )
            )
            break

    # Update required_angles to reflect all assignments
    for i in range(n_angle_rows):
        required_angles[i] = 0  # All should be assigned


def run_tmr_row_assignment(
    required_angles_per_row: list[int],
    available_factories_per_row: list[int],
) -> dict:
    """
    Execute the two-phase matching algorithm for magic state assignment.
    All required angles must be assigned to available factories.
    Each factory can hold multiple angle requirements.

    Args:
        required_angles_per_row: list of required angles at each row (must all become 0)
        available_factories_per_row: list of available factory slots at each row (can hold multiple angles)
    Returns:
        result: dictionary containing:
            - required_angles_original: original required angles
            - available_factories_original: original factory capacities
            - required_angles: final required angles (should all be 0)
            - available_factories: final factory capacities
            - assignments: list of assignments for each angle row [(factory_row, amount, phase_type), ...]
            - n_angles: number of angle rows
            - n_factories: number of factory rows
    """
    # Store originals
    required_angles_original = required_angles_per_row[:]
    available_factories_original = available_factories_per_row[:]
    n_angles = len(required_angles_per_row)
    n_factories = len(available_factories_per_row)

    # Working copies
    required_angles = required_angles_per_row[:]
    available_factories = available_factories_per_row[:]

    # Assignments: assignments[i] = [(factory_row_j, amount, phase_type), ...]
    assignments = [[] for _ in range(n_angles)]

    # Run phases
    phase1_exact_match(required_angles, available_factories, assignments)
    phase2_best_fit(required_angles, available_factories, assignments)

    return {
        "required_angles_original": required_angles_original,
        "available_factories_original": available_factories_original,
        "required_angles": required_angles,
        "available_factories": available_factories,
        "assignments": assignments,
        "n_angles": n_angles,
        "n_factories": n_factories,
    }


def assign_factories_for_batch(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    batch_angles: dict[int, dict[int, int]],
):
    """
    Assign factories to prepare the given batch of angles.

    Args:
        factory_pool: FactoryPool
        batch_angles: List of (level, success_rate, angle, target_qubit)

    Returns:
        factory_assignments: List of (angle, qubit, success_rate) for each factory
    """
    # TODO: assignment based on location
    idle_factories = factory_pool.get_idle_factories()
    idx = 0
    # print("assign factory")
    for qubit, demands in batch_angles.items():
        for level, demand in demands.items():
            angle = qubit_trackers[qubit].target_angle * pow(2, level)
            success_rate = calculate_success_rate(angle)
            for i in range(demand):
                factory = idle_factories[idx]
                qubit_trackers[qubit].add_factory(factory.id, angle)
                # print(factory.id, angle, qubit, success_rate)
                factory_pool.assign_factory(factory.id, angle, qubit, success_rate)
                idx += 1
