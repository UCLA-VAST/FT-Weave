from collections import defaultdict
import heapq
from itertools import count

from src.ds.device_state import FactoryPool, QubitAngleTracker

from src.simulation import (
    calculate_success_rate,
)


def phase1_exact_match(
    required_angles: dict[int, int],
    available_factories: dict[int, int],
    assignments: dict[int, list[tuple[int, int, str]]],
) -> None:
    """
    Phase 1: Exact one-to-one match
    If required_angles[i] = available_factories[j], assign required_angles[i] to available_factories[j]

    Args:
        required_angles: working copy of required angles per row (modified in place)
        available_factories: working copy of available factories per row (modified in place)
        assignments: dict of assignments for each angle row (modified in place)
    Returns:
    """

    # Build a map from available factory capacity -> list of factory indices
    # Only include factories with positive capacity. This lets us find an exact
    # match for a required angle in O(1) average time per angle, making the
    # whole phase O(n).
    value_to_indices = defaultdict(list)
    for j, cap in available_factories.items():
        if cap > 0:
            value_to_indices[cap].append(j)

    # For each angle row, check whether there's a factory with exactly the
    # same remaining capacity. If so, assign and remove that factory from
    # the map (its capacity becomes zero).
    for i in list(required_angles.keys()):
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
        if i not in assignments:
            assignments[i] = []
        assignments[i].append((j, match_amount, "exact"))

        # Update remaining capacity (factory becomes zero since exact match)
        required_angles[i] = 0
        available_factories[j] = 0

        # Clean up map entry if no more factories with that capacity
        if not indices:
            del value_to_indices[req]


def phase2_best_fit(
    required_angles: dict[int, int],
    available_factories: dict[int, int],
    assignments: dict[int, list[tuple[int, int, str]]],
) -> None:
    """
    Phase 2: Best-fit with sorted assignment
    Always work on the largest remaining required angles first.
    If angles don't fit completely in one factory, split and re-insert into sorted list.
    Continue until all angles are assigned.

    Args:
        required_angles: working copy of required angles per row (modified in place)
        available_factories: working copy of available factories per row (modified in place)
        assignments: dict of assignments for each angle row (modified in place)
    Returns:
    """

    # Create a max-heap of (-amount, counter, angle_row) for unassigned angles
    # Using a heap gives O(log n) insertion and removal compared to O(n)
    # list insert/pop(0) used previously.
    heap = []
    unique_counter = count()
    for i, amount in required_angles.items():
        if amount > 0:
            # push negative amount to simulate max-heap
            heapq.heappush(heap, (-amount, next(unique_counter), i))

    # Keep working until all angles are assigned
    while heap:
        # Pop the largest remaining angle requirement
        neg_amount, _, angle_row = heapq.heappop(heap)
        angle_amount = -neg_amount

        # Find best factory (least leftover space)
        best_j = -1
        min_leftover = float("inf")

        for j, capacity in available_factories.items():
            if capacity >= angle_amount:
                leftover = capacity - angle_amount
                if leftover < min_leftover:
                    min_leftover = leftover
                    best_j = j

        # If no complete fit, pick largest capacity
        if best_j == -1:
            max_capacity = 0
            for j, capacity in available_factories.items():
                if capacity > max_capacity:
                    max_capacity = capacity
                    best_j = j

        # Make assignment if valid factory found
        if best_j != -1 and available_factories[best_j] > 0:
            amount = min(angle_amount, available_factories[best_j])

            # Record assignment
            if angle_row not in assignments:
                assignments[angle_row] = []
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
    for i in required_angles:
        required_angles[i] = 0  # All should be assigned


def run_tmr_row_assignment(
    required_angles_per_row: dict[int, int],
    available_factories_per_row: dict[int, int],
) -> dict:
    """
    Execute the two-phase matching algorithm for magic state assignment.
    All required angles must be assigned to available factories.
    Each factory can hold multiple angle requirements.

    Args:
        required_angles_per_row: dict of required angles at each row (must all become 0)
        available_factories_per_row: dict of available factory slots at each row (can hold multiple angles)
    Returns:
        result: dictionary containing:
            - required_angles_original: original required angles
            - available_factories_original: original factory capacities
            - required_angles: final required angles (should all be 0)
            - available_factories: final factory capacities
            - assignments: dict of assignments for each angle row [(factory_row, amount, phase_type), ...]
            - n_angle_rows: number of angle rows
            - n_factory_rows: number of factory rows
    """
    # Store originals
    required_angles_original = required_angles_per_row.copy()
    available_factories_original = available_factories_per_row.copy()
    n_angle_rows = len(required_angles_per_row)
    n_factory_rows = len(available_factories_per_row)

    # Working copies
    required_angles = required_angles_per_row.copy()
    available_factories = available_factories_per_row.copy()

    # Assignments: assignments[i] = [(factory_row_j, amount, phase_type), ...]
    assignments = {i: [] for i in required_angles_per_row.keys()}

    # Run phases
    phase1_exact_match(required_angles, available_factories, assignments)
    phase2_best_fit(required_angles, available_factories, assignments)

    return {
        "required_angles_original": required_angles_original,
        "available_factories_original": available_factories_original,
        "required_angles": required_angles,
        "available_factories": available_factories,
        "assignments": assignments,
        "n_angle_rows": n_angle_rows,
        "n_factory_rows": n_factory_rows,
    }


def assign_factories_for_batch(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    batch_angles: dict[int, dict[int, int]],
):
    """
    Assign factories to prepare the given batch of angles.

    Args:
        factory_pool: FactoryPool
        batch_angles: List of (level, success_rate, angle, target_qubit)
        logic_qubit_locations: (x, y) location for each logical qubit

    Returns:
        factory_assignments: List of (angle, qubit, success_rate) for each factory
    """
    # TODO: assignment based on location, i.e., distance
    idle_factories = factory_pool.get_idle_factories()
    idx = 0
    # print("assign factory")
    required_angles_per_level_row = dict()
    available_factories_per_row = defaultdict(int)
    qubit_by_row = defaultdict(list)
    factory_by_row = defaultdict(list)

    for qubit, demands in batch_angles.items():
        x, y = logic_qubit_locations[qubit]
        qubit_by_row[y].append((x, qubit))
        for level, demand in demands.items():
            if level not in required_angles_per_level_row:
                required_angles_per_level_row[level] = defaultdict(int)
            required_angles_per_level_row[level][y] += demand

    for y in qubit_by_row.keys():
        qubit_by_row[y] = sorted(qubit_by_row[y], key=lambda x: x[0])

    for factory in idle_factories:
        x, y = factory.location
        available_factories_per_row[y] += 1
        factory_by_row[y].append((x, factory.id))

    for y in factory_by_row.keys():
        factory_by_row[y] = sorted(factory_by_row[y], key=lambda x: x[0])

    used_factories = set()
    for level, required_angles_per_row in required_angles_per_level_row.items():
        result = run_tmr_row_assignment(
            required_angles_per_row, available_factories_per_row
        )

        # from src.util import print_tmr_assignment_results

        # print_tmr_assignment_results(result)

        # Assign angles to factories based on result using row-based assignments
        # Process assignments from result
        # print("batch_angles")
        # print(batch_angles)
        for angle_row, factory_assignments in result["assignments"].items():
            qubits_at_row = qubit_by_row[angle_row]
            qubit_idx = 0
            fulfilled_demand = 0
            for factory_row, amount, _ in factory_assignments:
                # Get factories at factory_row
                factories_at_row = factory_by_row[factory_row]

                # Assign 'amount' angles from this angle_row to factories at factory_row
                angles_assigned = 0
                idx = 0

                while angles_assigned < amount:
                    # Get the qubit and factory
                    _, qubit = qubits_at_row[qubit_idx]
                    while factories_at_row[idx][1] in used_factories:
                        idx += 1
                    _, factory_id = factories_at_row[idx]

                    # Assign this angle to the factory
                    angle = qubit_trackers[qubit].target_angle * pow(2, level)
                    success_rate = calculate_success_rate(angle)
                    qubit_trackers[qubit].add_factory(factory_id, angle)
                    factory_pool.assign_factory(factory_id, angle, qubit, success_rate)
                    used_factories.add(factory_id)
                    # print(
                    #     f"assign factory {factory_id} for qubit {qubit} with level {level}"
                    # )
                    available_factories_per_row[factory_row] -= 1
                    angles_assigned += 1
                    fulfilled_demand += 1
                    idx += 1
                    # print("used_factories")
                    # print(used_factories)
                    # print(
                    #     f"fulfilled_demand: {fulfilled_demand}, required demand: {batch_angles[qubit][level]}"
                    # )
                    if fulfilled_demand == batch_angles[qubit][level]:
                        qubit_idx += 1
                        fulfilled_demand = 0
    # input()
