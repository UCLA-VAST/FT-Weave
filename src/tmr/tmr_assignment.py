from collections import defaultdict
import heapq
from itertools import count
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.ds.device_state import FactoryPool, QubitAngleTracker
from src.ds.architecture import move_duration
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
    for j, cap in sorted(available_factories.items()):
        if cap > 0:
            value_to_indices[cap].append(j)

    # For each angle row, check whether there's a factory with exactly the
    # same remaining capacity. If so, assign and remove that factory from
    # the map (its capacity becomes zero).
    for i, req in sorted(required_angles.items()):
        if req <= 0:
            continue

        indices = value_to_indices.get(req)
        if not indices:
            continue

        # Pop the first matching factory index (use pop(0) to preserve order).
        j = indices.pop(0)

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


def run_tmr_assignment(
    required_angles_per_position: dict[int, int],
    available_factories_per_position: dict[int, int],
    column: bool = False,
) -> dict:
    """
    Execute the two-phase matching algorithm for magic state assignment.
    All required angles must be assigned to available factories.
    Each factory can hold multiple angle requirements.

    Args:
        required_angles_per_position: dict of required angles at each row/column (must all become 0)
        available_factories_per_position: dict of available factory slots at each row/column (can hold multiple angles)
        column: if False, use row-based grouping (y-coordinate); if True, use column-based grouping (x-coordinate)
    Returns:
        result: dictionary containing:
            - required_angles_original: original required angles
            - available_factories_original: original factory capacities
            - required_angles: final required angles (should all be 0)
            - available_factories: final factory capacities
            - assignments: dict of assignments for each angle position [(factory_position, amount, phase_type), ...]
            - n_angle_positions: number of angle rows/columns
            - n_factory_positions: number of factory rows/columns
            - dimension: 'column' if column=True, else 'row'
    """
    # Store originals
    required_angles_original = required_angles_per_position.copy()
    available_factories_original = available_factories_per_position.copy()
    n_angle_positions = len(required_angles_per_position)
    n_factory_positions = len(available_factories_per_position)

    # Working copies
    required_angles = required_angles_per_position.copy()
    available_factories = available_factories_per_position.copy()

    # Assignments: assignments[i] = [(factory_position_j, amount, phase_type), ...]
    assignments = {i: [] for i in required_angles_per_position.keys()}

    # Run phases
    phase1_exact_match(required_angles, available_factories, assignments)
    phase2_best_fit(required_angles, available_factories, assignments)

    return {
        "required_angles_original": required_angles_original,
        "available_factories_original": available_factories_original,
        "required_angles": required_angles,
        "available_factories": available_factories,
        "assignments": assignments,
        "n_angle_positions": n_angle_positions,
        "n_factory_positions": n_factory_positions,
        "dimension": "column" if column else "row",
    }


def assign_factories_for_batch_naive(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    batch_angles: dict[int, dict[int, int]],
    column: bool = False,
):
    """
    Assign factories to prepare the given batch of angles.

    Args:
        factory_pool: FactoryPool
        qubit_trackers: dict of QubitAngleTracker indexed by qubit id
        logic_qubit_locations: (x, y) location for each logical qubit
        batch_angles: dict of required angles per level and qubit
        column: if False, use row-based matching (group by y-coordinate);
                if True, use column-based matching (group by x-coordinate)

    Returns:
        None (modifies factory_pool and qubit_trackers in place)
    """
    idle_factories = factory_pool.get_idle_factories()
    required_angles_per_level_position = dict()
    available_factories_per_position = defaultdict(int)
    qubit_by_position = defaultdict(list)
    factory_by_position = defaultdict(list)

    # Group by column (x) if column=True, otherwise by row (y)
    for qubit, demands in batch_angles.items():
        x, y = logic_qubit_locations[qubit]
        pos = x if column else y
        other_coord = y if column else x
        qubit_by_position[pos].append((other_coord, qubit))
        for level, demand in demands.items():
            if level not in required_angles_per_level_position:
                required_angles_per_level_position[level] = defaultdict(int)
            required_angles_per_level_position[level][pos] += demand

    for pos in qubit_by_position.keys():
        qubit_by_position[pos] = sorted(qubit_by_position[pos], key=lambda x: x[0])

    for factory in idle_factories:
        x, y = factory.location
        pos = x if column else y
        other_coord = y if column else x
        available_factories_per_position[pos] += 1
        factory_by_position[pos].append((other_coord, factory.id))

    for pos in factory_by_position.keys():
        factory_by_position[pos] = sorted(factory_by_position[pos], key=lambda x: x[0])

    used_factories = set()
    for (
        level,
        required_angles_per_position,
    ) in required_angles_per_level_position.items():
        result = run_tmr_assignment(
            required_angles_per_position,
            available_factories_per_position,
            column=column,
        )

        # Assign angles to factories based on result
        for angle_pos, factory_assignments in result["assignments"].items():
            qubits_at_pos = qubit_by_position[angle_pos]
            qubit_idx = 0
            fulfilled_demand = 0
            for factory_pos, amount, _ in factory_assignments:
                # Get factories at factory_pos
                factories_at_pos = factory_by_position[factory_pos]

                # Assign 'amount' angles from this angle_pos to factories at factory_pos
                angles_assigned = 0
                idx = 0

                while angles_assigned < amount:
                    # Get the qubit and factory
                    _, qubit = qubits_at_pos[qubit_idx]
                    while factories_at_pos[idx][1] in used_factories:
                        idx += 1
                    _, factory_id = factories_at_pos[idx]

                    # Assign this angle to the factory
                    angle = qubit_trackers[qubit].target_angle * pow(2, level)
                    success_rate = calculate_success_rate(angle)
                    qubit_trackers[qubit].add_factory(factory_id, angle)
                    factory_pool.assign_factory(factory_id, angle, qubit, success_rate)
                    used_factories.add(factory_id)
                    available_factories_per_position[factory_pos] -= 1
                    angles_assigned += 1
                    fulfilled_demand += 1
                    idx += 1
                    if fulfilled_demand == batch_angles[qubit][level]:
                        qubit_idx += 1
                        fulfilled_demand = 0


def assign_factories_for_batch_matching(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    batch_angles: dict[int, dict[int, int]],
):
    """
    Assign factories to prepare the given batch of angles.

    Args:
        factory_pool: FactoryPool
        qubit_trackers: dict of QubitAngleTracker indexed by qubit id
        logic_qubit_locations: (x, y) location for each logical qubit
        batch_angles: dict of required angles per level and qubit

    Returns:
        None (modifies factory_pool and qubit_trackers in place)
    """
    idle_factories = factory_pool.get_idle_factories()

    cost_matrix = np.zeros((len(idle_factories), len(idle_factories)))
    assignment_idx_to_qubit_anlge_pair = dict()
    level_constant = 10  # constant to prioritize lower level angles
    for i, factory in enumerate(idle_factories):
        x_src, y_src = factory.location
        idx = 0
        for qubit, demands in batch_angles.items():
            x_dst, y_dst = logic_qubit_locations[qubit]
            for level, demand in demands.items():
                # Manhattan distance as cost
                angle = qubit_trackers[qubit].target_angle * pow(2, level)
                for _ in range(demand):
                    cost_matrix[i, idx] = (
                        move_duration(x_src, y_src, x_dst, y_dst)
                        + level * level_constant
                    )
                    assignment_idx_to_qubit_anlge_pair[idx] = (qubit, angle)
                    idx += 1
    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # assign factory based on solution
    for i, j in zip(row_ind, col_ind):
        factory = idle_factories[i]
        qubit, angle = assignment_idx_to_qubit_anlge_pair[j]
        success_rate = calculate_success_rate(angle)
        qubit_trackers[qubit].add_factory(factory.id, angle)
        factory_pool.assign_factory(factory.id, angle, qubit, success_rate)


def assign_factories_for_batch(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    batch_angles: dict[int, dict[int, int]],
    column: bool = False,
    method: str = "matching",
):
    """
    Assign factories to prepare the given batch of angles.

    Args:
        factory_pool: FactoryPool
        qubit_trackers: dict of QubitAngleTracker indexed by qubit id
        logic_qubit_locations: (x, y) location for each logical qubit
        batch_angles: dict of required angles per level and qubit
        column: if False, use row-based matching (group by y-coordinate);
                if True, use column-based matching (group by x-coordinate)

    Returns:
        None (modifies factory_pool and qubit_trackers in place)
    """
    if method == "naive":
        assign_factories_for_batch_naive(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
            batch_angles,
            column,
        )
    elif method == "matching":
        assign_factories_for_batch_matching(
            factory_pool,
            qubit_trackers,
            logic_qubit_locations,
            batch_angles,
        )
    else:
        raise ValueError(f"Unknown assignment method: {method}")
