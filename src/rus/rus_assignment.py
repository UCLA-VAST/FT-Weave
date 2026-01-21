from collections import defaultdict
from typing import Dict, List, Tuple
from src.ds.device_state import FactoryPool, QubitAngleTracker
from src.ds.architecture import move_duration
from src.rus.angle_factory_index import AngleFactoryIndex
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_teleportation_with_sharing(
    successful_qubits: set[int],
    qubit_trackers: Dict[int, QubitAngleTracker],
    factory_pool: FactoryPool,
    logic_qubit_locations: List[Tuple[int, int]],
    angle_factory_index: AngleFactoryIndex,
) -> List[Tuple[int, int]]:
    """
    Assign factories to qubits for RUS injection with cross-qubit factory sharing.

    This enhanced version can assign factories from other qubits if:
    1. Those factories prepare the same angle
    2. The other qubit's factories failed
    3. Minimizing the total moving distance via vertex matching

    Args:
        successful_qubits: Set of qubits that have already been successfully injected
        qubit_trackers: Dict mapping qubit_id to QubitAngleTracker with angle info
        factory_pool: Pool of available factories
        logic_qubit_locations: List of (x, y) locations for each logical qubit
        angle_factory_index: Optional pre-built index of angles to factories.
                            If None, will be built from qubit_trackers.

    Returns:
        List of (qubit, factory_id) assignment pairs with optimal distance
    """

    # Organize qubits by their required angle
    angle_to_qubits: Dict[float, List[int]] = defaultdict(list)

    for qubit in qubit_trackers:
        if qubit in successful_qubits:
            continue

        # Get the angle for this qubit (first generation ready for injection)
        tracker = qubit_trackers[qubit]
        angle = tracker.target_angle
        angle_to_qubits[angle].append(qubit)

    qubit_factory_pairs = []
    assigned_qubits = set()
    # print(f"angle_to_qubits: {angle_to_qubits}")
    # Process each angle group separately
    for angle, qubits_for_angle in angle_to_qubits.items():
        # Get all factories preparing this angle
        factories_for_angle = angle_factory_index.get_factories_for_angle(angle)

        # Convert to dict format: factory_id -> location
        available_factories = {}
        for factory_id, _ in factories_for_angle:
            factory = factory_pool.get_factory_by_id(factory_id)
            if factory is not None:
                available_factories[factory_id] = factory.location

        # print(f"available_factories: {available_factories}")
        if not available_factories:
            continue

        # Get locations of qubits needing this angle
        qubit_locations_dict = {q: logic_qubit_locations[q] for q in qubits_for_angle}

        # Find optimal matching using vertex matching algorithm
        matches = find_optimal_factory_assignment(
            qubits_for_angle,
            available_factories,
            qubit_locations_dict,
        )

        qubit_factory_pairs.extend(matches)
        assigned_qubits.update([q for q, _ in matches])

    # Handle any remaining qubits not in angle groups (shouldn't happen in normal flow)
    # for qubit in qubit_trackers:
    #     if qubit in successful_qubits or qubit in assigned_qubits:
    #         continue
    #     else:
    #         tracker = qubit_trackers[qubit]
    #         for factory, angle in tracker.factories:
    #             if np.isclose(angle, tracker.target_angle):
    #                 assert False

    return qubit_factory_pairs


def find_optimal_factory_assignment(
    qubits_needing_rus: List[int],
    available_factories: Dict[int, Tuple[int, int]],
    qubit_locations: Dict[int, Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Find optimal factory-to-qubit assignment minimizing total moving distance.

    Uses scipy's linear_sum_assignment (Hungarian algorithm) for maximal cardinality
    matching with minimal weight.

    When num_factories >= num_qubits: Find best matching for all qubits.
    When num_factories < num_qubits: Assign only the closest qubits to factories.

    Args:
        qubits_needing_rus: List of qubit IDs that need RUS injection
        available_factories: Dict mapping factory_id to (x, y) location
        qubit_locations: Dict mapping qubit_id to (x, y) location

    Returns:
        List of (qubit_id, factory_id) assignment pairs
    """
    if not qubits_needing_rus or not available_factories:
        return []

    # Create cost matrix: rows are qubits, columns are factories
    qubit_list = list(qubits_needing_rus)
    factory_list = list(available_factories.keys())

    cost_matrix = np.zeros((len(qubit_list), len(factory_list)))

    for i, qubit_id in enumerate(qubit_list):
        x_q, y_q = qubit_locations[qubit_id]
        for j, factory_id in enumerate(factory_list):
            x_f, y_f = available_factories[factory_id]
            # Manhattan distance as cost
            cost_matrix[i, j] = move_duration(x_q, y_q, x_f, y_f)

    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert indices back to qubit/factory IDs
    assignments = []
    for i, j in zip(row_ind, col_ind):
        qubit_id = qubit_list[i]
        factory_id = factory_list[j]
        assignments.append((qubit_id, factory_id))

    return assignments
