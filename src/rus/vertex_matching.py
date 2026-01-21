"""
Vertex matching algorithm for optimal factory assignment.

This module implements optimal bipartite matching to assign factories to qubits
using scipy's Hungarian algorithm, minimizing total Manhattan distance.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


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
            distance = abs(x_q - x_f) + abs(y_q - y_f)
            cost_matrix[i, j] = distance

    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert indices back to qubit/factory IDs
    assignments = []
    for i, j in zip(row_ind, col_ind):
        qubit_id = qubit_list[i]
        factory_id = factory_list[j]
        assignments.append((qubit_id, factory_id))

    return assignments
