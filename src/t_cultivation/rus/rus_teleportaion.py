from src.star.rus.rus_routing import two_layer_routing
from src.ds import TFactoryPool, move_duration
import numpy as np
from scipy.optimize import linear_sum_assignment


def rus_teleportation(
    qubits_to_teleport: list[int],
    logic_qubit_locations: list[tuple[int, int]],
    factory_pool: TFactoryPool,
    available_factory_ids: list[int],
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int, int, int, int, int]]]]:

    available_factories = {}
    for factory_id in available_factory_ids:
        factory = factory_pool.get_factory_by_id(factory_id)
        available_factories[factory_id] = factory.location
    qubit_factory_pairs = assign_teleportation(
        qubits_to_teleport,
        logic_qubit_locations,
        available_factories,
    )
    if not qubit_factory_pairs:
        return [], []

    routing_batches = two_layer_routing(
        factory_pool, logic_qubit_locations, qubit_factory_pairs
    )

    return qubit_factory_pairs, routing_batches


def assign_teleportation(
    qubits_to_teleport: list[int],
    qubit_locations: list[tuple[int, int]],
    available_factories: dict[int, tuple[int, int]],
) -> list[tuple[int, int]]:
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
    if not qubits_to_teleport or not available_factories:
        return []

    # Create cost matrix: rows are qubits, columns are factories
    qubit_list = list(qubits_to_teleport)
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
