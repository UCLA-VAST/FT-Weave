from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.ds import TFactoryPool, move_duration
from src.star.rus.rus_routing import two_layer_routing
from src.t_cultivation.config import RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT


def rus_teleportation(
    qubits_to_teleport: list[int],
    logic_qubit_locations: list[tuple[int, int]],
    factory_pool: TFactoryPool,
    available_factory_ids: list[int],
    qubit_to_node: Optional[dict[int, int]] = None,
    longest_path_to_sink: Optional[list[int]] = None,
    critical_path_weight: Optional[float] = None,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int, int, int, int, int]]]]:

    available_factories = {}
    for factory_id in available_factory_ids:
        factory = factory_pool.get_factory_by_id(factory_id)
        available_factories[factory_id] = factory.location
    w = (
        RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT
        if critical_path_weight is None
        else critical_path_weight
    )
    qubit_factory_pairs = assign_teleportation(
        qubits_to_teleport,
        logic_qubit_locations,
        available_factories,
        qubit_to_node=qubit_to_node,
        longest_path_to_sink=longest_path_to_sink,
        critical_path_weight=w,
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
    qubit_to_node: Optional[dict[int, int]] = None,
    longest_path_to_sink: Optional[list[int]] = None,
    critical_path_weight: float = 0.0,
) -> list[tuple[int, int]]:
    """
    Find optimal factory-to-qubit assignment minimizing total cost.

    Uses scipy's linear_sum_assignment (Hungarian algorithm) for maximal cardinality
    matching with minimal weight.

    Cost per (qubit, factory) is::
        move_duration(qubit, factory)
        + critical_path_weight * (batch_max_lp - lp[instruction_node])

    where ``lp[node]`` is the longest path length (in instructions) from that T gate
    node to a sink in the circuit DAG. The most critical T in the batch has
    ``batch_max_lp - lp == 0`` (no extra cost); less critical gates pay a higher cost
    so the matcher favors shorter moves for instructions on the critical path.

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

    use_critical = (
        critical_path_weight > 0
        and longest_path_to_sink is not None
        and qubit_to_node is not None
    )
    lp_per_row: list[int] = []
    batch_max_lp = 1
    if use_critical:
        assert qubit_to_node is not None and longest_path_to_sink is not None
        for qubit_id in qubit_list:
            nid = qubit_to_node.get(qubit_id)
            if nid is None or nid < 0 or nid >= len(longest_path_to_sink):
                lp_per_row.append(1)
            else:
                lp_per_row.append(longest_path_to_sink[nid])
        batch_max_lp = max(lp_per_row) if lp_per_row else 1

    cost_matrix = np.zeros((len(qubit_list), len(factory_list)))

    for i, qubit_id in enumerate(qubit_list):
        x_q, y_q = qubit_locations[qubit_id]
        for j, factory_id in enumerate(factory_list):
            x_f, y_f = available_factories[factory_id]
            dist = move_duration(x_q, y_q, x_f, y_f)
            if use_critical:
                crit = batch_max_lp - lp_per_row[i]
                cost_matrix[i, j] = dist + critical_path_weight * crit
            else:
                cost_matrix[i, j] = dist

    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert indices back to qubit/factory IDs
    assignments = []
    for i, j in zip(row_ind, col_ind):
        qubit_id = qubit_list[i]
        factory_id = factory_list[j]
        assignments.append((qubit_id, factory_id))

    return assignments
