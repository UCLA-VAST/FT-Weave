from src.ds import FactoryPool, QubitAngleTracker, move_duration
from src.simulation import (
    calculate_success_rate,
)

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_factories_for_batch_matching(
    factory_pool: FactoryPool,
    factories: list,
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
    cost_matrix = np.zeros((len(factories), len(factories)))
    assignment_idx_to_qubit_anlge_pair = dict()
    # level_constant = 10  # constant to prioritize lower level angles
    # print("batch_angles:")
    # print(batch_angles)
    for i, factory in enumerate(factories):
        x_src, y_src = factory.location
        idx = 0
        for qubit, demands in batch_angles.items():
            x_dst, y_dst = logic_qubit_locations[qubit]
            for level, demand in demands.items():
                # Manhattan distance as cost
                angle = qubit_trackers[qubit].target_angle * pow(2, level)
                for _ in range(demand):
                    cost_matrix[i, idx] = move_duration(x_src, y_src, x_dst, y_dst) / (
                        level + 1
                    )
                    assignment_idx_to_qubit_anlge_pair[idx] = (qubit, angle)
                    idx += 1
    # print("cost_matrix:")
    # print(cost_matrix)
    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # for i, j in zip(row_ind, col_ind):
    #     print(
    #         "Factory {} assigned to qubit {}, angle {}".format(
    #             idle_factories[i].id,
    #             assignment_idx_to_qubit_anlge_pair[j][0],
    #             assignment_idx_to_qubit_anlge_pair[j][1],
    #         )
    #     )
    # input()
    # assign factory based on solution
    for i, j in zip(row_ind, col_ind):
        factory = factories[i]
        qubit, angle = assignment_idx_to_qubit_anlge_pair[j]
        success_rate = calculate_success_rate(angle)
        qubit_trackers[qubit].add_factory(factory.id, angle)
        factory_pool.assign_factory(factory.id, angle, qubit, success_rate)
