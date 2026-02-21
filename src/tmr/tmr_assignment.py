from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.ds import FactoryPool, QubitAngleTracker, move_duration, FactoryState

from src.tmr.tmr_assignment_method import (
    assign_factories_for_batch_naive,
    assign_factories_for_batch_matching,
)

from src.config import TMR_P, TMR_Q, CNOT_TIME, SE_TIME, ANGLE_S


def assign_factories_for_batch(
    factory_pool: FactoryPool,
    factories: list,
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
            factories,
            qubit_trackers,
            logic_qubit_locations,
            batch_angles,
            column,
        )
    elif method == "matching":
        assign_factories_for_batch_matching(
            factory_pool,
            factories,
            qubit_trackers,
            logic_qubit_locations,
            batch_angles,
        )
    else:
        raise ValueError(f"Unknown assignment method: {method}")


def reassign_factories(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
):
    """
    Release factories that are no longer needed.
    """
    # return
    busy_factories = factory_pool.get_wait_for_rus_factories()
    cost_matrix = np.zeros((len(busy_factories), len(qubit_trackers)))
    angles_to_factory = defaultdict(list)
    for i, factory in enumerate(busy_factories):
        angles_to_factory[factory.angle].append((i, factory.id))

    # print("angles_to_factory")
    # print(angles_to_factory)
    duration_threshold = SE_TIME * (TMR_P + TMR_Q) - CNOT_TIME

    assignment_idx_to_qubit_anlge_pair = dict()
    i = 0
    for qubit_idx, qubit in qubit_trackers.items():
        x_dst, y_dst = logic_qubit_locations[qubit_idx]
        angle = qubit.target_angle
        assignment_idx_to_qubit_anlge_pair[i] = (qubit_idx, angle)
        for j, factory_id in angles_to_factory[angle]:
            factory = factory_pool.get_factory_by_id(factory_id)
            x_src, y_src = factory.location
            duration = move_duration(x_src, y_src, x_dst, y_dst)
            if duration < duration_threshold:
                cost_matrix[j, i] = duration
        i += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # print("assignment_idx_to_qubit_anlge_pair")
    # print(assignment_idx_to_qubit_anlge_pair)
    # print("cost_matrix")
    # print(cost_matrix)

    used_factories = [False for _ in range(len(busy_factories))]
    # print("Reassignment results:")
    for i, j in zip(row_ind, col_ind):
        factory = busy_factories[i]
        used_factories[i] = True
        qubit, angle = assignment_idx_to_qubit_anlge_pair[j]
        if factory.qubit != qubit:
            if factory.qubit in qubit_trackers:
                qubit_trackers[factory.qubit].remove_factory(factory.id, factory.angle)
            factory.update_qubit(qubit)
            # success_rate = calculate_success_rate(angle)
            qubit_trackers[qubit].add_factory(factory.id, angle)
        # print(
        #     f"Reassigning factory {factory.id} at location {factory.location} preparing angle {factory.angle} to qubit {qubit} who need angle {angle}"
        # )
        # factory_pool.assign_factory(factory.id, angle, qubit, success_rate)

    for i, used in enumerate(used_factories):
        if not used:
            factory = busy_factories[i]
            factory_pool.free_factory(factory.id)
            # print(
            #     f"Releasing factory {factory.id} at location {factory.location} preparing angle {factory.angle}"
            # )
    # input()


def release_useless_factories(
    factory_pool: FactoryPool,
    qubit_trackers: dict[int, QubitAngleTracker],
):
    """
    Release factories that are no longer needed.
    """
    # return
    angles_required_by_qubits = set()
    for qubit, tracker in qubit_trackers.items():
        begin_level = 0
        if tracker.waiting_for_rus:
            begin_level = 1
        for level in range(begin_level, 3):
            angle = tracker.target_angle * pow(2, level)
            if np.isclose(angle, 0.0, atol=1e-8):
                continue
            if ANGLE_S < angle:
                angle -= ANGLE_S
            angles_required_by_qubits.add(angle)

    print(
        "in release_useless_factories: angles_required_by_qubits: ",
        angles_required_by_qubits,
    )
    print(factory_pool.get_factory_by_id(6))
    for factory in factory_pool.factories:
        if (
            factory.state == FactoryState.WAIT_FOR_RUS
            and factory.angle not in angles_required_by_qubits
        ):
            print(
                f"Releasing factory {factory.id} preparing angle {factory.angle} which is not required by any qubit"
            )
            if factory.qubit is not None and factory.qubit in qubit_trackers:
                qubit_trackers[factory.qubit].remove_factory(factory.id, factory.angle)
            factory_pool.free_factory(factory.id)
