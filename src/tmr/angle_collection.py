import math
from collections import defaultdict
import numpy as np
from src.ds import QubitAngleTracker

from src.simulation import (
    calculate_success_rate,
)


def get_angles_for_preparation(
    qubit_trackers: dict[int, QubitAngleTracker],
    n_available_factories: int,
    code_distance: int,
) -> dict[int, dict[int, int]]:
    #  compute demand for qubits
    qubit_demands = defaultdict(int)
    sum_demands = 0
    for qubit, tracker in qubit_trackers.items():
        level_demand = 1
        if tracker.waiting_for_rus:
            level_demand = 2

        for level in range(0, 2):
            angle = tracker.get_angle_level(level)
            if np.isclose(angle, 0.0, atol=1e-8):
                continue
            success_rate = calculate_success_rate(angle, code_distance)
            demand = level_demand / success_rate
            level_demand /= 2
            qubit_demands[qubit] += demand
            sum_demands += demand

    # allocate factories for qubit based on the level 1 and level 2 demands (due to expectation value)
    ideal_factory_allocation = {
        qubit: n_available_factories * demand / sum_demands
        for qubit, demand in qubit_demands.items()
    }
    # print("ideal_factory_allocation:")
    # print(ideal_factory_allocation)
    allocation = integer_allocation(n_available_factories, ideal_factory_allocation)
    # print("allocation")
    # print(allocation)
    qubit_angle_factories = {}
    level_threshold = 3
    # print("Allocating factories for qubits:")
    # print(allocation)
    for qubit, tracker in qubit_trackers.items():
        if qubit not in allocation or allocation[qubit] < 1:
            continue
        level_demand = 1
        demands = {}
        sum_demands = 0
        for level in range(0, level_threshold):
            angle = tracker.get_angle_level(level)
            if np.isclose(angle, 0.0, atol=1e-8):
                continue
            # assert angle > 0
            success_rate = calculate_success_rate(angle, code_distance)
            # if qubit == 7:
            #     print("angle")
            #     print(angle)
            #     print("success_rate")
            #     print(success_rate)
            demand = level_demand / success_rate
            if level > 1 and allocation[qubit] < sum_demands:
                break
            demands[level] = demand
            sum_demands += demand
            level_demand /= 2
        for level, demand in demands.items():
            demands[level] = demand / sum_demands * allocation[qubit]
        # if qubit == 7:
        #     print("demands for qubit 7")
        #     print(demands)
        # print("demands")
        # print(demands)
        qubit_angle_factories[qubit] = integer_allocation(allocation[qubit], demands)
    # print("qubit_angle_factories")
    # print(qubit_angle_factories)
    # assert n_available_factories == 5
    count = 0
    for allocation in qubit_angle_factories.values():
        count += sum(allocation.values())
    assert count == n_available_factories
    # input()
    return qubit_angle_factories


def integer_allocation(n: int, demands: dict[int, float]) -> dict[int, int]:
    # allocate factories for qubit based on the level 1 and level 2 demands (due to expectation value)
    # Largest Remainder Method
    # Compute the ideal fractional allocation
    # Take the floor of each
    # Distribute the remaining factories to the qubits with the largest fractional remainders
    # Step 2: take floors
    allocation = {idx: math.floor(v) for idx, v in demands.items()}

    # Step 3: distribute remaining factories
    remaining = n - sum(allocation.values())

    # sort by fractional remainder (descending)
    remainders = sorted(
        demands.items(),
        key=lambda x: x[1] - math.floor(x[1]),
        reverse=True,
    )

    for qubit, _ in remainders[:remaining]:
        allocation[qubit] += 1

    return allocation
