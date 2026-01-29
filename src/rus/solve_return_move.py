from collections import defaultdict

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from src.ds.architecture import move_duration
from src.rus.two_layer_routing import chain_decomposition_matching
from src.ds.device_state import FactoryPool


def solve_return_move(routing_batches: list[list[tuple]], factory_pool: FactoryPool):
    """
    Relax routing: used to move factories back to the empty spot after CNOT.
    In this case, we don't need to worry ghost spot.
    Args;
    routing_batch: list of list of tuple (qubit, x_q, y_q, factory_id, x_f, y_f, reverse)
    """
    # print("enter solve_return_move")
    empty_locations = []  # (x, y)
    initial_locations = []  # (factory_id, x, y)
    for batches in routing_batches:
        for _, x_q, y_q, factory_id, x_f, y_f in batches:
            empty_locations.append((x_f, y_f))
            initial_locations.append((factory_id, x_q, y_q))
    # print("empty_locations")
    # print(empty_locations)
    # print("initial_locations")
    # print(initial_locations)
    assignments = find_return_location(initial_locations, empty_locations)
    # update factory assignment
    for factory_id, x_src, y_src, x_dst, y_dst in assignments:
        factory = factory_pool.get_factory_by_id(factory_id)
        factory.set_location((x_dst, y_dst))

    # route
    return_routing_batches = gready_routing(assignments)
    # print("return_routing_batches")
    # print(return_routing_batches)
    # print("leave solve_return_move")
    return return_routing_batches


def find_return_location(
    initial_locations: list[tuple],
    empty_locations: list[tuple],
) -> list[tuple[int, int, int, int, int]]:
    """
    Find optimal factory-to-empty-spot assignment minimizing total moving distance.

    Uses scipy's linear_sum_assignment (Hungarian algorithm) for maximal cardinality
    matching with minimal weight.

    Args:
        empty_locations: lists of empty locations (x, y)
        initial_locations: lists of initial locations for the factories (factory_id, x, y)

    Returns:
        List of (factory_id, x_src, y_src, x_dst, y_dst) assignment pairs
    """

    # Create cost matrix: rows are inital locations, columns are dst locations
    cost_matrix = np.zeros((len(initial_locations), len(empty_locations)))

    for i, (factory_id, x_src, y_src) in enumerate(initial_locations):
        for j, (x_dst, y_dst) in enumerate(empty_locations):
            # Manhattan distance as cost
            cost_matrix[i, j] = move_duration(x_src, y_src, x_dst, y_dst)

    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert indices back to qubit/factory IDs
    assignments = []
    for i, j in zip(row_ind, col_ind):
        factory_id, x_src, y_src = initial_locations[i]
        x_dst, y_dst = empty_locations[j]
        assignments.append((factory_id, x_src, y_src, x_dst, y_dst))

    return assignments


def compatible_1d(a_src, a_dst, b_src, b_dst):
    if a_src == b_src:
        return a_dst == b_dst
    return (a_src < b_src) == (a_dst < b_dst)


def compatible_move(
    batch: list[tuple[int, int, int, int, int, int]],
    x_src: int,
    y_src: int,
    x_dst: int,
    y_dst: int,
) -> bool:
    for _, x1_dst, y1_dst, factory_id, x1_src, y1_src in batch:
        # same src x diff dst x
        if not (
            compatible_1d(x_src, x_dst, x1_src, x1_dst)
            and compatible_1d(y_src, y_dst, y1_src, y1_dst)
        ):
            return False
    return True


def compatible_transfer(
    batch: list[tuple[int, int, int, int, int, int]],
    occupation_matrix: npt.NDArray,
) -> tuple[list, list]:
    # print("occupation_matrix")
    # print(occupation_matrix)
    removed_vec = set()
    xy_idx = dict()
    for idx, (_, _, _, _, x, y) in enumerate(batch):
        xy_idx[(x, y)] = idx
    solve = False
    while not solve:
        conflict_count = defaultdict(int)
        has_conflicts = False

        for i in range(len(batch)):
            if i in removed_vec:
                continue
            _, _, _, _, x0, y0 = batch[i]
            for j in range(i + 1, len(batch)):
                if j in removed_vec:
                    continue
                _, _, _, _, x1, y1 = batch[j]
                cross_0 = (x0, y1)
                cross_1 = (x1, y0)
                if occupation_matrix[x0, y1]:
                    if cross_0 in xy_idx and xy_idx[cross_0] in removed_vec:
                        has_conflicts = True
                if occupation_matrix[x1, y0]:
                    if cross_1 in xy_idx and xy_idx[cross_1] in removed_vec:
                        has_conflicts = True
                if has_conflicts:
                    conflict_count[(x0, y0)] += 1
                    conflict_count[(x1, y1)] += 1
        solve = not has_conflicts
        assert solve
        if has_conflicts:
            # remove the activation involved in the most conflicts
            worst = max(conflict_count.items(), key=lambda kv: kv[1])[0]
            idx = xy_idx[worst]
            removed_vec.add(idx)
    assert len(removed_vec) == 0
    new_batch = [element for i, element in enumerate(batch) if i not in removed_vec]

    return new_batch, list(removed_vec)


def gready_routing(
    assignments: list[tuple[int, int, int, int, int]],
) -> list[list[tuple[int, int, int, int, int, int]]]:
    """
    Two layer routing
    """
    # print("assignments")
    # print(assignments)
    # extract all source coordinates
    xs = [x_src for _, x_src, y_src, _, _ in assignments]
    ys = [y_src for _, x_src, y_src, _, _ in assignments]

    # matrix size (assumes 0-based indexing)
    max_x = max(xs)
    max_y = max(ys)

    # initialize matrix with zeros
    occupation_matrix = np.zeros((max_x + 1, max_y + 1), dtype=bool)

    # fill in ones
    for _, x_src, y_src, _, _ in assignments:
        occupation_matrix[x_src, y_src] = True

    remaining_assignment_idx = list(range(len(assignments)))
    routing_batches = []

    while remaining_assignment_idx:
        print(len(remaining_assignment_idx))
        new_remaining_assignment_idx = []
        batch = []
        for idx in remaining_assignment_idx:
            factory_id, x_src, y_src, x_dst, y_dst = assignments[idx]
            if compatible_move(
                batch,
                x_src,
                y_src,
                x_dst,
                y_dst,
            ):
                batch.append((-1, x_dst, y_dst, factory_id, x_src, y_src))
            else:
                new_remaining_assignment_idx.append(idx)
        batch, tmp_remaining_assignment = compatible_transfer(batch, occupation_matrix)
        routing_batches.append(batch)
        remaining_assignment_idx = tmp_remaining_assignment + [
            i for i in new_remaining_assignment_idx
        ]
    return routing_batches
