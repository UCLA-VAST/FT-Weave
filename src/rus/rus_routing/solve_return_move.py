from collections import defaultdict

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from src.ds import move_duration, FactoryPool


def solve_return_move(
    routing_batches: list[list[tuple]],
    factory_pool: FactoryPool,
    decompose_move: bool = True,
) -> list[list[tuple[int, int, int, int, int, int]]]:
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

    if decompose_move:
        return_routing_batches = decompose_return_move(
            return_routing_batches, factory_pool
        )
    # print("return_routing_batches")
    # print(return_routing_batches)
    # print("leave solve_return_move")
    return return_routing_batches


def decompose_return_move(
    routing_batches: list[list[tuple[int, int, int, int, int, int]]],
    factory_pool: FactoryPool,
):
    """
    Decompose return move into multiple parallel shifts to reduce maximum move duration.

    This function allows using intermediate factory positions to create parallel
    movements that reduce the maximum move distance. For example, instead of moving
    a factory from (3,0) to (0,0) directly (distance 3), we can:
    - Move factory at (2,0) to (0,0) (distance 2)
    - Move factory at (3,0) to (2,0) (distance 1)
    Both happen in parallel, reducing max duration from 3 to 2.

    Args:
        routing_batches: List of movement batches, each containing tuples of
                        (_, x_dst, y_dst, factory_id, x_src, y_src)
        factory_pool: Pool of all factories to check intermediate positions

    Returns:
        List of decomposed routing batches
    """
    new_routing_batches = []

    # Build map of all factory locations
    all_factory_locations = {}  # (x, y) -> factory_id
    for factory in factory_pool.factories:
        all_factory_locations[factory.location] = factory.id

    for batch in routing_batches:
        if not batch:
            continue

        # Check if all moves in batch have the same direction vector
        _, x_dst, y_dst, factory_id, x_src, y_src = batch[0]
        move_vec = (x_dst - x_src, y_dst - y_src)
        same_direction = True

        for _, x_d, y_d, fid, x_s, y_s in batch[1:]:
            if (x_d - x_s, y_d - y_s) != move_vec:
                same_direction = False
                break

        # Only try to decompose if all moves have same direction and distance > 1
        if same_direction and (abs(move_vec[0]) > 1 or abs(move_vec[1]) > 1):
            decomposed = try_decompose_parallel_shift(
                batch, all_factory_locations, move_vec
            )
            if decomposed and len(decomposed[0]) > 1:
                # print("try_decompose_parallel_shift: ", batch)
                # print("decomposed: ", decomposed)
                # Successfully decomposed into multiple batches
                new_routing_batches.extend(decomposed)
            else:
                # Decomposition not beneficial, keep original
                new_routing_batches.append(batch)
        else:
            # Can't or don't need to decompose
            new_routing_batches.append(batch)
        # print("new_routing_batches: ", new_routing_batches)
        # Update factory locations and empty spots
        for _, x_d, y_d, factory_id, x_s, y_s in batch:
            all_factory_locations[(x_d, y_d)] = factory_id
            if (x_s, y_s) in all_factory_locations:
                del all_factory_locations[(x_s, y_s)]

    return new_routing_batches


def try_decompose_parallel_shift(
    batch: list[tuple[int, int, int, int, int, int]],
    all_factory_locations: dict[tuple[int, int], int],
    move_vec: tuple[int, int],
) -> list[list[tuple[int, int, int, int, int, int]]] | None:
    """
    Try to decompose a batch into a single batch with more parallel movements.

    For movements in the same direction, we can include intermediate factories
    to reduce the maximum move distance. Supports horizontal, vertical, and
    diagonal shifts (including non-perfect diagonals).

    For non-perfect diagonals, the number of shifts is min(abs(dx), abs(dy)),
    and step sizes are calculated proportionally for each axis.

    Example (horizontal):
    - Original: Factory at (3,0) -> (0,0), max distance = 3
    - Decomposed: Factory at (2,0) -> (0,0) AND Factory at (3,0) -> (2,0)
      Both in same batch (parallel), max distance = 2

    Example (perfect diagonal):
    - Original: Factory at (3,3) -> (0,0), max distance = 3
    - Decomposed: Factory at (2,2) -> (0,0) AND Factory at (3,3) -> (2,2)
      Both in same batch (parallel), max distance = 2

    Example (non-perfect diagonal):
    - Original: Factory at (4,0) -> (0,2), dx=-4, dy=2, distance = 3
    - shift_amount = min(4,2) = 2, step_x = -2, step_y = 1
    - Intermediate at (2,1): Factory at (2,1) -> (0,2) AND Factory at (4,0) -> (2,1)
      Both in same batch (parallel), max distance = 1.5

    Args:
        batch: Current batch of movements
        all_factory_locations: Map of (x,y) -> factory_id for all factories
        move_vec: Movement vector (dx, dy)

    Returns:
        List containing single decomposed batch if successful, None otherwise
    """
    dx, dy = move_vec

    # Determine shift type and amount
    if dx != 0 and dy != 0:
        # Diagonal shift (including non-perfect diagonals)
        # Number of shifts is determined by the minimum of abs(dx) and abs(dy)
        axis = "diagonal"
        shift_amount = min(abs(dx), abs(dy))

        # Calculate step size for each dimension
        # The larger dimension will have a step > 1 to cover the distance
        if shift_amount > 0:
            step_x = dx / shift_amount
            step_y = dy / shift_amount
        else:
            return None
    elif dx != 0:
        # Horizontal shift
        axis = "x"
        shift_amount = abs(dx)
        step_x = 1 if dx > 0 else -1
        step_y = 0
    else:
        # Vertical shift
        axis = "y"
        shift_amount = abs(dy)
        step_x = 0
        step_y = 1 if dy > 0 else -1

    if shift_amount <= 1:
        return None

    # Build set of factories involved in current batch movements
    involved_factories = {factory_id for _, _, _, factory_id, _, _ in batch}

    # Create decomposed batch with all movements happening in parallel
    # For each original movement, we add movements to the new batch:
    # 1. Intermediate factory moves to final destination
    # 2. Original factory moves to intermediate position
    new_batch = []

    for _, x_dst, y_dst, factory_id, x_src, y_src in batch:
        # Calculate intermediate positions along the path
        mid_factory_ids = [factory_id]
        mid_positions = [(x_src, y_src)]

        for shift in range(1, shift_amount):
            if axis == "x":
                mid_x = int(x_src + step_x * shift)
                mid_y = y_src
            elif axis == "y":
                mid_x = x_src
                mid_y = int(y_src + step_y * shift)
            else:  # diagonal (including non-perfect)
                mid_x = int(x_src + step_x * shift)
                mid_y = int(y_src + step_y * shift)

            mid_pos = (mid_x, mid_y)
            if mid_pos in all_factory_locations:
                mid_factory_ids.append(all_factory_locations[mid_pos])
                mid_positions.append(mid_pos)
        mid_positions.append((x_dst, y_dst))

        # Check if intermediate position has a factory
        # print("all_factory_locations: ", all_factory_locations)
        # print("mid_factory_ids: ", mid_factory_ids)
        # print("mid_positions: ", mid_positions)
        # print("involved_factories: ", involved_factories)
        if mid_factory_ids:
            # Check if this intermediate factory is not already moving in this batch
            for i, mid_factory_id in enumerate(mid_factory_ids):
                if (
                    mid_factory_id not in involved_factories
                    or mid_factory_id == factory_id
                ):
                    # Add intermediate factory movement to destination
                    mid_pos0 = mid_positions[i]
                    mid_pos1 = mid_positions[i + 1]
                    new_batch.append(
                        (
                            -1,
                            mid_pos1[0],
                            mid_pos1[1],
                            mid_factory_id,
                            mid_pos0[0],
                            mid_pos0[1],
                        )
                    )
                else:
                    # Can't use this intermediate position (factory already moving)
                    return None
        else:
            # No factory at intermediate position, can't decompose
            return None

    # Return single batch with all parallel movements
    # The moves are compatible because they all shift in the same direction
    # print("new_batch: ", new_batch)
    if new_batch:
        return [new_batch]

    return None


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
        # print(len(remaining_assignment_idx))
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
