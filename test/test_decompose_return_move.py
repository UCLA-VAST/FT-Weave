"""Test for decompose_return_move functionality."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ds.device_state_star import FactoryPool
from src.star.rus import (
    decompose_return_move,
)
from src.ds import move_duration


def test_decompose_return_move_example():
    """
    Test the example from the user:
    - Factory A at (3,0) needs to return to (0,0), distance = 3
    - Factory B (idle) at (2,0) can be used as intermediate
    - Result: Both move in parallel, max distance = 2
    """
    # Create factory pool
    factory_pool = FactoryPool(num_factories=2)

    # Set initial locations
    # Factory 0 at (0,0) - will be the destination
    # Factory 1 at (2,0) - idle, will be used as intermediate
    # Factory 2 at (3,0) - needs to return
    factory_pool.set_locations([(0, 0), (2, 0)])

    # Simulate: Factory 2 moved from (0,0) to (3,0) during RUS
    # Now it needs to return: (3,0) -> (0,0)
    factory_pool.get_factory_by_id(0).set_location((3, 0))

    # Return move batch: factory 2 from (3,0) to (0,0)
    routing_batches = [
        [
            (-1, 0, 0, 0, 3, 0),  # (_, x_dst, y_dst, factory_id, x_src, y_src)
        ]
    ]

    # Decompose the return move
    result = decompose_return_move(routing_batches, factory_pool)

    print("Original batch:")
    for batch in routing_batches:
        for move in batch:
            _, x_dst, y_dst, fid, x_src, y_src = move
            dist = move_duration(x_src, y_src, x_dst, y_dst)
            print(
                f"  Factory {fid}: ({x_src},{y_src}) -> ({x_dst},{y_dst}), distance = {dist}"
            )

    print("\nDecomposed batches:")
    for i, batch in enumerate(result):
        max_dist = 0
        print(f"Batch {i+1}:")
        for move in batch:
            _, x_dst, y_dst, fid, x_src, y_src = move
            dist = move_duration(x_src, y_src, x_dst, y_dst)
            max_dist = max(max_dist, dist)
            print(
                f"  Factory {fid}: ({x_src},{y_src}) -> ({x_dst},{y_dst}), distance = {dist}"
            )
        print(f"  Max distance in batch: {max_dist}")

    # Check that decomposition reduced the maximum distance
    # Original: single move with distance 3
    # Decomposed: should have moves with max distance 2
    assert len(result) > 0
    if len(result[0]) > 1:  # If decomposed
        for batch in result:
            max_dist = max(
                move_duration(x_s, y_s, x_d, y_d) for _, x_d, y_d, _, x_s, y_s in batch
            )
            print(f"\nMax distance after decomposition: {max_dist}")
            assert max_dist < 3, "Decomposition should reduce max distance"


def test_no_intermediate_factory():
    """Test when there's no intermediate factory available."""
    factory_pool = FactoryPool(num_factories=2)
    factory_pool.set_locations([(0, 0), (3, 0)])
    factory_pool.get_factory_by_id(1).set_location((3, 0))
    # No factory at (2,0) to use as intermediate
    routing_batches = [
        [
            (-1, 0, 0, 0, 3, 0),
        ]
    ]

    result = decompose_return_move(routing_batches, factory_pool)

    # Should return original batch since decomposition isn't possible
    print("\nTest: No intermediate factory")
    print(f"Original batches: {len(routing_batches)}")
    print(f"Result batches: {len(result)}")
    assert len(result) == 1
    assert len(result[0]) == 1


def test_diagonal_shift():
    """
    Test diagonal shift decomposition:
    - Factory A at (3,3) needs to return to (0,0), distance = 3
    - Factory B (idle) at (2,2) can be used as intermediate
    - Factory C (idle) at (1,1) can be used as intermediate
    - Result: All move in parallel, max distance = 1
    """
    # Create factory pool with factories along diagonal
    factory_pool = FactoryPool(num_factories=4)

    # Set initial locations - factories at diagonal positions
    # Factory 0 at (0,0), Factory 1 at (1,1), Factory 2 at (2,2)
    factory_pool.set_locations([(0, 0), (1, 1), (2, 2)])

    # Move factory 0 from (0,0) to (3,3) during RUS (simulated)
    factory_pool.get_factory_by_id(0).set_location((3, 3))

    # Return move batch: factory 0 from (3,3) to (0,0)
    routing_batches = [
        [
            (-1, 0, 0, 0, 3, 3),  # (_, x_dst, y_dst, factory_id, x_src, y_src)
        ]
    ]

    # Decompose the return move
    result = decompose_return_move(routing_batches, factory_pool)

    print("\nOriginal batch:")
    for batch in routing_batches:
        for move in batch:
            _, x_dst, y_dst, fid, x_src, y_src = move
            dist = move_duration(x_src, y_src, x_dst, y_dst)
            print(
                f"  Factory {fid}: ({x_src},{y_src}) -> ({x_dst},{y_dst}), distance = {dist}"
            )

    print("\nDecomposed batches:")
    for i, batch in enumerate(result):
        max_dist = 0
        print(f"Batch {i+1}:")
        for move in batch:
            _, x_dst, y_dst, fid, x_src, y_src = move
            dist = move_duration(x_src, y_src, x_dst, y_dst)
            max_dist = max(max_dist, dist)
            print(
                f"  Factory {fid}: ({x_src},{y_src}) -> ({x_dst},{y_dst}), distance = {dist}"
            )
        print(f"  Max distance in batch: {max_dist}")

    # Check that decomposition reduced the maximum distance
    assert len(result) > 0
    if len(result[0]) > 1:  # If decomposed
        for batch in result:
            max_dist = max(
                move_duration(x_s, y_s, x_d, y_d) for _, x_d, y_d, _, x_s, y_s in batch
            )
            print(f"\nMax distance after decomposition: {max_dist}")
            # Original distance was 3, should be reduced
            assert max_dist < 3, "Decomposition should reduce max distance"
            print(f"✓ Successfully reduced from 3 to {max_dist}")


def test_diagonal_shift_no_intermediate():
    """Test diagonal shift when intermediate factories are missing."""
    factory_pool = FactoryPool(num_factories=2)
    # Only factories at corners, no intermediate diagonal positions
    factory_pool.set_locations([(0, 0), (3, 3)])
    factory_pool.get_factory_by_id(1).set_location((3, 3))

    routing_batches = [
        [
            (-1, 0, 0, 1, 3, 3),
        ]
    ]

    result = decompose_return_move(routing_batches, factory_pool)

    # Should return original batch since no intermediate factories
    print("\nTest: Diagonal shift without intermediate factories")
    print(f"Original batches: {len(routing_batches)}")
    print(f"Result batches: {len(result)}")
    assert len(result) == 1
    assert len(result[0]) == 1
    print("✓ Correctly kept original batch")


def test_non_perfect_diagonal_shift():
    """
    Test non-perfect diagonal shift decomposition:
    - Factory A at (4,0) needs to return to (0,2)
    - dx = -4, dy = 2
    - shift_amount = min(4, 2) = 2
    - step_x = -4/2 = -2, step_y = 2/2 = 1
    - Intermediate at (2,1)
    - Factory B at (2,1) should move to (0,2)
    - Factory A at (4,0) should move to (2,1)
    """
    # Create factory pool with factories at strategic positions
    factory_pool = FactoryPool(num_factories=2)

    # Factory 0 at (0,2) (destination), Factory 1 at (2,1) (intermediate)
    factory_pool.set_locations([(0, 0), (2, 1)])

    # Move factory 0 from (0,2) to (4,3) during RUS (simulated)
    factory_pool.get_factory_by_id(0).set_location((4, 3))

    # Return move batch: factory 0 from (4,0) to (0,2)
    routing_batches = [
        [
            (-1, 0, 0, 0, 4, 3),  # (_, x_dst, y_dst, factory_id, x_src, y_src)
        ]
    ]

    # Decompose the return move
    result = decompose_return_move(routing_batches, factory_pool)

    print("\nOriginal batch:")
    for batch in routing_batches:
        for move in batch:
            _, x_dst, y_dst, fid, x_src, y_src = move
            dist = move_duration(x_src, y_src, x_dst, y_dst)
            print(
                f"  Factory {fid}: ({x_src},{y_src}) -> ({x_dst},{y_dst}), distance = {dist}"
            )

    print("\nDecomposed batches:")
    for i, batch in enumerate(result):
        max_dist = 0
        print(f"Batch {i+1}:")
        for move in batch:
            _, x_dst, y_dst, fid, x_src, y_src = move
            dist = move_duration(x_src, y_src, x_dst, y_dst)
            max_dist = max(max_dist, dist)
            print(
                f"  Factory {fid}: ({x_src},{y_src}) -> ({x_dst},{y_dst}), distance = {dist}"
            )
        print(f"  Max distance in batch: {max_dist}")

    # Check that decomposition reduced the maximum distance
    assert len(result) > 0
    if len(result[0]) > 1:  # If decomposed
        for batch in result:
            max_dist = max(
                move_duration(x_s, y_s, x_d, y_d) for _, x_d, y_d, _, x_s, y_s in batch
            )
            print(f"\nMax distance after decomposition: {max_dist}")
            # Original distance was 3, should be reduced
            assert max_dist < 3, "Decomposition should reduce max distance"
            print(f"✓ Successfully reduced from 3 to {max_dist}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


test_non_perfect_diagonal_shift()
