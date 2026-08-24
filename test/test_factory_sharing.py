"""
Unit tests for factory sharing features.

Tests cover:
- AngleFactoryIndex data structure
- Vertex matching algorithms
- Enhanced assignment functions
- Integration with existing system
"""

import pytest
from src.ds import FactoryPool, QubitAngleTracker
from src.star.rus import (
    AngleFactoryIndex,
    assign_teleportation_with_sharing,
    find_optimal_factory_assignment,
)


# ============================================================================
# TESTS FOR AngleFactoryIndex
# ============================================================================


class TestAngleFactoryIndex:
    """Test suite for AngleFactoryIndex data structure."""

    def test_add_and_retrieve_factory(self):
        """Test adding a factory and retrieving it by angle."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=1)

        factories = index.get_factories_for_angle(0.5)
        assert len(factories) == 1
        assert factories[0] == (0, 1)

    def test_multiple_factories_same_angle(self):
        """Test multiple factories preparing the same angle."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        index.add_factory(factory_id=1, angle=0.5, qubit_id=1)
        index.add_factory(factory_id=2, angle=0.5, qubit_id=2)

        factories = index.get_factories_for_angle(0.5)
        assert len(factories) == 3
        assert (0, 0) in factories
        assert (1, 1) in factories
        assert (2, 2) in factories

    def test_multiple_angles(self):
        """Test factories preparing different angles."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        index.add_factory(factory_id=1, angle=1.0, qubit_id=1)
        index.add_factory(factory_id=2, angle=0.5, qubit_id=2)

        factories_05 = index.get_factories_for_angle(0.5)
        factories_10 = index.get_factories_for_angle(1.0)

        assert len(factories_05) == 2
        assert len(factories_10) == 1

    def test_remove_factory(self):
        """Test removing a factory from index."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        index.add_factory(factory_id=1, angle=0.5, qubit_id=1)

        index.remove_factory(factory_id=0)

        factories = index.get_factories_for_angle(0.5)
        assert len(factories) == 1
        assert factories[0] == (1, 1)

    def test_get_angle_for_factory(self):
        """Test retrieving angle for a factory."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        index.add_factory(factory_id=1, angle=1.0, qubit_id=1)

        assert index.get_angle_for_factory(0) == 0.5
        assert index.get_angle_for_factory(1) == 1.0

    def test_get_all_angles(self):
        """Test getting all angles in index."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        index.add_factory(factory_id=1, angle=1.0, qubit_id=1)
        index.add_factory(factory_id=2, angle=0.5, qubit_id=2)

        angles = index.get_all_angles()
        assert angles == {0.5, 1.0}

    def test_clear(self):
        """Test clearing all entries from index."""
        index = AngleFactoryIndex()
        index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        index.add_factory(factory_id=1, angle=1.0, qubit_id=1)

        index.clear()

        assert len(index.get_all_angles()) == 0
        assert index.get_factories_for_angle(0.5) == []


# ============================================================================
# TESTS FOR Vertex Matching
# ============================================================================


class TestVertexMatching:
    """Test suite for vertex matching algorithms."""

    def test_find_optimal_assignment_greedy(self):
        """Test optimal assignment using greedy method."""
        qubits = [0, 1, 2]
        factories = {
            0: (0, 0),
            1: (10, 10),
        }
        qubit_locs = {
            0: (0, 0),
            1: (5, 5),
            2: (10, 10),
        }

        assignments = find_optimal_factory_assignment(qubits, factories, qubit_locs)

        assert len(assignments) <= 2


# ============================================================================
# TESTS FOR Integration Functions
# ============================================================================


class TestFactorySharingIntegration:
    """Test suite for factory sharing integration utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create factory pool
        self.factory_pool = FactoryPool(num_factories=5)
        self.factory_pool.set_locations([(0, 0), (5, 5), (10, 10), (15, 15), (20, 20)])

        # Create qubit trackers
        self.qubit_trackers = {
            0: QubitAngleTracker(
                qubit_id=0, target_angle=0.5, factory_limit=2, code_distance=7
            ),
            1: QubitAngleTracker(
                qubit_id=1, target_angle=0.5, factory_limit=2, code_distance=7
            ),
            2: QubitAngleTracker(
                qubit_id=2, target_angle=1.0, factory_limit=2, code_distance=7
            ),
        }

        # Add factories to trackers
        self.qubit_trackers[0].add_factory(factory_id=0, angle=0.5)
        self.qubit_trackers[0].add_factory(factory_id=1, angle=0.5)

        self.qubit_trackers[1].add_factory(factory_id=2, angle=0.5)
        self.qubit_trackers[1].add_factory(factory_id=3, angle=0.5)

        self.qubit_trackers[2].add_factory(factory_id=4, angle=1.0)


# ============================================================================
# TESTS FOR Assignment with Sharing
# ============================================================================


class TestAssignmentWithSharing:
    """Test suite for assignment with factory sharing."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create factory pool
        self.factory_pool = FactoryPool(num_factories=4)
        self.factory_pool.set_locations([(0, 0), (10, 0), (0, 10), (10, 10)])

        # Create qubit trackers with same angle
        self.qubit_trackers = {
            0: QubitAngleTracker(
                qubit_id=0, target_angle=0.5, factory_limit=2, code_distance=7
            ),
            1: QubitAngleTracker(
                qubit_id=1, target_angle=0.5, factory_limit=2, code_distance=7
            ),
        }

        # Qubit 0 has factories 0, 1 for angle 0.5
        # Qubit 1 has factories 2, 3 for angle 0.5
        self.qubit_trackers[0].add_factory(factory_id=0, angle=0.5)
        self.qubit_trackers[0].add_factory(factory_id=1, angle=0.5)

        self.qubit_trackers[1].add_factory(factory_id=2, angle=0.5)
        self.qubit_trackers[1].add_factory(factory_id=3, angle=0.5)

        self.logic_qubit_locations = [(0, 0), (10, 10)]

    def test_assignment_same_angle_sharing(self):
        """Test assignment when qubits share angle and factories."""

        angle_index = AngleFactoryIndex()
        angle_index.add_factory(factory_id=0, angle=0.5, qubit_id=0)
        angle_index.add_factory(factory_id=1, angle=0.5, qubit_id=0)
        angle_index.add_factory(factory_id=2, angle=0.5, qubit_id=1)
        angle_index.add_factory(factory_id=3, angle=0.5, qubit_id=1)

        assignments = assign_teleportation_with_sharing(
            qubit_trackers=self.qubit_trackers,
            factory_pool=self.factory_pool,
            logic_qubit_locations=self.logic_qubit_locations,
            angle_factory_index=angle_index,
        )

        # Should have 2 assignments (one per qubit)
        assert len(assignments) == 2

        # Check all qubits are assigned
        assigned_qubits = {q for q, _ in assignments}
        assert assigned_qubits == {0, 1}

        # Check no two assignments use the same factory
        used_factories = [f for _, f in assignments]
        assert len(used_factories) == len(set(used_factories))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
