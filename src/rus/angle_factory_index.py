"""
AngleFactoryIndex: Data structure for mapping angles to factories across all qubits.

This module enables factory sharing across qubits by maintaining a global index
that allows quick lookup of which factories are preparing a specific angle,
regardless of which qubit they're assigned to.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple


class AngleFactoryIndex:
    """
    Index mapping angles to factories preparing them.

    This enables efficient lookup of factories by angle, allowing factory sharing
    across multiple qubits that require the same angle.

    Attributes:
        angle_to_factories: Dict[float, List[Tuple[int, int]]] mapping angle to list of (factory_id, qubit_id)
        factory_to_angle: Dict[int, float] mapping factory_id to the angle it's preparing
    """

    def __init__(self):
        """Initialize empty index."""
        self.angle_to_factories: Dict[float, List[Tuple[int, int]]] = defaultdict(list)
        self.angle_to_qubits: Dict[float, int] = defaultdict(int)
        self.factory_to_angle: Dict[int, float] = {}

    def add_qubit_for_angle(self, angle: float):
        """
        Args:
            angle: The angle being prepared
        """
        self.angle_to_qubits[angle] += 1

    def remove_qubit_for_angle(self, angle: float):
        """
        Args:
            angle: The angle being prepared
        """
        self.angle_to_qubits[angle] -= 1

    def add_factory(self, factory_id: int, angle: float, qubit_id: int):
        """
        Register a factory preparing a specific angle for a qubit.

        Args:
            factory_id: Unique factory identifier
            angle: The angle being prepared
            qubit_id: The qubit this factory is initially assigned to
        """
        # Only add if factory not already indexed
        if factory_id not in self.factory_to_angle:
            self.angle_to_factories[angle].append((factory_id, qubit_id))
            self.factory_to_angle[factory_id] = angle

    def remove_factory(self, factory_id: int):
        """
        Remove a factory from the index (e.g., when it's freed).

        Args:
            factory_id: Factory to remove
        """
        if factory_id in self.factory_to_angle:
            angle = self.factory_to_angle[factory_id]
            self.angle_to_factories[angle] = [
                (fid, qid)
                for fid, qid in self.angle_to_factories[angle]
                if fid != factory_id
            ]
            if not self.angle_to_factories[angle]:
                del self.angle_to_factories[angle]
            del self.factory_to_angle[factory_id]

    def get_factories_for_angle(self, angle: float) -> List[Tuple[int, int]]:
        """
        Get all factories preparing a specific angle.

        Args:
            angle: The angle to query

        Returns:
            List of (factory_id, qubit_id) tuples preparing this angle
        """
        return self.angle_to_factories.get(angle, [])

    def get_angle_for_factory(self, factory_id: int) -> float:
        """
        Get the angle being prepared by a factory.

        Args:
            factory_id: The factory to query

        Returns:
            The angle being prepared, or None if factory not found
        """
        return self.factory_to_angle[factory_id]

    def get_all_angles(self) -> Set[float]:
        """Get set of all angles currently being prepared."""
        return set(self.angle_to_factories.keys())

    def clear(self):
        """Clear all entries."""
        self.angle_to_factories.clear()
        self.factory_to_angle.clear()

    def __repr__(self):
        return f"AngleFactoryIndex\n {len(self.factory_to_angle)} factories:\n {self.angle_to_factories},\n {len(self.angle_to_factories)} angles\n: {self.angle_to_qubits}"
