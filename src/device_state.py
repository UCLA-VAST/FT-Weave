from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import PRECISION

# ============================================================================
# DATA STRUCTURES
# ============================================================================


class QubitAngleTracker:
    """
    Tracks angle preparation state for a single qubit.

    Attributes:
        qubit_id: Identifier for the qubit
        target_angle: Current target angle to prepare
        factories: List of (factory_id, angle) tuples working on this qubit
        angle_counts: Counter tracking how many factories are preparing each angle
    """

    def __init__(self, qubit_id, target_angle):
        self.qubit_id = round(qubit_id, PRECISION)
        self.target_angle = target_angle
        self.factories = []  # List of (factory_id, angle)
        self.angle_counts = Counter()  # angle -> count

    def get_generation(self, angle):
        """Calculate generation level: 0 for original, 1 for 2x, 2 for 4x, etc."""
        if angle == self.target_angle:
            return 0
        # Calculate how many times we've doubled: log2(angle/original)
        import math

        ratio = angle // self.target_angle
        if ratio >= 1:  # Check if power of 2
            return int(math.log2(ratio))
        return 999  # Large number for non-power-of-2 angles

    def add_factory(self, factory_id, angle):
        """Add a factory working on a specific angle."""
        self.factories.append((factory_id, angle))
        self.angle_counts[angle] += 1

    def remove_factory(self, factory_id, angle):
        """Remove a factory from tracking."""
        self.factories = [(fid, a) for fid, a in self.factories if fid != factory_id]
        self.angle_counts[angle] -= 1
        if self.angle_counts[angle] == 0:
            del self.angle_counts[angle]

    def remove_angle(self, angle):
        """Remove all factories working on a specific angle."""
        removed_factories = [(fid, a) for fid, a in self.factories if a == angle]
        self.factories = [(fid, a) for fid, a in self.factories if a != angle]
        if angle in self.angle_counts:
            del self.angle_counts[angle]
        return removed_factories

    def clear_all(self):
        """Clear all factory assignments."""
        self.factories = []
        self.angle_counts.clear()

    def get_factories_for_angle(self, angle):
        """Get list of factory IDs working on a specific angle."""
        return [fid for fid, a in self.factories if a == angle]

    def get_angle_count(self, angle):
        """Get number of factories working on a specific angle."""
        return self.angle_counts.get(angle, 0)

    def get_sorted_factories(self):
        """Get factories sorted by angle (ascending)."""
        return sorted(self.factories, key=lambda x: x[1])

    def has_active_factories(self):
        """Check if any factories are working on this qubit."""
        return len(self.factories) > 0

    def double_target_angle(self):
        self.target_angle *= 2

    def __repr__(self):
        return f"QubitTracker(qubit={self.qubit_id}, target={self.target_angle}, factories={len(self.factories)})"


@dataclass
class Factory:
    id: int
    angle: Optional[float] = None
    qubit: Optional[int] = None
    success_rate: Optional[float] = None
    state: int = 0  # 0: 'idle', 1: 'tmr', 2: 'wait', 3: 'rus'
    tmr_state: Optional[bool] = None
    rus_state: Optional[bool] = None
    location: tuple[int, int] = (0, 0)  # (x, y) coordinates

    def free(self):
        """Mark the factory as idle."""
        self.angle = None
        self.qubit = None
        self.success_rate = None
        self.tmr_state = None
        self.rus_state = None
        self.state = 0

    def update(self, angle, qubit, success_rate):
        """Update the factory with new assignment."""
        self.angle = angle
        self.qubit = qubit
        self.success_rate = success_rate
        self.state = 1
        self.tmr_state = None
        self.rus_state = None

    def set_tmr_state(self, state: bool):
        """Set the TMR preparation state."""
        self.tmr_state = state
        self.state = 2

    def set_to_rus(self):
        """Set the factory state to RUS injection."""
        self.state = 3

    def set_rus_state(self, state: bool):
        """Set the RUS injection state."""
        self.rus_state = state
        self.state = 0

    def set_location(self, loc: tuple[int, int]):
        self.location = loc


@dataclass
class FactoryPool:
    """
    Manages a pool of factories.

    Attributes:
        factories: List of Factory objects
        num_factories: Total number of factories
        num_idle_factories: Number of idle factories
        num_busy_factories: Number of busy factories
    """

    num_factories: int
    factories: list[Factory] = field(default_factory=list)
    num_idle_factories: int = field(init=False)
    num_busy_factories: int = 0
    avaliable_factories_per_row: defaultdict[int, int] = field(
        default_factory=defaultdict
    )

    def __post_init__(self):
        self.factories = [Factory(id=i) for i in range(self.num_factories)]
        self.num_idle_factories = self.num_factories
        self.avaliable_factories_per_row = defaultdict(int)

    def set_locations(self, locations: list[tuple[int, int]]):
        for factory, loc in zip(self.factories, locations):
            factory.set_location(loc)
            self.avaliable_factories_per_row[loc[1]] += 1

    def get_num_idle_factories(self):
        return self.num_idle_factories

    def get_num_busy_factories(self):
        return self.num_busy_factories

    def get_idle_factories(self) -> list[Factory]:
        """Return a list of idle factories."""
        return [factory for factory in self.factories if factory.angle is None]

    def get_tmr_factories(self) -> list[Factory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == 1]

    def get_wait_factories(self) -> list[Factory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == 2]

    def get_rus_factories(self) -> list[Factory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == 3]

    def get_factory_by_id(self, factory_id: int) -> Factory:
        """Get a factory by its ID."""
        assert 0 <= factory_id < self.num_factories, "Factory ID out of range."
        return self.factories[factory_id]

    def free_factory(self, factory_id: int):
        """Free a factory by its ID."""
        factory = self.get_factory_by_id(factory_id)
        if factory and factory.angle is not None:
            factory.free()
            self.num_idle_factories += 1
            loc = factory.location
            self.avaliable_factories_per_row[loc[1]] += 1

    def assign_factory(
        self, factory_id: int, angle: float, qubit: int, success_rate: float
    ):
        """Assign a factory by its ID."""
        factory = self.get_factory_by_id(factory_id)
        if factory and factory.angle is None:
            factory.update(angle, qubit, success_rate)
            self.num_idle_factories -= 1
            self.num_busy_factories += 1
            loc = factory.location
            self.avaliable_factories_per_row[loc[1]] -= 1
