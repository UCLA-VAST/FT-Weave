from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from ..config import PRECISION


from src.config import (
    CNOT_TIME,
    SE_TIME,
    TMR_P,
    TMR_Q,
    ANGLE_S,
    TMR_PREPARATION_TIME,
    # THRESHOLD_HIGH_TMR,
    # THRESHOLD_HIGH_RUS,
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================


class FactoryState(Enum):
    """Represents the state of a factory during angle preparation."""

    IDLE = 0
    TMR_BEFORE_RZ = 1
    TMR_AFTER_RZ = 2
    WAIT_FOR_RUS = 3
    RUS = 4


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
        self.factories: list[tuple[int, float]] = []  # List of (factory_id, angle)
        self.angle_counts = Counter()  # angle -> count
        self.waiting_for_rus = False

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

    def set_target_angle(self, angle):
        self.target_angle = angle

    def set_waiting_for_rus(self):
        self.waiting_for_rus = True

    def update_rus_state(self, success: bool) -> tuple[bool, bool]:
        insert_s_gate = False
        if success:
            self.clear_all()
        else:
            _ = self.remove_angle(self.target_angle)
            self.double_target_angle()
            diff_s = self.target_angle - ANGLE_S
            if np.isclose(diff_s, 0.0, atol=1e-8):
                insert_s_gate = True
                success = True  # Treat as successful teleportation for state tracking
                self.clear_all()
            elif diff_s > 0:
                insert_s_gate = True
                self.set_target_angle(self.target_angle - ANGLE_S)

        self.waiting_for_rus = False

        return success, insert_s_gate

    def __repr__(self):
        return f"QubitTracker(qubit={self.qubit_id}, target={self.target_angle}, factories={len(self.factories)})"


@dataclass
class Factory:
    id: int
    angle: Optional[float] = None
    qubit: Optional[int] = None
    success_rate: Optional[float] = None
    state: FactoryState = FactoryState.IDLE
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
        self.state = FactoryState.IDLE

    def update(self, angle, qubit, success_rate):
        """Update the factory with new assignment."""
        self.angle = angle
        self.qubit = qubit
        self.success_rate = success_rate
        self.state = FactoryState.TMR_AFTER_RZ
        self.tmr_state = None
        self.rus_state = None

    def update_qubit(self, qubit):
        """Update the factory with new assignment."""
        self.qubit = qubit

    def set_pre_tmr_state(self):
        """Set the TMR preparation state."""
        self.state = FactoryState.TMR_BEFORE_RZ

    def set_tmr_state(self, state: bool):
        """Set the TMR preparation state."""
        self.tmr_state = state
        self.state = FactoryState.WAIT_FOR_RUS

    def set_to_rus(self):
        """Set the factory state to RUS injection."""
        self.state = FactoryState.RUS

    def set_rus_state(self, state: bool):
        """Set the RUS injection state."""
        self.rus_state = state
        # self.state = FactoryState.IDLE

    def set_location(self, loc: tuple[int, int]):
        self.location = loc

    def __repr__(self):
        return f"Factory(id={self.id}, angle={self.angle}, qubit={self.qubit}, state={self.state}, location={self.location})"


@dataclass
class FactoryPool:
    """
    Manages a pool of factories.

    Attributes:
        factories: List of Factory objects
        num_factories: Total number of factories
    """

    num_factories: int
    factories: list[Factory] = field(default_factory=list)
    avaliable_factories_per_row: defaultdict[int, int] = field(
        default_factory=defaultdict
    )

    def __post_init__(self):
        self.factories = [Factory(id=i) for i in range(self.num_factories)]
        self.avaliable_factories_per_row = defaultdict(int)

    def set_locations(self, locations: list[tuple[int, int]]):
        for factory, loc in zip(self.factories, locations):
            factory.set_location(loc)
            self.avaliable_factories_per_row[loc[1]] += 1

    def get_idle_factories(self) -> list[Factory]:
        """Return a list of idle factories."""
        return [
            factory for factory in self.factories if factory.state == FactoryState.IDLE
        ]

    def get_busy_factories(self) -> list[Factory]:
        """Return a list of idle factories."""
        return [factory for factory in self.factories if factory.angle is not None]

    def get_tmr_before_rz_factories(self) -> list[Factory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == FactoryState.TMR_BEFORE_RZ]

    def get_tmr_after_rz_factories(self) -> list[Factory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == FactoryState.TMR_AFTER_RZ]

    def get_wait_for_rus_factories(self) -> list[Factory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == FactoryState.WAIT_FOR_RUS]

    # def get_rus_factories(self) -> list[Factory]:
    #     """Return a list of active factories."""
    #     return [f for f in self.factories if f.state == FactoryState.RUS]

    def get_factory_by_id(self, factory_id: int) -> Factory:
        """Get a factory by its ID."""
        assert 0 <= factory_id < self.num_factories, "Factory ID out of range."
        return self.factories[factory_id]

    def free_factory(self, factory_id: int):
        """Free a factory by its ID."""
        factory = self.get_factory_by_id(factory_id)
        if factory and factory.angle is not None:
            factory.free()
            loc = factory.location
            self.avaliable_factories_per_row[loc[1]] += 1

    def assign_factory(
        self, factory_id: int, angle: float, qubit: int, success_rate: float
    ):
        """Assign a factory by its ID."""
        factory = self.get_factory_by_id(factory_id)
        if factory and factory.angle is None:
            factory.update(angle, qubit, success_rate)
            loc = factory.location
            self.avaliable_factories_per_row[loc[1]] -= 1

    def __repr__(self):
        return f"FactoryPool(num_factories={self.num_factories})"
