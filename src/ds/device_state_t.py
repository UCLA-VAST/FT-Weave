from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================================
# DATA STRUCTURES
# ============================================================================
class FactoryStateT(Enum):
    """Represents the state of a factory during angle preparation."""

    IDLE = 0
    STAGE_1 = 1
    STAGE_2 = 2
    WAIT_FOR_RUS = 3
    RUS = 4


@dataclass
class TFactory:
    id: int
    state: FactoryStateT = FactoryStateT.IDLE
    stage_1_state: Optional[bool] = None
    stage_2_state: Optional[bool] = None
    location: tuple[int, int] = (0, 0)  # (x, y) coordinates

    def free(self):
        """Mark the factory as idle."""
        self.stage_1_state = None
        self.stage_2_state = None
        self.state = FactoryStateT.IDLE

    def restart(self):
        """restart the factory."""
        self.stage_1_state = None
        self.stage_2_state = None
        self.state = FactoryStateT.STAGE_1

    def set_stage_1_state(self):
        """Set to state 1."""
        self.state = FactoryStateT.STAGE_1

    def set_stage_2_state(self):
        """Set to state 2."""
        self.state = FactoryStateT.STAGE_2

    def set_to_rus(self):
        """Set the factory state to RUS teleportation."""
        self.state = FactoryStateT.RUS

    def set_rus_state(self, state: bool):
        """Set the RUS teleportation state."""
        self.rus_state = state

    def set_location(self, loc: tuple[int, int]):
        self.location = loc

    def __repr__(self):
        return f"Factory(id={self.id}, state={self.state}, location={self.location})"


@dataclass
class TFactoryPool:
    """
    Manages a pool of factories.

    Attributes:
        factories: List of Factory objects
        num_factories: Total number of factories
    """

    num_factories: int
    factories: list[TFactory] = field(default_factory=list)

    def __post_init__(self):
        self.factories = [TFactory(id=i) for i in range(self.num_factories)]

    def set_locations(self, locations: list[tuple[int, int]]):
        for factory, loc in zip(self.factories, locations):
            factory.set_location(loc)

    def get_idle_factories(self) -> list[TFactory]:
        """Return a list of idle factories."""
        return [
            factory for factory in self.factories if factory.state == FactoryStateT.IDLE
        ]

    def get_busy_factories(self) -> list[TFactory]:
        """Return a list of idle factories."""
        return [
            factory
            for factory in self.factories
            if factory.state is not FactoryStateT.IDLE
        ]

    def get_stage_1_factories(self) -> list[TFactory]:
        return [f for f in self.factories if f.state == FactoryStateT.STAGE_1]

    def get_stage_2_factories(self) -> list[TFactory]:
        return [f for f in self.factories if f.state == FactoryStateT.STAGE_2]

    def get_wait_for_rus_factories(self) -> list[TFactory]:
        """Return a list of active factories."""
        return [f for f in self.factories if f.state == FactoryStateT.WAIT_FOR_RUS]

    def get_factory_by_id(self, factory_id: int) -> TFactory:
        """Get a factory by its ID."""
        assert 0 <= factory_id < self.num_factories, "Factory ID out of range."
        return self.factories[factory_id]

    def free_factory(self, factory_id: int):
        """Free a factory by its ID."""
        factory = self.get_factory_by_id(factory_id)
        factory.free()

    def restart_factory(self, factory_id: int):
        """Free a factory by its ID."""
        factory = self.get_factory_by_id(factory_id)
        factory.restart()

    def free_factories(self, factory_ids: list[int]):
        """Free a factory by its ID."""
        for factory_id in factory_ids:
            factory = self.get_factory_by_id(factory_id)
            factory.free()

    def __repr__(self):
        return f"FactoryPool(num_factories={self.num_factories})"
