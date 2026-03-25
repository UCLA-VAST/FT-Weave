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
class TSubFactory:
    """One parallel stage-1 line inside a physical T-factory."""

    index: int
    stage_1_success: Optional[bool] = None


def _make_subfactories(num_subfactories: int) -> list[TSubFactory]:
    return [TSubFactory(index=i) for i in range(num_subfactories)]


@dataclass
class TFactory:
    id: int
    state: FactoryStateT = FactoryStateT.IDLE
    stage_2_state: Optional[bool] = None
    location: tuple[int, int] = (0, 0)  # (x, y) coordinates
    subfactories: list[TSubFactory] = field(default_factory=list)

    def stage_1_passed(self) -> bool:
        return any(sf.stage_1_success is True for sf in self.subfactories)

    def clear_subfactory_stage1(self) -> None:
        for sf in self.subfactories:
            sf.stage_1_success = None

    def free(self):
        """Mark the factory as idle."""
        self.stage_2_state = None
        self.clear_subfactory_stage1()
        self.state = FactoryStateT.IDLE

    def restart(self):
        """restart the factory."""
        self.stage_2_state = None
        self.clear_subfactory_stage1()
        self.state = FactoryStateT.STAGE_1

    def set_stage_1_state(self):
        """Set to state 1."""
        self.state = FactoryStateT.STAGE_1

    def set_stage_2_state(self):
        """Set to state 2."""
        self.state = FactoryStateT.STAGE_2

    def set_stage_2_state_outcome(self, outcome: bool):
        """Set state 2 outcome."""
        self.stage_2_state = outcome

    def set_to_wait_for_rus(self):
        """Set the factory state to RUS teleportation."""
        self.state = FactoryStateT.WAIT_FOR_RUS

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
        num_subfactories: Parallel stage-1 lines per physical factory
    """

    num_factories: int
    num_subfactories: int = 1
    factories: list[TFactory] = field(default_factory=list)

    def __post_init__(self):
        k = max(1, self.num_subfactories)
        self.num_subfactories = k
        self.factories = [
            TFactory(id=i, subfactories=_make_subfactories(k))
            for i in range(self.num_factories)
        ]

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
