"""Protocol for factory-to-angle assignment objectives."""

from __future__ import annotations

from typing import Protocol


class AssignmentObjective(Protocol):
    """Evaluates total cost and marginal cost changes for assignments."""

    def total_cost(self, assignment: dict[int, float]) -> float:
        """Return total objective value for factory_id -> angle assignment."""
        ...

    def delta_cost(
        self,
        assignment: dict[int, float],
        factory_id: int,
        new_angle: float,
    ) -> float:
        """Return cost(new) - cost(old) if factory_id moves to new_angle."""
        ...
