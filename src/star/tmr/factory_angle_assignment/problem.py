"""Problem data for factory-to-angle assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.star.config import PRECISION
from src.star.simulation import calculate_success_rate

from ....ds import QubitAngleTracker


def manhattan_distance(loc_a: tuple[int, int], loc_b: tuple[int, int]) -> int:
    """Raw Manhattan distance: |col_a - col_b| + |row_a - row_b|."""
    return abs(loc_a[0] - loc_b[0]) + abs(loc_a[1] - loc_b[1])


def normalize_angle(angle: float) -> float:
    return round(angle, PRECISION)


@dataclass(frozen=True)
class CoverageTerm:
    """One weighted stochastic-coverage demand at a qubit location."""

    term_index: int
    qubit_id: int
    qubit_index: int
    angle: float
    weight: float


@dataclass
class AssignmentProblem:
    """Immutable inputs for factory-to-angle optimization."""

    factory_ids: list[int]
    factory_index: dict[int, int]
    qubit_ids: list[int]
    qubit_index: dict[int, int]
    distances: np.ndarray
    coverage_terms: list[CoverageTerm]
    angles: list[float]
    p_theta: dict[float, float]
    terms_by_angle: dict[float, list[int]] = field(default_factory=dict)
    qubits_by_angle: dict[float, list[int]] = field(default_factory=dict)
    r_max: int = 0

    @property
    def n_factories(self) -> int:
        return len(self.factory_ids)


def build_coverage_terms(
    qubit_trackers: dict[int, QubitAngleTracker],
    qubit_ids: list[int],
    code_distance: int,
    *,
    include_lookahead: bool,
    allocation_level_range: range | None = None,
) -> list[CoverageTerm]:
    """Build weighted coverage terms with optional lookahead weighting."""
    import numpy as np

    if allocation_level_range is None:
        allocation_level_range = range(0, 2) if include_lookahead else range(0, 1)

    qubit_index = {qid: idx for idx, qid in enumerate(qubit_ids)}
    terms: list[CoverageTerm] = []
    term_index = 0

    for qubit_id in qubit_ids:
        tracker = qubit_trackers[qubit_id]
        level_demand = 2.0 if tracker.waiting_for_rus else 1.0
        levels = allocation_level_range if include_lookahead else range(0, 1)

        for level in levels:
            angle = normalize_angle(tracker.get_angle_level(level))
            if np.isclose(angle, 0.0, atol=1e-8):
                continue

            if include_lookahead:
                success_rate = calculate_success_rate(angle, code_distance)
                weight = level_demand / success_rate
                level_demand /= 2.0
            else:
                weight = 1.0

            terms.append(
                CoverageTerm(
                    term_index=term_index,
                    qubit_id=qubit_id,
                    qubit_index=qubit_index[qubit_id],
                    angle=angle,
                    weight=weight,
                )
            )
            term_index += 1

    return terms


def build_assignment_problem(
    factories: list,
    qubit_trackers: dict[int, QubitAngleTracker],
    logic_qubit_locations: list[tuple[int, int]],
    code_distance: int,
    *,
    include_lookahead: bool,
    allocation_level_range: range | None = None,
) -> AssignmentProblem | None:
    """Construct an assignment problem from live simulation state."""
    if not factories or not qubit_trackers:
        return None

    factory_ids = [factory.id for factory in factories]
    factory_index = {fid: idx for idx, fid in enumerate(factory_ids)}
    qubit_ids = sorted(qubit_trackers.keys())

    coverage_terms = build_coverage_terms(
        qubit_trackers,
        qubit_ids,
        code_distance,
        include_lookahead=include_lookahead,
        allocation_level_range=allocation_level_range,
    )
    if not coverage_terms:
        return None

    n_factories = len(factory_ids)
    n_qubits = len(qubit_ids)
    distances = np.zeros((n_factories, n_qubits), dtype=np.int32)

    qubit_index = {qid: idx for idx, qid in enumerate(qubit_ids)}
    factory_by_id = {factory.id: factory for factory in factories}

    for fid, fi in factory_index.items():
        fx, fy = factory_by_id[fid].location
        for qid, qi in qubit_index.items():
            qx, qy = logic_qubit_locations[qid]
            distances[fi, qi] = abs(fx - qx) + abs(fy - qy)

    angles = sorted({term.angle for term in coverage_terms})
    p_theta = {angle: calculate_success_rate(angle, code_distance) for angle in angles}

    terms_by_angle: dict[float, list[int]] = {angle: [] for angle in angles}
    qubits_by_angle: dict[float, list[int]] = {angle: [] for angle in angles}
    for term in coverage_terms:
        terms_by_angle[term.angle].append(term.term_index)
        if term.qubit_id not in qubits_by_angle[term.angle]:
            qubits_by_angle[term.angle].append(term.qubit_id)

    r_max = int(distances.max()) if n_factories and n_qubits else 0

    return AssignmentProblem(
        factory_ids=factory_ids,
        factory_index=factory_index,
        qubit_ids=qubit_ids,
        qubit_index={qid: idx for idx, qid in enumerate(qubit_ids)},
        distances=distances,
        coverage_terms=coverage_terms,
        angles=angles,
        p_theta=p_theta,
        terms_by_angle=terms_by_angle,
        qubits_by_angle=qubits_by_angle,
        r_max=r_max,
    )
