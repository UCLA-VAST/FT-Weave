"""Logical-qubit syndrome-extraction (SE) scheduler for Rz rounds.

This is a small bookkeeping helper used by the STAR / T-cultivation Rz schedulers
to interleave per-logical-qubit SE rounds into the existing factory-driven log
without reserving any extra AOD time.

Behavior:
    - A logical qubit is considered idle since its last "active" event (CNOT,
      S, H, Rz, T/Tdg). Each such event resets the qubit's idle timer because
      the logical gate already implies a syndrome extraction.
    - At every factory SE or CNOT moment, ``piggyback`` is called. Any qubit
      that has been idle for at least ``interval - 2`` cycles then gets a
      logical SE event written sharing the host operation's ``aod_id``.
    - ``interval`` is a hard maximum idle gap. If no piggy-back opportunity
      occurs by then, ``force_due`` writes all missing logical SE rounds at
      their deadlines so every logical qubit receives SE at least once per
      ``interval`` idle cycles.

Logical SE events are emitted with ``operation="SE_q"`` (distinct from the
factory ``"SE"`` operation) so downstream consumers (validators, fidelity
simulators, animators, plotters) can unambiguously route them to the
logical-qubit row even when qubit and factory ids collide.
"""

from __future__ import annotations

from typing import Iterable

from src.execution_log.event_helpers import build_event
from src.star.config import SE_TIME


class LogicalSEScheduler:
    """Track per-logical-qubit idle time and emit piggy-backed SE events."""

    def __init__(
        self,
        qubit_ids: Iterable[int],
        interval: int | None,
        start_time: float = 0.0,
    ) -> None:
        self.enabled = interval is not None and interval > 0
        self.interval = int(interval) if interval is not None else 0
        self.lower = max(0, self.interval - 2)
        self.upper = self.interval
        self.last_active: dict[int, float] = {
            int(q): float(start_time) for q in qubit_ids
        }

    def register_qubits(self, qubit_ids: Iterable[int], t: float) -> None:
        """Register additional qubits (e.g. discovered mid-run) at time ``t``."""
        for q in qubit_ids:
            self.last_active.setdefault(int(q), float(t))

    def reset(self, qubits, t: float) -> None:
        """Mark ``qubits`` as just having executed an SE-bearing gate at ``t``."""
        if not self.enabled or qubits is None:
            return
        if isinstance(qubits, (int,)):
            qubits = [qubits]
        for q in qubits:
            if q is None:
                continue
            try:
                qid = int(q)
            except (TypeError, ValueError):
                continue
            self.last_active[qid] = float(t)

    def _emit_grouped(
        self,
        t: float,
        aod_id: int | None,
        execution_log: list,
        due_qubits: list[int],
    ) -> None:
        """Append a single SE_q entry covering every qubit in ``due_qubits``.

        All qubits piggy-backing onto (or forced at) the same moment share the
        same start/end time and AOD, so we coalesce them into one log entry
        with a multi-qubit ``targets`` list rather than emitting one entry per
        qubit.
        """
        if not due_qubits:
            return
        targets = sorted(set(int(q) for q in due_qubits))
        end_time = float(t) + SE_TIME
        execution_log.append(
            build_event(
                start_time=float(t),
                end_time=end_time,
                factories=[],
                operation="SE_q",
                aod_assignment=aod_id,
                targets=targets,
                move_vecs=None,
            )
        )
        for qid in targets:
            self.last_active[qid] = end_time

    def piggyback(
        self,
        t: float,
        aod_id: int | None,
        execution_log: list,
    ) -> None:
        """Emit one SE_q entry at ``t`` covering every qubit whose idle >= lower.

        Before using the current piggy-back opportunity, catch up any hard
        deadlines that have already passed. This prevents a late factory-SE or
        CNOT opportunity from stretching a qubit's idle gap beyond ``interval``.
        """
        if not self.enabled:
            return
        self.force_due(t, aod_id, execution_log)
        due = [qid for qid, last in self.last_active.items() if t - last >= self.lower]
        self._emit_grouped(t, aod_id, execution_log, due)

    def force_due(
        self,
        t_now: float,
        aod_id: int | None,
        execution_log: list,
        *,
        include_current: bool = True,
    ) -> None:
        """Emit all SE_q entries needed to satisfy the hard interval bound.

        This guards against a long stretch with no factory-SE/CNOT piggy-back
        opportunity. If a qubit is overdue by multiple intervals, this emits
        multiple rounds at successive deadlines rather than only one late round.
        Set ``include_current=False`` before a CNOT target reset so a CNOT that
        starts exactly at the hard deadline can count as the target qubit's SE
        round without also emitting a same-qubit SE_q at that time.
        """
        if not self.enabled:
            return
        t_limit = float(t_now)
        while True:
            due_by_time: dict[float, list[int]] = {}
            for qid, last in self.last_active.items():
                due_time = float(last) + self.upper
                is_due = due_time <= t_limit if include_current else due_time < t_limit
                if is_due:
                    due_by_time.setdefault(due_time, []).append(qid)
            if not due_by_time:
                return
            for due_time in sorted(due_by_time):
                self._emit_grouped(due_time, aod_id, execution_log, due_by_time[due_time])
