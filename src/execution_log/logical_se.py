"""Logical-qubit syndrome-extraction (SE) scheduler for Rz rounds.

This is a small bookkeeping helper used by the STAR / T-cultivation Rz schedulers
to interleave per-logical-qubit SE rounds into the existing factory-driven log
without reserving any extra AOD time.

Behavior:
    - A logical qubit is considered idle since its last "active" event (CNOT,
      S, H, Rz, T/Tdg). Each such event resets the qubit's idle timer because
      the logical gate already implies a syndrome extraction.
    - At every factory SE moment (single SE in STAR, or per-cycle expansion
      inside SE_stage_1 / SE_stage_2 in T-cultivation), ``piggyback`` is called.
      Any qubit that has been idle for at least ``interval - 2`` cycles then
      gets a logical SE event written sharing the factory's ``aod_id``.
    - If a logical qubit's idle ever exceeds ``interval + 2`` without a
      factory SE moment to piggy-back onto, ``force_due`` issues an SE at the
      requested time on the supplied AOD to keep the maximum gap bounded.

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
        self.upper = self.interval + 2
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
        """Emit one SE_q entry at ``t`` covering every qubit whose idle >= lower."""
        if not self.enabled:
            return
        due = [qid for qid, last in self.last_active.items() if t - last >= self.lower]
        self._emit_grouped(t, aod_id, execution_log, due)

    def force_due(
        self,
        t_now: float,
        aod_id: int | None,
        execution_log: list,
    ) -> None:
        """Emit one SE_q entry covering every qubit whose idle has exceeded ``upper``.

        This guards against a long stretch with no factory SE event; it keeps
        the maximum gap between SE rounds at roughly ``interval + 2`` cycles.
        """
        if not self.enabled:
            return
        due = [qid for qid, last in self.last_active.items() if t_now - last > self.upper]
        self._emit_grouped(t_now, aod_id, execution_log, due)
