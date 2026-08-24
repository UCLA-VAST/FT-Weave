"""Tests for logical-qubit SE rounds piggy-backed on factory SE.

Covers three layers:
1. Unit tests of :class:`src.execution_log.LogicalSEScheduler` (enable/disable,
   piggy-back lower bound, reset, force_due hard interval bound).
2. STAR integration (both sequential and parallel) — driving one TFIM Trotter
   layer with ``LOGICAL_SE_INTERVAL`` set and asserting that logical SE entries
   appear with the canonical schema and that ``validate_execution_log`` accepts
   them.
3. T-cultivation integration — same checks via the per-Rz-round T-cultivation
   pipeline.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path so `src` is importable when running tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.ds import get_microarchitecture
from src.execution_log import LogicalSEScheduler, validate_execution_log
from src.star.config import update_config as star_update
from src.star.tfim_star import generate_one_layer_2d_tfim_circuit_star
from src.t_cultivation.config import update_config as t_update
from src.t_cultivation.tfim_t import generate_one_layer_2d_tfim_circuit_t_cultivation


# ============================================================================
# HELPERS
# ============================================================================


def _is_logical_se(entry: dict) -> bool:
    """Logical-qubit SE entries are emitted with operation='SE_q' and empty factories."""
    return entry.get("operation") == "SE_q"


def _count_logical_se(layer_logs) -> int:
    n = 0
    for layer in layer_logs:
        if not isinstance(layer, list):
            continue
        for entry in layer:
            if _is_logical_se(entry):
                n += 1
    return n


def _all_logical_se_entries(layer_logs):
    for layer in layer_logs:
        if not isinstance(layer, list):
            continue
        for entry in layer:
            if _is_logical_se(entry):
                yield entry


# ============================================================================
# UNIT TESTS: LogicalSEScheduler
# ============================================================================


class TestLogicalSEScheduler:
    def test_disabled_when_interval_none(self):
        scheduler = LogicalSEScheduler([0, 1, 2], interval=None)
        log: list = []
        scheduler.piggyback(t=10.0, aod_id=0, execution_log=log)
        scheduler.force_due(t_now=100.0, aod_id=0, execution_log=log)
        scheduler.reset([0, 1], t=5.0)
        assert log == []
        assert not scheduler.enabled

    def test_disabled_when_interval_zero(self):
        scheduler = LogicalSEScheduler([0, 1], interval=0)
        log: list = []
        scheduler.piggyback(t=10.0, aod_id=0, execution_log=log)
        scheduler.force_due(t_now=100.0, aod_id=0, execution_log=log)
        assert log == []
        assert not scheduler.enabled

    def test_piggyback_emits_at_lower_bound(self):
        # interval = 6 => lower = 4
        scheduler = LogicalSEScheduler([0, 1], interval=6)
        log: list = []
        # idle=3 < 4 -> nothing
        scheduler.piggyback(t=3.0, aod_id=0, execution_log=log)
        assert log == []
        # idle=4 == lower -> a single grouped entry with both qubits
        scheduler.piggyback(t=4.0, aod_id=0, execution_log=log)
        assert len(log) == 1
        entry = log[0]
        assert entry["operation"] == "SE_q"
        assert entry["factories"] == []
        assert entry["aod_assignment"] == 0
        assert entry["targets"] == [0, 1]
        assert entry["end_time"] > entry["start_time"]

    def test_piggyback_groups_only_due_qubits(self):
        scheduler = LogicalSEScheduler([0, 1, 2], interval=6)
        log: list = []
        # qubit 1 was just used at t=3, so at t=4 only qubits 0 and 2 are due.
        scheduler.reset([1], t=3.0)
        scheduler.piggyback(t=4.0, aod_id=0, execution_log=log)
        assert len(log) == 1
        assert log[0]["targets"] == [0, 2]

    def test_piggyback_resets_last_active(self):
        scheduler = LogicalSEScheduler([0], interval=6)
        log: list = []
        scheduler.piggyback(t=4.0, aod_id=0, execution_log=log)
        assert len(log) == 1
        # last_active is now t + SE_TIME = 5; the next piggy-back at t=5
        # has idle = 0, so nothing should be emitted.
        scheduler.piggyback(t=5.0, aod_id=0, execution_log=log)
        assert len(log) == 1

    def test_reset_blocks_immediate_piggyback(self):
        scheduler = LogicalSEScheduler([0, 1], interval=6)
        log: list = []
        scheduler.reset([0], t=3.0)  # qubit 0 active at t=3
        scheduler.piggyback(t=4.0, aod_id=0, execution_log=log)
        # qubit 0: idle = 1 -> no emit
        # qubit 1: idle = 4 -> emit
        assert len(log) == 1
        assert log[0]["targets"] == [1]

    def test_reset_accepts_scalar_and_none(self):
        scheduler = LogicalSEScheduler([0], interval=6)
        scheduler.reset(0, t=2.5)
        assert scheduler.last_active[0] == 2.5
        # None and missing values are no-ops.
        scheduler.reset(None, t=10.0)
        scheduler.reset([None, "garbage"], t=10.0)
        assert scheduler.last_active[0] == 2.5

    def test_force_due_emits_at_hard_interval(self):
        # interval = 6 is a hard maximum idle gap.
        scheduler = LogicalSEScheduler([0], interval=6)
        log: list = []
        # idle = 5 < interval -> no force
        scheduler.force_due(t_now=5.0, aod_id=1, execution_log=log)
        assert log == []
        # idle = 6 == interval -> force-emit exactly at the deadline
        scheduler.force_due(t_now=6.0, aod_id=1, execution_log=log)
        assert len(log) == 1
        entry = log[0]
        assert entry["operation"] == "SE_q"
        assert entry["factories"] == []
        assert entry["targets"] == [0]
        assert entry["aod_assignment"] == 1
        assert entry["start_time"] == 6.0

    def test_force_due_catches_up_multiple_missed_rounds(self):
        scheduler = LogicalSEScheduler([0], interval=6)
        log: list = []
        scheduler.force_due(t_now=20.0, aod_id=1, execution_log=log)
        # SE_TIME = 1, so after starts at 6 and 13, the next hard deadline is 20.
        assert [entry["start_time"] for entry in log] == [6.0, 13.0, 20.0]
        assert all(entry["targets"] == [0] for entry in log)

    def test_register_new_qubits_midrun(self):
        scheduler = LogicalSEScheduler([0], interval=6)
        scheduler.register_qubits([1, 2], t=10.0)
        log: list = []
        scheduler.piggyback(t=14.0, aod_id=0, execution_log=log)
        # Existing qubit 0 is caught up at hard deadlines. Newly registered
        # qubits 1 and 2 piggy-back together once they reach the lower bound.
        assert [entry["start_time"] for entry in log] == [6.0, 13.0, 14.0]
        assert log[-1]["targets"] == [1, 2]

    def test_register_qubits_does_not_overwrite_existing(self):
        scheduler = LogicalSEScheduler([0], interval=6)
        scheduler.reset([0], t=5.0)
        scheduler.register_qubits([0, 1], t=10.0)
        # Existing qubit 0 keeps its 5.0 timestamp; only new qubit 1 starts at 10.0.
        assert scheduler.last_active[0] == 5.0
        assert scheduler.last_active[1] == 10.0


# ============================================================================
# INTEGRATION: STAR
# ============================================================================


_STAR_LAYOUT = (4, 4)
_STAR_N_QUBITS = _STAR_LAYOUT[0] * _STAR_LAYOUT[1]


def _build_star_config(n_aods: int = 1) -> dict:
    return {
        "n_aods": n_aods,
        "consider_skip_rus": 0,
        "tmr_assignment_method": "matching",
        "trivial_return": False,
        "decompose_move": False,
        "rng": np.random.default_rng(0),
    }


def _run_star(parallel: bool, interval: int | None, n_aods: int = 1):
    star_update(LOGICAL_SE_INTERVAL=interval)
    return generate_one_layer_2d_tfim_circuit_star(
        n_qubits=_STAR_N_QUBITS,
        qubit_layout=_STAR_LAYOUT,
        placement="col_based",
        J=1.0,
        h=1.0,
        dt=0.05,
        code_distance=3,
        config=_build_star_config(n_aods=n_aods),
        parallel_execution=parallel,
        analyze_result=False,
    )


class TestStarLogicalSE:
    def teardown_method(self, _method):
        # Always restore the global knob so other tests aren't perturbed.
        star_update(LOGICAL_SE_INTERVAL=None)

    def test_disabled_sequential_emits_no_logical_se(self):
        _, layer_logs, _ = _run_star(parallel=False, interval=None)
        assert _count_logical_se(layer_logs) == 0

    def test_disabled_parallel_emits_no_logical_se(self):
        _, layer_logs, _ = _run_star(parallel=True, interval=None, n_aods=2)
        assert _count_logical_se(layer_logs) == 0

    def test_enabled_sequential_emits_logical_se(self):
        _, layer_logs, _ = _run_star(parallel=False, interval=6)
        assert _count_logical_se(layer_logs) > 0

    def test_enabled_parallel_emits_logical_se(self):
        _, layer_logs, _ = _run_star(parallel=True, interval=6, n_aods=2)
        assert _count_logical_se(layer_logs) > 0

    def test_logical_se_entry_schema(self):
        _, layer_logs, _ = _run_star(parallel=False, interval=6)
        entries = list(_all_logical_se_entries(layer_logs))
        assert entries, "expected at least one logical SE entry"
        for entry in entries:
            assert entry["operation"] == "SE_q"
            assert entry["factories"] == []
            targets = entry["targets"]
            assert isinstance(targets, list) and len(targets) >= 1
            for qid in targets:
                assert isinstance(qid, int)
                assert 0 <= qid < _STAR_N_QUBITS
            assert len(set(targets)) == len(targets), "targets must be unique"
            assert entry["end_time"] > entry["start_time"]

    def test_validate_execution_log_accepts_sequential_log(self):
        _, layer_logs, _ = _run_star(parallel=False, interval=6)
        _, mloc = get_microarchitecture(
            n_qubits=_STAR_N_QUBITS,
            n_factories=_STAR_N_QUBITS,
            qubit_layout=_STAR_LAYOUT,
            placement="col_based",
        )
        # validate_execution_log raises on any inconsistency; running it on each
        # Rz-round log doubles as the assertion.
        for layer in layer_logs:
            if isinstance(layer, list) and any(
                entry.get("operation") in ("SE", "SE_q") for entry in layer
            ):
                validate_execution_log(layer, mloc, n_aods=1)

    def test_parallel_pipeline_internal_validation_passes(self):
        # factory_angle_execution_parallel calls validate_execution_log internally;
        # a successful run with the feature on means concurrent factory + logical
        # SE writes are accepted.
        _, layer_logs, _ = _run_star(parallel=True, interval=6, n_aods=2)
        assert _count_logical_se(layer_logs) > 0


# ============================================================================
# INTEGRATION: T cultivation
# ============================================================================


_T_LAYOUT = (4, 4)
_T_N_QUBITS = _T_LAYOUT[0] * _T_LAYOUT[1]


def _build_t_config() -> dict:
    return {
        "n_aods": 2,
        "rng": np.random.default_rng(0),
        "to_decompose": False,
        "print_profile": False,
    }


def _run_t_cultivation(interval: int | None):
    t_update(LOGICAL_SE_INTERVAL=interval)
    return generate_one_layer_2d_tfim_circuit_t_cultivation(
        n_qubits=_T_N_QUBITS,
        qubit_layout=_T_LAYOUT,
        placement="col_based",
        J=1.0,
        h=1.0,
        dt=0.05,
        code_distance=3,
        config=_build_t_config(),
        analyze_result=False,
    )


class TestTCultivationLogicalSE:
    def teardown_method(self, _method):
        t_update(LOGICAL_SE_INTERVAL=None)

    def test_disabled_emits_no_logical_se(self):
        _, full_logs, _ = _run_t_cultivation(interval=None)
        assert _count_logical_se(full_logs) == 0

    def test_enabled_emits_logical_se(self):
        _, full_logs, _ = _run_t_cultivation(interval=6)
        assert _count_logical_se(full_logs) > 0

    def test_logical_se_entry_schema(self):
        _, full_logs, _ = _run_t_cultivation(interval=6)
        entries = list(_all_logical_se_entries(full_logs))
        assert entries, "expected at least one logical SE entry"
        for entry in entries:
            assert entry["operation"] == "SE_q"
            assert entry["factories"] == []
            targets = entry["targets"]
            assert isinstance(targets, list) and len(targets) >= 1
            for qid in targets:
                assert isinstance(qid, int)
                assert 0 <= qid < _T_N_QUBITS
            assert len(set(targets)) == len(targets), "targets must be unique"
            assert entry["end_time"] > entry["start_time"]

    def test_logical_se_window_during_stage_blocks(self):
        """Logical SEs in T cultivation should align with integer cycles inside
        the SE_stage_1/SE_stage_2 blocks they piggy-back on."""
        from src.t_cultivation import config as tcfg

        _, full_logs, _ = _run_t_cultivation(interval=6)
        # Build per-layer (stage_1_start, stage_1_end) and (stage_2_start, stage_2_end)
        for layer in full_logs:
            if not isinstance(layer, list):
                continue
            stage_intervals: list[tuple[float, float, str]] = []
            for entry in layer:
                op = entry.get("operation")
                if op in ("SE_stage_1", "SE_stage_2"):
                    stage_intervals.append((entry["start_time"], entry["end_time"], op))
            if not stage_intervals:
                continue
            for entry in layer:
                if not _is_logical_se(entry):
                    continue
                # Each logical SE's start_time must fall inside one of the stage
                # blocks (it can also coincide with the block end since one cycle
                # is added on top via SE_TIME).
                t = entry["start_time"]
                inside_any = any(s <= t < e for (s, e, _) in stage_intervals)
                # Logical SE may also be triggered by force_due after the block,
                # so we only check that *if* it's inside a block, the time is an
                # integer offset (consistent with per-cycle expansion).
                if inside_any:
                    # find the block containing it
                    s_block = next(s for (s, e, _) in stage_intervals if s <= t < e)
                    # offset should be a non-negative integer cycle.
                    offset = t - s_block
                    assert offset >= 0
                    assert abs(offset - round(offset)) < 1e-9


# ============================================================================
# CADENCE SANITY CHECK
# ============================================================================


class TestLogicalSECadence:
    """Statistical sanity check: most consecutive activity events for a qubit
    should be separated by approximately the configured interval ``x``.

    Piggy-back opportunities may schedule logical SE slightly before ``x``,
    but the scheduler's hard-deadline path prevents idle gaps from exceeding
    the configured interval.
    """

    def test_star_sequential_idle_gap_never_exceeds_interval(self):
        try:
            _, layer_logs, _ = _run_star(parallel=False, interval=6)
        finally:
            star_update(LOGICAL_SE_INTERVAL=None)

        from collections import defaultdict

        idle_gaps: list[float] = []
        for layer in layer_logs:
            if not isinstance(layer, list):
                continue
            events_per_q: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for entry in layer:
                op = entry.get("operation")
                targets = entry.get("targets")
                factories = entry.get("factories")
                start = entry.get("start_time")
                end = entry.get("end_time")
                if op == "SE_q":
                    for q in targets or []:
                        events_per_q[q].append((start, end))
                elif op == "CNOT" and factories:
                    for q in targets or []:
                        if isinstance(q, int):
                            events_per_q[q].append((start, end))
                elif op == "S":
                    for q in targets or []:
                        if isinstance(q, int):
                            events_per_q[q].append((start, end))
            for times in events_per_q.values():
                times.sort()
                for (_prev_start, prev_end), (next_start, _next_end) in zip(
                    times, times[1:]
                ):
                    idle_gaps.append(next_start - prev_end)

        assert idle_gaps, "expected to gather some inter-event gaps"
        max_gap = max(idle_gaps)
        assert max_gap <= 6, f"max idle gap {max_gap} exceeds interval 6"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
