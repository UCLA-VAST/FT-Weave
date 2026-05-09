from __future__ import annotations

import ast
import re
from typing import Any

from src.ds import Architecture
from src.execution_log.event_helpers import normalize_factories
from src.writer.ft_zair_writer import FTZAIRWriter
from src.writer.zair_writer import ZAIRWriter

_PAIR_RE = re.compile(r"^\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*$")

# Keep in sync with Animator.FPS / Animator.MUS_PER_FRM in animator_matplotlib.py
_ANIMATOR_FPS = 10
_ANIMATOR_MUS_PER_FRM = 150.0 / _ANIMATOR_FPS
# Defaults tuned to match matplotlib Animator sampling (~MUS_PER_FRM per frame).
_MIN_ANIMATOR_FRAMES_COMPACT = 2  # 1qGate, rydberg: brief but still visible
_MIN_ANIMATOR_FRAMES_REARRANGE = 8  # whole rearrangeJob; substeps weighted toward move*
# FT animator: SE/CNOT and zero-time markers need enough packed duration that
# frame sampling (ceil + >=1 frame each) never skips an instruction.
_MIN_ANIMATOR_FRAMES_FT_ENTANGLE = 6  # SE, SE_stage_*, CNOT (logical zone highlight)
_MIN_ANIMATOR_FRAMES_FT_MARKER = 6  # RUS_fail, TMR_fail (event-only; must still be visible)
_DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT = 4.0


def default_init_locs_for_logic_grid(
    logic_qubit_locations: list[tuple[int, int]],
    slm_id: int = 0,
) -> list[list[int]]:
    return [[i, slm_id, y, x] for i, (x, y) in enumerate(logic_qubit_locations)]


def default_init_locs_with_factories(
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    slm_id: int = 0,
) -> list[list[int]]:
    """Init ZAIR sites: logical qubits ``0..L-1`` then factories ``L..L+F-1`` (left trap)."""
    init = default_init_locs_for_logic_grid(logic_qubit_locations, slm_id=slm_id)
    base = len(init)
    for j, (x, y) in enumerate(magic_state_locations):
        init.append([base + j, slm_id, y, x])
    return init


def _safe_interval(
    start: float, end: float, min_dt: float = 1e-6
) -> tuple[float, float]:
    if end < start:
        end = start
    if end - start < min_dt:
        end = start + min_dt
    return start, end


def _parse_parenthesized_pair(s: str) -> tuple[int, int]:
    m = _PAIR_RE.match(s.strip())
    if not m:
        raise ValueError(f"Cannot parse grid pair from string: {s!r}")
    a, b = m.group(1).strip(), m.group(2).strip()
    try:
        return int(float(a)), int(float(b))
    except ValueError:
        return int(ast.literal_eval(a)), int(ast.literal_eval(b))


def _parse_move_endpoints(pair: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    """Grid (x, y) endpoints in chronological order: ``start`` then ``end``.

    STAR/RUS string pairs from ``execute_movement`` are ``[factory_site, qubit_site]``
    on forward ``move`` and ``[qubit_site, factory_home]`` on ``return_move``. Tuple
    pairs follow Clifford CNOT logs (mobile then pivot, or reversed for return).
    """
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        a, b = pair[0], pair[1]
        if isinstance(a, str) and isinstance(b, str):
            start = _parse_parenthesized_pair(a)
            end = _parse_parenthesized_pair(b)
            return start, end
        if (
            isinstance(a, (list, tuple))
            and isinstance(b, (list, tuple))
            and len(a) == 2
            and len(b) == 2
        ):
            return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    raise ValueError(f"Unsupported move_vec pair: {pair!r}")


def _nearest_qubit_index(gx: int, gy: int, logic_locs: list[tuple[int, int]]) -> int:
    best_i = 0
    best_d = float("inf")
    for i, (lx, ly) in enumerate(logic_locs):
        d = (gx - lx) ** 2 + (gy - ly) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _factory_particle_id(
    factory_field: Any, pair_index: int, n_logic: int
) -> int | None:
    """Map log ``factories`` entry to init particle id ``L + factory_id`` when unambiguous."""
    ids = normalize_factories(factory_field)
    if not ids:
        return None
    fid: int | None
    if pair_index < len(ids):
        fid = int(ids[pair_index])
    elif len(ids) == 1 and pair_index == 0:
        fid = int(ids[0])
    else:
        return None
    if fid < 0:
        return None
    return n_logic + fid


def _nearest_particle_index(
    gx: int,
    gy: int,
    logic_locs: list[tuple[int, int]],
    factory_locs: list[tuple[int, int]],
) -> int:
    """Nearest site among logical qubits (indices ``0..L-1``) then factories ``L..``."""
    n_log = len(logic_locs)
    best_i = 0
    best_d = float("inf")
    for i, (lx, ly) in enumerate(logic_locs):
        d = (gx - lx) ** 2 + (gy - ly) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    for j, (lx, ly) in enumerate(factory_locs):
        d = (gx - lx) ** 2 + (gy - ly) ** 2
        if d < best_d:
            best_d = d
            best_i = n_log + j
    return best_i


def _cnot_pairs_from_targets(targets: Any) -> list[tuple[int, int]] | None:
    if not targets or not isinstance(targets, (list, tuple)):
        return None
    pairs: list[tuple[int, int]] = []
    for t in targets:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            pairs.append((int(t[0]), int(t[1])))
    return pairs or None


def _mobile_qubit_for_move_vec(
    op: str,
    start: tuple[int, int],
    end: tuple[int, int],
    cnot_pairs: list[tuple[int, int]] | None,
    logic_locs: list[tuple[int, int]],
) -> int | None:
    """Resolve which logical qubit is *moving* using CNOT (mobile, pivot) pairs when present."""
    if not cnot_pairs:
        return None
    sx, sy = start
    ex, ey = end
    for mobile, pivot in cnot_pairs:
        mh = logic_locs[mobile]
        ph = logic_locs[pivot]
        if op == "move":
            if mh == (sx, sy) and ph == (ex, ey):
                return mobile
        elif op == "return_move":
            if ph == (sx, sy) and mh == (ex, ey):
                return mobile
    return None


def _targets_as_int_qubits(targets: Any) -> list[int]:
    if targets is None:
        return []
    if isinstance(targets, (list, tuple)):
        out: list[int] = []
        for t in targets:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                continue
            if t is None:
                continue
            if isinstance(t, (int, float, str)):
                out.append(int(t))
        return out
    if isinstance(targets, (int, float, str)):
        return [int(targets)]
    return []


def _scalar_floats_from_targets(targets: Any) -> list[float]:
    """Scalars in ``targets`` (skips CNOT-style pairs). Used for Rz angles in STAR TMR logs."""
    if targets is None:
        return []
    if isinstance(targets, (list, tuple)):
        out: list[float] = []
        for t in targets:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                continue
            if isinstance(t, (list, tuple, dict, set)):
                continue
            if t is None:
                continue
            if isinstance(t, bool):
                continue
            if isinstance(t, (int, float)):
                out.append(float(t))
                continue
            if isinstance(t, str):
                try:
                    out.append(float(t))
                except ValueError:
                    continue
        return out
    if isinstance(targets, bool) or isinstance(targets, (list, tuple, dict, set)):
        return []
    if isinstance(targets, (int, float)):
        return [float(targets)]
    if isinstance(targets, str):
        try:
            return [float(targets)]
        except ValueError:
            return []
    return []


def _factory_particle_ids(factory_field: Any, n_logic: int) -> list[int]:
    """Map ``factories`` field to init particle indices ``L + factory_id``."""
    ids = normalize_factories(factory_field)
    return [n_logic + int(fid) for fid in ids if fid is not None and int(fid) >= 0]


def execution_log_to_zair_instructions(
    execution_log: list[dict[str, Any]],
    *,
    architecture: Architecture,
    logic_qubit_locations: list[tuple[int, int]],
    init_locs: list[list[int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
    time_scale: float = 1.0,
    skip_barriers: bool = True,
) -> list[dict[str, Any]]:
    factory_xy = list(magic_state_locations) if magic_state_locations else []
    if init_locs is None:
        if factory_xy:
            init_locs = default_init_locs_with_factories(
                logic_qubit_locations, factory_xy
            )
        else:
            init_locs = default_init_locs_for_logic_grid(logic_qubit_locations)

    writer = ZAIRWriter(architecture=architecture)
    prompts_init = {"type": "init", "id": 0, "init_locs": init_locs}
    instructions: list[dict[str, Any]] = [writer.write_instruction(prompts_init)]

    current_locs = {qid: list(loc) for qid, *loc in init_locs}
    next_id = 1

    sorted_log = sorted(
        execution_log,
        key=lambda e: (
            float(e.get("start_time", 0.0)),
            float(e.get("end_time", 0.0)),
            str(e.get("operation", "")),
        ),
    )

    for entry in sorted_log:
        op = str(entry.get("operation", ""))
        if skip_barriers and op == "Barrier":
            continue

        t0 = float(entry.get("start_time", 0.0)) * time_scale
        t1 = float(entry.get("end_time", t0)) * time_scale
        t0, t1 = _safe_interval(t0, t1)
        aod_raw = entry.get("aod_assignment")
        aod_id = 0 if aod_raw is None else int(aod_raw)

        if op in ("move", "return_move"):
            move_vecs = entry.get("move_vecs")
            if not isinstance(move_vecs, list) or not move_vecs:
                continue
            flat_rows: list[tuple[int, int, list[int], list[int]]] = []
            cnot_pairs = _cnot_pairs_from_targets(entry.get("targets"))
            n_logic = len(logic_qubit_locations)
            for mi, pair in enumerate(move_vecs):
                (sx, sy), (ex, ey) = _parse_move_endpoints(pair)
                qid = _mobile_qubit_for_move_vec(
                    op, (sx, sy), (ex, ey), cnot_pairs, logic_qubit_locations
                )
                if qid is None:
                    qid = _factory_particle_id(entry.get("factories"), mi, n_logic)
                if qid is None:
                    # Clifford / fallback: nearest site (mobile qubit or factory) at path endpoints.
                    if op == "return_move":
                        qid = _nearest_particle_index(
                            ex, ey, logic_qubit_locations, factory_xy
                        )
                    else:
                        qid = _nearest_particle_index(
                            sx, sy, logic_qubit_locations, factory_xy
                        )

                # Execution logs omit trap side:
                # - default/init is left trap (SLM 0)
                # - forward movement goes to right trap (SLM 1)
                # - return movement goes back to left trap (SLM 0)
                if op == "move":
                    new_loc = [1, ey, ex]
                    default_old = [0, sy, sx]
                else:
                    # return: start (e.g. pivot/factory on right) -> end (mobile home on left)
                    new_loc = [0, ey, ex]
                    default_old = [1, sy, sx]

                old_loc = current_locs.get(qid)
                if old_loc is None:
                    old_loc = list(default_old)
                else:
                    old_loc = list(old_loc)

                begin_loc = [qid, old_loc[0], old_loc[1], old_loc[2]]
                end_loc = [qid, new_loc[0], new_loc[1], new_loc[2]]
                flat_rows.append((old_loc[1], qid, begin_loc, end_loc))
                current_locs[qid] = new_loc

            # Row-major grouping so one activate can pick up all participating rows/cols.
            row_order: list[int] = []
            row_groups: dict[int, dict[str, list]] = {}
            for row_y, qid, b_loc, e_loc in flat_rows:
                if row_y not in row_groups:
                    row_order.append(row_y)
                    row_groups[row_y] = {"aod_qubits": [], "begin_locs": [], "end_locs": []}
                grp = row_groups[row_y]
                grp["aod_qubits"].append(qid)
                grp["begin_locs"].append(b_loc)
                grp["end_locs"].append(e_loc)

            aod_qubits = [row_groups[y]["aod_qubits"] for y in row_order]
            begin_locs = [row_groups[y]["begin_locs"] for y in row_order]
            end_locs = [row_groups[y]["end_locs"] for y in row_order]
            prompt = {
                "type": "rearrangeJob",
                "id": next_id,
                "aod_id": aod_id,
                "aod_qubits": aod_qubits,
                "begin_locs": begin_locs,
                "end_locs": end_locs,
                "dependency": {"qubit": [next_id - 1]},
            }
            inst = writer.write_instruction(prompt)
            inst["begin_time"] = t0
            inst["end_time"] = t1
            if inst.get("insts"):
                n_sub = max(1, len(inst["insts"]))
                dt = (t1 - t0) / n_sub
                for idx, sub in enumerate(inst["insts"]):
                    sub["begin_time"] = t0 + idx * dt
                    sub["end_time"] = t0 + (idx + 1) * dt
            instructions.append(inst)
            next_id += 1
            continue

        if op == "Rz":
            n_logic = len(logic_qubit_locations)
            qs = _factory_particle_ids(entry.get("factories"), n_logic)
            if not qs:
                continue
            angles = _scalar_floats_from_targets(entry.get("targets"))
            while len(angles) < len(qs):
                angles.append(0.0)
            angles = angles[: len(qs)]
            mapping = {q: current_locs.get(q, [0, 0, 0]) for q in qs}
            prompt = {
                "type": "1qGate",
                "id": next_id,
                "unitary": op,
                "gates": [{"q": q, "angle": ang} for q, ang in zip(qs, angles)],
                "mapping": mapping,
                "dependency": {"qubit": [next_id - 1]},
            }
            inst = writer.write_instruction(prompt)
            inst["begin_time"] = t0
            inst["end_time"] = t1
            instructions.append(inst)
            next_id += 1
            continue

        if op in ("H", "S"):
            qs = _targets_as_int_qubits(entry.get("targets"))
            if not qs:
                continue
            mapping = {q: current_locs.get(q, [0, 0, 0]) for q in qs}
            prompt = {
                "type": "1qGate",
                "id": next_id,
                "unitary": op,
                "gates": [{"q": q, "angle": 0.0} for q in qs],
                "mapping": mapping,
                "dependency": {"qubit": [next_id - 1]},
            }
            inst = writer.write_instruction(prompt)
            inst["begin_time"] = t0
            inst["end_time"] = t1
            instructions.append(inst)
            next_id += 1
            continue

        if op in (
            "SE",
            "SE_q",
            "SE_stage_1",
            "SE_stage_2",
            "CNOT",
            # "tmr_fail",
            # "TMR_fail",
            # "RUS_success",
            # "RUS_fail",
        ):
            prompt = {
                "type": "rydberg",
                "id": next_id,
                "zone_id": 0,
                "gates": [],
                "dependency": {"qubit": [next_id - 1]},
            }
            inst = writer.write_instruction(prompt)
            inst["begin_time"] = t0
            inst["end_time"] = t1
            instructions.append(inst)
            next_id += 1
            continue

    return instructions


def _resplit_rearrange_subtimes_weighted(
    inst: dict[str, Any],
    *,
    move_weight: float = _DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT,
) -> None:
    """Partition rearrangeJob time across substeps; ``move*`` steps get ``move_weight`` share."""
    subs = inst.get("insts")
    if not subs:
        return
    t0 = float(inst["begin_time"])
    t1 = float(inst["end_time"])
    span = t1 - t0
    if span <= 0:
        return
    weights: list[float] = []
    for s in subs:
        st = s.get("type", "")
        w = move_weight if isinstance(st, str) and st.startswith("move") else 1.0
        weights.append(w)
    wsum = sum(weights) or 1.0
    acc = t0
    for i, (sub, w) in enumerate(zip(subs, weights)):
        dt = span * (w / wsum)
        sub["begin_time"] = acc
        sub["end_time"] = acc + dt if i < len(subs) - 1 else t1
        acc = sub["end_time"]


def pack_timeline_for_matplotlib_animator(
    instructions: list[dict[str, Any]],
    *,
    min_compact_duration: float | None = None,
    min_rearrange_duration: float | None = None,
    rearrange_move_substep_weight: float = _DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT,
) -> float:
    """Repack instruction times so the animator samples each step; movement gets more wall time.

    Non-``rearrangeJob`` instructions use ``min_compact_duration``. ``rearrangeJob`` uses
    ``min_rearrange_duration`` and splits substeps with extra weight on ``move*`` phases
    so trajectories get more frames than activate/deactivate.
    """
    if min_compact_duration is None:
        min_compact_duration = _ANIMATOR_MUS_PER_FRM * _MIN_ANIMATOR_FRAMES_COMPACT
    if min_rearrange_duration is None:
        min_rearrange_duration = _ANIMATOR_MUS_PER_FRM * _MIN_ANIMATOR_FRAMES_REARRANGE
    if len(instructions) <= 1:
        return float(instructions[0].get("end_time", 0.0)) if instructions else 0.0

    t_cursor = 0.0
    for inst in instructions[1:]:
        b = float(inst.get("begin_time", 0.0))
        e = float(inst.get("end_time", b))
        raw = e - b
        if inst.get("type") == "rearrangeJob":
            dur = max(raw, min_rearrange_duration)
        else:
            dur = max(raw, min_compact_duration)
        inst["begin_time"] = t_cursor
        inst["end_time"] = t_cursor + dur
        if inst.get("type") == "rearrangeJob":
            _resplit_rearrange_subtimes_weighted(
                inst, move_weight=rearrange_move_substep_weight
            )
        t_cursor = inst["end_time"]
    return t_cursor


def execution_log_to_animator_code(
    execution_log: list[dict[str, Any]],
    *,
    architecture: Architecture,
    logic_qubit_locations: list[tuple[int, int]],
    init_locs: list[list[int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
    time_scale: float = 1.0,
    name: str = "execution_log",
    skip_barriers: bool = True,
    pack_timeline_for_animator: bool = True,
    min_animator_compact_instruction_duration: float | None = None,
    min_animator_rearrange_duration: float | None = None,
    rearrange_move_substep_weight: float = _DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT,
) -> dict[str, Any]:
    """Build animator ``code`` dict from an execution log."""
    instructions = execution_log_to_zair_instructions(
        execution_log,
        architecture=architecture,
        logic_qubit_locations=logic_qubit_locations,
        init_locs=init_locs,
        magic_state_locations=magic_state_locations,
        time_scale=time_scale,
        skip_barriers=skip_barriers,
    )
    runtime: float
    if pack_timeline_for_animator:
        runtime = pack_timeline_for_matplotlib_animator(
            instructions,
            min_compact_duration=min_animator_compact_instruction_duration,
            min_rearrange_duration=min_animator_rearrange_duration,
            rearrange_move_substep_weight=rearrange_move_substep_weight,
        )
    else:
        end_times = [float(inst.get("end_time", 0.0)) for inst in instructions[1:]]
        runtime = max(end_times) if end_times else 0.0

    return {
        "name": name,
        "architecture_spec_path": None,
        "instructions": instructions,
        "runtime": runtime,
        "n_logic_qubits": len(logic_qubit_locations),
    }


def execution_log_to_ft_zair_instructions(
    execution_log: list[dict[str, Any]],
    *,
    architecture: Architecture,
    logic_qubit_locations: list[tuple[int, int]],
    init_locs: list[list[int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
    time_scale: float = 1.0,
    skip_barriers: bool = True,
) -> list[dict[str, Any]]:
    """Build FT-ZAIR logical instructions from execution log.

    FT-ZAIR keeps logical operation identity (e.g. ``SE_stage_1``) and stores
    a ZAIR-compatible realization payload for downstream synthesis/animation.
    """
    factory_xy = list(magic_state_locations) if magic_state_locations else []
    if init_locs is None:
        if factory_xy:
            init_locs = default_init_locs_with_factories(
                logic_qubit_locations, factory_xy
            )
        else:
            init_locs = default_init_locs_for_logic_grid(logic_qubit_locations)

    ft_writer = FTZAIRWriter(architecture=architecture)
    instructions: list[dict[str, Any]] = [
        ft_writer.write_instruction(
            {
                "type": "init",
                "id": 0,
                "init_locs": init_locs,
                "dependency": {},
                "begin_time": 0.0,
                "end_time": 0.0,
            }
        )
    ]

    current_locs = {qid: list(loc) for qid, *loc in init_locs}
    next_id = 1
    sorted_log = sorted(
        execution_log,
        key=lambda e: (
            float(e.get("start_time", 0.0)),
            float(e.get("end_time", 0.0)),
            str(e.get("operation", "")),
        ),
    )
    for entry in sorted_log:
        op = str(entry.get("operation", ""))
        if skip_barriers and op == "Barrier":
            continue
        t0 = float(entry.get("start_time", 0.0)) * time_scale
        t1 = float(entry.get("end_time", t0)) * time_scale
        t0, t1 = _safe_interval(t0, t1)
        aod_raw = entry.get("aod_assignment")
        aod_id = 0 if aod_raw is None else int(aod_raw)

        if op in ("move", "return_move"):
            move_vecs = entry.get("move_vecs")
            if not isinstance(move_vecs, list) or not move_vecs:
                continue
            flat_rows: list[tuple[int, int, list[int], list[int]]] = []
            cnot_pairs = _cnot_pairs_from_targets(entry.get("targets"))
            n_logic = len(logic_qubit_locations)
            for mi, pair in enumerate(move_vecs):
                (sx, sy), (ex, ey) = _parse_move_endpoints(pair)
                qid = _mobile_qubit_for_move_vec(
                    op, (sx, sy), (ex, ey), cnot_pairs, logic_qubit_locations
                )
                if qid is None:
                    qid = _factory_particle_id(entry.get("factories"), mi, n_logic)
                if qid is None:
                    qid = _nearest_particle_index(
                        ex if op == "return_move" else sx,
                        ey if op == "return_move" else sy,
                        logic_qubit_locations,
                        factory_xy,
                    )
                if op == "move":
                    new_loc = [1, ey, ex]
                    default_old = [0, sy, sx]
                else:
                    new_loc = [0, ey, ex]
                    default_old = [1, sy, sx]
                old_loc = list(current_locs.get(qid, default_old))
                begin_loc = [qid, old_loc[0], old_loc[1], old_loc[2]]
                end_loc = [qid, new_loc[0], new_loc[1], new_loc[2]]
                flat_rows.append((old_loc[1], qid, begin_loc, end_loc))
                current_locs[qid] = new_loc

            row_order: list[int] = []
            row_groups: dict[int, dict[str, list]] = {}
            for row_y, qid, b_loc, e_loc in flat_rows:
                if row_y not in row_groups:
                    row_order.append(row_y)
                    row_groups[row_y] = {"aod_qubits": [], "begin_locs": [], "end_locs": []}
                grp = row_groups[row_y]
                grp["aod_qubits"].append(qid)
                grp["begin_locs"].append(b_loc)
                grp["end_locs"].append(e_loc)
            rearrange_prompt = {
                "type": "rearrangeJob",
                "id": next_id,
                "aod_id": aod_id,
                "aod_qubits": [row_groups[y]["aod_qubits"] for y in row_order],
                "begin_locs": [row_groups[y]["begin_locs"] for y in row_order],
                "end_locs": [row_groups[y]["end_locs"] for y in row_order],
                "dependency": {"qubit": [next_id - 1]},
            }
            rearrange_job = ft_writer.write_zair_instruction(rearrange_prompt)
            rearrange_job["begin_time"] = t0
            rearrange_job["end_time"] = t1
            if rearrange_job.get("insts"):
                n_sub = max(1, len(rearrange_job["insts"]))
                dt = (t1 - t0) / n_sub
                for idx, sub in enumerate(rearrange_job["insts"]):
                    sub["begin_time"] = t0 + idx * dt
                    sub["end_time"] = t0 + (idx + 1) * dt
            instructions.append(
                ft_writer.write_instruction(
                    {
                        "type": op,
                        "id": next_id,
                        "aod_id": aod_id,
                        "targets": entry.get("targets"),
                        "factories": normalize_factories(entry.get("factories")),
                        "move_vecs": move_vecs,
                        "dependency": {"qubit": [next_id - 1]},
                        "begin_time": t0,
                        "end_time": t1,
                        "rearrange_job": rearrange_job,
                    }
                )
            )
            next_id += 1
            continue

        if op == "Rz":
            n_logic = len(logic_qubit_locations)
            qs = _factory_particle_ids(entry.get("factories"), n_logic)
            if not qs:
                continue
            angles = _scalar_floats_from_targets(entry.get("targets"))
            while len(angles) < len(qs):
                angles.append(0.0)
            angles = angles[: len(qs)]
            mapping = {q: current_locs.get(q, [0, 0, 0]) for q in qs}
            gate_prompt = {
                "type": "1qGate",
                "id": next_id,
                "unitary": op,
                "gates": [{"q": q, "angle": ang} for q, ang in zip(qs, angles)],
                "mapping": mapping,
                "dependency": {"qubit": [next_id - 1]},
            }
            gate_inst = ft_writer.write_zair_instruction(gate_prompt)
            gate_inst["begin_time"] = t0
            gate_inst["end_time"] = t1
            instructions.append(
                ft_writer.write_instruction(
                    {
                        "type": op,
                        "id": next_id,
                        "targets": qs,
                        "factories": normalize_factories(entry.get("factories")),
                        "dependency": {"qubit": [next_id - 1]},
                        "begin_time": t0,
                        "end_time": t1,
                        "gate_inst": gate_inst,
                    }
                )
            )
            next_id += 1
            continue

        if op in ("H", "S"):
            qs = _targets_as_int_qubits(entry.get("targets"))
            if not qs:
                continue
            gate_prompt = {
                "type": "1qGate",
                "id": next_id,
                "unitary": op,
                "gates": [{"q": q, "angle": 0.0} for q in qs],
                "mapping": {q: current_locs.get(q, [0, 0, 0]) for q in qs},
                "dependency": {"qubit": [next_id - 1]},
            }
            gate_inst = ft_writer.write_zair_instruction(gate_prompt)
            gate_inst["begin_time"] = t0
            gate_inst["end_time"] = t1
            instructions.append(
                ft_writer.write_instruction(
                    {
                        "type": op,
                        "id": next_id,
                        "targets": qs,
                        "dependency": {"qubit": [next_id - 1]},
                        "begin_time": t0,
                        "end_time": t1,
                        "gate_inst": gate_inst,
                    }
                )
            )
            next_id += 1
            continue

        if op in ("RUS_fail", "TMR_fail", "tmr_fail"):
            ft_op = "TMR_fail" if str(op).lower() == "tmr_fail" else "RUS_fail"
            instructions.append(
                ft_writer.write_instruction(
                    {
                        "type": ft_op,
                        "id": next_id,
                        "factories": normalize_factories(entry.get("factories")),
                        "targets": entry.get("targets"),
                        "aod_assignment": entry.get("aod_assignment"),
                        "dependency": {"qubit": [next_id - 1]},
                        "begin_time": t0,
                        "end_time": t1,
                    }
                )
            )
            next_id += 1
            continue

        if op in ("CNOT", "SE", "SE_q", "SE_stage_1", "SE_stage_2"):
            zair_prompt = {
                "type": "rydberg",
                "id": next_id,
                "zone_id": 0,
                "gates": [],
                "dependency": {"qubit": [next_id - 1]},
            }
            zair_inst = ft_writer.write_zair_instruction(zair_prompt)
            zair_inst["begin_time"] = t0
            zair_inst["end_time"] = t1
            instructions.append(
                ft_writer.write_instruction(
                    {
                        "type": op,
                        "id": next_id,
                        "targets": entry.get("targets"),
                        "dependency": {"qubit": [next_id - 1]},
                        "begin_time": t0,
                        "end_time": t1,
                        "zair_inst": zair_inst,
                    }
                )
            )
            next_id += 1
            continue

    return instructions


def execution_log_to_ft_animator_code(
    execution_log: list[dict[str, Any]],
    *,
    architecture: Architecture,
    logic_qubit_locations: list[tuple[int, int]],
    init_locs: list[list[int]] | None = None,
    magic_state_locations: list[tuple[int, int]] | None = None,
    time_scale: float = 1.0,
    name: str = "execution_log_ft",
    skip_barriers: bool = True,
    pack_timeline_for_animator: bool = True,
    min_animator_compact_instruction_duration: float | None = None,
    min_animator_logical_entangle_duration: float | None = None,
    min_animator_marker_duration: float | None = None,
    min_animator_rearrange_duration: float | None = None,
    rearrange_move_substep_weight: float = _DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT,
) -> dict[str, Any]:
    instructions = execution_log_to_ft_zair_instructions(
        execution_log,
        architecture=architecture,
        logic_qubit_locations=logic_qubit_locations,
        init_locs=init_locs,
        magic_state_locations=magic_state_locations,
        time_scale=time_scale,
        skip_barriers=skip_barriers,
    )
    if min_animator_compact_instruction_duration is None:
        min_animator_compact_instruction_duration = (
            _ANIMATOR_MUS_PER_FRM * _MIN_ANIMATOR_FRAMES_COMPACT
        )
    if min_animator_logical_entangle_duration is None:
        min_animator_logical_entangle_duration = (
            _ANIMATOR_MUS_PER_FRM * _MIN_ANIMATOR_FRAMES_FT_ENTANGLE
        )
    if min_animator_marker_duration is None:
        min_animator_marker_duration = (
            _ANIMATOR_MUS_PER_FRM * _MIN_ANIMATOR_FRAMES_FT_MARKER
        )
    if min_animator_rearrange_duration is None:
        min_animator_rearrange_duration = (
            _ANIMATOR_MUS_PER_FRM * _MIN_ANIMATOR_FRAMES_REARRANGE
        )
    if pack_timeline_for_animator:
        t_cursor = 0.0
        min_dt = 1e-6
        entangle_ops = frozenset(
            {"CNOT", "SE", "SE_q", "SE_stage_1", "SE_stage_2"}
        )
        marker_ops = frozenset({"RUS_fail", "TMR_fail"})
        for inst in instructions[1:]:
            b = float(inst.get("begin_time", 0.0))
            e = float(inst.get("end_time", b))
            raw = max(0.0, e - b)
            op = str(inst.get("type", ""))
            if op in ("move", "return_move"):
                min_len = float(min_animator_rearrange_duration)
            elif op in entangle_ops:
                min_len = float(min_animator_logical_entangle_duration)
            elif op in marker_ops:
                min_len = float(min_animator_marker_duration)
            else:
                min_len = float(min_animator_compact_instruction_duration)
            dur = max(raw, min_len, min_dt)
            inst["begin_time"] = t_cursor
            inst["end_time"] = t_cursor + dur
            if op in ("move", "return_move") and isinstance(
                inst.get("rearrange_job"), dict
            ):
                rj = inst["rearrange_job"]
                rj["begin_time"] = inst["begin_time"]
                rj["end_time"] = inst["end_time"]
                _resplit_rearrange_subtimes_weighted(
                    rj, move_weight=rearrange_move_substep_weight
                )
            t_cursor = float(inst["end_time"])
        runtime = t_cursor
    else:
        runtime = max(
            (float(inst.get("end_time", 0.0)) for inst in instructions[1:]),
            default=0.0,
        )
    return {
        "name": name,
        "architecture_spec_path": None,
        "instructions": instructions,
        "runtime": runtime,
        "n_logic_qubits": len(logic_qubit_locations),
    }


__all__ = [
    "default_init_locs_for_logic_grid",
    "default_init_locs_with_factories",
    "execution_log_to_animator_code",
    "execution_log_to_ft_animator_code",
    "execution_log_to_ft_zair_instructions",
    "execution_log_to_zair_instructions",
    "pack_timeline_for_matplotlib_animator",
]
