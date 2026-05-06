from __future__ import annotations

import ast
import re
from typing import Any

from src.ds import Architecture
from src.writer.zair_writer import ZAIRWriter

_PAIR_RE = re.compile(r"^\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*$")

# Keep in sync with Animator.FPS / Animator.MUS_PER_FRM in animator_matplotlib.py
_ANIMATOR_FPS = 15
_ANIMATOR_MUS_PER_FRM = 150.0 / _ANIMATOR_FPS
# Defaults tuned to match matplotlib Animator sampling (~MUS_PER_FRM per frame).
_MIN_ANIMATOR_FRAMES_COMPACT = 2  # 1qGate, rydberg: brief but still visible
_MIN_ANIMATOR_FRAMES_REARRANGE = 15  # whole rearrangeJob; move substeps get most via weighting
_DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT = 4.0


def default_init_locs_for_logic_grid(
    logic_qubit_locations: list[tuple[int, int]],
    slm_id: int = 0,
) -> list[list[int]]:
    return [[i, slm_id, y, x] for i, (x, y) in enumerate(logic_qubit_locations)]


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
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        a, b = pair[0], pair[1]
        if isinstance(a, str) and isinstance(b, str):
            end = _parse_parenthesized_pair(a)
            start = _parse_parenthesized_pair(b)
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


def _move_vec_is_string_pair(pair: Any) -> bool:
    return (
        isinstance(pair, (list, tuple))
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
    )


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


def execution_log_to_zair_instructions(
    execution_log: list[dict[str, Any]],
    *,
    architecture: Architecture,
    logic_qubit_locations: list[tuple[int, int]],
    init_locs: list[list[int]] | None = None,
    time_scale: float = 1.0,
    skip_barriers: bool = True,
) -> list[dict[str, Any]]:
    if init_locs is None:
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
            aod_qubits: list[int] = []
            begin_locs: list[list[int]] = []
            end_locs: list[list[int]] = []
            cnot_pairs = _cnot_pairs_from_targets(entry.get("targets"))
            for pair in move_vecs:
                (sx, sy), (ex, ey) = _parse_move_endpoints(pair)
                str_pair = _move_vec_is_string_pair(pair)
                qid = _mobile_qubit_for_move_vec(
                    op, (sx, sy), (ex, ey), cnot_pairs, logic_qubit_locations
                )
                if qid is None:
                    if op == "return_move":
                        # Tuple logs: atom starts at pivot/interaction site; string STAR logs:
                        # start is mobile home, end is factory — mobile index from home.
                        qid = (
                            _nearest_qubit_index(sx, sy, logic_qubit_locations)
                            if str_pair
                            else _nearest_qubit_index(ex, ey, logic_qubit_locations)
                        )
                    else:
                        qid = _nearest_qubit_index(sx, sy, logic_qubit_locations)

                # Execution logs omit trap side:
                # - default/init is left trap (SLM 0)
                # - forward movement goes to right trap (SLM 1)
                # - return movement goes back to left trap (SLM 0)
                if op == "move":
                    new_loc = [1, ey, ex]
                    default_old = [0, sy, sx]
                else:
                    if str_pair:
                        # e.g. execute_movement: [factory, qubit_home] -> parsed start=home, end=factory
                        new_loc = [0, sy, sx]
                        default_old = [1, ey, ex]
                    else:
                        # e.g. Clifford layer: ((pivot), (mobile_home))
                        new_loc = [0, ey, ex]
                        default_old = [1, sy, sx]

                old_loc = current_locs.get(qid)
                if old_loc is None:
                    old_loc = list(default_old)
                else:
                    old_loc = list(old_loc)

                aod_qubits.append(qid)
                begin_locs.append([qid, old_loc[0], old_loc[1], old_loc[2]])
                end_locs.append([qid, new_loc[0], new_loc[1], new_loc[2]])
                current_locs[qid] = new_loc
            prompt = {
                "type": "rearrangeJob",
                "id": next_id,
                "aod_id": aod_id,
                "aod_qubits": [aod_qubits],
                "begin_locs": [begin_locs],
                "end_locs": [end_locs],
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

        if op in ("H", "S", "Rz"):
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
    time_scale: float = 1.0,
    name: str = "execution_log",
    skip_barriers: bool = True,
    pack_timeline_for_animator: bool = True,
    min_animator_compact_instruction_duration: float | None = None,
    min_animator_rearrange_duration: float | None = None,
    rearrange_move_substep_weight: float = _DEFAULT_REARRANGE_MOVE_SUBSTEP_WEIGHT,
) -> dict[str, Any]:
    instructions = execution_log_to_zair_instructions(
        execution_log,
        architecture=architecture,
        logic_qubit_locations=logic_qubit_locations,
        init_locs=init_locs,
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
    }


__all__ = [
    "default_init_locs_for_logic_grid",
    "execution_log_to_animator_code",
    "execution_log_to_zair_instructions",
    "pack_timeline_for_matplotlib_animator",
]
