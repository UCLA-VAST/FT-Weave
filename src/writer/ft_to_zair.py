from __future__ import annotations

from typing import Any

from src.ds import Architecture, LogicalGridManager
from src.execution_log.event_helpers import normalize_factories
from src.writer.zair_writer import ZAIRWriter


def _flatten_locs(locs: list) -> list[list[int]]:
    if not locs:
        return []
    if isinstance(locs[0], (list, tuple)) and locs[0] and isinstance(
        locs[0][0], (list, tuple)
    ):
        return [list(loc) for row in locs for loc in row]
    return [list(loc) for loc in locs]


def _copy_time(src: dict[str, Any], dst: dict[str, Any]) -> None:
    dst["begin_time"] = float(src.get("begin_time", 0.0))
    dst["end_time"] = float(src.get("end_time", dst["begin_time"]))


def _set_rearrange_subtimes(inst: dict[str, Any]) -> None:
    subs = inst.get("insts")
    if not subs:
        return
    t0 = float(inst.get("begin_time", 0.0))
    t1 = float(inst.get("end_time", t0))
    dt = (t1 - t0) / max(1, len(subs))
    for idx, sub in enumerate(subs):
        sub["begin_time"] = t0 + idx * dt
        sub["end_time"] = t0 + (idx + 1) * dt


def _patch_sites(
    grid: LogicalGridManager,
    slm: int,
    logical_r: int,
    logical_c: int,
    *,
    include_check_qubits: bool,
) -> list[tuple[str, tuple[int, int, int]]]:
    data = [
        ("data", site)
        for site in grid.physical_data_sites_from_logical_index(
            slm, logical_r, logical_c
        )
    ]
    if not include_check_qubits:
        return data
    x_checks = [
        ("x_check", site)
        for site in grid.physical_x_check_sites_from_logical_index(
            slm, logical_r, logical_c
        )
    ]
    z_checks = [
        ("z_check", site)
        for site in grid.physical_z_check_sites_from_logical_index(
            slm, logical_r, logical_c
        )
    ]
    return data + x_checks + z_checks


def build_rotated_surface_code_layout(
    init_locs: list[list[int]],
    *,
    architecture: Architecture,
    code_distance: int,
    include_check_qubits: bool = True,
) -> dict[str, Any]:
    """Expand FT logical init sites to rotated surface-code physical sites.

    Each logical qubit expands to ``d^2`` data qubits plus ``d^2 - 1`` check
    qubits, for ``2*d^2 - 1`` physical qubits in total. The layout follows
    :class:`LogicalGridManager`: data sites first, then X checks, then Z checks.
    """
    if code_distance <= 0:
        raise ValueError("code_distance must be positive")

    grid = LogicalGridManager(architecture, code_distance)
    patch_size = (
        grid.num_total_qubit if include_check_qubits else grid.num_data_qubit
    )
    logical_to_physical: dict[int, dict[str, Any]] = {}
    physical_qubits: list[dict[str, Any]] = []
    physical_init_locs: list[list[int]] = []

    for loc in init_locs:
        logical_id, slm, logical_r, logical_c = map(int, loc[:4])
        sites = _patch_sites(
            grid,
            slm,
            logical_r,
            logical_c,
            include_check_qubits=include_check_qubits,
        )
        physical_ids: list[int] = []
        data_ids: list[int] = []
        x_check_ids: list[int] = []
        z_check_ids: list[int] = []
        init_for_logical: list[list[int]] = []
        for offset, (role, (site_slm, site_r, site_c)) in enumerate(sites):
            pid = logical_id * patch_size + offset
            phys_loc = [pid, site_slm, site_r, site_c]
            physical_ids.append(pid)
            init_for_logical.append(phys_loc)
            physical_init_locs.append(phys_loc)
            if role == "data":
                data_ids.append(pid)
            elif role == "x_check":
                x_check_ids.append(pid)
            elif role == "z_check":
                z_check_ids.append(pid)
            physical_qubits.append(
                {
                    "id": pid,
                    "logical_id": logical_id,
                    "patch_offset": offset,
                    "role": role,
                    "loc": [site_slm, site_r, site_c],
                }
            )
        logical_to_physical[logical_id] = {
            "physical_ids": physical_ids,
            "data_ids": data_ids,
            "x_check_ids": x_check_ids,
            "z_check_ids": z_check_ids,
            "init_locs": init_for_logical,
            "logical_loc": [slm, logical_r, logical_c],
        }

    return {
        "code_distance": code_distance,
        "include_check_qubits": include_check_qubits,
        "patch_size": patch_size,
        "grid": grid,
        "logical_to_physical": logical_to_physical,
        "physical_qubits": physical_qubits,
        "init_locs": physical_init_locs,
    }


def _physical_locs_for_logical_loc(
    layout: dict[str, Any],
    logical_loc: list[int],
) -> list[list[int]]:
    logical_id, slm, logical_r, logical_c = map(int, logical_loc[:4])
    patch = layout["logical_to_physical"][logical_id]
    sites = _patch_sites(
        layout["grid"],
        slm,
        logical_r,
        logical_c,
        include_check_qubits=bool(layout["include_check_qubits"]),
    )
    return [
        [pid, site_slm, site_r, site_c]
        for pid, (_, (site_slm, site_r, site_c)) in zip(
            patch["physical_ids"], sites
        )
    ]


def _group_locs_by_row(
    begin_locs: list[list[int]],
    end_locs: list[list[int]],
) -> tuple[list[list[int]], list[list[list[int]]], list[list[list[int]]]]:
    row_order: list[int] = []
    row_groups: dict[int, dict[str, list]] = {}
    for b_loc, e_loc in zip(begin_locs, end_locs):
        row = int(b_loc[2])
        if row not in row_groups:
            row_order.append(row)
            row_groups[row] = {"aod_qubits": [], "begin_locs": [], "end_locs": []}
        row_groups[row]["aod_qubits"].append(int(b_loc[0]))
        row_groups[row]["begin_locs"].append(b_loc)
        row_groups[row]["end_locs"].append(e_loc)
    return (
        [row_groups[row]["aod_qubits"] for row in row_order],
        [row_groups[row]["begin_locs"] for row in row_order],
        [row_groups[row]["end_locs"] for row in row_order],
    )


def _expand_rearrange(
    inst: dict[str, Any],
    *,
    layout: dict[str, Any],
    writer: ZAIRWriter,
    current_locs: dict[int, list[int]],
) -> dict[str, Any] | None:
    rj = inst.get("rearrange_job") if isinstance(inst.get("rearrange_job"), dict) else inst
    logical_begin = _flatten_locs(rj.get("begin_locs", []))
    logical_end = _flatten_locs(rj.get("end_locs", []))
    if not logical_begin or not logical_end:
        return None

    physical_begin: list[list[int]] = []
    physical_end: list[list[int]] = []
    for b_logical, e_logical in zip(logical_begin, logical_end):
        b_locs = _physical_locs_for_logical_loc(layout, b_logical)
        e_locs = _physical_locs_for_logical_loc(layout, e_logical)
        for b_loc, e_loc in zip(b_locs, e_locs):
            pid = int(b_loc[0])
            cur_loc = current_locs.get(pid)
            physical_begin.append([pid, *cur_loc] if cur_loc is not None else b_loc)
            physical_end.append(e_loc)
            current_locs[pid] = [int(e_loc[1]), int(e_loc[2]), int(e_loc[3])]

    aod_qubits, begin_locs, end_locs = _group_locs_by_row(
        physical_begin, physical_end
    )
    prompt = {
        "type": "rearrangeJob",
        "id": inst.get("id", rj.get("id", 0)),
        "aod_id": inst.get("aod_id", rj.get("aod_id", 0)),
        "aod_qubits": aod_qubits,
        "begin_locs": begin_locs,
        "end_locs": end_locs,
        "dependency": dict(inst.get("dependency", rj.get("dependency", {}))),
    }
    out = writer.write_instruction(prompt)
    _copy_time(inst, out)
    _set_rearrange_subtimes(out)
    return out


def _logical_ids_from_gate_inst(inst: dict[str, Any]) -> list[tuple[int, float]]:
    gate_inst = inst.get("gate_inst")
    if not isinstance(gate_inst, dict):
        return [(int(q), 0.0) for q in inst.get("targets", [])]
    targets: list[tuple[int, float]] = []
    if isinstance(gate_inst.get("inst"), list):
        for block in gate_inst["inst"]:
            angle = float(block.get("angle", 0.0))
            for loc in block.get("locs", []):
                if loc:
                    targets.append((int(loc[0]), angle))
    elif isinstance(gate_inst.get("gates"), list):
        for gate in gate_inst["gates"]:
            targets.append((int(gate["q"]), float(gate.get("angle", 0.0))))
    return targets


def _expand_1q_gate(
    inst: dict[str, Any],
    *,
    layout: dict[str, Any],
    writer: ZAIRWriter,
    current_locs: dict[int, list[int]],
) -> dict[str, Any] | None:
    gates: list[dict[str, Any]] = []
    mapping: dict[int, list[int]] = {}
    for logical_id, angle in _logical_ids_from_gate_inst(inst):
        patch = layout["logical_to_physical"].get(logical_id)
        if not patch:
            continue
        for pid in patch["data_ids"]:
            gates.append({"q": pid, "angle": angle})
            mapping[pid] = current_locs.get(pid, patch["init_locs"][pid % layout["patch_size"]][1:])
    if not gates:
        return None
    prompt = {
        "type": "1qGate",
        "id": inst.get("id", 0),
        "unitary": inst.get("type", inst.get("unitary", "")),
        "gates": gates,
        "mapping": mapping,
        "dependency": dict(inst.get("dependency", {})),
    }
    out = writer.write_instruction(prompt)
    _copy_time(inst, out)
    return out


def _target_pairs(targets: Any) -> list[tuple[int, int]]:
    if not isinstance(targets, (list, tuple)):
        return []
    pairs: list[tuple[int, int]] = []
    for t in targets:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            pairs.append((int(t[0]), int(t[1])))
    return pairs


def _expand_rydberg(
    inst: dict[str, Any],
    *,
    layout: dict[str, Any],
    writer: ZAIRWriter,
    n_logic_qubits: int | None,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    pairs = _target_pairs(inst.get("targets"))
    if pairs:
        for a, b in pairs:
            pa = layout["logical_to_physical"].get(a)
            pb = layout["logical_to_physical"].get(b)
            if not pa or not pb:
                continue
            for qa, qb in zip(pa["data_ids"], pb["data_ids"]):
                gates.append({"q0": qa, "q1": qb})
    else:
        factories = normalize_factories(inst.get("factories"))
        targets = inst.get("targets")
        target_ids = (
            [int(t) for t in targets]
            if isinstance(targets, list)
            else ([] if targets is None else [int(targets)])
        )
        if factories and target_ids and n_logic_qubits is not None:
            for fid, target in zip(factories, target_ids):
                pf = layout["logical_to_physical"].get(n_logic_qubits + int(fid))
                pt = layout["logical_to_physical"].get(int(target))
                if not pf or not pt:
                    continue
                for qf, qt in zip(pf["data_ids"], pt["data_ids"]):
                    gates.append({"q0": qf, "q1": qt})
        else:
            for logical_id in target_ids:
                patch = layout["logical_to_physical"].get(int(logical_id))
                if patch:
                    for pid in patch["data_ids"]:
                        gates.append({"q": pid})
    prompt = {
        "type": "rydberg",
        "id": inst.get("id", 0),
        "zone_id": int(inst.get("zair_inst", {}).get("zone_id", 0))
        if isinstance(inst.get("zair_inst"), dict)
        else 0,
        "gates": gates,
        "dependency": dict(inst.get("dependency", {})),
    }
    out = writer.write_instruction(prompt)
    _copy_time(inst, out)
    out.setdefault("ft_source_type", inst.get("type"))
    return out


def ft_zair_to_zair_instructions(
    ft_instructions: list[dict[str, Any]],
    *,
    architecture: Architecture | None = None,
    code_distance: int | None = None,
    include_check_qubits: bool = True,
    n_logic_qubits: int | None = None,
) -> list[dict[str, Any]]:
    """Interface for FT-ZAIR -> ZAIR synthesis.

    If ``architecture`` and ``code_distance`` are provided, synthesize logical
    FT-ZAIR instructions into physical ZAIR over rotated surface-code patches.
    Otherwise, preserve the original pass-through bridge for embedded ZAIR
    payloads (``zair_inst``, ``rearrange_job``, ``gate_inst``).
    """
    if architecture is None or code_distance is None:
        return _passthrough_ft_payloads(ft_instructions)

    if not ft_instructions or ft_instructions[0].get("type") != "init":
        raise ValueError("FT-ZAIR synthesis requires an initial init instruction")

    writer = ZAIRWriter(architecture=architecture)
    init_inst = ft_instructions[0]
    layout = build_rotated_surface_code_layout(
        [list(loc) for loc in init_inst.get("init_locs", [])],
        architecture=architecture,
        code_distance=code_distance,
        include_check_qubits=include_check_qubits,
    )
    current_locs = {
        int(loc[0]): [int(loc[1]), int(loc[2]), int(loc[3])]
        for loc in layout["init_locs"]
    }

    init_out = {
        "type": "init",
        "id": init_inst["id"],
        "init_locs": layout["init_locs"],
        "dependency": dict(init_inst.get("dependency", {})),
        "begin_time": float(init_inst.get("begin_time", 0.0)),
        "end_time": float(init_inst.get("end_time", 0.0)),
        "ft_zair": {
            "code_distance": code_distance,
            "patch_size": layout["patch_size"],
            "include_check_qubits": include_check_qubits,
            "logical_to_physical": {
                logical_id: {
                    key: value
                    for key, value in patch.items()
                    if key != "init_locs"
                }
                for logical_id, patch in layout["logical_to_physical"].items()
            },
            "physical_qubits": layout["physical_qubits"],
        },
    }
    out: list[dict[str, Any]] = []
    out.append(init_out)
    for inst in ft_instructions[1:]:
        itype = str(inst.get("type", ""))
        if itype in ("move", "return_move"):
            expanded = _expand_rearrange(
                inst, layout=layout, writer=writer, current_locs=current_locs
            )
            if expanded is not None:
                out.append(expanded)
            continue
        if itype in ("H", "S", "Rz"):
            expanded = _expand_1q_gate(
                inst, layout=layout, writer=writer, current_locs=current_locs
            )
            if expanded is not None:
                out.append(expanded)
            continue
        if itype in ("CNOT", "SE", "SE_q", "SE_stage_1", "SE_stage_2"):
            out.append(
                _expand_rydberg(
                    inst,
                    layout=layout,
                    writer=writer,
                    n_logic_qubits=n_logic_qubits,
                )
            )
            continue
        if itype in ("RUS_fail", "TMR_fail"):
            continue
    return out


def _passthrough_ft_payloads(ft_instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for inst in ft_instructions:
        itype = str(inst.get("type", ""))
        if itype == "init":
            out.append(
                {
                    "type": "init",
                    "id": inst["id"],
                    "init_locs": list(inst.get("init_locs", [])),
                    "dependency": dict(inst.get("dependency", {})),
                    "begin_time": float(inst.get("begin_time", 0.0)),
                    "end_time": float(inst.get("end_time", 0.0)),
                }
            )
            continue
        for key in ("zair_inst", "rearrange_job", "gate_inst"):
            payload = inst.get(key)
            if isinstance(payload, dict):
                out.append(dict(payload))
                break
    return out

