from __future__ import annotations

from collections import defaultdict
from typing import Any


def rearrange_rowbyrow(job, architecture, parking_dist: float = 1.0) -> list[dict]:
    begin_locs = job.begin_locs
    end_locs = job.end_locs
    details: list[dict] = []

    if not begin_locs or not end_locs:
        return details

    begin_coord: list[list[dict]] = []
    end_coord: list[list[dict]] = []
    all_col_x: list[float] = []
    row_y_begin: list[float] = []
    row_y_end: list[float] = []
    row_loc_begin: list[list[int]] = []
    row_loc_end: list[list[int]] = []

    for b_locs, e_locs in zip(begin_locs, end_locs):
        if not b_locs or not e_locs:
            continue
        b0 = architecture.exact_SLM_location(b_locs[0][1], b_locs[0][2], b_locs[0][3])
        e0 = architecture.exact_SLM_location(e_locs[0][1], e_locs[0][2], e_locs[0][3])
        row_y_begin.append(b0[1])
        row_y_end.append(e0[1])
        row_loc_begin.append([b_locs[0][1], b_locs[0][2]])
        row_loc_end.append([e_locs[0][1], e_locs[0][2]])

        b_row: list[dict] = []
        e_row: list[dict] = []
        for b_loc, e_loc in zip(b_locs, e_locs):
            b_exact = architecture.exact_SLM_location(b_loc[1], b_loc[2], b_loc[3])
            e_exact = architecture.exact_SLM_location(e_loc[1], e_loc[2], e_loc[3])
            b_row.append({"id": b_loc[0], "x": b_exact[0], "y": b_exact[1]})
            e_row.append({"id": b_loc[0], "x": e_exact[0], "y": e_exact[1]})
            all_col_x.append(b_exact[0])
        begin_coord.append(b_row)
        end_coord.append(e_row)

    all_col_x = sorted(set(all_col_x))
    col_x_to_id = {x: i for i, x in enumerate(all_col_x)}
    col_id: list[int] = []
    col_x_begin: list[float] = []
    col_x_end: list[float] = []
    col_loc_begin: list[list[int]] = []
    col_loc_end: list[list[int]] = []

    seen_cols: set[int] = set()
    for b_locs, e_locs in zip(begin_locs, end_locs):
        for b_loc, e_loc in zip(b_locs, e_locs):
            b_exact = architecture.exact_SLM_location(b_loc[1], b_loc[2], b_loc[3])
            c_id = col_x_to_id[b_exact[0]]
            if c_id in seen_cols:
                continue
            seen_cols.add(c_id)
            col_id.append(c_id)
            col_x_begin.append(b_exact[0])
            col_loc_begin.append([b_loc[1], b_loc[3]])
            e_exact = architecture.exact_SLM_location(e_loc[1], e_loc[2], e_loc[3])
            col_x_end.append(e_exact[0])
            col_loc_end.append([e_loc[1], e_loc[3]])

    activate = {
        "type": "activate",
        "row_id": list(range(len(row_y_begin))),
        "row_y": row_y_begin,
        "row_loc": row_loc_begin,
        "col_id": col_id,
        "col_x": col_x_begin,
        "col_loc": col_loc_begin,
    }
    big_move = {
        "type": "move:big",
        "move_type": "big",
        "row_id": list(range(len(row_y_begin))),
        "row_y_begin": row_y_begin,
        "row_y_end": row_y_end,
        "row_loc_begin": row_loc_begin,
        "row_loc_end": row_loc_end,
        "col_id": col_id,
        "col_x_begin": col_x_begin,
        "col_x_end": col_x_end,
        "col_loc_begin": col_loc_begin,
        "col_loc_end": col_loc_end,
        "begin_coord": begin_coord,
        "end_coord": end_coord,
    }
    deactivate = {
        "type": "deactivate",
        "row_id": list(range(len(row_y_begin))),
        "col_id": col_id,
    }

    details.append(activate)
    details.append(big_move)
    details.append(deactivate)
    for inst_counter, detail_inst in enumerate(details):
        detail_inst["id"] = inst_counter
    return details


def coalesce_same_angle(gates: list[dict]) -> list[dict]:
    if not gates:
        return []
    coalesced: list[dict] = []
    loc_dependency: dict[int, int] = defaultdict(int)
    for gate in gates:
        angle = gate["angle"]
        locs = gate["locs"]
        loc_id = locs[0][0] if isinstance(locs[0], list) else locs[0]
        dependent_idx = loc_dependency.get(loc_id, 0)
        found = False
        for idx, inst in enumerate(coalesced[dependent_idx:], start=dependent_idx):
            if inst["angle"] == angle:
                inst["locs"].append(locs[0])
                loc_dependency[loc_id] = idx
                found = True
                break
        if not found:
            coalesced.append({"angle": angle, "locs": [locs[0]]})
            loc_dependency[loc_id] = len(coalesced) - 1
    return coalesced


def flatten_aod_qubits(aod_qubits: list) -> list[int]:
    from itertools import chain
    if not aod_qubits:
        return []
    if isinstance(aod_qubits[0], list):
        return list(chain.from_iterable(aod_qubits))
    return list(aod_qubits)


def flatten_locs(locs: list) -> list:
    from itertools import chain
    if not locs:
        return []
    if locs and isinstance(locs[0], list) and locs[0] and isinstance(locs[0][0], list):
        return list(chain.from_iterable(locs))
    return list(locs)


DEFAULT_EXPANSIONS: dict[str, Any] = {
    "rearrangeJob": rearrange_rowbyrow,
    "1qGate": coalesce_same_angle,
}

