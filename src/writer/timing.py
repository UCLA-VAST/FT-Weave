from __future__ import annotations


def get_begin_time(insts: list[dict], cur_inst_idx: int, dependency: dict) -> float:
    begin_time = 0.0
    for dep_type, dep_value in dependency.items():
        if isinstance(dep_value, int):
            begin_time = max(begin_time, insts[dep_value]["end_time"])
        elif isinstance(dep_value, list):
            if dep_type == "site":
                for inst_idx in dep_value:
                    begin_time = max(begin_time, insts[inst_idx]["end_time"])
            else:
                for inst_idx in dep_value:
                    if inst_idx == cur_inst_idx:
                        continue
                    begin_time = max(begin_time, insts[inst_idx]["end_time"])
    return begin_time


def get_duration(architecture, inst: dict) -> float:
    detail_insts = inst.get("insts", [])
    duration = 0.0
    for detail in detail_insts:
        inst_type = detail["type"].split(":")[0]
        detail["begin_time"] = duration
        if inst_type in ("activate", "deactivate"):
            duration += architecture.time_atom_transfer
            detail["end_time"] = duration
        elif inst_type == "move":
            move_duration = 0.0
            row_begins = detail.get("row_y_begin", [])
            row_ends = detail.get("row_y_end", [])
            col_begins = detail.get("col_x_begin", [])
            col_ends = detail.get("col_x_end", [])
            for row_begin, row_end in zip(row_begins, row_ends):
                for col_begin, col_end in zip(col_begins, col_ends):
                    move_duration = max(
                        move_duration,
                        architecture.movement_duration(col_begin, row_begin, col_end, row_end),
                    )
            detail["end_time"] = duration + move_duration
            duration += move_duration
        else:
            raise ValueError(f"Unknown sub-instruction type: {inst_type}")
    return duration


def apply_timing(insts: list[dict], architecture) -> None:
    for idx, inst in enumerate(insts):
        dependency = inst.get("dependency", {})
        begin_time = get_begin_time(insts, idx, dependency)
        inst["begin_time"] = begin_time
        inst_type = inst.get("type", "")
        if inst_type == "rearrangeJob" and "insts" in inst:
            inst["end_time"] = begin_time + get_duration(architecture, inst)
        elif inst_type == "rydberg":
            inst["end_time"] = begin_time + getattr(architecture, "time_rydberg", 0.0)
        elif inst_type == "1qGate":
            inst["end_time"] = begin_time + getattr(architecture, "time_1qGate", 0.0)
        elif inst_type == "init":
            inst["end_time"] = begin_time
        elif "end_time" not in inst:
            inst["end_time"] = begin_time


def compute_total_duration(insts: list[dict]) -> float:
    return max((inst.get("end_time", 0.0) for inst in insts), default=0.0)

