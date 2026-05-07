from __future__ import annotations

from copy import deepcopy

from src.ds import Architecture


def build_compact_architecture(
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    n_aods: int,
) -> Architecture:
    """Build a compact two-trap-per-site architecture from used sites."""
    used_points = logic_qubit_locations + magic_state_locations
    max_x = max(x for x, _ in used_points)
    max_y = max(y for _, y in used_points)

    site_dx = 12
    site_dy = 10
    right_slm_offset_x = 2

    n_rows = max_y + 1
    n_cols = max_x + 1

    slm0_x_min = 0
    slm0_x_max = (n_cols - 1) * site_dx
    slm1_x_min = right_slm_offset_x
    slm1_x_max = right_slm_offset_x + (n_cols - 1) * site_dx
    y_min = 0
    y_max = (n_rows - 1) * site_dy

    padding_x = site_dx
    padding_y = site_dy
    arch_min_x = min(slm0_x_min, slm1_x_min) - padding_x
    arch_max_x = max(slm0_x_max, slm1_x_max) + padding_x
    arch_min_y = y_min - padding_y
    arch_max_y = y_max + padding_y

    architecture_spec = {
        "name": "compact_microarchitecture",
        "operation_duration": {
            "rydberg": 0.36,
            "1qGate": 52,
            "atom_transfer": 15,
        },
        "storage_zones": [],
        "entanglement_zones": [
            {
                "zone_id": 0,
                "slms": [
                    {
                        "id": 0,
                        "site_seperation": [site_dx, site_dy],
                        "r": n_rows,
                        "c": n_cols,
                        "location": [0, 0],
                    },
                    {
                        "id": 1,
                        "site_seperation": [site_dx, site_dy],
                        "r": n_rows,
                        "c": n_cols,
                        "location": [right_slm_offset_x, 0],
                    },
                ],
                "offset": [0, 0],
                "dimension": [arch_max_x - arch_min_x, arch_max_y - arch_min_y],
            }
        ],
        "aods": [
            {"id": i, "site_seperation": 2, "r": n_rows, "c": n_cols}
            for i in range(n_aods)
        ],
        "arch_range": [[arch_min_x, arch_min_y], [arch_max_x, arch_max_y]],
        "rydberg_range": [[[arch_min_x, arch_min_y], [arch_max_x, arch_max_y]]],
    }
    return Architecture(architecture_spec)


def merge_layer_logs_with_global_timeline(
    layer_logs: list[list[dict]],
    inter_layer_gap: float = 0.0,
) -> list[dict]:
    """Concatenate per-layer logs into one monotonically increasing timeline."""
    merged: list[dict] = []
    current_time = 0.0
    for layer in layer_logs:
        if not layer:
            continue
        layer_start = min(float(entry.get("start_time", 0.0)) for entry in layer)
        layer_end = max(
            float(entry.get("end_time", entry.get("start_time", 0.0)))
            for entry in layer
        )
        shift = current_time - layer_start
        for entry in layer:
            new_entry = deepcopy(entry)
            start = float(new_entry.get("start_time", 0.0)) + shift
            end = float(new_entry.get("end_time", start)) + shift
            new_entry["start_time"] = start
            new_entry["end_time"] = end
            merged.append(new_entry)
        current_time += (layer_end - layer_start) + inter_layer_gap
    return merged

