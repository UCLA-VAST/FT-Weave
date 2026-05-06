import os
import sys
from copy import deepcopy
from typing import Any

import numpy as np

# Ensure repository root is on sys.path so `src` and `script` are importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator.animator_matplotlib import Animator
from src.ds import Architecture, get_microarchitecture
from src.star.tfim_star import generate_one_layer_2d_tfim_circuit_star
from src.writer.execution_log_to_zair import execution_log_to_animator_code


def build_compact_architecture(
    logic_qubit_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
    n_aods: int,
) -> Architecture:
    """Build a compact two-trap-per-site architecture from used microarchitecture sites.

    Trap model:
    - Left trap  : SLM 0
    - Right trap : SLM 1 (x offset +2 um from left at the same site)
    - Site pitch : dx=12 um, dy=10 um
    """
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


def main() -> None:
    # Fixed parameters for quick local testing (edit these directly as needed).
    n_cols = 2
    n_rows = 2
    placement = "col_based"
    code_distance = 3
    n_aods = 1
    parallel_execution = False
    seed = 0
    J = 1.0
    h = 0.7
    dt = 0.1
    inter_layer_gap = 0.0
    output_path = "output/animation/tfim_log_adapter_test.mp4"
    ffmpeg_path = "ffmpeg"
    figure_scaling = 16
    figure_font = 12
    # Encode FPS: higher than 15 shortens wall-clock playback (same frame count).
    animator_fps = 30.0
    # Larger => fewer frames for the same simulated runtime. None = class default (~10).
    animator_mus_per_frm: float | None = 12.0

    n_qubits = n_cols * n_rows
    qubit_layout = (n_cols, n_rows)

    config = {
        "n_aods": n_aods,
        "consider_skip_rus": 0,
        "tmr_assignment_method": "matching",
        "trivial_return": False,
        "decompose_move": False,
        "rng": np.random.default_rng(seed),
    }

    _, raw_layer_logs, _ = generate_one_layer_2d_tfim_circuit_star(
        n_qubits=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
        J=J,
        h=h,
        dt=dt,
        code_distance=code_distance,
        config=config,
        parallel_execution=parallel_execution,
        analyze_result=False,
    )

    layer_logs: list[list[dict[str, Any]]] = []
    for layer in raw_layer_logs:
        if isinstance(layer, list) and all(isinstance(entry, dict) for entry in layer):
            layer_logs.append([dict(entry) for entry in layer])
    merged_log = merge_layer_logs_with_global_timeline(
        layer_logs=layer_logs,
        inter_layer_gap=inter_layer_gap,
    )

    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        n_qubits=n_qubits,
        n_factories=n_qubits,
        qubit_layout=qubit_layout,
        placement=placement,
    )
    architecture = build_compact_architecture(
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
        n_aods=n_aods,
    )

    code = execution_log_to_animator_code(
        merged_log,
        architecture=architecture,
        logic_qubit_locations=logic_qubit_locations,
        name="tfim-log-adapter-test",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Animator().animate(
        code=code,
        architecture=architecture,
        output=output_path,
        ffmpeg=ffmpeg_path,
        scaling_factor=figure_scaling,
        font=figure_font,
        mus_per_frm=animator_mus_per_frm,
        fps=animator_fps,
    )
    print(f"Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
