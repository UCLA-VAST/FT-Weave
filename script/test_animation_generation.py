import os
import sys
from typing import Any

import numpy as np

# Ensure repository root is on sys.path so `src` and `script` are importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator import FTAnimator
from src.animator.animator_matplotlib import Animator
from src.animator.pipeline_utils import (
    build_compact_architecture,
    merge_layer_logs_with_global_timeline,
)
from src.ds import get_microarchitecture
from src.star.tfim_star import generate_one_layer_2d_tfim_circuit_star
from src.writer.execution_log_to_zair import (
    execution_log_to_animator_code,
    execution_log_to_ft_animator_code,
)


def main() -> None:
    # Fixed parameters for quick local testing (edit these directly as needed).
    n_cols = 4
    n_rows = 4
    placement = "col_based"
    code_distance = 3
    n_aods = 1
    parallel_execution = False
    seed = 0
    J = 1.0
    h = 1
    dt = 0.1

    l1 = 1
    alpha = 2
    omega = 1
    p_ph = 7.3e-4
    dt = l1 * alpha * p_ph / omega

    inter_layer_gap = 0.0
    output_path = "output/animation/tfim_log_adapter_test.mp4"
    ffmpeg_path = "ffmpeg"
    figure_scaling = 16
    figure_font = 12
    # Encode FPS: higher than 15 shortens wall-clock playback (same frame count).
    animator_fps = 20.0
    # Larger => fewer frames for the same simulated runtime. None = class default (~10).
    animator_mus_per_frm: float | None = 12.0
    use_ft_zair = True

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
    for layer in raw_layer_logs[4:]:
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

    if use_ft_zair:
        code = execution_log_to_ft_animator_code(
            merged_log,
            architecture=architecture,
            logic_qubit_locations=logic_qubit_locations,
            magic_state_locations=magic_state_locations,
            name="tfim-log-adapter-test-ft",
        )
        animator = FTAnimator()
    else:
        code = execution_log_to_animator_code(
            merged_log,
            architecture=architecture,
            logic_qubit_locations=logic_qubit_locations,
            magic_state_locations=magic_state_locations,
            name="tfim-log-adapter-test",
        )
        animator = Animator()

    print("merged_log")
    for log in merged_log:
        print(log)
    print("code")
    for entry in code["instructions"]:
        print(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    animator.animate(
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
