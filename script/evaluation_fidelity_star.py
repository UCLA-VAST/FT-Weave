"""STAR fidelity and runtime evaluation sweep (paper data generator).

Running this module writes the four CSVs consumed by the paper figure scripts:

``output/evaluation/fidelity/raw_fidelity_results.csv``
    Uncorrected physical-qubit baseline (appendix fidelity figures).
``output/evaluation/fidelity/star_fidelity_results.csv``
    End-to-end logical fidelity per STAR compile setting.
``output/evaluation/fidelity/star_full_trotter_profiling_results.csv``
    Per-round execution-time profiling (figures 8 and 9).

Reproduce with::

    uv run script/evaluation_fidelity_star.py
"""

import argparse
import csv
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from script.script_utils import reset_csv_files
from src.error_model import LogicalErrorModel, PhysicalErrorModel
from src.fidelity_simulation import (
    simluate_trotter_2d_tfim_fidelity,
    simluate_trotter_2d_tfim_fidelity_star,
)
from src.star.config import (
    get_config as get_star_config,
    update_config as update_star_config,
)
from src.star.tfim_star import generate_one_layer_2d_tfim_circuit_star
from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz

# A STAR compile setting is
# ``(placement, prepare_lookahead_angles, trivial_return, consider_skip_rus,
#    decompose_move, parallel_execution)``:
#
# placement                 magic-state microarchitecture (``src.ds.get_microarchitecture``)
# prepare_lookahead_angles  speculatively prepare the next rotation's angles
# trivial_return            return magic states along the reverse of the forward path
# consider_skip_rus         operation rematerialization: 0 = off, 2 = skip whole RUS round
# decompose_move            split a transfer into independently scheduled AOD legs
# parallel_execution        asynchronous (per-factory) instead of synchronous execution
#
# ``SETTINGS`` is the sweep run by ``__main__``; the first four entries are the
# compilation strategies compared in figures 8 and 9 (mirrored, in the same
# order, by ``SETTINGS`` in ``script/process_csv_paper.py``). The fifth is the
# high-parallelism strategy used for the runtime profile and for every
# fidelity comparison in the appendix.
SETTINGS = [
    ("seperate_region_row", False, True, 0, False, False),  # Baseline
    ("seperate_region_row", False, False, 0, True, False),  # + Routing opt.
    ("col_based", True, False, 2, False, True),  # Greedy exec. (+ microarch. opt.)
    ("col_based", False, False, 2, True, False),  # High parallelism exec.
    ("col_based", True, False, 2, True, False),  # Runtime profile / fidelity reference
]

# Setting used for every STAR curve in the appendix fidelity figures. Must stay
# a member of ``SETTINGS``; ``script/figures/compare_fidelity.py`` imports it.
MAIN_SETTINGS = [
    ("col_based", True, False, 2, True, False),
]

# Sweep axes shared by the fidelity and profiling CSVs.
QUBIT_LAYOUTS = [(4, 4), (6, 6), (8, 8), (10, 10)]  # 16, 36, 64, 100 qubits
CODE_DISTANCES = [7, 9, 13]
N_AODS = [1, 2, 3, 5]
TRIALS_PER_CONFIG = 10
# Logical-qubit syndrome-extraction cadence, in cycles. Drives the ``SE_q``
# events that the simulator folds into ``fidelity_idle``.
LOGICAL_SE_INTERVAL = 10


# ! only consider 1 trotter for now
def run_evaluation_raw(params: dict, physical_error_model: PhysicalErrorModel):
    """Run comprehensive evaluation with different parameter combinations."""

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("EVALUATION SCRIPT - RAW FIDELITY SIMULATION")
    print("=" * 80 + "\n")
    print("Current run settings:")
    print(f"  qubit_layout: {params.get('qubit_layout')}")
    print(f"  tfim: {params.get('tfim')}")
    print(f"  physical_error_model: {physical_error_model}")

    results_path = os.path.join(output_dir, "raw_fidelity_results.csv")
    raw_header_needed = (not os.path.exists(results_path)) or (
        os.path.getsize(results_path) == 0
    )
    with open(results_path, "a", newline="") as csvfile:
        fieldnames = [
            "n_qubit",
            "n_trotter_steps",
            "fidelity",
            "total_duration",
            "fidelity_cz",
            "fidelity_1q",
            "fidelity_move",
            "fidelity_idle",
            "fidelity_init",
            "fidelity_measurement",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if raw_header_needed:
            writer.writeheader()

        # simulate raw physical fidelity for different qubit layouts and trotter steps
        for qubit_layout in params["qubit_layout"]:
            n_qubits = qubit_layout[0] * qubit_layout[1]
            for J, h, dt, n_trotter in params["tfim"]:
                qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
                    n_qubits=n_qubits,
                    qubit_layout=qubit_layout,
                    J=J,
                    h=h,
                    dt=dt,
                    order=2,
                )
                # for n_trotter_steps in range(1, n_trotter):
                for n_trotter_steps in range(1, 2):
                    fidelity_profile = simluate_trotter_2d_tfim_fidelity(
                        n_qubits=n_qubits,
                        qubit_layout=qubit_layout,
                        n_trotter_steps=n_trotter_steps,
                        qc_one_layer=qc_one_layer,
                        physical_error_model=physical_error_model,
                    )
                    writer.writerow(
                        {
                            "n_qubit": n_qubits,
                            "n_trotter_steps": n_trotter_steps,
                            "fidelity": fidelity_profile["fidelity"],
                            "total_duration": fidelity_profile["total_duration"],
                            "fidelity_cz": fidelity_profile["fidelity_cz"],
                            "fidelity_1q": fidelity_profile["fidelity_1q"],
                            "fidelity_move": fidelity_profile["fidelity_move"],
                            "fidelity_idle": fidelity_profile["fidelity_idle"],
                            "fidelity_init": fidelity_profile["fidelity_init"],
                            "fidelity_measurement": fidelity_profile[
                                "fidelity_measurement"
                            ],
                        }
                    )


def run_evaluation_star(params: dict, logical_error_models, analyze_result: bool):
    """Run comprehensive evaluation with different parameter combinations.

    ``params`` may contain ``logical_se_interval`` to drive the logical-qubit
    SE scheduler. With it set, the per-layer execution log carries ``SE_q``
    events and the simulator returns a non-trivial ``fidelity_idle`` factor
    folded into ``fidelity``.
    """

    # Support evaluating one or multiple code distances in a single run.
    if not isinstance(logical_error_models, (list, tuple)):
        logical_error_models = [logical_error_models]

    print("=" * 80)
    print("EVALUATION SCRIPT - STAR FIDELITY SIMULATION")
    print("=" * 80)
    print("Current run settings:")
    print(f"  qubit_layout: {params.get('qubit_layout')}")
    print(f"  tfim: {params.get('tfim')}")
    print(f"  n_aods: {params.get('n_aods')}")
    print(f"  settings_count: {len(params.get('settings', []))}")
    print(f"  trials_per_config: {params.get('trials_per_config')}")
    print(f"  logical_se_interval: {params.get('logical_se_interval')}")
    print(
        "  code_distances:",
        [model.code_distance for model in logical_error_models],
    )
    print()

    # Create output directory
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "star_fidelity_results.csv")
    profiling_results_path = os.path.join(
        output_dir, "star_full_trotter_profiling_results.csv"
    )

    result_writer = None
    profiling_writer = None
    result_header_needed = (not os.path.exists(results_path)) or (
        os.path.getsize(results_path) == 0
    )
    profiling_header_needed = (not os.path.exists(profiling_results_path)) or (
        os.path.getsize(profiling_results_path) == 0
    )

    result_file = open(results_path, "a", newline="")
    profiling_file = (
        open(profiling_results_path, "a", newline="") if analyze_result else None
    )

    # Configure the logical-qubit SE scheduler globally for the duration of
    # this run; restore it afterwards so we don't leak settings into other
    # callers. ``get_star_config()`` returns derived read-only keys (e.g.
    # ``TMR_PREPARATION_TIME``) that ``update_star_config`` rejects, so we
    # only round-trip ``LOGICAL_SE_INTERVAL``.
    logical_se_interval = params.get("logical_se_interval")
    original_logical_se_interval = get_star_config().get("LOGICAL_SE_INTERVAL")
    update_star_config(LOGICAL_SE_INTERVAL=logical_se_interval)

    try:
        for logical_error_model in logical_error_models:
            for n_cols, n_rows in params["qubit_layout"]:
                for J, h, dt, n_trotter in params["tfim"]:
                    for n_aods in params["n_aods"]:
                        for (
                            placement,
                            prepare_lookahead_angles,
                            trivial_ret,
                            skip_rus,
                            decompose_move,
                            parallel_execution,
                        ) in params["settings"]:
                            for trial in range(params["trials_per_config"]):
                                rng = np.random.default_rng(42 + trial)
                                config = {
                                    "n_aods": n_aods,
                                    "consider_skip_rus": skip_rus,
                                    "trivial_return": trivial_ret,
                                    "decompose_move": decompose_move,
                                    "tmr_assignment_method": "matching",
                                    "prepare_lookahead_angles": prepare_lookahead_angles,
                                    "rng": rng,
                                    "save_log": False,
                                }
                                print(
                                    "Compile setting:",
                                    {**config, "placement": placement},
                                )
                                (
                                    qc_one_layer,
                                    full_logs,
                                    profiling_results_per_case,
                                ) = generate_one_layer_2d_tfim_circuit_star(
                                    n_qubits=n_cols * n_rows,
                                    qubit_layout=(n_rows, n_cols),
                                    placement=placement,
                                    J=J,
                                    h=h,
                                    dt=dt,
                                    code_distance=logical_error_model.code_distance,
                                    config=config,
                                    parallel_execution=parallel_execution,
                                    analyze_result=analyze_result,
                                    # result_path=log_dir + f"/trial_{trial}.pickle",
                                )
                                # print circuit execution diagram
                                # from src.animator import plot_circuit_execution

                                # for i, log in enumerate(full_logs):
                                #     plot_circuit_execution(
                                #         log,
                                #         n_cols * n_rows,
                                #         save_path=f"output/evaluation/circuit_execution/star_circuit_execution_{placement}_{trial}_{i}.pdf",
                                #     )
                                # assert False
                                if analyze_result:
                                    if (
                                        profiling_writer is None
                                        and profiling_results_per_case
                                    ):
                                        if profiling_file is None:
                                            raise RuntimeError(
                                                "profiling_file is None while analyze_result is True"
                                            )
                                        profiling_writer = csv.DictWriter(
                                            profiling_file,
                                            fieldnames=profiling_results_per_case[
                                                0
                                            ].keys(),
                                        )
                                        if profiling_header_needed:
                                            profiling_writer.writeheader()
                                    if profiling_writer is not None:
                                        for row in profiling_results_per_case:
                                            profiling_writer.writerow(row)

                                result = simluate_trotter_2d_tfim_fidelity_star(
                                    n_qubits=n_cols * n_rows,
                                    n_factories=n_cols * n_rows,
                                    qubit_layout=(n_rows, n_cols),
                                    n_trotter_steps=1,  # for simplicity, we only evaluate 1 trotter step for star compilation
                                    execution_logs=full_logs,
                                    logical_error_model=logical_error_model,
                                )
                                total_depth = None
                                if analyze_result and profiling_results_per_case:
                                    total_depth = float(
                                        sum(
                                            float(r.get("total_time", 0.0))
                                            for r in profiling_results_per_case
                                        )
                                    )
                                # ``result["fidelity"]`` already folds in
                                # ``fidelity_idle`` (computed from the
                                # SE_q events emitted by the logical-SE
                                # scheduler). All counts and component
                                # fidelities live in ``result``; we just
                                # add per-trial bookkeeping here.
                                row = {
                                    "trial": trial,
                                    "code_distance": logical_error_model.code_distance,
                                    "qubit_layout": (n_rows, n_cols),
                                    "n_trotter_steps": 1,
                                    "placement": placement,
                                    "n_aods": n_aods,
                                    "consider_skip_rus": skip_rus,
                                    "trivial_return": trivial_ret,
                                    "decompose_move": decompose_move,
                                    "parallel_execution": parallel_execution,
                                    "prepare_lookahead_angles": prepare_lookahead_angles,
                                    "logical_se_interval": logical_se_interval,
                                    "total_depth": total_depth,
                                    **result,
                                }
                                if result_writer is None:
                                    result_writer = csv.DictWriter(
                                        result_file,
                                        fieldnames=row.keys(),
                                    )
                                    if result_header_needed:
                                        result_writer.writeheader()
                                result_writer.writerow(row)
    finally:
        update_star_config(LOGICAL_SE_INTERVAL=original_logical_se_interval)
        result_file.close()
        if profiling_file is not None:
            profiling_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "STAR fidelity + runtime evaluation sweep. Writes the CSVs consumed "
            "by script/figures/compare_fidelity.py and script/process_csv_paper.py."
        )
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append to existing CSVs instead of rewriting them. Off by default so "
            "a repeated run reproduces the same data rather than duplicating rows."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=TRIALS_PER_CONFIG,
        help=f"Trials per configuration (default: {TRIALS_PER_CONFIG})",
    )
    args = parser.parse_args()

    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)
    if not args.append:
        reset_csv_files(
            os.path.join(output_dir, "raw_fidelity_results.csv"),
            os.path.join(output_dir, "star_fidelity_results.csv"),
            os.path.join(output_dir, "star_full_trotter_profiling_results.csv"),
        )

    physical_error_model = PhysicalErrorModel("lookahead")
    p_ph = physical_error_model.get_error_rate("p_ph")

    # One second-order Trotter layer of the 2D transverse-field Ising model.
    # The step size dt is chosen so the Trotter error per step is comparable to
    # the physical error rate: dt = l1 * alpha * p_ph / omega.
    l1, alpha, omega = 1, 2, 1
    tfim = []
    for j, h in [(1.0, 1.0)]:
        dt = l1 * alpha * p_ph / omega
        tfim.append((j, h, dt, int((10 / j) / dt)))

    # 1. Uncorrected physical baseline (appendix fidelity figures).
    run_evaluation_raw(
        params={"qubit_layout": QUBIT_LAYOUTS, "tfim": tfim},
        physical_error_model=physical_error_model,
    )

    # 2. STAR sweep over compile strategies x AOD count x code distance.
    logical_error_models = [
        LogicalErrorModel(physical_model=physical_error_model, code_distance=d)
        for d in CODE_DISTANCES
    ]
    run_evaluation_star(
        params={
            "qubit_layout": QUBIT_LAYOUTS,
            "tfim": tfim,
            "n_aods": N_AODS,
            "settings": SETTINGS,
            "trials_per_config": args.trials,
            "logical_se_interval": LOGICAL_SE_INTERVAL,
        },
        logical_error_models=logical_error_models,
        analyze_result=True,
    )
