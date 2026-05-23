import csv
import os
import sys
from dataclasses import dataclass

import numpy as np

# Make sure the repo root is on ``sys.path`` *before* importing
# ``script.script_utils``. When this module is launched directly via
# ``uv run script/evaluation_fidelity_t_cultivation.py``, Python sets
# ``sys.path[0]`` to ``script/`` rather than the repo root, so the dotted
# ``script.<...>`` import would otherwise fail with ``ModuleNotFoundError``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from script.script_utils import add_repo_root_to_syspath, ensure_csv_writer

add_repo_root_to_syspath(__file__)

from src.ds import get_microarchitecture
from src.error_model import LogicalErrorModel, PhysicalErrorModel
from src.fidelity_simulation import simluate_trotter_2d_tfim_fidelity_t_cultivation
from src.t_cultivation.config import get_config, update_config
from src.t_cultivation.tfim_t import generate_one_layer_2d_tfim_circuit_t_cultivation

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("src").setLevel(logging.WARNING)

# T-cultivation compile grid: ``(trivial_return, decompose_move,
# redistribute_stage1_success)``. When ``redistribute_stage1_success`` is False, stage-1
# outcomes are not patched across factories (see ``complete_stage1_preparation``).
# RUS skip, TMR assignment, and parallel angle execution are STAR-only.
COMPILE_SETTINGS = [
    (True, False, False),
    (False, True, False),
    (False, True, True),
]

_DEFAULT_COMPILE = (False, True, True)
# Former main default: ``decompose_move=True`` with stage-1 redistribution on.
MAIN_COMPILE_SETTING = (False, True, True)


@dataclass(frozen=True)
class TSetting:
    """Evaluation setting aligned with throughput evaluation."""

    fidelity_target: float
    factory_physical_size: int
    distance: int


def apply_setting(setting: TSetting, logical_se_interval: int | None = None) -> None:
    """Apply one T-cultivation setting to global config.

    ``logical_se_interval`` enables the logical-qubit syndrome-extraction
    scheduler so the per-layer execution log contains ``SE_q`` events; the
    fidelity simulator then folds them into ``fidelity_idle``. When ``None``
    the scheduler stays off and ``fidelity_idle`` is 1.0.
    """
    update_config(
        STAGE_2_FIDELITY_TARGET=setting.fidelity_target,
        FACTORY_PHYSICAL_SIZE=setting.factory_physical_size,
        STAGE_1_RESOURCE_UNITS=1,
        LOGICAL_SE_INTERVAL=logical_se_interval,
    )


def run_evaluation_t_cultivation(
    params: dict,
    physical_error_model: PhysicalErrorModel,
    analyze_result: bool = True,
    verbose: bool = True,
):
    """
    Evaluate T-cultivation execution and write CSV outputs.

    Structure mirrors ``run_evaluation_star`` in ``evaluation_fidelity.py``: one TFIM
    layer is produced; each ``Rz`` instruction is scheduled in its own round (same as
    STAR’s per-Rz logs). Full per-layer logs are passed to
    :func:`simluate_trotter_2d_tfim_fidelity_t_cultivation`
    (same nested structure as STAR).

    ``params`` may include ``compile_settings``: a list of
    ``(trivial_return, decompose_move, redistribute_stage1_success)`` tuples.
    STAR-only options (RUS skip, TMR assignment, parallel execution) are not used.
    When omitted, ``(False, False, True)`` is used (trivial/decompose off; redistribution on).

    Notes:
    - Logical T fidelity comes from ``setting.fidelity_target`` via
      ``LogicalErrorModel.integrate_t_cultivation_fidelity_target`` (default: target
      is a logical error rate ``p``, T fidelity ``= 1 - p``).
    - Circuit execution time is recorded for each trotter step in a dedicated CSV.
    - With ``verbose=True``, prints each configuration before it is simulated so you
      can see which case is running (setting, layout, TFIM parameters, placement,
      ``n_aods``, trial).
    """
    output_dir = "output/evaluation/fidelity"
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("EVALUATION SCRIPT - T-CULTIVATION FIDELITY SIMULATION")
        print("=" * 80)
        print("Current run settings:")
        print(f"  qubit_layout: {params.get('qubit_layout')}")
        print(f"  tfim: {params.get('tfim')}")
        print(f"  placement_methods: {params.get('placement_methods')}")
        print(f"  n_aods: {params.get('n_aods')}")
        print(
            "  compile_settings_count:",
            len(params.get("compile_settings") or [_DEFAULT_COMPILE]),
        )
        print(f"  settings_count: {len(params.get('settings', []))}")
        print(f"  trials_per_config: {params.get('trials_per_config')}")
        print(f"  logical_se_interval: {params.get('logical_se_interval')}")
        print(f"  epsilon: {params.get('epsilon', 1e-4)}")
        print(f"  physical_error_model: {physical_error_model}")
        print()

    fidelity_results_path = os.path.join(
        output_dir, "t_cultivation_fidelity_results.csv"
    )
    profiling_results_path = os.path.join(
        output_dir, "t_cultivation_fidelity_profiling_results.csv"
    )

    # Per-round execution profiling (from ``_build_t_profiling_row`` in ``tfim_t``).
    profiling_fields = [
        "n_qubits",
        "qubit_cols",
        "qubit_rows",
        "round",
        "code_distance",
        "placement",
        "n_aods",
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
        "total_time",
        "movement_time",
        "return_movement_time",
        "stage1_time",
        "stage2_time",
        "clifford_time",
        "TMR_round",
        "RUS_round",
        "n_cnot",
        "max_rus_per_qubit",
        "avg_rus_per_qubit",
        "initial_angle",
        "largest_angle",
        "tmr_total",
        "rus_total",
        "trial",
        "fidelity_target",
        "factory_physical_size",
    ]

    fidelity_fields = [
        "trial",
        "code_distance",
        "fidelity_target",
        "factory_physical_size",
        "qubit_layout",
        "placement",
        "n_qubits",
        "n_factories",
        "n_aods",
        "trivial_return",
        "decompose_move",
        "redistribute_stage1_success",
        "J",
        "h",
        "dt",
        "n_trotter_steps",
        "epsilon",
        "total_depth",
        # Counts produced by the updated T-cultivation simulator.
        "n_cnot",
        "n_cnot_teleportation",
        "n_h",
        "n_s",
        "n_t",
        "n_se_q",
        "n_rz_decomposition",
        # Fidelity components. ``fidelity_total`` already folds in
        # ``fidelity_idle`` (from SE_q), ``fidelity_synthesis`` (gridsynth),
        # and ``fidelity_of_t_gate`` (cultivated T-state imperfection per
        # teleportation CNOT).
        "fidelity_total",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_of_rz_h",
        "fidelity_of_t_gate",
        "fidelity_cnot",
        "fidelity_h",
        "fidelity_idle",
        "fidelity_synthesis",
        "synthesis_epsilon",
    ]

    fidelity_file, fidelity_writer = ensure_csv_writer(
        fidelity_results_path, fidelity_fields
    )

    profiling_file = None
    profiling_writer: csv.DictWriter | None = None
    if analyze_result:
        profiling_file, profiling_writer = ensure_csv_writer(
            profiling_results_path, profiling_fields
        )

    original_cfg = get_config().copy()
    compile_settings = params.get("compile_settings") or [_DEFAULT_COMPILE]
    try:
        for setting in params["settings"]:
            apply_setting(
                setting,
                logical_se_interval=params.get("logical_se_interval"),
            )
            for n_cols, n_rows in params["qubit_layout"]:
                n_qubits = n_cols * n_rows
                n_factories = n_qubits  # follow star convention
                for J, h, dt, n_trotter_steps in params["tfim"]:
                    for placement in params["placement_methods"]:
                        logic_qubit_locations, magic_state_locations = (
                            get_microarchitecture(
                                n_qubits=n_qubits,
                                n_factories=n_factories,
                                qubit_layout=(n_rows, n_cols),
                                placement=placement,
                            )
                        )
                        for n_aods in params["n_aods"]:
                            for (
                                trivial_ret,
                                decompose_move,
                                redistribute_s1,
                            ) in compile_settings:
                                for trial in range(params["trials_per_config"]):
                                    trial_seed = 42 + trial
                                    rng = np.random.default_rng(trial_seed)
                                    # ``epsilon`` is the gridsynth approximation
                                    # accuracy (per-Rz target operator distance).
                                    # The same value is fed to the simulator as
                                    # ``synthesis_epsilon`` so the per-Rz state
                                    # infidelity ``epsilon ** 2`` is folded into
                                    # ``fidelity_total``.
                                    epsilon = float(params.get("epsilon", 1e-4))
                                    config = {
                                        "n_aods": n_aods,
                                        "rng": rng,
                                        "to_decompose": False,
                                        "print_profile": False,
                                        "epsilon": epsilon,
                                        "trivial_return": trivial_ret,
                                        "decompose_move": decompose_move,
                                        "redistribute_stage1_success": redistribute_s1,
                                    }
                                    if verbose:
                                        n_trials = params["trials_per_config"]
                                        print(
                                            "[evaluation_fidelity_t_cultivation] "
                                            f"trial {trial + 1}/{n_trials} | "
                                            f"distance={setting.distance} "
                                            f"fidelity_target={setting.fidelity_target:g} "
                                            f"factory_physical_size={setting.factory_physical_size} | "
                                            f"layout=({n_rows},{n_cols}) n_qubits={n_qubits} | "
                                            f"placement={placement} n_aods={n_aods} | "
                                            f"J={J} h={h} dt={dt:g} "
                                            f"n_trotter_steps={n_trotter_steps}",
                                            flush=True,
                                        )
                                        print("Compile setting:", config, flush=True)

                                    logical_error_model = LogicalErrorModel(
                                        physical_model=physical_error_model,
                                        code_distance=setting.distance,
                                    )
                                    logical_error_model.integrate_t_cultivation_fidelity_target(
                                        setting.fidelity_target
                                    )

                                    (
                                        qc_one_layer,
                                        full_logs,
                                        profiling_results_per_case,
                                    ) = generate_one_layer_2d_tfim_circuit_t_cultivation(
                                        n_qubits=n_qubits,
                                        qubit_layout=(n_rows, n_cols),
                                        placement=placement,
                                        J=J,
                                        h=h,
                                        dt=dt,
                                        code_distance=setting.distance,
                                        config=config,
                                        analyze_result=analyze_result,
                                        logic_qubit_locations=logic_qubit_locations,
                                        magic_state_locations=magic_state_locations,
                                    )
                                    if analyze_result and profiling_writer is not None:
                                        for row in profiling_results_per_case:
                                            row["trial"] = trial
                                            row["code_distance"] = setting.distance
                                            row["fidelity_target"] = (
                                                setting.fidelity_target
                                            )
                                            row["factory_physical_size"] = (
                                                setting.factory_physical_size
                                            )
                                            profiling_writer.writerow(
                                                {
                                                    field: row.get(field)
                                                    for field in profiling_fields
                                                }
                                            )

                                    n_trotter_fidelity = 1
                                    fprof = (
                                        simluate_trotter_2d_tfim_fidelity_t_cultivation(
                                            n_qubits=n_qubits,
                                            n_factories=n_factories,
                                            qubit_layout=(n_rows, n_cols),
                                            n_trotter_steps=n_trotter_fidelity,
                                            qc_one_layer=qc_one_layer,
                                            execution_logs=full_logs,
                                            logical_error_model=logical_error_model,
                                            synthesis_epsilon=epsilon,
                                        )
                                    )

                                    total_depth = None
                                    if analyze_result and profiling_results_per_case:
                                        total_depth = float(
                                            sum(
                                                float(r.get("total_time", 0.0))
                                                for r in profiling_results_per_case
                                            )
                                        )

                                    # The simulator now natively accounts for
                                    # idle (SE_q-based), gridsynth synthesis
                                    # error, and cultivated-T-state imperfection
                                    # (one F_T factor per teleportation CNOT),
                                    # so ``fprof['fidelity']`` is the canonical
                                    # end-to-end logical fidelity.
                                    fidelity_writer.writerow(
                                        {
                                            "trial": trial,
                                            "code_distance": setting.distance,
                                            "fidelity_target": setting.fidelity_target,
                                            "factory_physical_size": setting.factory_physical_size,
                                            "qubit_layout": (n_rows, n_cols),
                                            "placement": placement,
                                            "n_qubits": n_qubits,
                                            "n_factories": n_factories,
                                            "n_aods": n_aods,
                                            "trivial_return": trivial_ret,
                                            "decompose_move": decompose_move,
                                            "redistribute_stage1_success": redistribute_s1,
                                            "J": J,
                                            "h": h,
                                            "dt": dt,
                                            "n_trotter_steps": n_trotter_steps,
                                            "epsilon": epsilon,
                                            "total_depth": total_depth,
                                            "n_cnot": fprof["n_cnot"],
                                            "n_cnot_teleportation": fprof[
                                                "n_cnot_teleportation"
                                            ],
                                            "n_h": fprof["n_h"],
                                            "n_s": fprof["n_s"],
                                            "n_t": fprof["n_t"],
                                            "n_se_q": fprof["n_se_q"],
                                            "n_rz_decomposition": fprof[
                                                "n_rz_decomposition"
                                            ],
                                            "fidelity_total": fprof["fidelity"],
                                            "fidelity_of_rz_teleportaion": fprof[
                                                "fidelity_of_rz_teleportaion"
                                            ],
                                            "fidelity_of_rz_s": fprof[
                                                "fidelity_of_rz_s"
                                            ],
                                            "fidelity_of_rz_h": fprof[
                                                "fidelity_of_rz_h"
                                            ],
                                            "fidelity_of_t_gate": fprof[
                                                "fidelity_of_t_gate"
                                            ],
                                            "fidelity_cnot": fprof["fidelity_cnot"],
                                            "fidelity_h": fprof["fidelity_h"],
                                            "fidelity_idle": fprof["fidelity_idle"],
                                            "fidelity_synthesis": fprof[
                                                "fidelity_synthesis"
                                            ],
                                            "synthesis_epsilon": fprof[
                                                "synthesis_epsilon"
                                            ],
                                        }
                                    )

                                    if not analyze_result:
                                        continue
    finally:
        update_config(**original_cfg)
        fidelity_file.close()
        if profiling_file is not None:
            profiling_file.close()


if __name__ == "__main__":
    qubit_layout = [
        (4, 4),
        (6, 6),
        (8, 8),
        (10, 10),
    ]
    t_settings = [
        TSetting(fidelity_target=1e-8, factory_physical_size=2, distance=7),
        TSetting(fidelity_target=1e-8, factory_physical_size=2, distance=9),
        TSetting(fidelity_target=1e-8, factory_physical_size=4, distance=13),
        # TSetting(fidelity_target=1e-9, factory_physical_size=4, distance=13),
    ]

    physical_error_model: PhysicalErrorModel = PhysicalErrorModel("lookahead")
    p_ph = physical_error_model.get_error_rate("p_ph")
    j_h = [(1.0, 1.0)]
    tfim = []
    l1 = 1
    alpha = 2
    omega = 1
    for j, h in j_h:
        dt = l1 * alpha * p_ph / omega
        T = 10 / j
        # n_trotter = int(T / dt)
        n_trotter = 1
        tfim.append((j, h, dt, n_trotter))

    common_params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        # Gridsynth approximation accuracy (per-Rz target operator distance).
        # Per-Rz state infidelity is ``epsilon ** 2``.
        "epsilon": 1e-4,
        # Logical-qubit SE cadence (in cycles). ``None`` disables the
        # scheduler; with a value set, ``SE_q`` events are emitted and the
        # simulator's ``fidelity_idle`` term contributes to the total.
        "logical_se_interval": 10,
        # Compilation knobs (see ``MAIN_COMPILE_SETTING``).
        "compile_settings": [MAIN_COMPILE_SETTING],
    }

    # Main fidelity/runtime sweep, matching the STAR script structure.
    # params = {
    #     **common_params,
    #     "placement_methods": ["col_based"],
    #     "n_aods": [2, 3, 4],
    #     "settings": t_settings,
    #     "trials_per_config": 5,
    # }
    # run_evaluation_t_cultivation(
    #     params=params,
    #     physical_error_model=physical_error_model,
    #     analyze_result=True,
    # )

    # # Architecture study.
    # params = {
    #     **common_params,
    #     "placement_methods": ["seperate_region_row", "checkerboard"],
    #     "n_aods": [1, 5],
    #     "settings": t_settings,
    #     "trials_per_config": 5,
    # }
    # run_evaluation_t_cultivation(
    #     params=params,
    #     physical_error_model=physical_error_model,
    #     analyze_result=True,
    # )

    # # Ablation study. This includes AOD=1 so compare_fidelity.py can use the
    # # same col-based, one-AOD architecture subset for STAR and T-cultivation.
    params = {
        **common_params,
        "placement_methods": ["col_based"],
        "n_aods": [1, 5],
        "settings": t_settings,
        "trials_per_config": 10,
        "compile_settings": COMPILE_SETTINGS,
    }
    run_evaluation_t_cultivation(
        params=params,
        physical_error_model=physical_error_model,
        analyze_result=True,
    )

    params = {
        **common_params,
        "placement_methods": ["seperate_region_row"],
        "n_aods": [1, 5],
        "settings": t_settings,
        "trials_per_config": 10,
        "compile_settings": [COMPILE_SETTINGS[0]],
    }
    run_evaluation_t_cultivation(
        params=params,
        physical_error_model=physical_error_model,
        analyze_result=True,
    )

    params = {
        **common_params,
        "placement_methods": ["col_based"],
        "n_aods": [2, 3, 4],
        "settings": t_settings,
        "trials_per_config": 10,
        "compile_settings": [MAIN_COMPILE_SETTING],
    }
    run_evaluation_t_cultivation(
        params=params,
        physical_error_model=physical_error_model,
        analyze_result=True,
    )
