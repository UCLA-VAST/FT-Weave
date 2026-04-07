import csv
import os
import sys
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

# Ensure repository root is on sys.path so `src` is importable when running this script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ds import get_microarchitecture
from src.error_model import LogicalErrorModel, PhysicalErrorModel
from src.fidelity_simulation import simluate_trotter_2d_tfim_fidelity_t_cultivation
from src.t_cultivation.config import get_config, update_config
from src.t_cultivation.tfim_t import generate_one_layer_2d_tfim_circuit_t_cultivation
from src.t_cultivation.t_cultivation import t_cultivation_execution
from src.util import analyze_execution_log

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("src").setLevel(logging.INFO)


@dataclass(frozen=True)
class TSetting:
    """Evaluation setting aligned with throughput evaluation."""

    fidelity_target: float
    factory_physical_size: int
    distance: int


def apply_setting(setting: TSetting) -> None:
    """Apply one T-cultivation setting to global config."""
    update_config(
        STAGE_2_FIDELITY_TARGET=setting.fidelity_target,
        FACTORY_PHYSICAL_SIZE=setting.factory_physical_size,
        STAGE_1_RESOURCE_UNITS=1,
    )


def _ensure_writer(csv_path: str, fieldnames: list[str]):
    header_needed = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if header_needed:
        writer.writeheader()
    return csv_file, writer


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
    STAR’s per-Rz logs).     Per-Rz ``rz_logs`` are passed to :func:`simluate_trotter_2d_tfim_fidelity_t_cultivation`
    (same nested structure as STAR).

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
        print(f"  settings_count: {len(params.get('settings', []))}")
        print(f"  trials_per_config: {params.get('trials_per_config')}")
        print(f"  physical_error_model: {physical_error_model}")
        print()

    fidelity_results_path = os.path.join(
        output_dir, "t_cultivation_fidelity_results.csv"
    )
    profiling_results_path = os.path.join(
        output_dir, "t_cultivation_fidelity_profiling_results.csv"
    )

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
        "J",
        "h",
        "dt",
        "n_trotter_steps",
        "fidelity_total",
        "fidelity_t_injection",
        "fidelity_t_teleportation",
        "fidelity_clifford",
    ]

    fidelity_file, fidelity_writer = _ensure_writer(
        fidelity_results_path, fidelity_fields
    )

    profiling_file = (
        open(profiling_results_path, "a", newline="") if analyze_result else None
    )
    profiling_writer: csv.DictWriter | None = None
    profiling_header_needed = (not os.path.exists(profiling_results_path)) or (
        os.path.getsize(profiling_results_path) == 0
    )

    original_cfg = get_config().copy()
    try:
        for setting in params["settings"]:
            apply_setting(setting)
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
                            for trial in range(params["trials_per_config"]):
                                trial_seed = 42 + trial
                                rng = np.random.default_rng(trial_seed)
                                config = {
                                    "n_aods": n_aods,
                                    "rng": rng,
                                    "to_decompose": False,
                                    "print_profile": False,
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
                                    rz_logs,
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
                                if analyze_result and profiling_file is not None:
                                    for row in profiling_results_per_case:
                                        if profiling_writer is None:
                                            profiling_writer = csv.DictWriter(
                                                profiling_file,
                                                fieldnames=list(row.keys()),
                                            )
                                            if profiling_header_needed:
                                                profiling_writer.writeheader()
                                        if profiling_writer is not None:
                                            profiling_writer.writerow(row)

                                n_trotter_fidelity = 1
                                fprof = simluate_trotter_2d_tfim_fidelity_t_cultivation(
                                    n_qubits=n_qubits,
                                    n_factories=n_factories,
                                    qubit_layout=(n_rows, n_cols),
                                    n_trotter_steps=n_trotter_fidelity,
                                    qc_one_layer=qc_one_layer,
                                    execution_logs=rz_logs,
                                    logical_error_model=logical_error_model,
                                )
                                fidelity_clifford = (
                                    fprof["fidelity_cnot"] * fprof["fidelity_1q"]
                                )
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
                                        "J": J,
                                        "h": h,
                                        "dt": dt,
                                        "n_trotter_steps": n_trotter_steps,
                                        "fidelity_total": fprof["fidelity"],
                                        "fidelity_t_injection": fprof[
                                            "fidelity_of_rz_injection"
                                        ],
                                        "fidelity_t_teleportation": fprof[
                                            "fidelity_of_rz_teleportaion"
                                        ],
                                        "fidelity_clifford": fidelity_clifford,
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
    n_aods = [2, 3, 4, 5]
    settings = [
        TSetting(fidelity_target=1e-8, factory_physical_size=2, distance=7),
        TSetting(fidelity_target=1e-8, factory_physical_size=4, distance=13),
        TSetting(fidelity_target=1e-10, factory_physical_size=4, distance=13),
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

    params = {
        "qubit_layout": qubit_layout,
        "tfim": tfim,
        "placement_methods": ["col_based"],
        "n_aods": n_aods,
        "settings": settings,
        "trials_per_config": 5,
    }

    run_evaluation_t_cultivation(
        params=params,
        physical_error_model=physical_error_model,
        analyze_result=True,
    )
