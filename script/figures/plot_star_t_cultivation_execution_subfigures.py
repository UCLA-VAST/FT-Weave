"""Stacked STAR and T-cultivation execution timelines (1 logical qubit, 2 factories)."""

import argparse
import logging
import os
import sys

import numpy as np

# Repo root (two levels up from script/figures/) so ``src`` is importable when
# this file is run directly, e.g. ``uv run script/figures/<name>.py``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.animator.circuit_execution_visualization import (
    plot_star_t_cultivation_execution_subfigures,
)
from src.ds import FactoryPool, get_microarchitecture
from src.star.analog_rotation_execution import factory_angle_execution
from src.t_cultivation.config import get_config, update_config
from src.t_cultivation.t_cultivation import t_cultivation_execution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

N_LOGICAL_QUBITS = 1
N_FACTORIES = 2


def _run_star_log(*, seed: int) -> list:
    placement = "col_based"
    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        N_LOGICAL_QUBITS,
        N_FACTORIES,
        (N_FACTORIES, 1),
        placement,
    )
    target_qubits_angles = {0: 0.001}
    factory_pool = FactoryPool(num_factories=N_FACTORIES)
    rng = np.random.default_rng(seed)
    _total_time, log = factory_angle_execution(
        factory_pool,
        target_qubits_angles,
        logic_qubit_locations,
        magic_state_locations,
        code_distance=7,
        column_based_placement=True,
        n_aods=1,
        consider_skip_rus=False,
        tmr_assignment_method="matching",
        trivial_return=False,
        rng=rng,
        logical_se_interval=100,
    )
    return log


def _run_t_cultivation_log(*, seed: int) -> list:
    circuit = [
        {"gate": "T", "targets": [0], "params": {}},
        # {"gate": "T", "targets": [0], "params": {}},
    ]
    logic_qubit_locations = [(0, 0)]
    magic_state_locations = [(0, 1), (1, 1)]
    return t_cultivation_execution(
        circuit=circuit,
        n_factories=N_FACTORIES,
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
        rng=np.random.default_rng(seed),
        n_aods=1,
        to_decompose=False,
        logical_se_interval=100,
    )


def main(
    *,
    output_pdf: str,
    star_seed: int,
    t_seed: int,
    figure_width: float,
    figure_vertical_stretch: float,
    suptitle: str | None,
) -> None:
    original_cfg = get_config().copy()
    try:
        update_config(FACTORY_PHYSICAL_SIZE=2)
        star_log = _run_star_log(seed=star_seed)
        t_log = _run_t_cultivation_log(seed=t_seed)
    finally:
        update_config(
            SE_STAGE_1=original_cfg["SE_STAGE_1"],
            SE_STAGE_2=original_cfg["SE_STAGE_2"],
            CNOT_TIME=original_cfg["CNOT_TIME"],
            SE_TIME=original_cfg["SE_TIME"],
            TELEPORTATION_SUCCESS_RATE=original_cfg["TELEPORTATION_SUCCESS_RATE"],
            STAGE_1_SUCCESS_RATE=original_cfg["STAGE_1_SUCCESS_RATE"],
            STAGE_2_SUCCESS_RATE=original_cfg["STAGE_2_SUCCESS_RATE"],
            STAGE_2_FIDELITY_TARGET=original_cfg["STAGE_2_FIDELITY_TARGET"],
            ANGLE_S=original_cfg["ANGLE_S"],
            FACTORY_PHYSICAL_SIZE=original_cfg["FACTORY_PHYSICAL_SIZE"],
            STAGE_1_RESOURCE_UNITS=original_cfg["STAGE_1_RESOURCE_UNITS"],
            SYNCHRONIZE_FACTORY_EXECUTION=original_cfg["SYNCHRONIZE_FACTORY_EXECUTION"],
            RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT=original_cfg[
                "RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT"
            ],
        )

    if not star_log:
        raise RuntimeError("STAR execution log is empty")
    if not t_log:
        raise RuntimeError("T-cultivation execution log is empty")

    out_dir = os.path.dirname(output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_star_t_cultivation_execution_subfigures(
        star_log=star_log,
        t_log=t_log,
        n_logical_qubits=N_LOGICAL_QUBITS,
        n_factories=N_FACTORIES,
        star_row_title="STAR architecture",
        t_row_title="T-cultivation architecture",
        save_path=output_pdf,
        suptitle=suptitle,
        figure_width=figure_width,
        figure_vertical_stretch=figure_vertical_stretch,
    )
    logging.info(
        "Wrote STAR vs T-cultivation subfigures: %s and %s",
        output_pdf,
        output_pdf.replace(".pdf", "_no_text.pdf"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Two-row execution timeline: STAR (top) and T-cultivation (bottom), "
            "each with 1 logical qubit and 2 factories."
        )
    )
    parser.add_argument(
        "--output-pdf",
        default=("output/circuit_execution/" "star_t_cultivation_1q2f_subfigures.pdf"),
        help="Output path for the PDF",
    )
    parser.add_argument("--star-seed", type=int, default=40)
    parser.add_argument("--t-seed", type=int, default=18)
    parser.add_argument("--figure-width", type=float, default=22.5)
    parser.add_argument(
        "--figure-vertical-stretch",
        type=float,
        default=2.4,
        help="Multiplies stacked-panel height",
    )
    parser.add_argument(
        "--suptitle",
        default="Execution timeline: STAR vs T-cultivation",
        help="Figure suptitle (empty string to omit)",
    )
    args = parser.parse_args()
    st = args.suptitle.strip()
    main(
        output_pdf=args.output_pdf,
        star_seed=args.star_seed,
        t_seed=args.t_seed,
        figure_width=args.figure_width,
        figure_vertical_stretch=args.figure_vertical_stretch,
        suptitle=st if st else None,
    )
