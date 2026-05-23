"""Stacked STAR execution timelines for analog rotation (two subfigures).

Settings match the two ``test()`` calls at the bottom of
``script/test_analog_rotation_execution.py``: n9/f9, col_based, nAOD=3,
synchronous vs asynchronous execution modes.
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator.circuit_execution_visualization import plot_star_execution_subfigures
from src.ds import FactoryPool, get_microarchitecture
from src.star.analog_rotation_execution import factory_angle_execution
from src.star.analog_rotation_execution_parallel import factory_angle_execution_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _run_star_log(*, optimize_strategy: bool):
    """Mirror ``test()`` in test_analog_rotation_execution.py (active block)."""
    # n_qubits = 8
    # n_factories = 8
    # qubit_layout = (4, 2)
    # rng = np.random.default_rng(33)
    n_qubits = 8
    n_factories = 8
    qubit_layout = (4, 2)
    rng = np.random.default_rng(8)  # 8
    logical_se_interval = 100

    target_qubits_angles = {i: 0.001 for i in range(n_qubits)}

    factory_pool = FactoryPool(num_factories=n_factories)

    if optimize_strategy:
        rng = np.random.default_rng(33)
        placement = "col_based"

        logic_qubit_locations, magic_state_locations = get_microarchitecture(
            n_qubits,
            n_factories,
            qubit_layout,
            placement,
        )
        _total_time, log = factory_angle_execution(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            code_distance=7,
            column_based_placement=True,
            n_aods=2,
            consider_skip_rus=2,
            tmr_assignment_method="matching",
            trivial_return=False,
            decompose_move=True,
            rng=rng,
            logical_se_interval=logical_se_interval,
        )
    else:
        placement = "seperate_region_col"

        logic_qubit_locations, magic_state_locations = get_microarchitecture(
            n_qubits,
            n_factories,
            qubit_layout,
            placement,
        )
        _total_time, log = factory_angle_execution(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            code_distance=7,
            column_based_placement=True,
            n_aods=1,
            consider_skip_rus=0,
            tmr_assignment_method="matching",
            trivial_return=True,
            decompose_move=False,
            rng=rng,
            logical_se_interval=logical_se_interval,
        )
    return log, n_factories, n_qubits


def main(
    *,
    output_pdf: str,
    figure_width: float,
    figure_vertical_stretch: float,
    suptitle: str | None,
) -> None:
    row_plots = [
        (
            "Unoptimized strategy",
            *_run_star_log(optimize_strategy=False),
        ),
        (
            "Optimized strategy",
            *_run_star_log(optimize_strategy=True),
        ),
    ]

    out_dir = os.path.dirname(output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_star_execution_subfigures(
        row_plots,
        show_logical_qubits=False,
        figure_width=figure_width,
        figure_vertical_stretch=figure_vertical_stretch,
        suptitle=suptitle,
        save_path=output_pdf,
    )
    logging.info("Wrote STAR execution subfigures: %s", output_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Two stacked horizontal STAR timelines (n=9, f=9, col_based, nAOD=3): "
            "synchronous vs asynchronous execution, matching test_analog_rotation_execution.py."
        )
    )
    parser.add_argument(
        "--output-pdf",
        default=(
            "output/circuit_execution/"
            "star_n9f9_optimized_vs_unoptimized_subfigures.pdf"
        ),
        help="Output path for the PDF",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=40.0,
        help="Base figure width (same scale as plot_circuit_execution in tests)",
    )
    parser.add_argument(
        "--figure-vertical-stretch",
        type=float,
        default=2,
        help=(
            "Multiplies stacked-panel height (taller figure => taller timeline boxes). "
            "Default 1.45; try 1.7–2.0 for very large lanes."
        ),
    )
    parser.add_argument(
        "--suptitle",
        default="Execution Timeline for STAR Architecture",
        help="Figure suptitle (empty string to omit)",
    )
    args = parser.parse_args()
    st = args.suptitle.strip()
    main(
        output_pdf=args.output_pdf,
        figure_width=args.figure_width,
        figure_vertical_stretch=args.figure_vertical_stretch,
        suptitle=st if st else None,
    )
