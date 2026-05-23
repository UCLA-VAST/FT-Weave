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


def _run_star_log(*, parallel_execution: bool):
    """Mirror ``test()`` in test_analog_rotation_execution.py (active block)."""
    n_qubits = 9
    n_factories = 9
    qubit_layout = (3, 3)
    placement = "col_based"
    n_aods = 3
    n_aods_se = 3
    consider_skip_rus = False
    tmr_assignment_method = "matching"
    trivial_return = False
    logical_se_interval = 100

    target_qubits_angles = {i: 0.001 for i in range(n_qubits)}
    column_based_placement = placement in ("seperate_region_col", "col_based")
    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        n_qubits,
        n_factories,
        qubit_layout,
        placement,
    )

    factory_pool = FactoryPool(num_factories=n_factories)
    rng = np.random.default_rng(40)

    if parallel_execution:
        _total_time, log = factory_angle_execution_parallel(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            code_distance=7,
            column_based_placement=column_based_placement,
            n_aods=n_aods,
            n_aods_se=n_aods_se,
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method=tmr_assignment_method,
            trivial_return=trivial_return,
            rng=rng,
            logical_se_interval=logical_se_interval,
        )
    else:
        _total_time, log = factory_angle_execution(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            code_distance=7,
            column_based_placement=column_based_placement,
            n_aods=n_aods,
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method=tmr_assignment_method,
            trivial_return=trivial_return,
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
            "Synchronous execution",
            *_run_star_log(parallel_execution=False),
        ),
        (
            "Asynchronous Execution",
            *_run_star_log(parallel_execution=True),
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
    logging.info(
        "Wrote STAR execution subfigures: %s and %s",
        output_pdf,
        output_pdf.replace(".pdf", "_no_text.pdf"),
    )


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
            "star_n9f9_col_based_naod3_synchronous_vs_asynchronous_subfigures.pdf"
        ),
        help="Output path for the PDF",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=30.0,
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
