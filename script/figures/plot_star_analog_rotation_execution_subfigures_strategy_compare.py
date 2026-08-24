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

# Repo root (two levels up from script/figures/) so ``src`` is importable when
# this file is run directly, e.g. ``uv run script/figures/<name>.py``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.animator.circuit_execution_visualization import plot_star_execution_subfigures
from src.animator.rus_round_visualization import (
    TRAP_GRID_MOVEMENT_COLOR_EARLY,
    TRAP_GRID_MOVEMENT_COLOR_LATE,
    build_rus_move_shade_specs,
    plot_all_rus_rounds,
)
from src.ds import FactoryPool, get_microarchitecture
from src.star.analog_rotation_execution import factory_angle_execution
from src.star.analog_rotation_execution_parallel import factory_angle_execution_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Per-panel RUS rounds to highlight (move before CNOT + return after CNOT).
_ROW_RUS_MOVE_SHADE_ROUNDS = [
    [1],  # Unoptimized strategy
    [1],  # Optimized strategy w/o dropout
    [5],  # Optimized strategy w/ dropout
]

# Middle strategy row: shade the RUS teleportation window replaced by rematerialization
# in the bottom row (move → CNOT/teleportation → return, roughly t ≈ 60–66).
_REMATERIALIZATION_SHADE_ROW_INDEX = 1
_REMATERIALIZATION_TIME_WINDOW = (60.0, 67.0)
_REMATERIALIZATION_SHADE_COLOR = "#FDAE6B"
_REMATERIALIZATION_SHADE_LABEL = "Operation rematerialization"

# Trap-grid movement arc/label colors (earlier batch → later batch). Override here or in
# rus_round_visualization.TRAP_GRID_MOVEMENT_COLOR_* for all callers.
RUS_TRAP_GRID_MOVEMENT_COLOR_EARLY = TRAP_GRID_MOVEMENT_COLOR_EARLY
RUS_TRAP_GRID_MOVEMENT_COLOR_LATE = TRAP_GRID_MOVEMENT_COLOR_LATE


def _run_star_log(
    *, optimize_strategy: bool, consider_skip_rus: int = 0, rng: np.random.Generator
):
    """Mirror ``test()`` in test_analog_rotation_execution.py (active block)."""
    # n_qubits = 8
    # n_factories = 8
    # qubit_layout = (4, 2)
    rng = np.random.default_rng(1233)
    n_qubits = 25
    n_factories = 25
    qubit_layout = (5, 5)
    logical_se_interval = 100

    target_qubits_angles = {i: 0.002 for i in range(n_qubits)}

    factory_pool = FactoryPool(num_factories=n_factories)

    prepare_lookahead_angles = True
    # prepare_lookahead_angles = False

    if optimize_strategy:
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
            n_aods=1,
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method="matching",
            trivial_return=False,
            decompose_move=True,
            rng=rng,
            prepare_lookahead_angles=prepare_lookahead_angles,
            logical_se_interval=logical_se_interval,
        )
    else:
        placement = "seperate_region_col"
        # rng = np.random.default_rng(50)
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
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method="matching",
            trivial_return=True,
            decompose_move=False,
            rng=rng,
            logical_se_interval=logical_se_interval,
            prepare_lookahead_angles=False,
        )
    print(
        f"Execution type: {optimize_strategy}, consider skip RUS: {consider_skip_rus}, total time: {_total_time}"
    )
    return (
        log,
        n_factories,
        n_qubits,
        logic_qubit_locations,
        magic_state_locations,
        placement,
    )


def _plot_rus_rounds_for_rows(
    row_plots: list[tuple],
    *,
    rus_output_dir: str,
    code_distance: int,
    movement_overlay: str,
) -> None:
    os.makedirs(rus_output_dir, exist_ok=True)
    for label, log, _nf, _nq, logic_locs, magic_locs, placement in row_plots:
        slug = label.lower().replace(" ", "_").replace("/", "_").replace("w/o", "wo")
        base_path = os.path.join(rus_output_dir, f"{slug}_{placement}")
        logging.info("RUS trap-grid figures for %s -> %s", label, base_path)
        plot_all_rus_rounds(
            execution_log=log,
            logic_qubit_locations=logic_locs,
            magic_state_locations=magic_locs,
            base_path=base_path,
            style_variant="trap_grid",
            code_distance=code_distance,
            movement_overlay=movement_overlay,  # type: ignore[arg-type]
            movement_color_early=RUS_TRAP_GRID_MOVEMENT_COLOR_EARLY,
            movement_color_late=RUS_TRAP_GRID_MOVEMENT_COLOR_LATE,
        )


def main(
    *,
    output_pdf: str,
    figure_width: float,
    figure_vertical_stretch: float,
    suptitle: str | None,
    rng: np.random.Generator,
    plot_rus_rounds: bool,
    rus_output_dir: str,
    rus_code_distance: int,
    rus_movement_overlay: str,
    shade_rus_moves: bool,
    rus_shade_alpha: float,
    row_rus_shade_rounds: list[list[int]],
) -> None:
    row_plots = [
        (
            "Unoptimized strategy",
            *_run_star_log(optimize_strategy=False, rng=rng),
        ),
        (
            "Optimized strategy w/o operation rematerialization",
            *_run_star_log(optimize_strategy=True, rng=rng),
        ),
        (
            "Optimized strategy w/ operation rematerialization",
            *_run_star_log(optimize_strategy=True, consider_skip_rus=2, rng=rng),
        ),
    ]

    if plot_rus_rounds:
        _plot_rus_rounds_for_rows(
            row_plots,
            rus_output_dir=rus_output_dir,
            code_distance=rus_code_distance,
            movement_overlay=rus_movement_overlay,
        )

    out_dir = os.path.dirname(output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    timeline_rows = [(label, log, nf, nq) for label, log, nf, nq, *_rest in row_plots]
    row_time_shades: list[list[dict] | None] = [None] * len(timeline_rows)
    if shade_rus_moves:
        for row_idx, ((_, log, *_rest), rounds) in enumerate(
            zip(row_plots, row_rus_shade_rounds)
        ):
            specs = build_rus_move_shade_specs(log, rounds, alpha=rus_shade_alpha)
            row_time_shades[row_idx] = specs if specs else None

    remat_t0, remat_t1 = _REMATERIALIZATION_TIME_WINDOW
    remat_spec = {
        "time_start": remat_t0,
        "time_end": remat_t1,
        "color": _REMATERIALIZATION_SHADE_COLOR,
        "label": _REMATERIALIZATION_SHADE_LABEL,
        "alpha": rus_shade_alpha,
    }
    remat_row = _REMATERIALIZATION_SHADE_ROW_INDEX
    existing_shades = row_time_shades[remat_row]
    if existing_shades is None:
        row_time_shades[remat_row] = [remat_spec]
    else:
        row_time_shades[remat_row] = [*existing_shades, remat_spec]

    plot_star_execution_subfigures(
        timeline_rows,
        show_logical_qubits=False,
        figure_width=figure_width,
        figure_vertical_stretch=figure_vertical_stretch,
        suptitle=suptitle,
        save_path=output_pdf,
        row_time_shades=row_time_shades,
        show_factory_yticks=False,
        legend_row_index=len(timeline_rows) // 2,
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
        default=60.0,
        help="Base figure width (same scale as plot_circuit_execution in tests)",
    )
    parser.add_argument(
        "--figure-vertical-stretch",
        type=float,
        default=1,
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
    parser.add_argument(
        "--rng",
        type=int,
        default=9,
        help="RNG seed",
    )
    parser.add_argument(
        "--plot-rus-rounds",
        action="store_true",
        default=True,
        help="Also write per-RUS trap-grid move/return PDFs (sync-style isolation)",
    )
    parser.add_argument(
        "--rus-output-dir",
        default="output/rus_rounds_detailed/strategy_compare",
        help="Directory for trap-grid RUS round PDFs",
    )
    parser.add_argument(
        "--rus-code-distance",
        type=int,
        default=3,
        help="Code distance d for trap-grid site layout",
    )
    parser.add_argument(
        "--rus-movement-overlay",
        choices=("both", "arrows_only", "aod_only", "none"),
        default="both",
        help="Movement visualization on trap-grid figures",
    )
    parser.add_argument(
        "--no-rus-move-shade",
        action="store_true",
        help="Disable colored timeline shading for selected RUS move/return intervals",
    )
    parser.add_argument(
        "--rus-shade-alpha",
        type=float,
        default=0.22,
        help="Alpha for RUS move/return timeline shading",
    )
    parser.add_argument(
        "--row-rus-shade-rounds",
        default="1|1,5|5",
        help=(
            "Per-panel 1-based RUS rounds to shade, separated by '|' "
            "(default: 1|1,5|5 for the three strategy rows)"
        ),
    )
    args = parser.parse_args()
    st = args.suptitle.strip()
    rng = args.rng
    row_rus_shade_rounds = [
        [int(x.strip()) for x in panel.split(",") if x.strip()]
        for panel in args.row_rus_shade_rounds.split("|")
    ]
    if len(row_rus_shade_rounds) != len(_ROW_RUS_MOVE_SHADE_ROUNDS):
        parser.error(
            f"--row-rus-shade-rounds must have {len(_ROW_RUS_MOVE_SHADE_ROUNDS)} "
            f"panels (got {len(row_rus_shade_rounds)})"
        )
    main(
        output_pdf=args.output_pdf,
        figure_width=args.figure_width,
        figure_vertical_stretch=args.figure_vertical_stretch,
        suptitle=st if st else None,
        rng=np.random.default_rng(rng),
        plot_rus_rounds=args.plot_rus_rounds,
        rus_output_dir=args.rus_output_dir,
        rus_code_distance=args.rus_code_distance,
        rus_movement_overlay=args.rus_movement_overlay,
        shade_rus_moves=not args.no_rus_move_shade,
        rus_shade_alpha=args.rus_shade_alpha,
        row_rus_shade_rounds=row_rus_shade_rounds,
    )
