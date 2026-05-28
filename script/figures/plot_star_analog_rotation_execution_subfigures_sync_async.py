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

from src.animator.circuit_execution_visualization import (
    _collapse_star_tmr_blocks,
    plot_star_execution_subfigures,
)
from src.animator.rus_round_visualization import (
    plot_all_rus_rounds,
    plot_trap_grid_time_range,
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


def _entry_start(entry: dict) -> float:
    if "start_time" in entry:
        return float(entry["start_time"])
    return float(entry.get("start", 0.0))


def _entry_end(entry: dict) -> float:
    if "end_time" in entry:
        return float(entry["end_time"])
    return float(entry.get("end", _entry_start(entry)))


def _normalize_factories(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (tuple, list)):
        return [int(v) for v in value if isinstance(v, int)]
    return []


def _unique_sorted(values: list[float]) -> list[float]:
    return sorted({round(float(v), 9) for v in values})


def _sync_realtime_control_markers(execution_log: list[dict]) -> list[float]:
    """Sync markers:
    1) one line at the start of each TMR,
    2) one line right after each TMR (TMR end time),
    3) one line at the end of each consecutive return_move section that is
       immediately followed by a move section.
    """
    plot_log = _collapse_star_tmr_blocks(execution_log)
    events = sorted(plot_log, key=_entry_start)
    markers: list[float] = []
    for entry in events:
        if entry.get("operation") == "TMR":
            markers.append(_entry_start(entry))
            markers.append(_entry_end(entry))

    n = len(events)
    i = 0
    while i < n:
        if events[i].get("operation") != "return_move":
            i += 1
            continue
        # Consume one consecutive return_move section.
        section_end = _entry_end(events[i])
        i += 1
        while i < n and events[i].get("operation") == "return_move":
            section_end = max(section_end, _entry_end(events[i]))
            i += 1
        # Keep this marker only when next section starts with move.
        if i < n and events[i].get("operation") == "move":
            markers.append(section_end)
    return _unique_sorted(markers)


def _async_realtime_control_markers(execution_log: list[dict]) -> list[float]:
    """Async markers:
    - TMR start
    - start and end of every move and return_move
    - after CNOT and before return_move when CNOT is immediately followed by return_move
    """
    plot_log = _collapse_star_tmr_blocks(execution_log)
    events = sorted(plot_log, key=_entry_start)
    markers: list[float] = []
    for i, entry in enumerate(events):
        op = entry.get("operation")
        if op == "TMR":
            markers.append(_entry_start(entry))
        elif op in {"move", "return_move"}:
            markers.append(_entry_start(entry))
            markers.append(_entry_end(entry))
        elif op == "CNOT":
            markers.append(_entry_end(entry))  # after CNOT
            if i + 1 < len(events) and events[i + 1].get("operation") == "return_move":
                markers.append(_entry_start(events[i + 1]))  # before return_move
    return _unique_sorted(markers)


def _run_star_log(*, parallel_execution: bool):
    """Mirror ``test()`` in test_analog_rotation_execution.py (active block)."""
    n_qubits = 25
    n_factories = 25
    qubit_layout = (5, 5)
    placement = "col_based"
    n_aods = 3
    n_aods_se = 3
    consider_skip_rus = 2
    tmr_assignment_method = "matching"
    trivial_return = False
    logical_se_interval = 100
    prepare_lookahead_angles = False
    target_qubits_angles = {i: 0.001 for i in range(n_qubits)}
    column_based_placement = placement in ("seperate_region_col", "col_based")
    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        n_qubits,
        n_factories,
        qubit_layout,
        placement,
    )

    factory_pool = FactoryPool(num_factories=n_factories)
    rng = np.random.default_rng(47)

    if parallel_execution:
        _total_time, log = factory_angle_execution_parallel(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            prepare_lookahead_angles=prepare_lookahead_angles,
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
            prepare_lookahead_angles=prepare_lookahead_angles,
            code_distance=7,
            column_based_placement=column_based_placement,
            n_aods=n_aods,
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method=tmr_assignment_method,
            trivial_return=trivial_return,
            rng=rng,
            logical_se_interval=logical_se_interval,
            decompose_move=True,
        )
    return log, n_factories, n_qubits, logic_qubit_locations, magic_state_locations


def main(
    *,
    output_pdf: str,
    figure_width: float,
    figure_vertical_stretch: float,
    suptitle: str | None,
    add_communication_markers: bool,
    plot_rus_rounds: bool,
    rus_sync_output_dir: str,
    rus_async_output_dir: str,
    rus_code_distance: int,
    rus_movement_overlay: str,
    async_time_start: float,
    async_time_end: float,
) -> None:
    (
        sync_log,
        sync_nf,
        sync_nq,
        logic_locs,
        magic_locs,
    ) = _run_star_log(parallel_execution=False)
    async_log, async_nf, async_nq, _, _ = _run_star_log(parallel_execution=True)
    row_plots = [
        (
            "Synchronous execution",
            sync_log,
            sync_nf,
            sync_nq,
        ),
        (
            "Asynchronous Execution",
            async_log,
            async_nf,
            async_nq,
        ),
    ]
    row_time_markers = None
    marker_line_kwargs = None
    if add_communication_markers:
        row_time_markers = [
            _sync_realtime_control_markers(sync_log),
            _async_realtime_control_markers(async_log),
        ]
        marker_line_kwargs = {
            "color": "#D81B60",
            "linestyle": "-.",
            "linewidth": 1.5,
            "alpha": 0.6,
            "zorder": 80,
        }

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
        row_time_markers=row_time_markers,
        marker_line_kwargs=marker_line_kwargs,
        marker_legend_label="Realtime control event",
    )
    logging.info(
        "Wrote STAR execution subfigures: %s and %s",
        output_pdf,
        output_pdf.replace(".pdf", "_no_text.pdf"),
    )

    if plot_rus_rounds:
        logging.info("RUS trap-grid (sync, per-round) -> %s", rus_sync_output_dir)
        plot_all_rus_rounds(
            execution_log=sync_log,
            logic_qubit_locations=logic_locs,
            magic_state_locations=magic_locs,
            base_path=rus_sync_output_dir,
            style_variant="trap_grid",
            code_distance=rus_code_distance,
            movement_overlay=rus_movement_overlay,  # type: ignore[arg-type]
        )
        logging.info(
            "RUS trap-grid (async, t=%s–%s) -> %s",
            async_time_start,
            async_time_end,
            rus_async_output_dir,
        )
        plot_trap_grid_time_range(
            execution_log=async_log,
            logic_qubit_locations=logic_locs,
            magic_state_locations=magic_locs,
            time_start=async_time_start,
            time_end=async_time_end,
            base_path=rus_async_output_dir,
            title_prefix=(f"Async t∈[{async_time_start:g},{async_time_end:g}]"),
            code_distance=rus_code_distance,
            movement_overlay=rus_movement_overlay,  # type: ignore[arg-type]
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
        default=1.5,
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
        "--no-communication-markers",
        action="store_true",
        help="Disable realtime-control marker lines",
    )
    parser.add_argument(
        "--plot-rus-rounds",
        action="store_true",
        default=True,
        help=(
            "Trap-grid spatial PDFs: per-RUS rounds for sync; time-window slice for async"
        ),
    )
    parser.add_argument(
        "--rus-sync-output-dir",
        default="output/rus_rounds_detailed/sync_async/sync",
        help="Output directory for synchronous per-RUS trap-grid PDFs",
    )
    parser.add_argument(
        "--rus-async-output-dir",
        default="output/rus_rounds_detailed/sync_async/async",
        help="Output directory for asynchronous time-window trap-grid PDFs",
    )
    parser.add_argument(
        "--async-time-start",
        type=float,
        default=0.0,
        help="Async trap-grid window start time (same units as execution log)",
    )
    parser.add_argument(
        "--async-time-end",
        type=float,
        default=500.0,
        help="Async trap-grid window end time",
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
    args = parser.parse_args()
    st = args.suptitle.strip()
    main(
        output_pdf=args.output_pdf,
        figure_width=args.figure_width,
        figure_vertical_stretch=args.figure_vertical_stretch,
        suptitle=st if st else None,
        add_communication_markers=not args.no_communication_markers,
        plot_rus_rounds=args.plot_rus_rounds,
        rus_sync_output_dir=args.rus_sync_output_dir,
        rus_async_output_dir=args.rus_async_output_dir,
        rus_code_distance=args.rus_code_distance,
        rus_movement_overlay=args.rus_movement_overlay,
        async_time_start=args.async_time_start,
        async_time_end=args.async_time_end,
    )
