"""Paper figure 7: synchronous versus asynchronous STAR execution.

Runs identical-angle ``Rz`` gates on 25 logical qubits with 25 factories and
four AODs, under both execution policies, and writes a single two-row figure: the execution
timeline on the left and, on the right, the movement schedule for the window
highlighted on the timeline. Realtime-control events -- the points at which each
policy must observe hardware outcomes and re-decide -- are marked on the
timeline.

The per-policy movement schedules are also written on their own, so the right
panel can be re-laid out independently.
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

from src.animator.circuit_execution_visualization import (
    _collapse_star_tmr_blocks,
    _trap_grid_block_legend_handles,
    plot_star_execution_subfigures,
    plot_star_timeline_movement_combined,
)
from src.animator.rus_round_visualization import (
    build_rus_move_shade_specs,
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
    """Compile one execution policy's log for the figure 7 workload."""
    n_qubits = 25
    n_factories = 25
    qubit_layout = (5, 5)
    placement = "col_based"
    n_aods = 4
    n_aods_se = 4
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
    # rng = np.random.default_rng(20)
    rng = np.random.default_rng(65)

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
    timeline_subfigures_pdf: str | None,
    figure_width: float,
    figure_vertical_stretch: float,
    suptitle: str | None,
    add_communication_markers: bool,
    plot_rus_rounds: bool,
    rus_sync_output_dir: str,
    rus_async_output_dir: str,
    rus_code_distance: int,
    rus_movement_overlay: str,
    trap_window_start: float,
    trap_window_end: float,
    timeline_xmax: float | None,
    shade_rus_moves: bool,
    sync_rus_shade_rounds: list[int],
    rus_shade_alpha: float,
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
    row_marker_line_kwargs = None
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
        # Async markers are dense; keep them slightly lighter than sync.
        row_marker_line_kwargs = [
            None,
            {"alpha": 0.4},
        ]

    out_dir = os.path.dirname(output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    row_time_shades: list[list[dict] | None] | None = None
    if shade_rus_moves and sync_rus_shade_rounds:
        sync_specs = build_rus_move_shade_specs(
            sync_log, sync_rus_shade_rounds, alpha=rus_shade_alpha
        )
        row_time_shades = [sync_specs if sync_specs else None, None]

    trap_window = (trap_window_start, trap_window_end)
    timeline_kwargs = dict(
        show_logical_qubits=False,
        figure_width=figure_width,
        figure_vertical_stretch=figure_vertical_stretch,
        suptitle=suptitle,
        row_time_markers=row_time_markers,
        marker_line_kwargs=marker_line_kwargs,
        marker_legend_label="Real-time control event.",
        row_time_windows=[trap_window, trap_window],
        row_time_shades=row_time_shades,
        highlight_xticks=[trap_window_start, trap_window_end],
        time_window_legend_label="Movement window",
        show_factory_yticks=False,
    )
    combined_kwargs = dict(
        **timeline_kwargs,
        row_marker_line_kwargs=row_marker_line_kwargs,
        highlight_xtick_labels=[trap_window_start],
        timeline_xmax=timeline_xmax,
        movement_column_title=(
            f"QEC Cycles= {trap_window_start:g}–{trap_window_end:g}."
        ),
        panel_labels=["(a)", "(b)", "(c)", "(d)"],
        extra_legend_handles=_trap_grid_block_legend_handles(),
    )
    plot_star_timeline_movement_combined(
        row_plots,
        logic_qubit_locations=logic_locs,
        magic_state_locations=magic_locs,
        movement_time_start=trap_window_start,
        movement_time_end=trap_window_end,
        movement_code_distance=rus_code_distance,
        movement_overlay=rus_movement_overlay,
        save_path=output_pdf,
        **combined_kwargs,
    )
    logging.info(
        "Wrote timeline + movement figure: %s and %s",
        output_pdf,
        output_pdf.replace(".pdf", "_no_text.pdf"),
    )

    if timeline_subfigures_pdf:
        plot_star_execution_subfigures(
            row_plots,
            save_path=timeline_subfigures_pdf,
            **timeline_kwargs,
        )
        logging.info(
            "Wrote timeline-only subfigures: %s and %s",
            timeline_subfigures_pdf,
            timeline_subfigures_pdf.replace(".pdf", "_no_text.pdf"),
        )

    if plot_rus_rounds:
        for label, log, out_dir in (
            ("sync", sync_log, rus_sync_output_dir),
            ("async", async_log, rus_async_output_dir),
        ):
            logging.info(
                "RUS trap-grid (%s, t=%s–%s) -> %s",
                label,
                trap_window_start,
                trap_window_end,
                out_dir,
            )
            plot_trap_grid_time_range(
                execution_log=log,
                logic_qubit_locations=logic_locs,
                magic_state_locations=magic_locs,
                time_start=trap_window_start,
                time_end=trap_window_end,
                base_path=out_dir,
                title_prefix=(
                    f"{label.capitalize()} t=[{trap_window_start:g},{trap_window_end:g}]"
                ),
                code_distance=rus_code_distance,
                movement_overlay=rus_movement_overlay,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Paper figure 7: synchronous vs asynchronous STAR execution on 25 "
            "logical qubits with 25 factories and four AODs. Each row pairs the "
            "execution timeline (left) with the movement schedule (right) for a "
            "shared movement time window."
        )
    )
    parser.add_argument(
        "--output-pdf",
        default=("output/figures/fig07_synchronous_vs_asynchronous_execution.pdf"),
        help="Output path for the figure 7 timeline + movement PDF",
    )
    parser.add_argument(
        "--timeline-subfigures-pdf",
        default=None,
        help=(
            "If set, also write timeline-only stacked subfigures to this path "
            "(e.g. ..._subfigures.pdf)"
        ),
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
        default="Execution timeline and movement for STAR architecture",
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
        # default=False,
        default=True,
        help=(
            "Trap-grid spatial PDFs for sync and async over the trap-window time range"
        ),
    )
    parser.add_argument(
        "--rus-sync-output-dir",
        default="output/figures/fig07_movement_schedules/synchronous",
        help="Output directory for the synchronous movement-schedule PDF",
    )
    parser.add_argument(
        "--rus-async-output-dir",
        default="output/figures/fig07_movement_schedules/asynchronous",
        help="Output directory for the asynchronous movement-schedule PDF",
    )
    parser.add_argument(
        "--trap-window-start",
        "--async-time-start",
        type=float,
        default=17.0,
        dest="trap_window_start",
        help="Trap-grid / timeline highlight window start (QEC cycles)",
    )
    parser.add_argument(
        "--trap-window-end",
        "--async-time-end",
        type=float,
        default=18.0,
        dest="trap_window_end",
        help="Trap-grid / timeline highlight window end (QEC cycles)",
    )
    parser.add_argument(
        "--timeline-xmax",
        type=float,
        default=65.0,
        help=(
            "Timeline x-axis maximum (default 65 for n9f9 sync/async). "
            "Use a negative value for the full circuit time range."
        ),
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
        "--rus-move-shade",
        action="store_false",
        help="Disable RUS move/return shading on the synchronous panel",
    )
    parser.add_argument(
        "--sync-rus-shade-rounds",
        default="1",
        help="Comma-separated 1-based RUS round indices to shade on the sync panel (e.g. 1,5)",
    )
    parser.add_argument(
        "--rus-shade-alpha",
        type=float,
        default=0.22,
        help="Alpha for RUS move/return timeline shading",
    )
    args = parser.parse_args()
    st = args.suptitle.strip()
    sync_rus_rounds = [
        int(x.strip()) for x in args.sync_rus_shade_rounds.split(",") if x.strip()
    ]
    main(
        output_pdf=args.output_pdf,
        timeline_subfigures_pdf=args.timeline_subfigures_pdf,
        figure_width=args.figure_width,
        figure_vertical_stretch=args.figure_vertical_stretch,
        suptitle=st if st else None,
        add_communication_markers=not args.no_communication_markers,
        plot_rus_rounds=args.plot_rus_rounds,
        rus_sync_output_dir=args.rus_sync_output_dir,
        rus_async_output_dir=args.rus_async_output_dir,
        rus_code_distance=args.rus_code_distance,
        rus_movement_overlay=args.rus_movement_overlay,
        trap_window_start=args.trap_window_start,
        trap_window_end=args.trap_window_end,
        timeline_xmax=(None if args.timeline_xmax < 0 else float(args.timeline_xmax)),
        shade_rus_moves=not args.rus_move_shade,
        sync_rus_shade_rounds=sync_rus_rounds,
        rus_shade_alpha=args.rus_shade_alpha,
    )
