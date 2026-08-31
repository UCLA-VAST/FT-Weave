"""
RUS Round Visualization Module

Plots circuit execution grouped by RUS rounds (between RUS:success or RUS:fail markers).
For each round, displays a 2D layout of magic state factories and qubits with:
- Required angles for qubits
- Prepared angles for factories
- TMR success/failure status
- Assignment arrows showing factory-to-qubit connections
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
from typing import Dict, List, Literal, Optional, Tuple
import os
from src.animator.log_view_helpers import (
    normalize_factories,
    resolve_indexed,
    entry_start,
    entry_end,
)

# ---------------------------------------------------------------------------
# Trap-grid style defaults (adjust here)
# ---------------------------------------------------------------------------
TRAP_GRID_BLOCK_SPACING = 0.55
TRAP_GRID_SITE_STEP = 0.10
TRAP_GRID_TRAP_OFFSET = 0.028
TRAP_GRID_TRAP_RADIUS = 0.020
TRAP_GRID_BLOCK_PAD = 0.05

TRAP_GRID_LOGICAL_FACE = "#DDEEFF"
TRAP_GRID_LOGICAL_EDGE = "#2A6F9E"
TRAP_GRID_FACTORY_FACE = "#FFF2CC"
TRAP_GRID_FACTORY_EDGE = "#CF8A1E"
TRAP_GRID_MOVED_FACTORY_COLOR = "#CC7A1D"
TRAP_GRID_DEPART_BOX_ALPHA = 0.12
TRAP_GRID_ARRIVE_OVERLAY_ALPHA = 0.42
# Factory box on logical block shifts right by (left-to-right trap spacing); traps stay put.
TRAP_GRID_FACTORY_BOX_OFFSET_TRAP_SPACINGS = 2.0

TRAP_GRID_AOD_LINEWIDTH = 0.6
TRAP_GRID_AOD_SRC_ALPHA_FACTOR = 0.22
TRAP_GRID_AOD_ALPHA_EARLY = 0.95
TRAP_GRID_AOD_ALPHA_LATE = 0.40

# Movement arcs/labels: earlier batch → darker, later → lighter (matches execution-timeline AOD move purple).
TRAP_GRID_MOVEMENT_COLOR_EARLY = "#7E22CE"
# TRAP_GRID_MOVEMENT_COLOR_LATE =#B274F3 115 249)F3"
TRAP_GRID_MOVEMENT_COLOR_LATE = "#8C5CC0"
TRAP_GRID_ARROW_LINEWIDTH = 1.2
TRAP_GRID_ARROW_MUTATION_SCALE = 14
TRAP_GRID_ARROW_LABEL_FONTSIZE = 9
TRAP_GRID_ARROW_LABEL_OFFSET = 0.06

TRAP_GRID_MOVEMENT_OVERLAY: Literal["both", "arrows_only", "aod_only", "none"] = "both"
TRAP_GRID_FIGSIZE_SCALE_X = 1.15
TRAP_GRID_FIGSIZE_SCALE_Y = 1.05
TRAP_GRID_FIGSIZE_PAD_X = 1.2
TRAP_GRID_FIGSIZE_PAD_Y = 0.8
TRAP_GRID_AXIS_MARGIN = 0.15
TRAP_GRID_SAVE_PAD_INCHES = 0.25

TRAP_GRID_TMR_OVERLAY_ALPHA = 0.75
TRAP_GRID_TMR_LABEL_FONTSIZE = 6
TRAP_GRID_TMR_LABEL_COLOR = "#333333"

# Color mapping for AOD devices (for border colors)
aod_colors = [
    "#FF57579A",
    "#EF7432F4",
    "#F3C82DEB",
    "#F62ED1C2",
    "#C540BE79",
    "#B145D9E5",
    "#4825E8CC",
]


def get_aod_border_color(aod_idx: int) -> str:
    """Get border color for a given AOD index."""
    return aod_colors[aod_idx % len(aod_colors)]


def _normalize_entry_factories(factory_id):
    """Convert factory_id to list for consistency."""
    return normalize_factories(factory_id)


def _resolve_entry_value(values, idx: int):
    """Resolve per-factory value from entry."""
    return resolve_indexed(values, idx)


def _parse_location_xy(raw_loc) -> Optional[Tuple[int, int]]:
    """Parse location from "(x,y)", [x, y], or (x, y) into integer tuple."""
    if raw_loc is None:
        return None

    # String form: "(x,y)" or "x,y"
    if isinstance(raw_loc, str):
        loc_str = raw_loc.strip().strip("()")
        parts = [p.strip() for p in loc_str.split(",")]
        if len(parts) != 2:
            return None
        try:
            return int(float(parts[0])), int(float(parts[1]))
        except ValueError:
            return None

    # Sequence form: [x, y] or (x, y)
    if isinstance(raw_loc, (list, tuple)) and len(raw_loc) == 2:
        try:
            return int(float(raw_loc[0])), int(float(raw_loc[1]))
        except (TypeError, ValueError):
            return None

    return None


def _split_segment_into_rus_cycles(segment: List[dict]) -> List[List[dict]]:
    """
    Split one post-TMR log segment into individual RUS cycles.

    One RUS = move stage + CNOT layer + return stage (in log order).
    """
    entries = [e for e in segment if e.get("operation") != "Barrier"]
    if not entries:
        return []

    cycles: List[List[dict]] = []
    current: List[dict] = []
    after_return = False

    for entry in entries:
        op = entry.get("operation")
        if op == "move":
            if after_return and current:
                cycles.append(current)
                current = []
            after_return = False
            current.append(entry)
        elif op == "CNOT":
            after_return = False
            current.append(entry)
        elif op == "return_move":
            current.append(entry)
            after_return = True
        elif current:
            # RUS_success/fail, SE, etc. between CNOT and return_move
            current.append(entry)

    if current:
        cycles.append(current)
    return cycles


def _extract_macro_rus_segments(execution_log: List[dict]) -> List[List[dict]]:
    """Segments between TMR/RUS barriers (may contain multiple RUS cycles each)."""
    barrier_indices = [
        i
        for i, entry in enumerate(execution_log)
        if entry.get("operation") == "Barrier"
    ]
    if not barrier_indices:
        return [execution_log]

    segments: List[List[dict]] = []
    for i in range(1, len(barrier_indices), 2):
        rus_barrier_idx = barrier_indices[i]
        if i > 1:
            start_idx = barrier_indices[i - 2] + 1
        else:
            start_idx = 0
        end_idx = rus_barrier_idx
        segment = [
            execution_log[j]
            for j in range(start_idx, end_idx)
            if execution_log[j].get("operation") != "Barrier"
        ]
        if segment:
            segments.append(segment)
    return segments


def extract_rus_rounds(execution_log: List[dict]) -> List[List[dict]]:
    """
    Extract individual RUS rounds from the execution log.

    Each RUS is one move stage + one CNOT layer + one return stage. A macro
    segment between TMR barriers (from ``_execute_rus_until_blocked``) may
    contain several such RUS cycles; this function splits them.

    Args:
        execution_log: List of execution log entries

    Returns:
        List of RUS rounds (one list of log entries per move/CNOT/return cycle)
    """
    if not execution_log:
        return []

    rus_rounds: List[List[dict]] = []
    for segment in _extract_macro_rus_segments(execution_log):
        rus_rounds.extend(_split_segment_into_rus_cycles(segment))
    return rus_rounds


# Timeline shading for STAR subfigures (one color per highlighted RUS round).
RUS_ROUND_SHADE_COLORS = [
    "#6BAED6",
    "#FD8D3C",
    "#74C476",
    "#BCBDDC",
    "#E377C2",
    "#A1D99B",
    "#FDAE6B",
    "#9ECAE1",
]


def rus_round_legend_label(round_index_1based: int) -> str:
    """Human-readable legend label, e.g. ``1st RUS``, ``5th RUS``."""
    n = int(round_index_1based)
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} RUS"


def _movement_intervals_in_rus_cycle(cycle: List[dict]) -> List[Tuple[float, float]]:
    """Time spans for ``move`` (before CNOT) and ``return_move`` (after CNOT)."""
    intervals: List[Tuple[float, float]] = []
    for entry in cycle:
        op = entry.get("operation")
        if op not in ("move", "return_move"):
            continue
        t0 = entry_start(entry)
        t1 = entry_end(entry)
        if t1 < t0:
            t0, t1 = t1, t0
        intervals.append((t0, t1))
    return intervals


def build_rus_move_shade_specs(
    execution_log: List[dict],
    round_indices_1based: List[int],
    *,
    alpha: float = 0.22,
) -> List[dict]:
    """Build per-row timeline shade specs for selected RUS rounds.

    Each spec is a dict with keys ``time_start``, ``time_end``, ``color``,
    ``label``, and ``alpha``. Move-before-CNOT and return-move-after-CNOT
    intervals in the same round share one color and one legend entry.
    """
    rounds = extract_rus_rounds(execution_log)
    specs: List[dict] = []
    for round_idx in round_indices_1based:
        ri = int(round_idx)
        if ri < 1 or ri > len(rounds):
            continue
        color = RUS_ROUND_SHADE_COLORS[(ri - 1) % len(RUS_ROUND_SHADE_COLORS)]
        label = rus_round_legend_label(ri)
        for t0, t1 in _movement_intervals_in_rus_cycle(rounds[ri - 1]):
            specs.append(
                {
                    "time_start": t0,
                    "time_end": t1,
                    "color": color,
                    "label": label,
                    "alpha": alpha,
                }
            )
    return specs


def filter_log_by_time_range(
    execution_log: List[dict],
    time_start: float,
    time_end: float,
) -> List[dict]:
    """Keep log entries that overlap ``[time_start, time_end]``."""
    t0 = float(time_start)
    t1 = float(time_end)
    if t1 < t0:
        t0, t1 = t1, t0
    filtered: List[dict] = []
    for entry in execution_log:
        if entry.get("operation") == "Barrier":
            continue
        start_t = entry_start(entry)
        end_t = entry_end(entry)
        if end_t >= t0 and start_t <= t1:
            filtered.append(entry)
    return filtered


def replay_factory_locations_at_time(
    execution_log: List[dict],
    magic_state_locations: List[Tuple[int, int]],
    *,
    time: float,
) -> List[Tuple[int, int]]:
    """Factory grid positions after all completed moves/returns before ``time``."""
    locs = list(magic_state_locations)
    ordered = sorted(execution_log, key=lambda e: (entry_start(e), entry_end(e)))
    for entry in ordered:
        if entry_end(entry) > time:
            continue
        op = entry.get("operation")
        if op not in ("move", "return_move"):
            continue
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        move_vecs = entry.get("move_vecs")
        for idx, factory_id in enumerate(factory_ids):
            if factory_id < 0 or factory_id >= len(locs):
                continue
            factory_move_vecs = _resolve_entry_value(move_vecs, idx)
            if not (factory_move_vecs and len(factory_move_vecs) >= 2):
                continue
            dst = _parse_location_xy(factory_move_vecs[1])
            if dst is not None:
                locs[factory_id] = dst
    return locs


def _save_trap_grid_round_figures(
    *,
    fig_move: Optional[Figure],
    fig_return: Optional[Figure],
    base_path: str,
    name_prefix: str,
) -> None:
    for suffix, fig in (("move", fig_move), ("return", fig_return)):
        if fig is None:
            continue
        round_pdf_path = f"{base_path}/{name_prefix}_{suffix}.pdf"
        round_pdf_dir = os.path.dirname(round_pdf_path)
        if round_pdf_dir:
            os.makedirs(round_pdf_dir, exist_ok=True)
        fig.savefig(
            round_pdf_path,
            bbox_inches="tight",
            pad_inches=TRAP_GRID_SAVE_PAD_INCHES,
        )
        plt.close(fig)


def _save_trap_grid_single_figure(
    *,
    fig: Optional[Figure],
    base_path: str,
    name_prefix: str,
) -> None:
    if fig is None:
        return
    pdf_path = f"{base_path}/{name_prefix}.pdf"
    pdf_dir = os.path.dirname(pdf_path)
    if pdf_dir:
        os.makedirs(pdf_dir, exist_ok=True)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=TRAP_GRID_SAVE_PAD_INCHES,
    )
    plt.close(fig)


def _log_has_tmr_activity(entries: List[dict]) -> bool:
    for entry in entries:
        if entry.get("operation") not in ("SE", "Rz"):
            continue
        if _normalize_entry_factories(entry.get("factories")):
            return True
    return False


def _lookup_factory_rz_angle(
    execution_log: List[dict],
    factory_id: int,
    *,
    not_before: float,
    not_after: float | None = None,
) -> float | None:
    """First Rz preparing angle for ``factory_id`` at/after ``not_before``."""
    best_t = float("inf")
    best_angle: float | None = None
    for entry in execution_log:
        if entry.get("operation") != "Rz":
            continue
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        if factory_id not in factory_ids:
            continue
        start_t = entry_start(entry)
        if start_t < not_before - 1e-9:
            continue
        if not_after is not None and start_t > not_after + 1e-9:
            continue
        idx = factory_ids.index(factory_id)
        raw = _resolve_entry_value(entry.get("targets"), idx)
        if raw is None or start_t >= best_t:
            continue
        best_t = start_t
        best_angle = float(raw)
    return best_angle


def _extract_tmr_factory_angles(
    execution_log: List[dict],
    *,
    time_start: float,
    time_end: float,
) -> dict[int, float]:
    """
    Map factory_id -> Rz angle for TMR blocks overlapping the time window.

    Uses collapsed TMR blocks (SE/Rz sequences) so factories still in the SE
    preparation phase before their Rz event are included.
    """
    from src.animator.circuit_execution_visualization import _collapse_star_tmr_blocks

    w0 = min(float(time_start), float(time_end))
    w1 = max(float(time_start), float(time_end))
    result: dict[int, float] = {}

    for entry in _collapse_star_tmr_blocks(execution_log):
        if entry.get("operation") != "TMR":
            continue
        block_t0 = entry_start(entry)
        block_t1 = entry_end(entry)
        if block_t1 <= w0 or block_t0 > w1:
            continue
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        theta = entry.get("targets")
        if isinstance(theta, list):
            for idx, fid in enumerate(factory_ids):
                if idx < len(theta) and theta[idx] is not None:
                    result[fid] = float(theta[idx])
                else:
                    angle = _lookup_factory_rz_angle(
                        execution_log, fid, not_before=block_t0, not_after=block_t1
                    )
                    if angle is not None:
                        result[fid] = angle
        elif isinstance(theta, (int, float)):
            val = float(theta)
            for fid in factory_ids:
                result[fid] = val
        else:
            for fid in factory_ids:
                angle = _lookup_factory_rz_angle(
                    execution_log, fid, not_before=block_t0, not_after=block_t1
                )
                if angle is not None:
                    result[fid] = angle
    return result


def _draw_tmr_factory_markup(
    ax,
    *,
    factory_angles: dict[int, float],
    factory_locations: list[tuple[int, int]],
    logic_qubit_locations: list[tuple[int, int]],
    block_spacing: float,
    d: int,
    site_step: float,
    trap_offset: float,
    trap_radius: float,
    block_pad: float,
) -> None:
    """Transparent white overlay on factory blocks and ``Rz(θ)`` label."""
    if not factory_angles:
        return
    logic_loc_set = set(logic_qubit_locations)
    half = _block_half_extent(d, site_step, trap_offset, trap_radius, block_pad)
    block_w = 2 * half
    block_h = 2 * half
    factory_box_offset = TRAP_GRID_FACTORY_BOX_OFFSET_TRAP_SPACINGS * trap_offset

    for fid, angle in factory_angles.items():
        if fid < 0 or fid >= len(factory_locations):
            continue
        loc = factory_locations[fid]
        x, y = _display_xy(loc, block_spacing)
        overlay_x = x + factory_box_offset if loc in logic_loc_set else x
        ax.add_patch(
            FancyBboxPatch(
                (overlay_x - half, y - half),
                block_w,
                block_h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=(1.0, 1.0, 1.0, TRAP_GRID_TMR_OVERLAY_ALPHA),
                edgecolor=TRAP_GRID_FACTORY_EDGE,
                linewidth=1.0,
                zorder=25,
            )
        )
        ax.text(
            overlay_x,
            y,
            f"Rz({angle:g})",
            ha="center",
            va="center",
            fontsize=TRAP_GRID_TMR_LABEL_FONTSIZE,
            color=TRAP_GRID_TMR_LABEL_COLOR,
            fontweight="bold",
            zorder=26,
        )


def _draw_time_window_axis(
    ax,
    time_start: float,
    time_end: float,
) -> None:
    """Minimal time axis matching the STAR execution timeline (window ticks only)."""
    t0 = min(float(time_start), float(time_end))
    t1 = max(float(time_start), float(time_end))
    span = t1 - t0
    pad = max(0.05 * span, 0.1) if span > 0 else 0.1
    ax.set_xlim(t0 - pad, t1 + pad)
    ax.set_xticks([t0, t1])
    ax.set_xticklabels([f"{t0:g}", f"{t1:g}"])
    ax.set_yticks([])
    ax.set_xlabel("Time (QEC cycles)", fontsize=10, fontweight="bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")


def plot_trap_grid_time_range(
    execution_log: List[dict],
    logic_qubit_locations: List[Tuple[int, int]],
    magic_state_locations: List[Tuple[int, int]],
    *,
    time_start: float,
    time_end: float,
    base_path: str = "output/rus_time_window",
    title_prefix: str | None = None,
    code_distance: int = 3,
    block_spacing: float = TRAP_GRID_BLOCK_SPACING,
    movement_overlay: Literal[
        "both", "arrows_only", "aod_only", "none"
    ] = TRAP_GRID_MOVEMENT_OVERLAY,
) -> None:
    """
    Trap-grid spatial view for all activity overlapping a time window.

    Used when RUS rounds cannot be isolated (e.g. async execution with overlapping TMR).
    """
    filtered = filter_log_by_time_range(execution_log, time_start, time_end)
    if not filtered:
        print(f"No log events in time range [{time_start}, {time_end}]")
        return

    t0 = min(float(time_start), float(time_end))
    t1 = max(float(time_start), float(time_end))
    prefix = title_prefix or f"t={t0:.1f}–{t1:.1f}"
    initial_locs = replay_factory_locations_at_time(
        execution_log, magic_state_locations, time=t0
    )

    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"Trap-grid time window [{t0}, {t1}]: {len(filtered)} events " f"-> {base_path}"
    )

    tmr_angles = (
        _extract_tmr_factory_angles(execution_log, time_start=t0, time_end=t1)
        if _log_has_tmr_activity(filtered)
        else {}
    )

    fig, _, _ = _plot_rus_round_trap_grid_style(
        0,
        filtered,
        logic_qubit_locations,
        initial_locs,
        code_distance=code_distance,
        block_spacing=block_spacing,
        movement_overlay=movement_overlay,
        title_prefix=prefix,
        combined_time_window=True,
        tmr_factory_angles=tmr_angles,
        time_window=(t0, t1),
    )
    safe_name = f"window_{t0:g}_{t1:g}".replace(".", "p")
    _save_trap_grid_single_figure(
        fig=fig,
        base_path=base_path,
        name_prefix=safe_name,
    )


def draw_trap_grid_time_range_on_axes(
    ax: Axes,
    execution_log: List[dict],
    logic_qubit_locations: List[Tuple[int, int]],
    magic_state_locations: List[Tuple[int, int]],
    *,
    time_start: float,
    time_end: float,
    title: str | None = None,
    code_distance: int = 3,
    block_spacing: float = TRAP_GRID_BLOCK_SPACING,
    movement_overlay: Literal[
        "both", "arrows_only", "aod_only", "none"
    ] = TRAP_GRID_MOVEMENT_OVERLAY,
    axis_margin: float | None = None,
    arrow_label_offset: float | None = None,
) -> bool:
    """Draw trap-grid movement for ``[time_start, time_end]`` on an existing axes."""
    filtered = filter_log_by_time_range(execution_log, time_start, time_end)
    if not filtered:
        return False

    t0 = min(float(time_start), float(time_end))
    t1 = max(float(time_start), float(time_end))
    initial_locs = replay_factory_locations_at_time(
        execution_log, magic_state_locations, time=t0
    )
    tmr_angles = (
        _extract_tmr_factory_angles(execution_log, time_start=t0, time_end=t1)
        if _log_has_tmr_activity(filtered)
        else {}
    )
    _plot_rus_round_trap_grid_style(
        0,
        filtered,
        logic_qubit_locations,
        initial_locs,
        code_distance=code_distance,
        block_spacing=block_spacing,
        movement_overlay=movement_overlay,
        title_prefix=title,
        combined_time_window=True,
        tmr_factory_angles=tmr_angles,
        time_window=(t0, t1),
        target_ax=ax,
        axis_margin=axis_margin,
        arrow_label_offset=arrow_label_offset,
    )
    return True


def extract_assignments_from_round(
    rus_round: list, logic_qubit_locations: List[Tuple[int, int]]
) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]:
    """
    Extract factory-to-qubit assignments from a RUS round.

    Uses 'move' operations to identify which factory is assigned to which qubit.
        Each move_vec is ``[source_grid, dest_grid]`` (strings or nested coords). For
        factory ``move`` events, the factory moves onto the qubit grid: qubit site is
        the **destination** (index 1). For ``return_move``, index 0 is the qubit site
        the factory leaves and index 1 is factory home.

    Args:
        rus_round: List of execution log entries for one RUS round
        logic_qubit_locations: List of (x, y) coordinates for logical qubits

    Returns:
        Dict mapping factory_id -> qubit_id for assignments in this round.
        Each factory is assigned to at most one qubit, but a qubit may have
        multiple factories (for retry attempts).
    """
    assignments = {}
    return_assignments = {}

    # Create a mapping from qubit location to qubit_id
    location_to_qubit = {loc: idx for idx, loc in enumerate(logic_qubit_locations)}

    for entry in rus_round:
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")

        # Look for move operations or injection operations (CNOT/SE + RUS_success/RUS_fail)
        for idx, factory_id in enumerate(factory_ids):
            if operation == "move" and move_vecs:
                # move/return_move operations have move_vecs = [[from, to], [from, to], ...]
                factory_move_vecs = _resolve_entry_value(move_vecs, idx)
                if factory_move_vecs and len(factory_move_vecs) >= 2:
                    qubit_loc = _parse_location_xy(factory_move_vecs[1])
                    if qubit_loc in location_to_qubit:
                        qubit_id = location_to_qubit[qubit_loc]
                        assignments[factory_id] = qubit_id
            elif operation == "return_move" and move_vecs:
                # move/return_move operations have move_vecs = [[from, to], [from, to], ...]
                factory_move_vecs = _resolve_entry_value(move_vecs, idx)
                if factory_move_vecs and len(factory_move_vecs) >= 2:
                    return_loc = _parse_location_xy(factory_move_vecs[1])
                    if return_loc is not None:
                        return_assignments[factory_id] = return_loc
            elif operation in ["RUS_success", "RUS_fail"]:
                # Use the qubit from RUS result marker (qubit_value)
                factory_qubit = _resolve_entry_value(value, idx)
                if (
                    factory_qubit is not None
                    and factory_id is not None
                    and factory_id >= 0
                ):
                    assignments[factory_id] = factory_qubit

    return (assignments, return_assignments)


def extract_rus_results(rus_round: List[dict]) -> Dict[int, bool]:
    """
    Extract RUS results (success/fail) for each factory in a round.

    Args:
        rus_round: List of execution log entries for one RUS round

    Returns:
        Dict mapping factory_id -> success (True for RUS_success, False for RUS_fail)
    """
    results = {}

    for entry in rus_round:
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")

        if operation == "RUS_success":
            for factory_id in factory_ids:
                results[factory_id] = True
        elif operation == "RUS_fail":
            for factory_id in factory_ids:
                results[factory_id] = False

    return results


def extract_rus_failed_qubits(rus_round: List[dict]) -> set[int]:
    """Extract qubit IDs that encountered RUS_fail in this round."""
    failed_qubits = set()

    for entry in rus_round:
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")
        value = entry.get("targets")

        if operation != "RUS_fail":
            continue

        for idx, factory_id in enumerate(factory_ids):
            qubit_value = _resolve_entry_value(value, idx)
            if isinstance(qubit_value, int):
                failed_qubits.add(qubit_value)
            elif factory_id is not None:
                # Fallback for logs where qubit id is not carried in value.
                failed_qubits.add(factory_id)

    return failed_qubits


def extract_tmr_failures(rus_round: List[dict]) -> set:
    """
    Extract TMR failure information for factories in a round.

    Args:
        rus_round: List of execution log entries for one RUS round

    Returns:
        Set of factory_ids that failed TMR
    """
    tmr_failures = set()

    for entry in rus_round:
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")

        if operation == "TMR_fail":
            for factory_id in factory_ids:
                tmr_failures.add(factory_id)

    return tmr_failures


def extract_angles_from_round(
    rus_round: List[dict],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Extract angle information from a RUS round.

    Args:
        rus_round: List of execution log entries for one RUS round

    Returns:
        Tuple of (qubit_angles_dict, factory_angles_dict)
        - qubit_angles_dict: Maps qubit_id -> required angle
        - factory_angles_dict: Maps factory_id -> prepared angle
    """
    qubit_angles = {}
    factory_angles = {}

    for entry in rus_round:
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")
        value = entry.get("targets")

        if operation == "Rz":
            # Rz operation shows angle preparation
            for idx, factory_id in enumerate(factory_ids):
                factory_value = _resolve_entry_value(value, idx)
                factory_angles[factory_id] = factory_value
        elif operation in ["CNOT", "SE"]:
            # These operations have a qubit value
            if value is not None:
                for idx, factory_id in enumerate(factory_ids):
                    factory_value = _resolve_entry_value(value, idx)
                    if isinstance(factory_value, int):
                        qubit = factory_value
                        if factory_id in factory_angles:
                            if qubit in qubit_angles:
                                qubit_angles[qubit] = min(
                                    qubit_angles[qubit],
                                    factory_angles[factory_id],
                                )
                            else:
                                qubit_angles[qubit] = factory_angles[factory_id]
    return qubit_angles, factory_angles


def get_box_edge_point(center, box_size, direction_vector):
    """
    Calculate the intersection point between a line and a box boundary.

    Args:
        center: (x, y) center of the box
        box_size: (width, height) of the box
        direction_vector: (dx, dy) direction from center outward

    Returns:
        (x, y) point on the box boundary in the direction of direction_vector
    """
    dx, dy = direction_vector
    width, height = box_size

    # Normalize direction
    length = (dx**2 + dy**2) ** 0.5
    if length == 0:
        return center

    dx_norm = dx / length
    dy_norm = dy / length

    # Find which boundary we hit first
    half_width = width / 2
    half_height = height / 2

    # Calculate distances to each boundary
    if dx_norm != 0:
        t_x = half_width / abs(dx_norm)
    else:
        t_x = float("inf")

    if dy_norm != 0:
        t_y = half_height / abs(dy_norm)
    else:
        t_y = float("inf")

    # Take the minimum t value (we hit that boundary first)
    t = min(t_x, t_y)

    return (center[0] + dx_norm * t, center[1] + dy_norm * t)


def _batch_intensity(time_key, sorted_times) -> float:
    """Map batch order to [0, 1] intensity with strong separation."""
    if not sorted_times:
        return 0.5
    denom = max(len(sorted_times) - 1, 1)
    time_to_intensity = {t: i / denom for i, t in enumerate(sorted_times)}
    return time_to_intensity.get(time_key, 0.5)


def _arrow_style_from_intensity(intensity: float) -> tuple[float, float]:
    """Return (linewidth, alpha) with inverse thickness/alpha relationship."""
    # Slight nonlinear boost preserves clear batch separation.
    boosted = intensity**1.35
    linewidth = 4.2 - 3.2 * boosted
    alpha = 0.18 + 0.82 * boosted
    return linewidth, alpha


def plot_rus_round(
    round_idx: int,
    rus_round: List[dict],
    logic_qubit_locations: List[Tuple[int, int]],
    magic_state_locations: List[Tuple[int, int]],
    qubit_trackers: Optional[Dict] = None,
) -> Tuple[Optional[Figure], List[Tuple[int, int]]]:
    """
    Plot a single RUS round with 2D layout.

    Args:
        round_idx: Index of this RUS round
        rus_round: List of execution log entries for this round
        logic_qubit_locations: List of (x, y) coordinates for logical qubits
        magic_state_locations: List of (x, y) coordinates for magic state factories
        qubit_trackers: Optional dict of QubitAngleTracker objects for detailed angle info

    Returns:
        Tuple of (Figure object or None, updated magic_state_locations for next round)
    """
    if not rus_round:
        return (None, magic_state_locations)

    circuit_times = [entry_start(entry) for entry in rus_round]
    circuit_times.extend([entry_end(entry) for entry in rus_round])

    if circuit_times:
        duration = max(circuit_times) - min(circuit_times)
    else:
        duration = 0

    # Calculate duration and rus_time
    start_time = entry_start(rus_round[0])
    end_time = entry_end(rus_round[0])
    rus_time = 0
    qubit_cnot_count = {}  # Track CNOT count per qubit
    for entry in rus_round:
        evt_start = entry_start(entry)
        evt_end = entry_end(entry)
        start_time = min(start_time, evt_start)
        end_time = max(end_time, evt_end)
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")
        value = entry.get("targets")

        if operation == "CNOT":
            for idx, _factory_id in enumerate(factory_ids):
                qubit_id = _resolve_entry_value(value, idx)
                if isinstance(qubit_id, int):
                    qubit_cnot_count[qubit_id] = qubit_cnot_count.get(qubit_id, 0) + 1

    duration = end_time - start_time

    if qubit_cnot_count:
        rus_time = max(qubit_cnot_count.values())

    # Extract information from the round
    assignments, return_assignments = extract_assignments_from_round(
        rus_round, logic_qubit_locations
    )
    rus_results = extract_rus_results(rus_round)
    tmr_failures = extract_tmr_failures(rus_round)
    qubit_angles, factory_angles = extract_angles_from_round(rus_round)

    # Create figure
    if len(logic_qubit_locations) < 30:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig, ax = plt.subplots(figsize=(11, 13))

    # Determine axis limits with tighter padding.
    all_locations = logic_qubit_locations + magic_state_locations
    all_x = [loc[0] for loc in all_locations]
    all_y = [loc[1] for loc in all_locations]

    axis_padding = 0.45
    x_min, x_max = min(all_x) - axis_padding, max(all_x) + axis_padding
    y_min, y_max = min(all_y) - axis_padding, max(all_y) + axis_padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Color scheme
    color_qubit = "#E8F4F8"
    color_qubit_fail = "#9EC3CF"
    color_factory = "#FFF9E6"
    color_qubit_border = "#0099CC"
    color_factory_border = "#FF9900"
    color_forward_move = "#8B5CF6"
    color_return_move = "#16A34A"

    box_width = 0.6
    box_height = 0.5
    rus_failed_qubits = extract_rus_failed_qubits(rus_round)
    for factory_id, is_success in rus_results.items():
        if not is_success and factory_id in assignments:
            rus_failed_qubits.add(assignments[factory_id])

    # Draw logical qubits
    for qubit_id, (x, y) in enumerate(logic_qubit_locations):
        qubit_facecolor = (
            color_qubit_fail if qubit_id in rus_failed_qubits else color_qubit
        )

        # Qubit box
        box = FancyBboxPatch(
            (x - 0.3, y - 0.3),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            facecolor=qubit_facecolor,
            edgecolor=color_qubit_border,
            linewidth=2,
            zorder=10,
        )
        ax.add_patch(box)

        # Qubit label
        ax.text(
            x,
            y - 0.15,
            f"Q{qubit_id}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=11,
        )

        # Angle if available
        if qubit_id in qubit_angles:
            angle_str = f"θ={qubit_angles[qubit_id]:.3f}"
        elif qubit_trackers and qubit_id in qubit_trackers:
            angle_str = f"θ={qubit_trackers[qubit_id].target_angle:.3f}"
        else:
            angle_str = "?"

        ax.text(
            x,
            y + 0.1,
            angle_str,
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            zorder=11,
        )

    # Draw magic state factories
    for factory_id, (x, y) in enumerate(magic_state_locations):
        # Determine factory color based on TMR failure
        factory_facecolor = color_factory
        if factory_id in tmr_failures:
            # Use darker color for TMR failures
            factory_facecolor = "#D4AF99"  # Darker factory color

        # Factory box
        box = FancyBboxPatch(
            (x - 0.3, y - 0.3),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            facecolor=factory_facecolor,
            edgecolor=color_factory_border,
            linewidth=2,
            zorder=10,
        )
        ax.add_patch(box)

        # Factory label
        ax.text(
            x,
            y - 0.15,
            f"F{factory_id}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=11,
        )

        # Angle if available
        if factory_id in factory_angles:
            angle_str = f"θ={factory_angles[factory_id]:.4f}"
        else:
            angle_str = "?"

        ax.text(
            x,
            y + 0.1,
            angle_str,
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            zorder=11,
        )

    # Group arrows by time to assign shades
    # Collect move and return_move operations from the round
    move_times = set()
    return_move_times = set()
    move_info = {}  # (qubit, factory, time) -> move info
    return_move_info = {}  # (factory, time) -> return_move info
    move_aod_info = {}  # (qubit_id, factory_id) -> aod_assignment
    return_move_aod_info = {}  # factory_id -> aod_assignment

    # Create a mapping from qubit location to qubit_id
    location_to_qubit = {loc: idx for idx, loc in enumerate(logic_qubit_locations)}

    for entry in rus_round:
        start_time = entry_start(entry)
        end_time = entry_end(entry)
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        move_vecs = entry.get("move_vecs")

        if operation in ["move", "return_move"]:
            for idx, factory_id in enumerate(factory_ids):
                factory_move_vecs = _resolve_entry_value(move_vecs, idx)
                factory_aod = _resolve_entry_value(aod_assignment, idx)

                if not (factory_move_vecs and len(factory_move_vecs) >= 2):
                    continue

                qubit_loc = _parse_location_xy(
                    factory_move_vecs[1]
                    if operation == "move"
                    else factory_move_vecs[0]
                )
                if qubit_loc in location_to_qubit:
                    qubit_id = location_to_qubit[qubit_loc]
                    if operation == "move":
                        move_times.add((start_time, end_time))
                        move_info[(qubit_id, factory_id)] = (start_time, end_time)
                        move_aod_info[(qubit_id, factory_id)] = factory_aod
                    else:  # return_move
                        return_move_times.add((start_time, end_time))
                        return_move_info[factory_id] = (
                            start_time,
                            end_time,
                        )
                        return_move_aod_info[factory_id] = factory_aod

    # Draw assignment arrows
    sorted_times = sorted(move_times) if move_times else []
    for factory_id, qubit_id in assignments.items():
        if qubit_id < len(logic_qubit_locations) and factory_id < len(
            magic_state_locations
        ):
            x_q, y_q = logic_qubit_locations[qubit_id]
            x_f, y_f = magic_state_locations[factory_id]
            magic_state_locations[factory_id] = (x_q, y_q)

            # Calculate direction vector from factory to qubit
            dx = x_q - x_f
            dy = y_q - y_f

            # Get box edge points
            box_size = (0.6, 0.5)
            start_point = get_box_edge_point((x_f, y_f), box_size, (dx, dy))
            end_point = get_box_edge_point((x_q, y_q), box_size, (-dx, -dy))

            # Forward move color is independent of RUS pass/fail.
            base_color = color_forward_move

            # Adjust arrow prominence based on routing batch time.
            if (qubit_id, factory_id) in move_info:
                move_time = move_info[(qubit_id, factory_id)]
                intensity = _batch_intensity(move_time, sorted_times)
            else:
                intensity = 0.5
            linewidth, alpha = _arrow_style_from_intensity(intensity)

            # Draw arrow with AOD color outline
            arrow = FancyArrowPatch(
                start_point,
                end_point,
                arrowstyle="->",
                mutation_scale=25,
                linewidth=linewidth,
                color=base_color,
                # edgecolor=aod_edge_color,
                alpha=alpha,
                zorder=5,
                connectionstyle="arc3,rad=0.2",
            )
            ax.add_patch(arrow)

    sorted_times = sorted(return_move_times) if return_move_times else []
    for factory_id, (new_x, new_y) in return_assignments.items():
        old_x, old_y = magic_state_locations[factory_id]
        magic_state_locations[factory_id] = (new_x, new_y)
        # Calculate direction vector from factory to qubit
        dx = new_x - old_x
        dy = new_y - old_y

        # Get box edge points
        box_size = (0.6, 0.5)
        start_point = get_box_edge_point((old_x, old_y), box_size, (dx, dy))
        end_point = get_box_edge_point((new_x, new_y), box_size, (-dx, -dy))

        base_color = color_return_move

        # Adjust arrow prominence based on routing batch time.
        if factory_id in return_move_info:
            move_time = return_move_info[factory_id]
            intensity = _batch_intensity(move_time, sorted_times)
        else:
            intensity = 0.5
        linewidth, alpha = _arrow_style_from_intensity(intensity)

        # Draw arrow with AOD color outline
        arrow = FancyArrowPatch(
            start_point,
            end_point,
            arrowstyle="->",
            mutation_scale=25,
            linewidth=linewidth,
            color=base_color,
            # edgecolor=aod_edge_color,
            alpha=alpha,
            zorder=5,
            connectionstyle="arc3,rad=0.2",
        )
        ax.add_patch(arrow)

    # Labels and title
    ax.set_xlabel("X Position", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y Position", fontsize=13, fontweight="bold")
    ax.set_title(
        f"RUS round {round_idx + 1}: duration {duration}, RUS time: {rus_time}",
        fontsize=17,
        fontweight="bold",
    )
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Grid
    ax.grid(True, alpha=0.2, linestyle="--")

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor=color_qubit,
            edgecolor=color_qubit_border,
            linewidth=2,
            label="Logical Qubit",
        ),
        mpatches.Patch(
            facecolor=color_factory,
            edgecolor=color_factory_border,
            linewidth=2,
            label="Factory",
        ),
        mpatches.Patch(
            facecolor="#808080",
            edgecolor="black",
            linewidth=2,
            label="Fail Operation",
        ),
        mpatches.FancyArrow(
            0,
            0,
            1,
            0,
            width=0.1,
            color=color_forward_move,
            alpha=0.7,
            label="Forward Move",
        ),
        mpatches.FancyArrow(
            0,
            0,
            1,
            0,
            width=0.1,
            color=color_return_move,
            alpha=0.7,
            label="Return Move",
        ),
    ]
    # Widen the axes only; keep legend in a narrow right margin (same overall figsize).
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.92)
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        bbox_transform=fig.transFigure,
        ncol=1,
        fontsize=11,
        frameon=True,
        fancybox=True,
    )
    return fig, magic_state_locations


def _build_round_motion_data(
    rus_round: List[dict], logic_qubit_locations: List[Tuple[int, int]]
) -> tuple[
    dict[int, tuple[int, int]], dict[int, tuple[int, int]], list[dict], list[dict]
]:
    """Extract move/return targets and ordered AOD movement entries for one round."""
    location_to_qubit = {loc: idx for idx, loc in enumerate(logic_qubit_locations)}
    move_targets: dict[int, tuple[int, int]] = {}
    return_targets: dict[int, tuple[int, int]] = {}
    move_steps: list[dict] = []
    return_steps: list[dict] = []
    move_batch_id = 0
    return_batch_id = 0

    for entry in rus_round:
        op = entry.get("operation")
        if op not in ("move", "return_move"):
            continue
        start_t = entry_start(entry)
        if op == "move":
            move_batch_id += 1
            batch_id = move_batch_id
        else:
            return_batch_id += 1
            batch_id = return_batch_id
        factory_ids = _normalize_entry_factories(entry.get("factories"))
        move_vecs = entry.get("move_vecs")
        aod_assignment = entry.get("aod_assignment")
        for idx, factory_id in enumerate(factory_ids):
            factory_move_vecs = _resolve_entry_value(move_vecs, idx)
            if not (factory_move_vecs and len(factory_move_vecs) >= 2):
                continue
            src = _parse_location_xy(factory_move_vecs[0])
            dst = _parse_location_xy(factory_move_vecs[1])
            if src is None or dst is None:
                continue
            aod_idx = _resolve_entry_value(aod_assignment, idx)
            if not isinstance(aod_idx, int):
                aod_idx = 0
            if op == "move":
                move_targets[factory_id] = dst
                qid = location_to_qubit.get(dst)
                move_steps.append(
                    {
                        "time": start_t,
                        "batch_id": batch_id,
                        "factory_id": factory_id,
                        "src": src,
                        "dst": dst,
                        "aod_idx": aod_idx,
                        "qubit_id": qid,
                    }
                )
            else:
                return_targets[factory_id] = dst
                return_steps.append(
                    {
                        "time": start_t,
                        "batch_id": batch_id,
                        "factory_id": factory_id,
                        "src": src,
                        "dst": dst,
                        "aod_idx": aod_idx,
                        "qubit_id": location_to_qubit.get(src),
                    }
                )
    move_steps.sort(key=lambda x: (x["time"], x["factory_id"]))
    return_steps.sort(key=lambda x: (x["time"], x["factory_id"]))
    return move_targets, return_targets, move_steps, return_steps


def _expand_decomposed_handoff_returns(
    return_steps: list[dict],
    factory_locations: list[tuple[int, int]],
    magic_state_locations: list[tuple[int, int]],
) -> list[dict]:
    """
    Append paired handoff hops for decomposed returns.

    When a return lands on a cell occupied by another factory not at home,
    add a synthetic hop for that occupant to its architecture home (e.g.
    (8,4)->(7,4) paired with occupant at (7,4)->(5,4)).
    """
    expanded = list(return_steps)
    existing = {(s["factory_id"], s["src"], s["dst"]) for s in return_steps}

    for step in return_steps:
        dst = step["dst"]
        returning_fid = step["factory_id"]
        for fid, loc in enumerate(factory_locations):
            if fid == returning_fid or loc != dst:
                continue
            home = magic_state_locations[fid]
            if home == loc:
                continue
            key = (fid, loc, home)
            if key in existing:
                continue
            expanded.append(
                {
                    "time": step["time"],
                    "batch_id": step["batch_id"],
                    "factory_id": fid,
                    "src": loc,
                    "dst": home,
                    "aod_idx": step["aod_idx"],
                    "qubit_id": None,
                }
            )
            existing.add(key)

    expanded.sort(key=lambda x: (x["time"], x["batch_id"], x["factory_id"]))
    return expanded


def _round_time_bounds(rus_round: List[dict]) -> tuple[float, float]:
    if not rus_round:
        return 0.0, 0.0
    starts = [entry_start(e) for e in rus_round]
    ends = [entry_end(e) for e in rus_round]
    return min(starts), max(ends)


def _resolve_step_src_loc(
    step: dict,
    factory_locations: list[tuple[int, int]] | None,
) -> tuple[int, int]:
    """Movement source cell: actual factory position when it differs from the log."""
    src_loc = step["src"]
    if factory_locations is None:
        return src_loc
    fid = step["factory_id"]
    if 0 <= fid < len(factory_locations):
        actual = factory_locations[fid]
        if actual != src_loc:
            return actual
    return src_loc


def _movement_arrow_rad(
    src_loc: tuple[int, int],
    dst_loc: tuple[int, int],
    *,
    factory_locations: list[tuple[int, int]],
    logic_qubit_locations: list[tuple[int, int]],
) -> float:
    """Bend arrows on the same row so they do not pass through idle factory cells."""
    if src_loc[1] != dst_loc[1]:
        return 0.08
    x_lo, x_hi = sorted((src_loc[0], dst_loc[0]))
    if x_hi - x_lo <= 1:
        return 0.08
    for loc in factory_locations:
        if loc[1] != src_loc[1]:
            continue
        if x_lo < loc[0] < x_hi and loc not in (src_loc, dst_loc):
            return -0.16
    for loc in logic_qubit_locations:
        if loc[1] != src_loc[1]:
            continue
        if x_lo < loc[0] < x_hi and loc not in (src_loc, dst_loc):
            return -0.16
    return 0.08


def _factories_by_logical_loc(
    factory_locations: list[tuple[int, int]],
    logic_loc_set: set[tuple[int, int]],
) -> dict[tuple[int, int], list[int]]:
    from collections import defaultdict

    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fid, loc in enumerate(factory_locations):
        if loc in logic_loc_set:
            out[loc].append(fid)
    return out


def _display_xy(loc: tuple[int, int], block_spacing: float) -> tuple[float, float]:
    """Scale architecture grid coordinates for plotting."""
    return float(loc[0]) * block_spacing, float(loc[1]) * block_spacing


def _trap_grid_layout(d: int) -> tuple[float, float, float, float]:
    """Return site_step, trap_offset, trap_radius, block_pad for a d×d block."""
    del d  # geometry is uniform; d only affects site count when drawing
    return (
        TRAP_GRID_SITE_STEP,
        TRAP_GRID_TRAP_OFFSET,
        TRAP_GRID_TRAP_RADIUS,
        TRAP_GRID_BLOCK_PAD,
    )


def _trap_grid_aod_intensity(order: int, total: int) -> float:
    if total <= 1:
        return TRAP_GRID_AOD_ALPHA_EARLY
    t = order / (total - 1)
    return TRAP_GRID_AOD_ALPHA_EARLY - t * (
        TRAP_GRID_AOD_ALPHA_EARLY - TRAP_GRID_AOD_ALPHA_LATE
    )


def _movement_arrow_color(
    order: int,
    total: int,
    *,
    color_early: str = TRAP_GRID_MOVEMENT_COLOR_EARLY,
    color_late: str = TRAP_GRID_MOVEMENT_COLOR_LATE,
) -> tuple[float, float, float]:
    """Earlier movement → darker shade; later → lighter shade."""
    if total <= 1:
        t = 0.0
    else:
        t = order / (total - 1)
    early = mcolors.to_rgb(color_early)
    late = mcolors.to_rgb(color_late)
    return tuple(early[i] + t * (late[i] - early[i]) for i in range(3))


def _block_half_extent(
    d: int, site_step: float, trap_offset: float, trap_radius: float, block_pad: float
) -> float:
    inner = ((d - 1) / 2) * site_step + trap_offset + trap_radius
    return block_pad + inner


def _trap_grid_scene_extent(
    *,
    logic_qubit_locations: List[Tuple[int, int]],
    factory_locations: List[Tuple[int, int]],
    arrival_factory_locs: set[tuple[int, int]],
    steps: list[dict],
    block_spacing: float,
    half: float,
    factory_box_offset: float,
    axis_margin: float | None = None,
    arrow_label_offset: float | None = None,
) -> tuple[float, float, float, float]:
    """Axis limits covering all blocks, movement endpoints, overlays, and labels."""
    raw_locs: list[tuple[int, int]] = list(logic_qubit_locations) + list(
        factory_locations
    )
    raw_locs.extend(arrival_factory_locs)
    for step in steps:
        raw_locs.append(step["src"])
        raw_locs.append(step["dst"])

    if not raw_locs:
        return -1.0, 1.0, -1.0, 1.0

    margin = TRAP_GRID_AXIS_MARGIN if axis_margin is None else float(axis_margin)
    arrow_off = (
        TRAP_GRID_ARROW_LABEL_OFFSET
        if arrow_label_offset is None
        else float(arrow_label_offset)
    )
    pad = half + factory_box_offset + margin + arrow_off
    xs: list[float] = []
    ys: list[float] = []
    for loc in raw_locs:
        x, y = _display_xy(loc, block_spacing)
        xs.extend((x - half, x + half + factory_box_offset))
        ys.extend((y - half, y + half))
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def _trap_grid_scene_extent_resolved(
    *,
    logic_qubit_locations: List[Tuple[int, int]],
    factory_locations: List[Tuple[int, int]],
    arrival_factory_locs: set[tuple[int, int]],
    steps: list[dict],
    block_spacing: float,
    half: float,
    factory_box_offset: float,
    axis_margin: float | None = None,
    arrow_label_offset: float | None = None,
) -> tuple[float, float, float, float]:
    """Extent helper that uses resolved per-factory movement sources."""
    resolved_steps = [
        {**step, "src": _resolve_step_src_loc(step, factory_locations)}
        for step in steps
    ]
    return _trap_grid_scene_extent(
        logic_qubit_locations=logic_qubit_locations,
        factory_locations=factory_locations,
        arrival_factory_locs=arrival_factory_locs,
        steps=resolved_steps,
        block_spacing=block_spacing,
        half=half,
        factory_box_offset=factory_box_offset,
        axis_margin=axis_margin,
        arrow_label_offset=arrow_label_offset,
    )


def _figsize_from_extent(
    x_lo: float, x_hi: float, y_lo: float, y_hi: float
) -> tuple[float, float]:
    span_x = max(x_hi - x_lo, 0.5)
    span_y = max(y_hi - y_lo, 0.5)
    w = max(4.0, span_x * TRAP_GRID_FIGSIZE_SCALE_X + TRAP_GRID_FIGSIZE_PAD_X)
    h = max(3.5, span_y * TRAP_GRID_FIGSIZE_SCALE_Y + TRAP_GRID_FIGSIZE_PAD_Y)
    data_aspect = span_x / span_y
    fig_aspect = w / h
    if fig_aspect < data_aspect:
        h = w / data_aspect
    elif fig_aspect > data_aspect:
        w = h * data_aspect
    return w, h


def _draw_trap_grid_block(
    ax,
    center_x: float,
    center_y: float,
    *,
    d: int,
    facecolor: str,
    edgecolor: str,
    is_logical: bool,
    site_step: float,
    trap_offset: float,
    trap_radius: float,
    block_pad: float,
    moved_factory_color: str,
    box_alpha: float = 1.0,
    moved_into_right: Optional[set[tuple[int, int]]] = None,
    trap_alpha: float = 1.0,
    factory_overlay_alpha: Optional[float] = None,
    factory_overlay_offset_x: float = 0.0,
    factory_left_only: bool = False,
):
    """Render one block as d×d sites; each site has a left/right trap pair."""
    moved_into_right = moved_into_right or set()
    half = _block_half_extent(d, site_step, trap_offset, trap_radius, block_pad)
    block_w = 2 * half
    block_h = 2 * half
    face = mcolors.to_rgba(facecolor, box_alpha)
    block = FancyBboxPatch(
        (center_x - half, center_y - half),
        block_w,
        block_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=face,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=box_alpha,
        zorder=2,
    )
    ax.add_patch(block)

    # Factory box overlay only (shifted); trap markers stay on the logical block.
    if factory_overlay_alpha is not None:
        overlay_cx = center_x + factory_overlay_offset_x
        overlay = FancyBboxPatch(
            (overlay_cx - half, center_y - half),
            block_w,
            block_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=mcolors.to_rgba(TRAP_GRID_FACTORY_FACE, factory_overlay_alpha),
            edgecolor=TRAP_GRID_FACTORY_EDGE,
            linewidth=1.0,
            alpha=factory_overlay_alpha,
            zorder=3,
        )
        ax.add_patch(overlay)

    left_fill = "#111111"
    factory_fill = moved_factory_color
    for r in range(d):
        for c in range(d):
            site_x = center_x + (c - (d - 1) / 2) * site_step
            site_y = center_y + (r - (d - 1) / 2) * site_step
            left_xy = (site_x - trap_offset, site_y)
            right_xy = (site_x + trap_offset, site_y)
            if is_logical:
                ax.add_patch(
                    plt.Circle(
                        left_xy,
                        trap_radius,
                        facecolor=left_fill,
                        edgecolor="#000000",
                        linewidth=0.6,
                        alpha=trap_alpha,
                        zorder=5,
                    )
                )
            right_site = (r, c)
            if right_site in moved_into_right:
                ax.add_patch(
                    plt.Circle(
                        right_xy,
                        trap_radius,
                        facecolor=factory_fill,
                        edgecolor="#8A4F00",
                        linewidth=0.6,
                        alpha=trap_alpha,
                        zorder=6,
                    )
                )
            elif is_logical:
                ax.add_patch(
                    plt.Circle(
                        right_xy,
                        trap_radius,
                        facecolor="#FFFFFF",
                        edgecolor="#333333",
                        linewidth=0.5,
                        alpha=trap_alpha,
                        zorder=5,
                    )
                )
            else:
                # Factory at home: left trap occupied; right reserved (empty).
                # Factory at logical (CNOT): both traps occupied on shifted block.
                ax.add_patch(
                    plt.Circle(
                        left_xy,
                        trap_radius,
                        facecolor=factory_fill,
                        edgecolor="#8A4F00",
                        linewidth=0.5,
                        alpha=trap_alpha,
                        zorder=5,
                    )
                )
                if factory_left_only:
                    ax.add_patch(
                        plt.Circle(
                            right_xy,
                            trap_radius,
                            facecolor="#FFFFFF",
                            edgecolor="#333333",
                            linewidth=0.5,
                            alpha=trap_alpha,
                            zorder=5,
                        )
                    )
                else:
                    ax.add_patch(
                        plt.Circle(
                            right_xy,
                            trap_radius,
                            facecolor=factory_fill,
                            edgecolor="#8A4F00",
                            linewidth=0.5,
                            alpha=trap_alpha,
                            zorder=5,
                        )
                    )


def _draw_aod_grid_at_centers(
    ax,
    centers: set[tuple[int, int]],
    *,
    d: int,
    site_step: float,
    trap_offset: float,
    block_spacing: float,
    color: str,
    alpha: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    linewidth: float = TRAP_GRID_AOD_LINEWIDTH,
    zorder: int = 7,
):
    """Draw d horizontal + d vertical AOD lines for each block center."""
    drawn_h: set[float] = set()
    drawn_v: set[float] = set()
    for center in centers:
        bx, by = _display_xy(center, block_spacing)
        for r in range(d):
            y = by + (r - (d - 1) / 2) * site_step
            yk = round(y, 5)
            if yk in drawn_h:
                continue
            drawn_h.add(yk)
            ax.plot(
                [x_lo, x_hi],
                [y, y],
                color=color,
                linestyle="--",
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        for c in range(d):
            x = bx + (c - (d - 1) / 2) * site_step
            xk = round(x, 5)
            if xk in drawn_v:
                continue
            drawn_v.add(xk)
            ax.plot(
                [x, x],
                [y_lo, y_hi],
                color=color,
                linestyle="--",
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )


def _draw_aod_grid(
    ax,
    steps: list[dict],
    *,
    d: int,
    site_step: float,
    trap_offset: float,
    block_spacing: float,
    color: str,
    alpha: float,
    factory_locations: list[tuple[int, int]] | None = None,
    src_alpha_factor: float = TRAP_GRID_AOD_SRC_ALPHA_FACTOR,
    linewidth: float = TRAP_GRID_AOD_LINEWIDTH,
    zorder: int = 7,
):
    """
    Draw AOD grid lines; source block grids are more transparent than destination.
    """
    if not steps:
        return

    src_centers = {_resolve_step_src_loc(step, factory_locations) for step in steps}
    dst_centers = {step["dst"] for step in steps}
    all_centers = src_centers | dst_centers

    display_pts = [_display_xy(c, block_spacing) for c in all_centers]
    half = _block_half_extent(d, site_step, trap_offset, 0.02, 0.05) + 0.12
    x_lo = min(p[0] for p in display_pts) - half
    x_hi = max(p[0] for p in display_pts) + half
    y_lo = min(p[1] for p in display_pts) - half
    y_hi = max(p[1] for p in display_pts) + half

    _draw_aod_grid_at_centers(
        ax,
        src_centers,
        d=d,
        site_step=site_step,
        trap_offset=trap_offset,
        block_spacing=block_spacing,
        color=color,
        alpha=alpha * src_alpha_factor,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        linewidth=linewidth,
        zorder=zorder,
    )
    _draw_aod_grid_at_centers(
        ax,
        dst_centers,
        d=d,
        site_step=site_step,
        trap_offset=trap_offset,
        block_spacing=block_spacing,
        color=color,
        alpha=alpha,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        linewidth=linewidth,
        zorder=zorder + 1,
    )


def _draw_movement_arrows(
    ax,
    steps: list[dict],
    *,
    block_spacing: float,
    trap_offset: float,
    phase: str,
    factory_locations: list[tuple[int, int]] | None = None,
    logic_qubit_locations: list[tuple[int, int]] | None = None,
    movement_color_early: str = TRAP_GRID_MOVEMENT_COLOR_EARLY,
    movement_color_late: str = TRAP_GRID_MOVEMENT_COLOR_LATE,
    zorder: int = 8,
    linewidth: float = TRAP_GRID_ARROW_LINEWIDTH,
    mutation_scale: float = TRAP_GRID_ARROW_MUTATION_SCALE,
):
    """Draw arrows per movement batch; same batch shares label and color shade."""
    ordered = sorted(steps, key=lambda s: (s["batch_id"], s["factory_id"]))
    batch_ids = sorted({s["batch_id"] for s in ordered})
    batch_order = {bid: i for i, bid in enumerate(batch_ids)}
    n_batches = len(batch_ids)
    logic_locs = logic_qubit_locations or []
    factory_locs = factory_locations or []

    for step in ordered:
        src_loc = _resolve_step_src_loc(step, factory_locations)
        dst_loc = step["dst"]
        src = _display_xy(src_loc, block_spacing)
        dst = _display_xy(dst_loc, block_spacing)
        start = src
        end = dst
        batch_idx = batch_order[step["batch_id"]]
        arrow_color = _movement_arrow_color(
            batch_idx,
            n_batches,
            color_early=movement_color_early,
            color_late=movement_color_late,
        )
        rad = _movement_arrow_rad(
            src_loc,
            dst_loc,
            factory_locations=factory_locs,
            logic_qubit_locations=logic_locs,
        )
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=arrow_color,
            alpha=1.0,
            zorder=zorder,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)
        ax.text(
            start[0],
            start[1] - TRAP_GRID_ARROW_LABEL_OFFSET,
            str(step["batch_id"]),
            ha="center",
            va="bottom",
            fontsize=TRAP_GRID_ARROW_LABEL_FONTSIZE,
            color=arrow_color,
            fontweight="bold",
            zorder=zorder + 1,
        )


def _plot_rus_round_trap_grid_style(
    round_idx: int,
    rus_round: List[dict],
    logic_qubit_locations: List[Tuple[int, int]],
    magic_state_locations: List[Tuple[int, int]],
    *,
    code_distance: int = 3,
    block_spacing: float = TRAP_GRID_BLOCK_SPACING,
    movement_overlay: Literal[
        "both", "arrows_only", "aod_only", "none"
    ] = TRAP_GRID_MOVEMENT_OVERLAY,
    movement_color_early: str = TRAP_GRID_MOVEMENT_COLOR_EARLY,
    movement_color_late: str = TRAP_GRID_MOVEMENT_COLOR_LATE,
    title_prefix: str | None = None,
    architecture_magic_state_locations: List[Tuple[int, int]] | None = None,
    combined_time_window: bool = False,
    tmr_factory_angles: dict[int, float] | None = None,
    time_window: tuple[float, float] | None = None,
    target_ax: Axes | None = None,
    axis_margin: float | None = None,
    arrow_label_offset: float | None = None,
) -> tuple[Optional[Figure], Optional[Figure], List[Tuple[int, int]]]:
    """
    New trap-grid style:
    - no concrete x/y axes
    - each block has d*d sites and 2 traps per site
    - move / return rendered as separate figures with AOD overlays.
    - ``block_spacing`` controls distance between block centers in the figure.
    - ``movement_overlay``: ``both``, ``arrows_only``, ``aod_only``, or ``none``.
    """
    if not rus_round:
        return None, None, magic_state_locations

    d = max(int(code_distance), 1)
    move_targets, return_targets, move_steps, return_steps = _build_round_motion_data(
        rus_round, logic_qubit_locations
    )

    # Placement states for this round.
    before_move_locations = list(magic_state_locations)
    after_move_locations = list(magic_state_locations)
    for fid, dst in move_targets.items():
        if 0 <= fid < len(after_move_locations):
            after_move_locations[fid] = dst
    arch_magic = architecture_magic_state_locations or magic_state_locations
    return_steps = _expand_decomposed_handoff_returns(
        return_steps, after_move_locations, arch_magic
    )
    for step in return_steps:
        return_targets[step["factory_id"]] = step["dst"]
    after_return_locations = list(after_move_locations)
    for fid, dst in return_targets.items():
        if 0 <= fid < len(after_return_locations):
            after_return_locations[fid] = dst

    return_destination_locs = {step["dst"] for step in return_steps}
    return_final_locs = set(return_targets.values())

    site_step, trap_offset, trap_radius, block_pad = _trap_grid_layout(d)
    half = _block_half_extent(d, site_step, trap_offset, trap_radius, block_pad)
    show_aod = movement_overlay in ("both", "aod_only")
    show_arrows = movement_overlay in ("both", "arrows_only")

    logical_face = TRAP_GRID_LOGICAL_FACE
    logical_edge = TRAP_GRID_LOGICAL_EDGE
    factory_face = TRAP_GRID_FACTORY_FACE
    factory_edge = TRAP_GRID_FACTORY_EDGE
    moved_factory_color = TRAP_GRID_MOVED_FACTORY_COLOR
    depart_box_alpha = TRAP_GRID_DEPART_BOX_ALPHA
    arrive_overlay_alpha = TRAP_GRID_ARRIVE_OVERLAY_ALPHA

    def _aod_groups(steps: list[dict]) -> list[tuple[int, list[dict]]]:
        from collections import defaultdict

        grouped: dict[int, list[dict]] = defaultdict(list)
        for step in steps:
            grouped[step["aod_idx"]].append(step)
        return sorted(
            grouped.items(),
            key=lambda item: min(s["time"] for s in item[1]),
        )

    def _all_sites() -> set[tuple[int, int]]:
        return {(r, c) for r in range(d) for c in range(d)}

    def _draw_scene(
        ax,
        *,
        factory_locations: list[tuple[int, int]],
        steps: list[dict],
        departing_factory_ids: set[int],
        arrival_qubit_ids: set[int],
        arrival_factory_locs: set[tuple[int, int]],
        phase: str,
        return_steps: list[dict] | None = None,
        move_departing_factory_ids: set[int] | None = None,
        return_departing_factory_ids: set[int] | None = None,
        tmr_factory_angles: dict[int, float] | None = None,
    ):
        move_depart = (
            move_departing_factory_ids
            if move_departing_factory_ids is not None
            else departing_factory_ids
        )
        return_depart = (
            return_departing_factory_ids
            if return_departing_factory_ids is not None
            else departing_factory_ids
        )
        logic_loc_set = set(logic_qubit_locations)
        factories_on_logical = _factories_by_logical_loc(
            factory_locations, logic_loc_set
        )

        # Logical blocks: shade never changes; factory marks right traps only.
        for qid, (gx, gy) in enumerate(logic_qubit_locations):
            loc = (gx, gy)
            x, y = _display_xy(loc, block_spacing)
            co_located = factories_on_logical.get(loc, [])
            is_move_target = phase in ("move", "combined") and qid in arrival_qubit_ids
            is_move_source = phase in ("move", "combined") and any(
                fid in move_depart for fid in co_located
            )
            is_return_source = phase in ("return", "combined") and any(
                fid in return_depart for fid in co_located
            )
            is_carried_factory = (
                phase in ("move", "combined")
                and co_located
                and not is_move_target
                and not any(fid in move_depart for fid in co_located)
            )
            moved_right: set[tuple[int, int]] = set()
            overlay = None
            trap_alpha = 1.0
            overlay_offset_x = 0.0
            if is_move_target or is_carried_factory:
                moved_right = _all_sites()
                overlay = arrive_overlay_alpha
                overlay_offset_x = (
                    TRAP_GRID_FACTORY_BOX_OFFSET_TRAP_SPACINGS * trap_offset
                )
            elif is_move_source or is_return_source:
                moved_right = _all_sites()
                overlay = depart_box_alpha
                trap_alpha = 0.22
                overlay_offset_x = (
                    TRAP_GRID_FACTORY_BOX_OFFSET_TRAP_SPACINGS * trap_offset
                )
            _draw_trap_grid_block(
                ax,
                x,
                y,
                d=d,
                facecolor=logical_face,
                edgecolor=logical_edge,
                is_logical=True,
                site_step=site_step,
                trap_offset=trap_offset,
                trap_radius=trap_radius,
                block_pad=block_pad,
                moved_factory_color=moved_factory_color,
                box_alpha=1.0,
                moved_into_right=moved_right,
                trap_alpha=trap_alpha,
                factory_overlay_alpha=overlay,
                factory_overlay_offset_x=overlay_offset_x,
            )

        # Standalone factory blocks (not co-located with a logical block).
        from collections import defaultdict

        loc_to_fids: dict[tuple[int, int], list[int]] = defaultdict(list)
        for fid, (gx, gy) in enumerate(factory_locations):
            loc = (gx, gy)
            if loc in logic_loc_set:
                continue
            loc_to_fids[loc].append(fid)

        for loc, fids in loc_to_fids.items():
            x, y = _display_xy(loc, block_spacing)
            staying = [f for f in fids if f not in departing_factory_ids]
            leaving = [f for f in fids if f in departing_factory_ids]
            if staying:
                _draw_trap_grid_block(
                    ax,
                    x,
                    y,
                    d=d,
                    facecolor=factory_face,
                    edgecolor=factory_edge,
                    is_logical=False,
                    site_step=site_step,
                    trap_offset=trap_offset,
                    trap_radius=trap_radius,
                    block_pad=block_pad,
                    moved_factory_color=moved_factory_color,
                    box_alpha=1.0,
                    trap_alpha=1.0,
                    factory_left_only=True,
                )
            elif leaving:
                _draw_trap_grid_block(
                    ax,
                    x,
                    y,
                    d=d,
                    facecolor=factory_face,
                    edgecolor=factory_edge,
                    is_logical=False,
                    site_step=site_step,
                    trap_offset=trap_offset,
                    trap_radius=trap_radius,
                    block_pad=block_pad,
                    moved_factory_color=moved_factory_color,
                    box_alpha=depart_box_alpha,
                    trap_alpha=0.18,
                    factory_left_only=True,
                )

        # Return arrivals at factory home (and intermediate hops): solid at final dst.
        if phase in ("return", "combined"):
            for loc in arrival_factory_locs:
                if loc in logic_loc_set:
                    continue
                ax_x, ax_y = _display_xy(loc, block_spacing)
                is_final = loc in return_final_locs
                _draw_trap_grid_block(
                    ax,
                    ax_x,
                    ax_y,
                    d=d,
                    facecolor=factory_face,
                    edgecolor=factory_edge,
                    is_logical=False,
                    site_step=site_step,
                    trap_offset=trap_offset,
                    trap_radius=trap_radius,
                    block_pad=block_pad,
                    moved_factory_color=moved_factory_color,
                    box_alpha=1.0 if is_final else arrive_overlay_alpha,
                    trap_alpha=1.0 if is_final else 0.45,
                    factory_left_only=True,
                )

        if tmr_factory_angles:
            _draw_tmr_factory_markup(
                ax,
                factory_angles=tmr_factory_angles,
                factory_locations=factory_locations,
                logic_qubit_locations=logic_qubit_locations,
                block_spacing=block_spacing,
                d=d,
                site_step=site_step,
                trap_offset=trap_offset,
                trap_radius=trap_radius,
                block_pad=block_pad,
            )

        factory_box_offset = TRAP_GRID_FACTORY_BOX_OFFSET_TRAP_SPACINGS * trap_offset
        extent_steps = steps
        if phase == "combined" and return_steps:
            extent_steps = steps + return_steps
        x_lo, x_hi, y_lo, y_hi = _trap_grid_scene_extent_resolved(
            logic_qubit_locations=logic_qubit_locations,
            factory_locations=factory_locations,
            arrival_factory_locs=arrival_factory_locs,
            steps=extent_steps,
            block_spacing=block_spacing,
            half=half,
            factory_box_offset=factory_box_offset,
            axis_margin=axis_margin,
            arrow_label_offset=arrow_label_offset,
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        aod_groups = _aod_groups(steps)
        for order, (aod_idx, group) in enumerate(aod_groups):
            color = get_aod_border_color(aod_idx)
            alpha = _trap_grid_aod_intensity(order, len(aod_groups))
            if show_aod:
                _draw_aod_grid(
                    ax,
                    group,
                    d=d,
                    site_step=site_step,
                    trap_offset=trap_offset,
                    block_spacing=block_spacing,
                    color=color,
                    alpha=alpha,
                    linewidth=TRAP_GRID_AOD_LINEWIDTH,
                    factory_locations=factory_locations,
                )
        if phase == "combined" and return_steps:
            ret_aod_groups = _aod_groups(return_steps)
            base = len(aod_groups)
            for order, (aod_idx, group) in enumerate(ret_aod_groups):
                color = get_aod_border_color(aod_idx)
                alpha = _trap_grid_aod_intensity(
                    base + order, base + len(ret_aod_groups)
                )
                if show_aod:
                    _draw_aod_grid(
                        ax,
                        group,
                        d=d,
                        site_step=site_step,
                        trap_offset=trap_offset,
                        block_spacing=block_spacing,
                        color=color,
                        alpha=alpha,
                        linewidth=TRAP_GRID_AOD_LINEWIDTH,
                        factory_locations=factory_locations,
                    )
        if show_arrows:
            _draw_movement_arrows(
                ax,
                steps,
                block_spacing=block_spacing,
                trap_offset=trap_offset,
                phase=phase,
                factory_locations=factory_locations,
                logic_qubit_locations=logic_qubit_locations,
                movement_color_early=movement_color_early,
                movement_color_late=movement_color_late,
            )
            if phase == "combined" and return_steps:
                _draw_movement_arrows(
                    ax,
                    return_steps,
                    block_spacing=block_spacing,
                    trap_offset=trap_offset,
                    phase=phase,
                    factory_locations=factory_locations,
                    logic_qubit_locations=logic_qubit_locations,
                    movement_color_early=movement_color_early,
                    movement_color_late=movement_color_late,
                )

    move_departing = {s["factory_id"] for s in move_steps}
    move_arrival_qubits = {
        s["qubit_id"] for s in move_steps if isinstance(s.get("qubit_id"), int)
    }
    return_departing = {s["factory_id"] for s in return_steps}
    return_arrival_qubits = {
        s["qubit_id"] for s in return_steps if isinstance(s.get("qubit_id"), int)
    }

    move_arrival_factory_locs: set[tuple[int, int]] = set()
    return_arrival_factory_locs = return_destination_locs

    factory_box_offset = TRAP_GRID_FACTORY_BOX_OFFSET_TRAP_SPACINGS * trap_offset

    combined_extent = _trap_grid_scene_extent_resolved(
        logic_qubit_locations=logic_qubit_locations,
        factory_locations=before_move_locations,
        arrival_factory_locs=return_arrival_factory_locs,
        steps=move_steps + return_steps,
        block_spacing=block_spacing,
        half=half,
        factory_box_offset=factory_box_offset,
    )
    move_extent = (
        combined_extent
        if combined_time_window
        else _trap_grid_scene_extent_resolved(
            logic_qubit_locations=logic_qubit_locations,
            factory_locations=before_move_locations,
            arrival_factory_locs=move_arrival_factory_locs,
            steps=move_steps,
            block_spacing=block_spacing,
            half=half,
            factory_box_offset=factory_box_offset,
        )
    )
    return_extent = _trap_grid_scene_extent_resolved(
        logic_qubit_locations=logic_qubit_locations,
        factory_locations=after_move_locations,
        arrival_factory_locs=return_arrival_factory_locs,
        steps=return_steps,
        block_spacing=block_spacing,
        half=half,
        factory_box_offset=factory_box_offset,
    )

    if target_ax is not None:
        ax_move = target_ax
        ax_time = None
        fig_move = None
    elif combined_time_window and time_window is not None:
        w, h = _figsize_from_extent(*move_extent)
        fig_move = plt.figure(figsize=(w, h + 0.65))
        gs = fig_move.add_gridspec(2, 1, height_ratios=[10, 1], hspace=0.12)
        ax_move = fig_move.add_subplot(gs[0])
        ax_time = fig_move.add_subplot(gs[1])
    else:
        fig_move, ax_move = plt.subplots(figsize=_figsize_from_extent(*move_extent))
        ax_time = None

    _draw_scene(
        ax_move,
        factory_locations=before_move_locations,
        steps=move_steps,
        departing_factory_ids=(
            move_departing | return_departing
            if combined_time_window
            else move_departing
        ),
        arrival_qubit_ids=move_arrival_qubits,
        arrival_factory_locs=(
            return_arrival_factory_locs
            if combined_time_window
            else move_arrival_factory_locs
        ),
        phase="combined" if combined_time_window else "move",
        return_steps=return_steps if combined_time_window else None,
        move_departing_factory_ids=move_departing,
        return_departing_factory_ids=return_departing,
        tmr_factory_angles=tmr_factory_angles if combined_time_window else None,
    )
    move_title = (
        title_prefix
        if combined_time_window
        else (
            f"{title_prefix} - move before CNOT"
            if title_prefix
            else f"RUS round {round_idx + 1} - move before CNOT"
        )
    )
    if move_title:
        ax_move.set_title(move_title, fontsize=14)

    if (
        target_ax is None
        and combined_time_window
        and time_window is not None
        and ax_time is not None
    ):
        _draw_time_window_axis(ax_time, time_window[0], time_window[1])

    if target_ax is not None or combined_time_window:
        return fig_move, None, after_return_locations

    fig_return, ax_return = plt.subplots(figsize=_figsize_from_extent(*return_extent))
    _draw_scene(
        ax_return,
        factory_locations=after_move_locations,
        steps=return_steps,
        departing_factory_ids=return_departing,
        arrival_qubit_ids=return_arrival_qubits,
        arrival_factory_locs=return_arrival_factory_locs,
        phase="return",
    )
    return_title = (
        f"{title_prefix} - return move after CNOT"
        if title_prefix
        else f"RUS round {round_idx + 1} - return move after CNOT"
    )
    ax_return.set_title(return_title, fontsize=14)

    return fig_move, fig_return, after_return_locations


def plot_all_rus_rounds(
    execution_log: List[dict],
    logic_qubit_locations: List[Tuple[int, int]],
    magic_state_locations: List[Tuple[int, int]],
    base_path: str = "output",
    style_variant: str = "classic",
    code_distance: int = 3,
    block_spacing: float = TRAP_GRID_BLOCK_SPACING,
    movement_overlay: Literal[
        "both", "arrows_only", "aod_only", "none"
    ] = TRAP_GRID_MOVEMENT_OVERLAY,
    movement_color_early: str = TRAP_GRID_MOVEMENT_COLOR_EARLY,
    movement_color_late: str = TRAP_GRID_MOVEMENT_COLOR_LATE,
) -> None:
    """
    Plot all RUS rounds and save to separate PDF files.

    Creates one PDF file per RUS round, showing the 2D layout with factory-to-qubit
    assignments, angles, and success/failure status.

    Args:
        execution_log: List of execution log entries from factory_angle_execution
        logic_qubit_locations: List of (x, y) coordinates for logical qubits
        magic_state_locations: List of (x, y) coordinates for magic state factories
        pdf_path: Path template where PDFs will be saved (e.g., "output/rus_rounds_detailed_col_based/round")
    """

    # Extract RUS rounds
    rus_rounds = extract_rus_rounds(execution_log)

    if not rus_rounds:
        print("No RUS rounds found in execution log")
        return

    print(f"Found {len(rus_rounds)} RUS rounds")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get the base name from pdf_path (without extension)

    # Save each round as a separate PDF
    # print("init magic_state_locations")
    # print(magic_state_locations)
    arch_magic = list(magic_state_locations)
    current_locs = list(magic_state_locations)
    for round_idx, rus_round in enumerate(rus_rounds):
        before_locs = list(current_locs)
        if style_variant == "trap_grid":
            fig_move, fig_return, updated_locations = _plot_rus_round_trap_grid_style(
                round_idx,
                rus_round,
                logic_qubit_locations,
                before_locs,
                code_distance=code_distance,
                block_spacing=block_spacing,
                movement_overlay=movement_overlay,
                movement_color_early=movement_color_early,
                movement_color_late=movement_color_late,
                architecture_magic_state_locations=arch_magic,
            )
            current_locs = updated_locations

            _save_trap_grid_round_figures(
                fig_move=fig_move,
                fig_return=fig_return,
                base_path=base_path,
                name_prefix=f"round{round_idx + 1}",
            )
            for suffix in ("move", "return"):
                if (suffix == "move" and fig_move is None) or (
                    suffix == "return" and fig_return is None
                ):
                    continue
                print(
                    f"  Round {round_idx + 1} ({suffix}): {len(rus_round)} events, "
                    f"saved to {base_path}/round{round_idx + 1}_{suffix}.pdf"
                )
        else:
            fig, updated_locations = plot_rus_round(
                round_idx,
                rus_round,
                logic_qubit_locations,
                before_locs,
            )
            current_locs = updated_locations
            if fig is not None:
                # Generate individual PDF filename
                round_pdf_path = f"{base_path}/round{round_idx + 1}.pdf"

                # Create subdirectory if needed
                round_pdf_dir = os.path.dirname(round_pdf_path)
                if round_pdf_dir:
                    os.makedirs(round_pdf_dir, exist_ok=True)

                # Save the figure
                fig.savefig(round_pdf_path, bbox_inches="tight")
                plt.close(fig)
                print(
                    f"  Round {round_idx + 1}: {len(rus_round)} events, "
                    f"{len(extract_assignments_from_round(rus_round, logic_qubit_locations))} assignments, "
                    f"saved to {round_pdf_path}"
                )

    print(f"\nRUS rounds visualization saved to: {output_dir}")
