"""
RUS Round Visualization Module

Plots circuit execution grouped by RUS rounds (between RUS:success or RUS:fail markers).
For each round, displays a 2D layout of magic state factories and qubits with:
- Required angles for qubits
- Prepared angles for factories
- TMR success/failure status
- Assignment arrows showing factory-to-qubit connections
"""

from attr import Factory
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import Dict, List, Tuple, Optional
import os
from src.animator.log_view_helpers import (
    normalize_factories,
    resolve_indexed,
    entry_start,
    entry_end,
)

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


def extract_rus_rounds(execution_log: List[dict]) -> List[List[dict]]:
    """
    Extract RUS rounds from execution log.

    A RUS round is bounded by barriers in the log. The pattern is:
    - Barrier at index 0 (TMR barrier)
    - Barrier at index 1 (RUS barrier)
    - Barrier at index 2 (TMR barrier)
    - Barrier at index 3 (RUS barrier)
    - ...

    Each RUS round consists of all operations between odd-indexed barriers (RUS barriers).

    Args:
        execution_log: List of execution log entries (start_time, end_time, factory_id, operation, qubit, ...)

    Returns:
        List of RUS rounds, where each round is a list of log entries (excluding barrier entries)
    """
    if not execution_log:
        return []

    # Find all barrier positions
    barrier_indices = []
    for i, entry in enumerate(execution_log):
        operation = entry.get("operation")
        if operation == "Barrier":
            barrier_indices.append(i)

    if not barrier_indices:
        # No barriers found, return entire log as one round
        return [execution_log]

    rus_rounds = []

    # Iterate through barrier pairs to extract RUS rounds
    # RUS rounds are bounded by odd-indexed barriers (indices 1, 3, 5, ...)
    for i in range(1, len(barrier_indices), 2):
        rus_barrier_idx = barrier_indices[i]

        # Find the previous boundary (either previous barrier or start of log)
        if i > 1:
            prev_barrier_idx = barrier_indices[i - 2]
            start_idx = prev_barrier_idx + 1
        else:
            start_idx = 0

        # Collect all entries up to and including operations at the RUS barrier time
        end_idx = rus_barrier_idx

        # Extract entries between barriers (excluding barrier entries)
        round_entries = [
            execution_log[j]
            for j in range(start_idx, end_idx)
            if execution_log[j].get("operation") != "Barrier"
        ]

        if round_entries:
            rus_rounds.append(round_entries)
    return rus_rounds


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


def plot_all_rus_rounds(
    execution_log: List[dict],
    logic_qubit_locations: List[Tuple[int, int]],
    magic_state_locations: List[Tuple[int, int]],
    base_path: str = "output",
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
    current_magic_state_locations = magic_state_locations
    for round_idx, rus_round in enumerate(rus_rounds):
        fig, updated_locations = plot_rus_round(
            round_idx,
            rus_round,
            logic_qubit_locations,
            current_magic_state_locations,
        )
        # print("updated_locations")
        # print(updated_locations)
        # input()

        # Update locations for next round
        current_magic_state_locations = updated_locations

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
