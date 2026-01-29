"""
RUS Round Visualization Module

Plots circuit execution grouped by RUS rounds (between RUS:success or RUS:fail markers).
For each round, displays a 2D layout of magic state factories and qubits with:
- Required angles for qubits
- Prepared angles for factories
- TMR success/failure status
- Assignment arrows showing factory-to-qubit connections
"""

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import Dict, List, Tuple, Optional
import os


def extract_rus_rounds(execution_log: List[Tuple]) -> List[List[Tuple]]:
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
        if len(entry) >= 4 and entry[3] == "Barrier":
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
            if not (len(execution_log[j]) >= 4 and execution_log[j][3] == "Barrier")
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
    The move_vecs in move operations contain destination coordinates, which are
    mapped back to qubit IDs using logic_qubit_locations.

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
        if len(entry) < 4:
            continue
        operation = entry[3]
        factory_id = entry[2]
        move_vecs = entry[5] if len(entry) > 5 else None
        qubit_value = entry[4] if len(entry) > 4 else None

        # Look for move operations or injection operations (CNOT/SE + RUS_success/RUS_fail)
        if operation == "move" and move_vecs:
            # move/return_move operations have move_vecs = [from_loc_str, to_loc_str]
            # Extract destination location and find qubit_id
            if len(move_vecs) >= 2:
                to_loc_str = move_vecs[1]  # e.g., "(0,0)"
                # Parse the location string
                try:
                    loc_str = to_loc_str.strip("()")
                    x, y = map(int, loc_str.split(","))
                    qubit_loc = (x, y)
                    if qubit_loc in location_to_qubit:
                        qubit_id = location_to_qubit[qubit_loc]
                        assignments[factory_id] = qubit_id
                except (ValueError, IndexError):
                    pass
        elif operation == "return_move" and move_vecs:
            # move/return_move operations have move_vecs = [from_loc_str, to_loc_str]
            # Extract destination location and find qubit_id
            if len(move_vecs) >= 2:
                to_loc_str = move_vecs[1]  # e.g., "(0,0)"
                # Parse the location string
                try:
                    loc_str = to_loc_str.strip("()")
                    x, y = map(int, loc_str.split(","))
                    return_assignments[factory_id] = (x, y)
                except (ValueError, IndexError):
                    pass
        elif operation in ["RUS_success", "RUS_fail"]:
            # Use the qubit from RUS result marker (qubit_value)
            if qubit_value is not None and factory_id is not None and factory_id >= 0:
                assignments[factory_id] = qubit_value

    return (assignments, return_assignments)


def extract_rus_results(rus_round: List[Tuple]) -> Dict[int, bool]:
    """
    Extract RUS results (success/fail) for each factory in a round.

    Args:
        rus_round: List of execution log entries for one RUS round

    Returns:
        Dict mapping factory_id -> success (True for RUS_success, False for RUS_fail)
    """
    results = {}

    for entry in rus_round:
        if len(entry) < 4:
            continue
        operation = entry[3]
        factory_id = entry[2]

        if operation == "RUS_success":
            results[factory_id] = True
        elif operation == "RUS_fail":
            results[factory_id] = False

    return results


def extract_tmr_failures(rus_round: List[Tuple]) -> set:
    """
    Extract TMR failure information for factories in a round.

    Args:
        rus_round: List of execution log entries for one RUS round

    Returns:
        Set of factory_ids that failed TMR
    """
    tmr_failures = set()

    for entry in rus_round:
        if len(entry) < 4:
            continue
        operation = entry[3]
        factory_id = entry[2]

        if operation == "TMR_fail":
            tmr_failures.add(factory_id)

    return tmr_failures


def extract_angles_from_round(
    rus_round: List[Tuple],
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
        if len(entry) == 5:
            _, _, factory_id, operation, value = entry
        else:
            continue
        if operation == "Rz":
            # Rz operation shows angle preparation
            factory_angles[factory_id] = value
        elif operation in ["CNOT", "SE"]:
            # These operations have a qubit value
            if value is not None:
                qubit = value
                if factory_id in factory_angles:
                    if qubit in qubit_angles:
                        qubit_angles[qubit] = min(
                            qubit_angles[qubit], factory_angles[factory_id]
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


def plot_rus_round(
    round_idx: int,
    rus_round: List[Tuple],
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

    circuit_times = [entry[0] for entry in rus_round if len(entry) >= 2]
    circuit_times.extend([entry[1] for entry in rus_round if len(entry) >= 2])

    if circuit_times:
        duration = max(circuit_times) - min(circuit_times)
    else:
        duration = 0

    # Calculate duration and rus_time
    start_time = rus_round[0][0]
    end_time = rus_round[0][1]
    rus_time = 0
    qubit_cnot_count = {}  # Track CNOT count per qubit
    for entry in rus_round:
        start_time = min(start_time, entry[0])
        end_time = max(end_time, entry[1])
        if len(entry) >= 4 and entry[3] in ["CNOT"]:
            qubit_id = entry[4] if len(entry) > 4 else None
            if qubit_id is not None:
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
    fig, ax = plt.subplots(figsize=(10, 10))

    # Determine axis limits with padding (include both initial and current factory locations)
    all_locations = logic_qubit_locations + magic_state_locations
    all_x = [loc[0] for loc in all_locations]
    all_y = [loc[1] for loc in all_locations]

    x_min, x_max = min(all_x) - 1, max(all_x) + 1
    y_min, y_max = min(all_y) - 1, max(all_y) + 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Color scheme
    color_qubit = "#E8F4F8"
    color_factory = "#FFF9E6"
    color_qubit_border = "#0099CC"
    color_factory_border = "#FF9900"
    color_rus_success = "#00AA00"
    color_rus_fail = "#DD0000"
    color_return_move = "#b64abe"

    box_width = 0.6
    box_height = 0.5

    # Draw logical qubits
    for qubit_id, (x, y) in enumerate(logic_qubit_locations):
        # Qubit box
        box = FancyBboxPatch(
            (x - 0.3, y - 0.3),
            box_width,
            box_height,
            boxstyle="round,pad=0.05",
            facecolor=color_qubit,
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
            fontsize=9,
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
            fontsize=8,
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
            fontsize=9,
            fontweight="bold",
            zorder=11,
        )

        # Angle if available
        if factory_id in factory_angles:
            angle_str = f"θ={factory_angles[factory_id]:.3f}"
        else:
            angle_str = "?"

        ax.text(
            x,
            y + 0.1,
            angle_str,
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            zorder=11,
        )

    # Group arrows by time to assign shades
    # Collect move and return_move operations from the round
    move_times = set()
    return_move_times = set()
    move_info = {}  # (qubit, factory, time) -> move info
    return_move_info = {}  # (factory, time) -> return_move info

    # Create a mapping from qubit location to qubit_id
    location_to_qubit = {loc: idx for idx, loc in enumerate(logic_qubit_locations)}

    for entry in rus_round:
        if len(entry) >= 4 and entry[3] in ["move", "return_move"]:
            start_time = entry[0]
            end_time = entry[1]
            factory_id = entry[2]
            operation = entry[3]
            move_vecs = entry[5] if len(entry) > 5 else None

            if move_vecs and len(move_vecs) >= 2:
                # Extract destination location and find qubit_id
                to_loc_str = move_vecs[1]  # e.g., "(0,0)"
                try:
                    loc_str = to_loc_str.strip("()")
                    x, y = map(int, loc_str.split(","))
                    qubit_loc = (x, y)
                    if qubit_loc in location_to_qubit:
                        qubit_id = location_to_qubit[qubit_loc]
                        if operation == "move":
                            move_times.add((start_time, end_time))
                            move_info[(qubit_id, factory_id)] = (start_time, end_time)
                        else:  # return_move
                            return_move_times.add((start_time, end_time))
                            return_move_info[factory_id] = (
                                start_time,
                                end_time,
                            )
                except (ValueError, IndexError):
                    pass

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

            # Determine if this is a move or return_move operation
            assert factory_id in rus_results
            if rus_results[factory_id]:
                base_color = color_rus_success  # green
            else:
                base_color = color_rus_fail  # red

            # Adjust shade based on movement time
            if (qubit_id, factory_id) in move_info:
                move_time = move_info[(qubit_id, factory_id)]
                time_to_intensity = {
                    t: i / max(len(sorted_times), 1) for i, t in enumerate(sorted_times)
                }
                intensity = time_to_intensity.get(move_time, 0.5)
            else:
                intensity = 0.5

            # Draw arrow with clear direction indication
            arrow = FancyArrowPatch(
                start_point,
                end_point,
                arrowstyle="->",
                mutation_scale=25,
                linewidth=2.5 - 2 * intensity,  # Thicker for later moves
                color=base_color,
                alpha=0.3
                + 0.4 * intensity,  # Lighter for early moves, darker for later
                zorder=5,
                connectionstyle="arc3,rad=0.2",
            )
            ax.add_patch(arrow)

    sorted_times = sorted(return_move_times) if move_times else []
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

        # Determine if this is a move or return_move operation
        assert factory_id in rus_results
        base_color = color_return_move

        # Adjust shade based on movement time
        if factory_id in return_move_info:
            move_time = return_move_info[factory_id]
            sorted_times = sorted(move_times) if move_times else []
            time_to_intensity = {
                t: i / max(len(sorted_times), 1) for i, t in enumerate(sorted_times)
            }
            intensity = time_to_intensity.get(move_time, 0.5)
        else:
            intensity = 0.5

        # Draw arrow with clear direction indication
        arrow = FancyArrowPatch(
            start_point,
            end_point,
            arrowstyle="->",
            mutation_scale=25,
            linewidth=2.5 - 2 * intensity,  # Thicker for later moves
            color=base_color,
            alpha=0.3 + 0.4 * intensity,  # Lighter for early moves, darker for later
            zorder=5,
            connectionstyle="arc3,rad=0.2",
        )
        ax.add_patch(arrow)

    # Labels and title
    ax.set_xlabel("X Position", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y Position", fontsize=10, fontweight="bold")
    ax.set_title(
        f"RUS round {round_idx + 1}: duration {duration}, RUS time: {rus_time}",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.grid(True, alpha=0.2, linestyle="--")

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor=color_qubit,
            edgecolor=color_qubit_border,
            linewidth=2,
            label="$Q_L$",
        ),
        mpatches.Patch(
            facecolor=color_factory,
            edgecolor=color_factory_border,
            linewidth=2,
            label="$Q_F$",
        ),
        mpatches.Patch(
            facecolor="#D4AF99",
            edgecolor=color_factory_border,
            linewidth=2,
            label="TMR:Fail",
        ),
        mpatches.FancyArrow(
            0,
            0,
            1,
            0,
            width=0.1,
            color=color_rus_success,
            alpha=0.7,
            label="move:Success",
        ),
        mpatches.FancyArrow(
            0,
            0,
            1,
            0,
            width=0.1,
            color=color_rus_fail,
            alpha=0.7,
            label="move:Fail",
        ),
        mpatches.FancyArrow(
            0,
            0,
            1,
            0,
            width=0.1,
            color=color_return_move,
            alpha=0.7,
            label="return_move",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize=10,
        frameon=True,
        fancybox=True,
    )

    plt.tight_layout()
    return fig, magic_state_locations


def plot_all_rus_rounds(
    execution_log: List[Tuple],
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
    current_magic_state_locations = magic_state_locations
    for round_idx, rus_round in enumerate(rus_rounds):
        fig, updated_locations = plot_rus_round(
            round_idx,
            rus_round,
            logic_qubit_locations,
            current_magic_state_locations,
        )

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
