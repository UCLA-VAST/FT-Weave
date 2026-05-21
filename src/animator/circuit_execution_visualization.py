import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from src.animator.log_view_helpers import (
    normalize_factories,
    resolve_indexed,
    resolve_move_vecs,
    entry_start,
    entry_end,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
_FIG_FONT_SIZE = 18
_TCULT_EXEC_FIGURE_WIDTH = 7
_BOX_HEIGHT = 0.7
_BOX_Y_OFFSET = _BOX_HEIGHT / 2
_BOX_ALPHA = 0.9
_AOD_MOVE_PURPLE_SHADES = [
    "#b64abe",
    "#D8B4FE",
    "#C084FC",
    "#A855F7",
    "#7E22CE",
]
_AOD_RETURN_MOVE_GREEN_SHADES = [
    "#9edc6f",
    "#d9f7be",
    "#b7eb8f",
    "#73d13d",
    "#389e0d",
]
# Color mapping for operations
color_map = {
    "TUM": "#9ed76c",
    "SE": "#4681a9",
    "SE_q": "#4681a9",
    "S": "#a98546",
    "CNOT": "#e74c3c",
    "Rz": "#e7ab3c",
    "move": "#b64abe",
    "return_move": "#9edc6f",
    "Barrier": "#000000",
    "RUS_success": "#E01414",
    "RUS_fail": "#DDA413",
    "TMR_fail": "#007E15",
    "CNOT(T)": "#e74c3c",
    "T": "#9ed76c",
    "H": "#5dade2",
    "SE_stage_1": "#6E9EBE",
    "SE_stage_2": "#4F64C2",
}

_TCULT_OP_ALIASES = {
    "SE Stage 1": "SE_stage_1",
    "SE Stage 2": "SE_stage_2",
    "Move": "move",
    "Return Move": "return_move",
}

# Color mapping for AOD devices (for border colors)
aod_colors = [
    "#FF57579A",
    "#EF7432F4",
    "#F3C82DEB",
    "#python-color-picker",
    "#C540BE79",
    "#B145D9E5",
    "#4825E8CC",
]
ylim = 93
_BOX_BORDER_WIDTH = 1
_AXIS_LABEL_FONT_SIZE = 24
_TITLE_FONT_SIZE = 28
_LEGEND_FONT_SIZE = 20
# STAR stacked subfigures: same size for suptitle, per-panel title, and x-axis label.
_STAR_SUBFIG_HEADING_FONT_SIZE = 18
# Inches scaling for total figure height (larger => taller lanes / taller boxes on screen).
_STAR_SUBFIG_DEFAULT_VERTICAL_STRETCH = 1.8


def get_aod_border_color(aod_idx: int) -> str:
    """Get border color for a given AOD index."""
    return aod_colors[aod_idx % len(aod_colors)]


def _resolve_factory_value(values, idx: int):
    return resolve_indexed(values, idx)


def _resolve_factory_move_vecs(move_vecs, idx: int):
    return resolve_move_vecs(move_vecs, idx)


def _extract_qubits_from_targets(targets) -> list[int]:
    if targets is None:
        return []
    qubits: list[int] = []
    for target in targets:
        if isinstance(target, int):
            qubits.append(target)
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            q0, q1 = target
            if isinstance(q0, int):
                qubits.append(q0)
            if isinstance(q1, int):
                qubits.append(q1)
    return sorted(set(qubits))


def _extract_qubits_from_value(value) -> list[int]:
    """Extract logical qubit ids from STAR execution entry values."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    ):
        return [value[0], value[1]]
    if isinstance(value, list):
        qubits: list[int] = []
        for item in value:
            qubits.extend(_extract_qubits_from_value(item))
        return sorted(set(qubits))
    return []


def _canonical_tcult_operation(operation: str) -> str:
    return _TCULT_OP_ALIASES.get(operation, operation)


def _star_box_label(operation: str, value=None, move_vecs=None) -> str:
    if operation == "Rz":
        return f"RZ\nθ:{value}"
    if operation == "S":
        return f"S\nθ:{value}"
    if operation == "SE":
        return "SE"
    if operation == "SE_q":
        return "SE"
    if operation == "CNOT":
        if value is not None:
            return f"CNOT\nQ{value}"
        return "CNOT"
    if operation in {"move", "return_move"}:
        if move_vecs and len(move_vecs) >= 2:
            return f"{move_vecs[0]}\n$\\downarrow$\n{move_vecs[1]}"
        return ""
    if value is not None:
        return f"{operation}\nQ{value}"
    return operation


def _movement_color_for_aod(operation: str, aod_assignment) -> str:
    if operation == "return_move":
        shades = _AOD_RETURN_MOVE_GREEN_SHADES
    else:
        shades = _AOD_MOVE_PURPLE_SHADES

    if not isinstance(aod_assignment, int) or aod_assignment < 0:
        return shades[0]
    return shades[aod_assignment % len(shades)]


def _resolve_star_row_layout(
    execution_log,
    n_factories: int,
    n_logical_qubits: int | None,
    show_logical_qubits: bool,
):
    """Return (resolved_n_logical, row_names, y_pos, n_rows) for horizontal STAR plots."""
    if not show_logical_qubits:
        return None, None, None, n_factories
    nl = n_logical_qubits
    if nl is None:
        max_qubit = -1
        for entry in execution_log:
            value = entry.get("targets")
            qubits = _extract_qubits_from_value(value)
            if qubits:
                max_qubit = max(max_qubit, max(qubits))
        nl = max_qubit + 1 if max_qubit >= 0 else 0
    row_names = [f"q{i}" for i in range(nl)] + [f"f{i}" for i in range(n_factories)]
    y_pos = {name: idx for idx, name in enumerate(row_names)}
    return nl, row_names, y_pos, len(row_names)


def _star_execution_legend_handles():
    return [
        mpatches.Patch(facecolor=color_map["SE"], edgecolor="black", label="SE"),
        mpatches.Patch(facecolor=color_map["CNOT"], edgecolor="black", label="CNOT"),
        mpatches.Patch(facecolor=color_map["S"], edgecolor="black", label="S gate"),
        mpatches.Patch(facecolor=color_map["move"], edgecolor="black", label="Move"),
        mpatches.Patch(
            facecolor=color_map["return_move"],
            edgecolor="black",
            label="Return Move",
        ),
        mpatches.Patch(
            facecolor=color_map["RUS_success"],
            edgecolor="black",
            label="RUS:succsss",
        ),
        mpatches.Patch(
            facecolor=color_map["RUS_fail"],
            edgecolor="black",
            label="RUS:fail",
        ),
        mpatches.Patch(
            facecolor=color_map["TMR_fail"],
            edgecolor="black",
            label="TMR:fail",
        ),
    ]


def _plot_circuit_execution_on_ax(
    ax,
    execution_log,
    n_factories: int,
    n_logical_qubits: int | None,
    *,
    show_box_text: bool = True,
    show_logical_qubits: bool = False,
    title: str = "STAR Execution Timeline",
    shared_xmax: float | None = None,
    show_legend: bool = True,
    unified_heading_fontsize: float | None = None,
    title_pad: float | None = None,
    suppress_box_text_ops: frozenset[str] | None = None,
) -> float:
    """Draw horizontal STAR execution on *ax*; returns max end time in the log."""
    _, row_names, y_pos, n_rows = _resolve_star_row_layout(
        execution_log, n_factories, n_logical_qubits, show_logical_qubits
    )

    max_time = 0.0
    for entry in execution_log:
        start_time = entry_start(entry)
        end_time = entry_end(entry)
        factories = normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")
        max_time = max(max_time, end_time)
        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

        alpha = _BOX_ALPHA
        if operation in ["move", "return_move"]:
            border_color = "black"
            border_width = _BOX_BORDER_WIDTH
        else:
            border_color = "black"
            border_width = _BOX_BORDER_WIDTH

        if operation == "Barrier":
            continue
            ax.axvline(
                x=start_time,
                color=color,
                linestyle="--",
                linewidth=3,
                alpha=1,
                zorder=10,
            )
        else:
            no_text_operations = {
                "RUS_success",
                "RUS_fail",
                "TMR_fail",
            }
            if suppress_box_text_ops:
                no_text_operations = no_text_operations | set(suppress_box_text_ops)
            if operation in no_text_operations:
                zorder = 15
                start_time -= 0.05
            else:
                zorder = 0

            for idx, factory_id in enumerate(factories):
                factory_value = _resolve_factory_value(value, idx)
                factory_move_vecs = _resolve_factory_move_vecs(move_vecs, idx)
                factory_aod = _resolve_factory_value(aod_assignment, idx)
                y_coord = factory_id
                if show_logical_qubits:
                    row_name = f"f{factory_id}"
                    if row_name not in y_pos:
                        continue
                    y_coord = y_pos[row_name]

                box_color = color
                if operation in ["move", "return_move"]:
                    box_color = _movement_color_for_aod(operation, factory_aod)

                rect = mpatches.Rectangle(
                    (start_time, y_coord - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=box_color,
                    edgecolor=border_color,
                    linewidth=border_width,
                    zorder=zorder,
                    alpha=alpha,
                )
                ax.add_patch(rect)

                if show_box_text and operation not in no_text_operations:
                    text = _star_box_label(operation, factory_value, factory_move_vecs)

                    ax.text(
                        start_time + duration / 2,
                        y_coord,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                        linespacing=0.9,
                        clip_on=True,
                    )

            if show_logical_qubits and operation == "CNOT":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                        duration,
                        _BOX_HEIGHT,
                        facecolor=color,
                        edgecolor=border_color,
                        linewidth=border_width,
                        zorder=zorder,
                        alpha=alpha,
                    )
                    ax.add_patch(rect)

            if show_logical_qubits and operation == "SE_q":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                        duration,
                        _BOX_HEIGHT,
                        facecolor=color,
                        edgecolor=border_color,
                        linewidth=border_width,
                        zorder=zorder,
                        alpha=alpha,
                    )
                    ax.add_patch(rect)
                    if show_box_text and operation not in no_text_operations:
                        ax.text(
                            start_time + duration / 2,
                            y_pos[row_name],
                            "SE",
                            ha="center",
                            va="center",
                            fontsize=10,
                            fontweight="bold",
                            color="black",
                            linespacing=0.9,
                            clip_on=True,
                        )

    xmax = shared_xmax if shared_xmax is not None else max_time * 1.01
    ax.set_xlim(0, xmax)
    if show_logical_qubits:
        ax.set_ylim(-0.5, len(row_names) - 0.5)
    else:
        ax.set_ylim(-0.5, n_factories - 0.5)
    title_fs = (
        unified_heading_fontsize
        if unified_heading_fontsize is not None
        else _TITLE_FONT_SIZE
    )
    xlabel_fs = (
        unified_heading_fontsize
        if unified_heading_fontsize is not None
        else _AXIS_LABEL_FONT_SIZE
    )
    ax.set_xlabel(
        "Time (circuit moments)",
        fontsize=xlabel_fs,
        fontweight="bold",
    )
    if show_logical_qubits:
        ax.set_ylabel("")
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
    else:
        ax.set_ylabel("")
        ax.set_yticks(range(n_factories))
    _tpad = 6 if title_pad is None else title_pad
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=_tpad)
    ax.tick_params(axis="x", labelsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if show_legend:
        legend_handles = _star_execution_legend_handles()
        leg = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=len(legend_handles),
            fontsize=_LEGEND_FONT_SIZE,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        leg.set_zorder(20)
    return max_time


def plot_star_execution_subfigures(
    row_plots,
    *,
    show_logical_qubits: bool = True,
    show_box_text: bool = True,
    figure_width: float = 50.0,
    figure_vertical_stretch: float = _STAR_SUBFIG_DEFAULT_VERTICAL_STRETCH,
    suptitle: str | None = None,
    save_path: str = "output/circuit_execution/star_execution_subfigures.pdf",
):
    """Plot multiple horizontal STAR timelines as stacked subfigures (shared time axis).

    row_plots: list of (row_title, execution_log, n_factories, n_logical_qubits)
    figure_vertical_stretch: multiplies default height so each lane/box is taller on screen.
    """
    if not row_plots:
        print("No row plots to render")
        return

    row_count = len(row_plots)
    lane_counts = []
    for _, execution_log, nf, nlq in row_plots:
        _, _, _, n_rows = _resolve_star_row_layout(
            execution_log, nf, nlq, show_logical_qubits
        )
        lane_counts.append(n_rows)
    row_height_ratios = [max(0.8, lanes / 3.0) for lanes in lane_counts]

    global_xmax = (
        max(
            max(entry_end(entry) for entry in execution_log)
            for _, execution_log, _, _ in row_plots
        )
        * 1.01
    )

    fig_width = max(
        max(
            float(figure_width) * 0.45,
            len(execution_log) / 10 + 4.5,
        )
        for _, execution_log, _, _ in row_plots
    )
    fig_height = 0.9 + 0.90 * float(figure_vertical_stretch) * sum(row_height_ratios)
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": row_height_ratios},
    )
    heading_fs = _STAR_SUBFIG_HEADING_FONT_SIZE
    if row_count == 1:
        axes = [axes]

    for ax, (row_title, execution_log, n_factories, n_logical_qubits) in zip(
        axes, row_plots
    ):
        _plot_circuit_execution_on_ax(
            ax,
            execution_log,
            n_factories,
            n_logical_qubits,
            show_box_text=show_box_text,
            show_logical_qubits=show_logical_qubits,
            title=row_title,
            shared_xmax=global_xmax,
            show_legend=False,
            unified_heading_fontsize=heading_fs,
            title_pad=4.0,
        )

    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Let panels use most of the figure height; suptitle is placed after draw (below).
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98 if suptitle else 0.96))

    if suptitle:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb0 = axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        # Main title baseline just above the first row (including its subplot title).
        fig.suptitle(
            suptitle,
            fontsize=heading_fs,
            fontweight="bold",
            y=float(min(0.998, bb0.y1 + 0.008)),
            va="bottom",
        )

    axes[0].legend(
        handles=_star_execution_legend_handles(),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        fontsize=max(10, heading_fs - 2),
        frameon=True,
        framealpha=0.95,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSTAR execution subfigure plot saved to: {save_path}")


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================
def plot_circuit_execution(
    execution_log,
    n_factories,
    n_logical_qubits: int | None = None,
    save_path="output/circuit_execution.pdf",
    figure_width=16,
    show_box_text=True,
    show_logical_qubits: bool = False,
):
    """
    Plot circuit execution timeline showing operations on each factory.

    Args:
        execution_log: List of (start_time, end_time, factory_id, operation, qubit)
        n_factories: Number of factories
        save_path: Path to save the figure
        show_box_text: Whether to render text labels inside operation boxes
    """
    if not execution_log:
        print("No execution log to plot")
        return
    circuit_length = len(execution_log)
    fig_width = max(float(figure_width) * 0.45, circuit_length / 10 + 4.5)
    _, _, _, n_rows = _resolve_star_row_layout(
        execution_log, n_factories, n_logical_qubits, show_logical_qubits
    )
    fig, ax = plt.subplots(figsize=(fig_width, max(10, n_rows)))
    _plot_circuit_execution_on_ax(
        ax,
        execution_log,
        n_factories,
        n_logical_qubits,
        show_box_text=show_box_text,
        show_logical_qubits=show_logical_qubits,
        title="STAR Execution Timeline",
        shared_xmax=None,
        show_legend=True,
    )
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    base_path = save_path.split(".")[0]
    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCircuit execution plot saved to: {save_path}")


def plot_circuit_execution_vertical(
    execution_log,
    n_factories,
    n_logical_qubits: int | None = None,
    save_path="output/circuit_execution_vertical.pdf",
    figure_height=16,
    show_box_text=True,
    show_logical_qubits: bool = False,
):
    """
    Plot circuit execution timeline with vertical time axis (top to bottom).
    Factories are arranged horizontally (left to right).

    Args:
        execution_log: List of (start_time, end_time, factory_id, operation, qubit)
        n_factories: Number of factories
        save_path: Path to save the figure
        figure_height: Height of the figure in inches
        show_box_text: Whether to render text labels inside operation boxes
    """
    if not execution_log:
        print("No execution log to plot")
        return

    circuit_length = len(execution_log)
    if show_logical_qubits:
        if n_logical_qubits is None:
            max_qubit = -1
            for entry in execution_log:
                value = entry.get("targets")
                qubits = _extract_qubits_from_value(value)
                if qubits:
                    max_qubit = max(max_qubit, max(qubits))
            n_logical_qubits = max_qubit + 1 if max_qubit >= 0 else 0
        row_names = [f"q{i}" for i in range(n_logical_qubits)] + [
            f"f{i}" for i in range(n_factories)
        ]
        y_pos = {name: idx for idx, name in enumerate(row_names)}
        fig, ax = plt.subplots(
            figsize=(max(6, len(row_names) * 0.8), circuit_length / 5)
        )
    else:
        fig, ax = plt.subplots(figsize=(max(8, n_factories * 0.8), circuit_length / 5))

    # Plot each operation as a rectangle
    for entry in execution_log:
        start_time = entry_start(entry)
        end_time = entry_end(entry)
        factories = normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")

        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

        # Determine border color: use AOD color for move operations, else black
        alpha = _BOX_ALPHA
        if operation in ["move", "return_move"]:
            border_color = "black"
            border_width = _BOX_BORDER_WIDTH
        else:
            border_color = "black"
            border_width = _BOX_BORDER_WIDTH

        if operation == "Barrier":
            continue
            ax.axhline(
                y=start_time,
                color=color,
                linestyle="--",
                linewidth=3,
                alpha=1,
                zorder=10,
            )
        else:
            # Annotate with operation and qubit
            no_text_operations = {
                "RUS_success",
                "RUS_fail",
                "TMR_fail",
            }
            if operation in no_text_operations:
                zorder = 15
                start_time -= 0.05
            else:
                zorder = 0

            for idx, factory_id in enumerate(factories):
                factory_value = _resolve_factory_value(value, idx)
                factory_move_vecs = _resolve_factory_move_vecs(move_vecs, idx)
                factory_aod = _resolve_factory_value(aod_assignment, idx)
                x_coord = factory_id
                if show_logical_qubits:
                    row_name = f"f{factory_id}"
                    if row_name not in y_pos:
                        continue
                    x_coord = y_pos[row_name]

                box_color = color
                if operation in ["move", "return_move"]:
                    box_color = _movement_color_for_aod(operation, factory_aod)

                # Swap coordinates: factory_id on x-axis, time on y-axis
                # Rectangle: (x, y), width (horizontal = factory dimension), height (vertical = time dimension)
                rect = mpatches.Rectangle(
                    (x_coord - _BOX_Y_OFFSET, start_time),
                    _BOX_HEIGHT,
                    duration,
                    facecolor=box_color,
                    edgecolor=border_color,
                    alpha=alpha,  # Lighter for early moves, darker for later
                    linewidth=border_width,
                    zorder=zorder,
                )
                ax.add_patch(rect)

                if show_box_text and operation not in no_text_operations:
                    text = _star_box_label(operation, factory_value, factory_move_vecs)

                    ax.text(
                        x_coord,
                        start_time + duration / 2,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="black",
                        linespacing=0.9,
                        clip_on=True,
                    )

            if show_logical_qubits and operation == "CNOT":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (y_pos[row_name] - _BOX_Y_OFFSET, start_time),
                        _BOX_HEIGHT,
                        duration,
                        facecolor=color,
                        edgecolor=border_color,
                        alpha=alpha,
                        linewidth=border_width,
                        zorder=zorder,
                    )
                    ax.add_patch(rect)

            # Logical-qubit SE entries (no factories): render directly on q-rows.
            if show_logical_qubits and operation == "SE_q":
                qubit_targets = _extract_qubits_from_value(value)
                for qubit_id in qubit_targets:
                    row_name = f"q{qubit_id}"
                    if row_name not in y_pos:
                        continue
                    rect = mpatches.Rectangle(
                        (y_pos[row_name] - _BOX_Y_OFFSET, start_time),
                        _BOX_HEIGHT,
                        duration,
                        facecolor=color,
                        edgecolor=border_color,
                        alpha=alpha,
                        linewidth=border_width,
                        zorder=zorder,
                    )
                    ax.add_patch(rect)
                    if show_box_text:
                        ax.text(
                            y_pos[row_name],
                            start_time + duration / 2,
                            "SE",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color="black",
                            linespacing=0.9,
                            clip_on=True,
                        )

    # Configure axes
    # Invert y-axis so time goes from top to bottom
    ax.set_ylim(
        max(entry_end(e) for e in execution_log) * 1.01,
        0,
    )
    if show_logical_qubits:
        ax.set_xlim(-0.5, len(row_names) - 0.5)
    else:
        ax.set_xlim(-0.5, n_factories - 0.5)
    ax.set_ylabel("")
    if show_logical_qubits:
        ax.set_xlabel(
            "Qubits and Magic State Factories",
            fontsize=_AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_xticks(range(len(row_names)))
        ax.set_xticklabels(row_names)
        ax.set_title(
            "STAR Execution Timeline (Vertical Time)",
            fontsize=_TITLE_FONT_SIZE,
            fontweight="bold",
        )
    else:
        ax.set_xlabel(
            "Magic State Factory ID",
            fontsize=_AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_xticks(range(n_factories))
        ax.set_title(
            "STAR Execution Timeline (Vertical Time)",
            fontsize=_TITLE_FONT_SIZE,
            fontweight="bold",
        )
    ax.tick_params(axis="x", labelsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_map["SE"], edgecolor="black", label="SE"),
        mpatches.Patch(facecolor=color_map["CNOT"], edgecolor="black", label="CNOT"),
        mpatches.Patch(facecolor=color_map["S"], edgecolor="black", label="S gate"),
        mpatches.Patch(facecolor=color_map["move"], edgecolor="black", label="Move"),
        mpatches.Patch(
            facecolor=color_map["return_move"],
            edgecolor="black",
            label="Return Move",
        ),
        mpatches.Patch(
            facecolor=color_map["RUS_success"],
            edgecolor="black",
            label="RUS:success",
        ),
        mpatches.Patch(
            facecolor=color_map["RUS_fail"],
            edgecolor="black",
            label="RUS:fail",
        ),
        mpatches.Patch(
            facecolor=color_map["TMR_fail"],
            edgecolor="black",
            label="TMR:fail",
        ),
    ]
    leg = ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(legend_elements),
        fontsize=_LEGEND_FONT_SIZE,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    leg.set_zorder(20)
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    base_path = save_path.split(".")[0]
    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCircuit execution plot (vertical) saved to: {save_path}")


def plot_t_cultivation_execution(
    execution_log,
    n_qubits: int,
    n_factories: int,
    save_path="output/circuit_execution/t_cultivation_execution.pdf",
):
    """Plot a single T-cultivation timeline with qubit and factory lanes."""
    if not execution_log:
        print("No execution log to plot")
        return

    rows = [f"q{i}" for i in range(n_qubits)] + [f"f{i}" for i in range(n_factories)]
    fig, ax = plt.subplots(figsize=(10.0, max(4, 0.72 * len(rows))))
    _plot_t_cultivation_execution_on_ax(
        ax,
        execution_log,
        n_qubits=n_qubits,
        n_factories=n_factories,
        title="T-Cultivation Execution Timeline",
        show_legend=True,
    )

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nT-cultivation plot saved to: {save_path}")


def _t_cultivation_legend_handles():
    legend_ops = [
        ("H", "H"),
        ("CNOT", "CNOT"),
        ("S", "S"),
        ("T", "T"),
        ("SE_stage_1", "Prep. Stage 1"),
        ("SE_stage_2", "Prep. Stage 2"),
        ("move", "Move"),
        ("return_move", "Return Move"),
    ]
    return [
        mpatches.Patch(facecolor=color_map[key], edgecolor="black", label=label)
        for key, label in legend_ops
        if key in color_map
    ]


def _plot_t_cultivation_execution_on_ax(
    ax,
    execution_log,
    *,
    n_qubits: int,
    n_factories: int,
    title: str,
    show_legend: bool,
    shared_xmax: float | None = None,
    unified_heading_fontsize: float | None = None,
    title_pad: float | None = None,
    show_move_box_text: bool = False,
) -> None:
    rows = [f"q{i}" for i in range(n_qubits)] + [f"f{i}" for i in range(n_factories)]
    y_pos = {name: idx for idx, name in enumerate(rows)}
    max_time = max(entry_end(entry) for entry in execution_log)

    for entry in execution_log:
        start_time = entry_start(entry)
        end_time = entry_end(entry)
        factories = normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")

        operation = _canonical_tcult_operation(operation)

        duration = max(end_time - start_time, 0.1)
        color = color_map.get(operation, "#95a5a6")
        if operation in {"move", "return_move"}:
            box_color = _movement_color_for_aod(operation, aod_assignment)
        else:
            box_color = color

        for idx, factory_id in enumerate(factories):
            row_name = f"f{factory_id}"
            if row_name not in y_pos:
                continue
            y_coord = y_pos[row_name]
            rect = mpatches.Rectangle(
                (start_time, y_coord - _BOX_Y_OFFSET),
                duration,
                _BOX_HEIGHT,
                facecolor=box_color,
                edgecolor="black",
                linewidth=_BOX_BORDER_WIDTH,
                alpha=_BOX_ALPHA,
            )
            ax.add_patch(rect)
            if show_move_box_text and operation in {"move", "return_move"}:
                factory_move_vecs = _resolve_factory_move_vecs(move_vecs, idx)
                text = _star_box_label(operation, None, factory_move_vecs)
                if text:
                    ax.text(
                        start_time + duration / 2,
                        y_coord,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                        linespacing=0.9,
                        clip_on=True,
                    )

        qubits = _extract_qubits_from_targets(value)
        if not factories and qubits:
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                rect = mpatches.Rectangle(
                    (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=_BOX_BORDER_WIDTH,
                    alpha=_BOX_ALPHA,
                )
                ax.add_patch(rect)

        if factories and operation == "CNOT" and qubits:
            t_color = color_map.get("CNOT(T)", color)
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                rect = mpatches.Rectangle(
                    (start_time, y_pos[row_name] - _BOX_Y_OFFSET),
                    duration,
                    _BOX_HEIGHT,
                    facecolor=t_color,
                    edgecolor="black",
                    linewidth=_BOX_BORDER_WIDTH,
                    alpha=_BOX_ALPHA,
                )
                ax.add_patch(rect)

    ax.set_xlim(0, shared_xmax if shared_xmax is not None else max_time * 1.03)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    title_fs = (
        unified_heading_fontsize
        if unified_heading_fontsize is not None
        else _TITLE_FONT_SIZE
    )
    xlabel_fs = (
        unified_heading_fontsize
        if unified_heading_fontsize is not None
        else _AXIS_LABEL_FONT_SIZE
    )
    ax.set_xlabel(
        "Time (circuit moments)",
        fontsize=xlabel_fs,
        fontweight="bold",
    )
    ax.set_ylabel("")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(axis="x", labelsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE - 1)
    _tpad = 6 if title_pad is None else title_pad
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=_tpad)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(
            handles=_t_cultivation_legend_handles(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=_FIG_FONT_SIZE,
            frameon=True,
        )


def plot_t_cultivation_execution_subfigures(
    row_plots,
    save_path="output/circuit_execution/t_cultivation_execution_subfigures.pdf",
):
    """Plot multiple T-cultivation timelines as row-wise subfigures.

    row_plots: list of (row_title, execution_log, n_qubits, n_factories)
    """
    if not row_plots:
        print("No row plots to render")
        return

    row_count = len(row_plots)
    lane_counts = [n_qubits + n_factories for _, _, n_qubits, n_factories in row_plots]
    row_height_ratios = [max(0.8, lanes / 3.0) for lanes in lane_counts]
    global_xmax = (
        max(
            max(entry_end(entry) for entry in execution_log)
            for _, execution_log, _, _ in row_plots
        )
        * 1.03
    )

    fig_height = 0.9 + 0.90 * sum(row_height_ratios)
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(10.0, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": row_height_ratios},
    )
    heading_fs = _STAR_SUBFIG_HEADING_FONT_SIZE
    if row_count == 1:
        axes = [axes]

    for ax, (row_title, execution_log, n_qubits, n_factories) in zip(axes, row_plots):
        _plot_t_cultivation_execution_on_ax(
            ax,
            execution_log,
            n_qubits=n_qubits,
            n_factories=n_factories,
            title=row_title,
            show_legend=False,
            shared_xmax=global_xmax,
            unified_heading_fontsize=heading_fs,
            title_pad=4.0,
        )

    # Only keep the x-axis label/tick labels on the last row.
    for ax in axes[:-1]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    fig.legend(
        handles=_t_cultivation_legend_handles(),
        loc="center right",
        ncol=1,
        bbox_to_anchor=(0.97, 0.5),
        fontsize=max(10, _FIG_FONT_SIZE - 4),
        frameon=True,
    )

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout(rect=(0, 0.0, 1, 0.98))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nT-cultivation subfigure plot saved to: {save_path}")


def _legend_patch(color_key: str, label: str) -> mpatches.Patch:
    return mpatches.Patch(
        facecolor=color_map[color_key],
        edgecolor="black",
        label=label,
    )


def _combined_star_t_legend_handles():
    """Merged legend for STAR + T-cultivation subfigures (one entry per operation)."""
    merged: list[tuple[str, str]] = [
        ("S", "S gate"),
        ("CNOT", "CNOT"),
        ("SE", "SE"),
        ("SE_stage_1", "Check Stage"),
        ("SE_stage_2", "Escape Stage"),
        ("move", "Move"),
        ("return_move", "Return Move"),
        ("RUS_success", "RUS:success"),
        ("RUS_fail", "RUS:fail"),
        ("TMR_fail", "TMR:fail"),
    ]
    return [_legend_patch(key, label) for key, label in merged if key in color_map]


def plot_star_t_cultivation_execution_subfigures(
    *,
    star_log: list,
    t_log: list,
    n_logical_qubits: int = 1,
    n_factories: int = 2,
    star_row_title: str = "STAR architecture",
    t_row_title: str = "T-cultivation architecture",
    save_path: str = (
        "output/circuit_execution/star_t_cultivation_1q2f_subfigures.pdf"
    ),
    suptitle: str | None = None,
    figure_width: float = 16.0,
    figure_vertical_stretch: float = 1.8,
    show_box_text: bool = True,
) -> None:
    """Stacked STAR (top) and T-cultivation (bottom) timelines on a shared time axis.

    Both rows use ``n_logical_qubits`` logical lanes and ``n_factories`` factory lanes.
    """
    if not star_log or not t_log:
        print("STAR and T-cultivation execution logs are required")
        return

    n_qubits = int(n_logical_qubits)
    n_f = int(n_factories)
    _, _, _, star_lanes = _resolve_star_row_layout(
        star_log, n_f, n_qubits, show_logical_qubits=True
    )
    t_lanes = n_qubits + n_f
    row_height_ratios = [
        max(0.8, star_lanes / 3.0),
        max(0.8, t_lanes / 3.0),
    ]
    global_xmax = (
        max(
            max(entry_end(entry) for entry in star_log),
            max(entry_end(entry) for entry in t_log),
        )
        * 1.03
    )
    circuit_length = max(len(star_log), len(t_log))
    fig_width = max(float(figure_width) * 0.45, circuit_length / 10 + 4.5)
    fig_height = 0.9 + 0.90 * float(figure_vertical_stretch) * sum(row_height_ratios)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": row_height_ratios},
    )
    heading_fs = _STAR_SUBFIG_HEADING_FONT_SIZE

    _plot_circuit_execution_on_ax(
        axes[0],
        star_log,
        n_f,
        n_qubits,
        show_box_text=show_box_text,
        show_logical_qubits=True,
        title=star_row_title,
        shared_xmax=global_xmax,
        show_legend=False,
        unified_heading_fontsize=heading_fs,
        title_pad=4.0,
        suppress_box_text_ops=frozenset({"SE", "SE_q", "CNOT"}),
    )
    _plot_t_cultivation_execution_on_ax(
        axes[1],
        t_log,
        n_qubits=n_qubits,
        n_factories=n_f,
        title=t_row_title,
        show_legend=False,
        shared_xmax=global_xmax,
        unified_heading_fontsize=heading_fs,
        title_pad=4.0,
        show_move_box_text=True,
    )

    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", labelbottom=False)

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98 if suptitle else 0.96))

    if suptitle:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb0 = axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        fig.suptitle(
            suptitle,
            fontsize=heading_fs,
            fontweight="bold",
            y=float(min(0.998, bb0.y1 + 0.008)),
            va="bottom",
        )

    axes[0].legend(
        handles=_combined_star_t_legend_handles(),
        loc="upper right",
        bbox_to_anchor=(1.06, 1.0),
        ncol=2,
        fontsize=max(10, heading_fs - 2),
        frameon=True,
        framealpha=0.95,
        columnspacing=0.4,
        handletextpad=0.2,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSTAR vs T-cultivation subfigure plot saved to: {save_path}")
