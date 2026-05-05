import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

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
    "SE_stage_1": "#4681a9",
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


def get_aod_border_color(aod_idx: int) -> str:
    """Get border color for a given AOD index."""
    return aod_colors[aod_idx % len(aod_colors)]


def _normalize_factories(factory_id) -> list[int]:
    if isinstance(factory_id, int):
        return [factory_id]
    if factory_id is None:
        return []
    return list(factory_id)


def _resolve_factory_value(values, idx: int):
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return values


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
        fig, ax = plt.subplots(figsize=(fig_width, max(6, len(row_names) * 0.8)))
    else:
        fig, ax = plt.subplots(figsize=(fig_width, max(6, n_factories * 0.8)))

    # Plot each operation as a rectangle
    max_time = 0
    for entry in execution_log:
        start_time = entry.get("start_time", 0)
        end_time = entry.get("end_time", start_time)
        factories = _normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")
        move_vecs = entry.get("move_vecs")
        max_time = max(max_time, end_time)
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
            ax.axvline(
                x=start_time,
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
                factory_move_vecs = _resolve_factory_value(move_vecs, idx)
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

    # Configure axes
    ax.set_xlim(0, max_time * 1.01)
    if show_logical_qubits:
        ax.set_ylim(-0.5, len(row_names) - 0.5)
    else:
        ax.set_ylim(-0.5, n_factories - 0.5)
    ax.set_xlabel(
        "Time (circuit moments)",
        fontsize=_AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    if show_logical_qubits:
        ax.set_ylabel("")
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
        ax.set_title(
            "STAR Execution Timeline",
            fontsize=_TITLE_FONT_SIZE,
            fontweight="bold",
        )
    else:
        ax.set_ylabel("")
        ax.set_yticks(range(n_factories))
        ax.set_title(
            "STAR Execution Timeline",
            fontsize=_TITLE_FONT_SIZE,
            fontweight="bold",
        )
    ax.tick_params(axis="x", labelsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Legend
    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor=color_map["SE"], edgecolor="black", label="SE (State Preparation)"
        ),
        mpatches.Patch(
            facecolor=color_map["CNOT"], edgecolor="black", label="CNOT (Injection)"
        ),
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
        # plt.Line2D(
        #     [0],
        #     [0],
        #     color=color_map["Barrier"],
        #     linewidth=2.5,
        #     linestyle="--",
        #     label="Barrier",
        # ),
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
        fig, ax = plt.subplots(figsize=(max(6, n_factories * 0.8), circuit_length / 5))

    # Plot each operation as a rectangle
    for entry in execution_log:
        start_time = entry.get("start_time", 0)
        end_time = entry.get("end_time", start_time)
        factories = _normalize_factories(entry.get("factories"))
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
                factory_move_vecs = _resolve_factory_value(move_vecs, idx)
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

    # Configure axes
    # Invert y-axis so time goes from top to bottom
    ax.set_ylim(
        max(e.get("end_time", e.get("start_time", 0)) for e in execution_log) * 1.01, 0
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
        mpatches.Patch(
            facecolor=color_map["SE"], edgecolor="black", label="SE (State Preparation)"
        ),
        mpatches.Patch(
            facecolor=color_map["CNOT"], edgecolor="black", label="CNOT (Injection)"
        ),
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
        ("SE_stage_1", "SE Stage 1"),
        ("SE_stage_2", "SE Stage 2"),
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
) -> None:
    rows = [f"q{i}" for i in range(n_qubits)] + [f"f{i}" for i in range(n_factories)]
    y_pos = {name: idx for idx, name in enumerate(rows)}
    max_time = max(
        entry.get("end_time", entry.get("start_time", 0)) for entry in execution_log
    )

    for entry in execution_log:
        start_time = entry.get("start_time", 0)
        end_time = entry.get("end_time", start_time)
        factories = _normalize_factories(entry.get("factories"))
        operation = entry.get("operation")
        aod_assignment = entry.get("aod_assignment")
        value = entry.get("targets")

        operation = _canonical_tcult_operation(operation)

        duration = max(end_time - start_time, 0.1)
        color = color_map.get(operation, "#95a5a6")

        for factory_id in factories:
            row_name = f"f{factory_id}"
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
    ax.set_xlabel("Time", fontsize=_FIG_FONT_SIZE)
    ax.set_ylabel("")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(axis="x", labelsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE - 1)
    ax.set_title(title, fontsize=_FIG_FONT_SIZE, pad=2)
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
            max(
                entry.get("end_time", entry.get("start_time", 0))
                for entry in execution_log
            )
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
    fig.suptitle(
        "Execution Diagram for One Qubit with 10 T Gates",
        fontsize=_FIG_FONT_SIZE,
        y=0.972,
    )
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
