import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
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
max_aod = 5
alpha_constant = 0.3
ylim = 93


def get_aod_border_color(aod_idx: int) -> str:
    """Get border color for a given AOD index."""
    return aod_colors[aod_idx % len(aod_colors)]


def _parse_execution_entry(entry: tuple) -> tuple:
    if len(entry) == 5:
        start_time, end_time, factory_id, operation, aod_assignment = entry
        assert (
            operation == "Barrier"
        ), "Expected 5-element entry to be a Barrier operation"
        value = None
        move_vecs = None
    elif len(entry) == 6:
        start_time, end_time, factory_id, operation, aod_assignment, value = entry
        move_vecs = None
    elif len(entry) == 7:
        (
            start_time,
            end_time,
            factory_id,
            operation,
            aod_assignment,
            value,
            move_vecs,
        ) = entry
    else:
        raise ValueError(f"Unexpected log entry format: {entry}")

    if isinstance(factory_id, int):
        factories = [factory_id]
    elif factory_id is None:
        factories = []
    else:
        factories = list(factory_id)

    return start_time, end_time, factories, operation, aod_assignment, value, move_vecs


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


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================
def plot_circuit_execution(
    execution_log,
    n_factories,
    save_path="output/circuit_execution.pdf",
    figure_width=16,
):
    """
    Plot circuit execution timeline showing operations on each factory.

    Args:
        execution_log: List of (start_time, end_time, factory_id, operation, qubit)
        n_factories: Number of factories
        save_path: Path to save the figure
    """
    if not execution_log:
        print("No execution log to plot")
        return
    circuit_length = len(execution_log)
    fig, ax = plt.subplots(figsize=(circuit_length / 10 + 4, max(6, n_factories * 0.8)))

    # Plot each operation as a rectangle
    max_time = 0
    for entry in execution_log:
        (
            start_time,
            end_time,
            factories,
            operation,
            aod_assignment,
            value,
            move_vecs,
        ) = _parse_execution_entry(entry)
        max_time = max(max_time, end_time)
        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

        # Determine border color: use AOD color for move operations, else black
        alpha = 1
        if operation in ["move", "return_move"]:
            border_color = "black"
            border_width = 2
            alpha = aod_assignment / max_aod + alpha_constant
        else:
            border_color = "black"
            border_width = 1

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

                rect = mpatches.Rectangle(
                    (start_time, factory_id - 0.4),
                    duration,
                    0.8,
                    facecolor=color,
                    edgecolor=border_color,
                    linewidth=border_width,
                    zorder=zorder,
                    alpha=alpha,
                )
                ax.add_patch(rect)

                if operation not in no_text_operations:
                    if operation == "Rz":
                        text = f"{operation}\nθ:{factory_value}"
                    elif operation == "S":
                        text = f"{operation}\nθ:{factory_value}"
                    elif operation in ["move", "return_move"] and factory_move_vecs:
                        text = f"{operation}\n{factory_move_vecs[0]}\n->{factory_move_vecs[1]}"
                    elif factory_value is not None:
                        text = f"{operation}\nQ{factory_value}"
                    else:
                        text = f"{operation}"

                    ax.text(
                        start_time + duration / 2,
                        factory_id,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="black",
                    )

    # Configure axes
    ax.set_xlim(0, max_time * 1.01)
    ax.set_ylim(-0.5, n_factories - 0.5)
    ax.set_xlabel("Time (circuit moments)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Magic State Factory ID", fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_factories))
    ax.set_title(
        "Circuit Execution Timeline: Magic State Factories",
        fontsize=14,
        fontweight="bold",
    )
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
        mpatches.Patch(
            facecolor=color_map["move"], edgecolor="black", label="move (Forward)"
        ),
        mpatches.Patch(
            facecolor=color_map["return_move"],
            edgecolor="black",
            label="return_move (Return)",
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
    leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    leg.set_zorder(20)
    plt.tight_layout()
    base_path = save_path.split(".")[0]
    output_dir = os.path.dirname(base_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCircuit execution plot saved to: {save_path}")


def plot_circuit_execution_vertical(
    execution_log,
    n_factories,
    save_path="output/circuit_execution_vertical.pdf",
    figure_height=16,
):
    """
    Plot circuit execution timeline with vertical time axis (top to bottom).
    Factories are arranged horizontally (left to right).

    Args:
        execution_log: List of (start_time, end_time, factory_id, operation, qubit)
        n_factories: Number of factories
        save_path: Path to save the figure
        figure_height: Height of the figure in inches
    """
    if not execution_log:
        print("No execution log to plot")
        return

    circuit_length = len(execution_log)
    fig, ax = plt.subplots(figsize=(max(6, n_factories * 0.8), circuit_length / 5))

    # Plot each operation as a rectangle
    for entry in execution_log:
        (
            start_time,
            end_time,
            factories,
            operation,
            aod_assignment,
            value,
            move_vecs,
        ) = _parse_execution_entry(entry)

        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

        # Determine border color: use AOD color for move operations, else black
        alpha = 1
        if operation in ["move", "return_move"]:
            border_color = "black"
            border_width = 2
            alpha = aod_assignment / max_aod + alpha_constant
        else:
            border_color = "black"
            border_width = 1

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

                # Swap coordinates: factory_id on x-axis, time on y-axis
                # Rectangle: (x, y), width (horizontal = factory dimension), height (vertical = time dimension)
                rect = mpatches.Rectangle(
                    (factory_id - 0.4, start_time),
                    0.8,
                    duration,
                    facecolor=color,
                    edgecolor=border_color,
                    alpha=alpha,  # Lighter for early moves, darker for later
                    linewidth=border_width,
                    zorder=zorder,
                )
                ax.add_patch(rect)

                if operation not in no_text_operations:
                    if operation == "Rz":
                        text = f"{operation}\nθ:{factory_value}"
                    elif operation == "S":
                        text = f"{operation}\nθ:{factory_value}"
                    elif operation in ["move", "return_move"] and factory_move_vecs:
                        text = f"{operation}\n{factory_move_vecs[0]}\n->{factory_move_vecs[1]}"
                    elif factory_value is not None:
                        text = f"{operation}\nQ{factory_value}"
                    else:
                        text = f"{operation}"

                    ax.text(
                        factory_id,
                        start_time + duration / 2,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="black",
                    )

    # Configure axes
    # Invert y-axis so time goes from top to bottom
    ax.set_ylim(max(e[1] for e in execution_log) * 1.01, 0)
    ax.set_xlim(-0.5, n_factories - 0.5)
    ax.set_ylabel("Time (circuit moments)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Magic State Factory ID", fontsize=12, fontweight="bold")
    ax.set_xticks(range(n_factories))
    ax.set_title(
        "Circuit Execution Timeline: Magic State Factories (Vertical Time)",
        fontsize=14,
        fontweight="bold",
    )
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
        mpatches.Patch(
            facecolor=color_map["move"], edgecolor="black", label="move (Forward)"
        ),
        mpatches.Patch(
            facecolor=color_map["return_move"],
            edgecolor="black",
            label="return_move (Return)",
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
    leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    leg.set_zorder(20)
    plt.tight_layout()
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
    """Plot T-cultivation timeline with qubit and factory lanes.

    X-axis is time, Y-axis contains logical qubits (q*) and factories (f*).
    For factory-side CNOT in RUS teleportation, also draw CNOT(T) on target qubits.
    """
    if not execution_log:
        print("No execution log to plot")
        return

    rows = [f"q{i}" for i in range(n_qubits)] + [f"f{i}" for i in range(n_factories)]
    y_pos = {name: idx for idx, name in enumerate(rows)}

    max_time = max(entry[1] for entry in execution_log)
    fig, ax = plt.subplots(figsize=(14, max(4, 0.9 * len(rows))))

    no_text_operations = {"RUS_success", "RUS_fail", "TMR_fail"}

    for entry in execution_log:
        (
            start_time,
            end_time,
            factories,
            operation,
            aod_assignment,
            value,
            _move_vecs,
        ) = _parse_execution_entry(entry)

        duration = max(end_time - start_time, 0.1)
        color = color_map.get(operation, "#95a5a6")

        # Factory lanes
        for idx, factory_id in enumerate(factories):
            row_name = f"f{factory_id}"
            if row_name not in y_pos:
                continue
            rect = mpatches.Rectangle(
                (start_time, y_pos[row_name] - 0.35),
                duration,
                0.7,
                facecolor=color,
                edgecolor="black",
                linewidth=1 if operation not in {"move", "return_move"} else 2,
                alpha=0.9 if aod_assignment is None else 0.9,
            )
            ax.add_patch(rect)
            if operation not in no_text_operations:
                factory_value = _resolve_factory_value(value, idx)
                text = (
                    operation
                    if factory_value is None
                    else f"{operation}\nQ{factory_value}"
                )
                ax.text(
                    start_time + duration / 2,
                    y_pos[row_name],
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        qubits = _extract_qubits_from_targets(value)

        # Logical-qubit lanes
        if not factories and qubits:
            # Normal logical gate path.
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                rect = mpatches.Rectangle(
                    (start_time, y_pos[row_name] - 0.35),
                    duration,
                    0.7,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.9,
                )
                ax.add_patch(rect)
                if operation not in no_text_operations:
                    ax.text(
                        start_time + duration / 2,
                        y_pos[row_name],
                        operation,
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        # For factory-side CNOT (RUS teleportation), also plot target qubit with suffix.
        if factories and operation == "CNOT" and qubits:
            t_color = color_map.get("CNOT(T)", color)
            for qubit in qubits:
                row_name = f"q{qubit}"
                if row_name not in y_pos:
                    continue
                rect = mpatches.Rectangle(
                    (start_time, y_pos[row_name] - 0.35),
                    duration,
                    0.7,
                    facecolor=t_color,
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.9,
                )
                ax.add_patch(rect)
                ax.text(
                    start_time + duration / 2,
                    y_pos[row_name],
                    "CNOT(T)",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    ax.set_xlim(0, max_time * 1.03)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Qubits / Factories")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("T-Cultivation Execution Timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    legend_ops = [
        "H",
        "CNOT",
        "CNOT(T)",
        "S",
        "T",
        "Rz",
        "SE_stage_1",
        "SE_stage_2",
        "move",
        "return_move",
    ]
    legend_handles = [
        mpatches.Patch(facecolor=color_map[op], edgecolor="black", label=op)
        for op in legend_ops
        if op in color_map
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nT-cultivation plot saved to: {save_path}")
