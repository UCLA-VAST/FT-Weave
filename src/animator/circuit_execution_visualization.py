import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Color mapping for operations
color_map = {
    "TUM": "#9ed76c",
    "SE": "#4681a9",
    "CNOT": "#e74c3c",
    "Rz": "#e7ab3c",
    "move": "#b64abe",
    "return_move": "#9edc6f",
    "Barrier": "#000000",
    "RUS_success": "#E01414",
    "RUS_fail": "#DDA413",
    "TMR_fail": "#007E15",
}


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
    fig, ax = plt.subplots(figsize=(circuit_length / 8 + 4, max(6, n_factories * 0.8)))

    # Plot each operation as a rectangle
    for entry in execution_log:
        if len(entry) == 5:
            start_time, end_time, factory_id, operation, value = entry
            move_vecs = None
        else:
            start_time, end_time, factory_id, operation, value, move_vecs = entry
        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

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

            rect = mpatches.Rectangle(
                (start_time, factory_id - 0.4),
                duration,
                0.8,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                zorder=zorder,
            )
            ax.add_patch(rect)

            if operation not in no_text_operations:
                if operation == "Rz":
                    text = f"{operation}\nθ:{value}"
                elif operation in ["move", "return_move"] and move_vecs:
                    text = f"{operation}\n{move_vecs[0]}\n->{move_vecs[1]}"
                else:
                    text = f"{operation}\nQ{value}"

                ax.text(
                    start_time + duration / 2,
                    factory_id,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )

    # Configure axes
    ax.set_xlim(0, max(e[1] for e in execution_log) * 1.05)
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
    fig, ax = plt.subplots(figsize=(max(6, n_factories * 0.8), circuit_length / 10))

    # Plot each operation as a rectangle
    for entry in execution_log:
        if len(entry) == 5:
            start_time, end_time, factory_id, operation, value = entry
            move_vecs = None
        else:
            start_time, end_time, factory_id, operation, value, move_vecs = entry

        duration = end_time - start_time
        if duration == 0:
            duration = 0.1
        color = color_map.get(operation, "#95a5a6")

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

            # Swap coordinates: factory_id on x-axis, time on y-axis
            # Rectangle: (x, y), width (horizontal = factory dimension), height (vertical = time dimension)
            rect = mpatches.Rectangle(
                (factory_id - 0.4, start_time),
                0.8,
                duration,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                zorder=zorder,
            )
            ax.add_patch(rect)

            if operation not in no_text_operations:
                if operation == "Rz":
                    text = f"{operation}\nθ:{value}"
                elif operation in ["move", "return_move"] and move_vecs:
                    text = f"{operation}\n{move_vecs[0]}\n->{move_vecs[1]}"
                else:
                    text = f"{operation}\nQ{value}"

                ax.text(
                    factory_id,
                    start_time + duration / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
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
