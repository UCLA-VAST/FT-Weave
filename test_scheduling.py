from src.scheduling import factory_angle_execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================
def plot_circuit_execution(
    execution_log, n_factories, save_path="circuit_execution.pdf"
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

    fig, ax = plt.subplots(figsize=(14, max(6, n_factories * 0.8)))

    # Color mapping for operations
    color_map = {
        "TUM": "#9ed76c",
        "SE": "#4681a9",
        "CNOT": "#e74c3c",
        "Rz": "#e7ab3c",
        "Barrier": "#000000",
        "RUS_succsss": "#E01414",
        "RUS_fail": "#DDA413",
        "TUM_fail": "#007E15",
    }

    # Plot each operation as a rectangle
    for start_time, end_time, factory_id, operation, value in execution_log:
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
            no_text_operations = {"RUS_succsss", "RUS_fail", "TUM_fail"}
            if operation in no_text_operations:
                zorder = 10
                start_time -= 0.05
            else:
                zorder = 0

            rect = mpatches.Rectangle(
                (start_time, factory_id - 0.4),
                duration,
                0.8,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=zorder,
            )
            ax.add_patch(rect)

            if operation not in no_text_operations:
                if operation == "Rz":
                    text = f"{operation}\nθ:{value}"
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
            facecolor=color_map["RUS_succsss"],
            edgecolor="black",
            label="RUS:succsss",
        ),
        mpatches.Patch(
            facecolor=color_map["RUS_fail"],
            edgecolor="black",
            label="RUS:fail",
        ),
        mpatches.Patch(
            facecolor=color_map["TUM_fail"],
            edgecolor="black",
            label="TUM:fail",
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCircuit execution plot saved to: {save_path}")


# Example usage

if __name__ == "__main__":
    # Configuration
    n_factories = 5
    target_qubits_angles = {0: 0.001, 1: 0.002, 2: 0.003, 3: 0.004, 4: 0.005}

    print("=" * 70)
    print("MAGIC STATE FACTORY ANGLE EXECUTION SIMULATION")
    print("=" * 70)
    print(f"Number of factories: {n_factories}")
    print(f"Target qubit angles: {target_qubits_angles}")
    print("=" * 70 + "\n")

    # Run simulation
    total_time, log = factory_angle_execution(n_factories, target_qubits_angles.copy())

    print("\n" + "=" * 70)
    print(f"TOTAL CIRCUIT EXECUTION TIME: {total_time} moments")
    print("=" * 70)

    # Visualize
    plot_circuit_execution(log, n_factories)
