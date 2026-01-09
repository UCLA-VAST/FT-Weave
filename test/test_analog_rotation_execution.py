import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
import random


random.seed(1234)

# Ensure repository root is on sys.path so `src` is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analog_rotation_execution import factory_angle_execution
from src.ds.device_state import FactoryPool
from src.util import analyze_execution_log, print_execution_profile


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

    fig, ax = plt.subplots(figsize=(16, max(6, n_factories * 0.8)))

    # Color mapping for operations
    color_map = {
        "TUM": "#9ed76c",
        "SE": "#4681a9",
        "CNOT": "#e74c3c",
        "Rz": "#e7ab3c",
        "move": "#b64abe",
        "Barrier": "#000000",
        "RUS_success": "#E01414",
        "RUS_fail": "#DDA413",
        "TMR_fail": "#007E15",
    }

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
                elif operation == "move" and move_vecs:
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCircuit execution plot saved to: {save_path}")


# Example usage
def test_small():
    # Configuration
    n_factories = 5
    target_qubits_angles = {0: 0.001, 1: 0.002, 2: 0.003, 3: 0.004, 4: 0.005}

    # logic_qubit_locations: list[tuple[int, int]] = [
    #     (0, 0),
    #     (1, 0),
    #     (2, 0),
    #     (3, 0),
    #     (4, 0),
    # ]
    # magic_state_locations: list[tuple[int, int]] = [
    #     (0, 1),
    #     (1, 1),
    #     (2, 1),
    #     (3, 1),
    #     (4, 1),
    # ]
    logic_qubit_locations: list[tuple[int, int]] = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
    ]
    magic_state_locations: list[tuple[int, int]] = [
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (1, 3),
    ]

    print("=" * 70)
    print("MAGIC STATE FACTORY ANGLE EXECUTION SIMULATION")
    print("=" * 70)
    print(f"Number of factories: {n_factories}")
    print(f"Target qubit angles: {target_qubits_angles}")
    print("=" * 70 + "\n")

    # Run simulation
    factory_pool = FactoryPool(num_factories=n_factories)
    total_time, log = factory_angle_execution(
        factory_pool, target_qubits_angles, logic_qubit_locations, magic_state_locations
    )

    # reference_log = "[(0, 1, 0, 'SE', 4), (1, 2, 0, 'SE', 4), (2, 3, 0, 'SE', 4), (3, 4, 0, 'Rz', 0.005), (4, 5, 0, 'SE', 4), (5, 6, 0, 'SE', 4), (0, 1, 1, 'SE', 3), (1, 2, 1, 'SE', 3), (2, 3, 1, 'SE', 3), (3, 4, 1, 'Rz', 0.004), (4, 5, 1, 'SE', 3), (5, 6, 1, 'SE', 3), (0, 1, 2, 'SE', 2), (1, 2, 2, 'SE', 2), (2, 3, 2, 'SE', 2), (3, 4, 2, 'Rz', 0.003), (4, 5, 2, 'SE', 2), (5, 6, 2, 'SE', 2), (0, 1, 3, 'SE', 1), (1, 2, 3, 'SE', 1), (2, 3, 3, 'SE', 1), (3, 4, 3, 'Rz', 0.002), (4, 5, 3, 'SE', 1), (5, 6, 3, 'SE', 1), (0, 1, 4, 'SE', 0), (1, 2, 4, 'SE', 0), (2, 3, 4, 'SE', 0), (3, 4, 4, 'Rz', 0.001), (4, 5, 4, 'SE', 0), (5, 6, 4, 'SE', 0), (6, 6, -1, 'Barrier', None), (6, 6, 4, 'TMR_fail', 0), (6, 7, 3, 'CNOT', 1), (7, 8, 3, 'SE', 1), (8, 8, 3, 'RUS_succsss', 1), (6, 7, 2, 'CNOT', 2), (7, 8, 2, 'SE', 2), (8, 8, 2, 'RUS_fail', 2), (6, 6, 1, 'TMR_fail', 3), (6, 6, 0, 'TMR_fail', 4), (8, 8, -1, 'Barrier', None), (8, 9, 0, 'SE', 4), (9, 10, 0, 'SE', 4), (10, 11, 0, 'SE', 4), (11, 12, 0, 'Rz', 0.005), (12, 13, 0, 'SE', 4), (13, 14, 0, 'SE', 4), (8, 9, 1, 'SE', 3), (9, 10, 1, 'SE', 3), (10, 11, 1, 'SE', 3), (11, 12, 1, 'Rz', 0.004), (12, 13, 1, 'SE', 3), (13, 14, 1, 'SE', 3), (8, 9, 2, 'SE', 2), (9, 10, 2, 'SE', 2), (10, 11, 2, 'SE', 2), (11, 12, 2, 'Rz', 0.006), (12, 13, 2, 'SE', 2), (13, 14, 2, 'SE', 2), (8, 9, 3, 'SE', 2), (9, 10, 3, 'SE', 2), (10, 11, 3, 'SE', 2), (11, 12, 3, 'Rz', 0.006), (12, 13, 3, 'SE', 2), (13, 14, 3, 'SE', 2), (8, 9, 4, 'SE', 0), (9, 10, 4, 'SE', 0), (10, 11, 4, 'SE', 0), (11, 12, 4, 'Rz', 0.001), (12, 13, 4, 'SE', 0), (13, 14, 4, 'SE', 0), (14, 14, -1, 'Barrier', None), (14, 15, 4, 'CNOT', 0), (15, 16, 4, 'SE', 0), (16, 16, 4, 'RUS_succsss', 0), (14, 15, 2, 'CNOT', 2), (15, 16, 2, 'SE', 2), (16, 16, 2, 'RUS_succsss', 2), (14, 15, 1, 'CNOT', 3), (15, 16, 1, 'SE', 3), (16, 16, 1, 'RUS_succsss', 3), (14, 15, 0, 'CNOT', 4), (15, 16, 0, 'SE', 4), (16, 16, 0, 'RUS_fail', 4), (16, 16, -1, 'Barrier', None), (16, 17, 0, 'SE', 4), (17, 18, 0, 'SE', 4), (18, 19, 0, 'SE', 4), (19, 20, 0, 'Rz', 0.01), (20, 21, 0, 'SE', 4), (21, 22, 0, 'SE', 4), (16, 17, 1, 'SE', 4), (17, 18, 1, 'SE', 4), (18, 19, 1, 'SE', 4), (19, 20, 1, 'Rz', 0.01), (20, 21, 1, 'SE', 4), (21, 22, 1, 'SE', 4), (16, 17, 2, 'SE', 4), (17, 18, 2, 'SE', 4), (18, 19, 2, 'SE', 4), (19, 20, 2, 'Rz', 0.02), (20, 21, 2, 'SE', 4), (21, 22, 2, 'SE', 4), (16, 17, 3, 'SE', 4), (17, 18, 3, 'SE', 4), (18, 19, 3, 'SE', 4), (19, 20, 3, 'Rz', 0.02), (20, 21, 3, 'SE', 4), (21, 22, 3, 'SE', 4), (16, 17, 4, 'SE', 4), (17, 18, 4, 'SE', 4), (18, 19, 4, 'SE', 4), (19, 20, 4, 'Rz', 0.01), (20, 21, 4, 'SE', 4), (21, 22, 4, 'SE', 4), (22, 22, -1, 'Barrier', None), (22, 23, 0, 'CNOT', 4), (23, 24, 0, 'SE', 4), (24, 24, 0, 'RUS_succsss', 4), (24, 24, -1, 'Barrier', None)]"
    # assert str(log) == reference_log, "Execution log does not match reference log."
    print("\n" + "=" * 70)
    print(f"TOTAL CIRCUIT EXECUTION TIME: {total_time} moments")
    print("=" * 70)

    # Visualize
    plot_circuit_execution(log, n_factories)


def test(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    same_angle: bool = True,
    placement: str = "seperate_region",
):
    target_qubits_angles = {}
    if same_angle:
        angle = random.uniform(0.0001, 0.001)
        for i in range(n_qubits):
            target_qubits_angles[i] = angle
    else:
        for i in range(n_qubits):
            target_qubits_angles[i] = random.uniform(0.0001, 0.001)

    n_columns, n_rows = qubit_layout
    if placement == "seperate_region":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x, n_rows + y) for x, y in product(range(n_columns), range(n_rows))
        ]
    else:
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y)
            for x, y in product(range(0, 2 * n_columns, 2), range(0, 2 * n_rows, 2))
        ]
        magic_state_locations = [
            (x, y)
            for x, y in product(
                range(1, 2 * n_rows + 1, 2), range(1, 2 * n_rows + 1, 2)
            )
        ]

    print("=" * 70)
    print("MAGIC STATE FACTORY ANGLE EXECUTION SIMULATION")
    print("=" * 70)
    print(f"Number of factories: {n_factories}")
    print(f"Target qubit angles: {target_qubits_angles}")
    print("=" * 70 + "\n")

    # Run simulation
    factory_pool = FactoryPool(num_factories=n_factories)
    total_time, log = factory_angle_execution(
        factory_pool, target_qubits_angles, logic_qubit_locations, magic_state_locations
    )

    # reference_log = "[(0, 1, 0, 'SE', 4), (1, 2, 0, 'SE', 4), (2, 3, 0, 'SE', 4), (3, 4, 0, 'Rz', 0.005), (4, 5, 0, 'SE', 4), (5, 6, 0, 'SE', 4), (0, 1, 1, 'SE', 3), (1, 2, 1, 'SE', 3), (2, 3, 1, 'SE', 3), (3, 4, 1, 'Rz', 0.004), (4, 5, 1, 'SE', 3), (5, 6, 1, 'SE', 3), (0, 1, 2, 'SE', 2), (1, 2, 2, 'SE', 2), (2, 3, 2, 'SE', 2), (3, 4, 2, 'Rz', 0.003), (4, 5, 2, 'SE', 2), (5, 6, 2, 'SE', 2), (0, 1, 3, 'SE', 1), (1, 2, 3, 'SE', 1), (2, 3, 3, 'SE', 1), (3, 4, 3, 'Rz', 0.002), (4, 5, 3, 'SE', 1), (5, 6, 3, 'SE', 1), (0, 1, 4, 'SE', 0), (1, 2, 4, 'SE', 0), (2, 3, 4, 'SE', 0), (3, 4, 4, 'Rz', 0.001), (4, 5, 4, 'SE', 0), (5, 6, 4, 'SE', 0), (6, 6, -1, 'Barrier', None), (6, 6, 4, 'TMR_fail', 0), (6, 7, 3, 'CNOT', 1), (7, 8, 3, 'SE', 1), (8, 8, 3, 'RUS_succsss', 1), (6, 7, 2, 'CNOT', 2), (7, 8, 2, 'SE', 2), (8, 8, 2, 'RUS_fail', 2), (6, 6, 1, 'TMR_fail', 3), (6, 6, 0, 'TMR_fail', 4), (8, 8, -1, 'Barrier', None), (8, 9, 0, 'SE', 4), (9, 10, 0, 'SE', 4), (10, 11, 0, 'SE', 4), (11, 12, 0, 'Rz', 0.005), (12, 13, 0, 'SE', 4), (13, 14, 0, 'SE', 4), (8, 9, 1, 'SE', 3), (9, 10, 1, 'SE', 3), (10, 11, 1, 'SE', 3), (11, 12, 1, 'Rz', 0.004), (12, 13, 1, 'SE', 3), (13, 14, 1, 'SE', 3), (8, 9, 2, 'SE', 2), (9, 10, 2, 'SE', 2), (10, 11, 2, 'SE', 2), (11, 12, 2, 'Rz', 0.006), (12, 13, 2, 'SE', 2), (13, 14, 2, 'SE', 2), (8, 9, 3, 'SE', 2), (9, 10, 3, 'SE', 2), (10, 11, 3, 'SE', 2), (11, 12, 3, 'Rz', 0.006), (12, 13, 3, 'SE', 2), (13, 14, 3, 'SE', 2), (8, 9, 4, 'SE', 0), (9, 10, 4, 'SE', 0), (10, 11, 4, 'SE', 0), (11, 12, 4, 'Rz', 0.001), (12, 13, 4, 'SE', 0), (13, 14, 4, 'SE', 0), (14, 14, -1, 'Barrier', None), (14, 15, 4, 'CNOT', 0), (15, 16, 4, 'SE', 0), (16, 16, 4, 'RUS_succsss', 0), (14, 15, 2, 'CNOT', 2), (15, 16, 2, 'SE', 2), (16, 16, 2, 'RUS_succsss', 2), (14, 15, 1, 'CNOT', 3), (15, 16, 1, 'SE', 3), (16, 16, 1, 'RUS_succsss', 3), (14, 15, 0, 'CNOT', 4), (15, 16, 0, 'SE', 4), (16, 16, 0, 'RUS_fail', 4), (16, 16, -1, 'Barrier', None), (16, 17, 0, 'SE', 4), (17, 18, 0, 'SE', 4), (18, 19, 0, 'SE', 4), (19, 20, 0, 'Rz', 0.01), (20, 21, 0, 'SE', 4), (21, 22, 0, 'SE', 4), (16, 17, 1, 'SE', 4), (17, 18, 1, 'SE', 4), (18, 19, 1, 'SE', 4), (19, 20, 1, 'Rz', 0.01), (20, 21, 1, 'SE', 4), (21, 22, 1, 'SE', 4), (16, 17, 2, 'SE', 4), (17, 18, 2, 'SE', 4), (18, 19, 2, 'SE', 4), (19, 20, 2, 'Rz', 0.02), (20, 21, 2, 'SE', 4), (21, 22, 2, 'SE', 4), (16, 17, 3, 'SE', 4), (17, 18, 3, 'SE', 4), (18, 19, 3, 'SE', 4), (19, 20, 3, 'Rz', 0.02), (20, 21, 3, 'SE', 4), (21, 22, 3, 'SE', 4), (16, 17, 4, 'SE', 4), (17, 18, 4, 'SE', 4), (18, 19, 4, 'SE', 4), (19, 20, 4, 'Rz', 0.01), (20, 21, 4, 'SE', 4), (21, 22, 4, 'SE', 4), (22, 22, -1, 'Barrier', None), (22, 23, 0, 'CNOT', 4), (23, 24, 0, 'SE', 4), (24, 24, 0, 'RUS_succsss', 4), (24, 24, -1, 'Barrier', None)]"
    # assert str(log) == reference_log, "Execution log does not match reference log."
    print("\n" + "=" * 70)
    print(f"TOTAL CIRCUIT EXECUTION TIME: {total_time} moments")
    print("=" * 70)

    profiling_result = analyze_execution_log(log, n_factories=n_factories)
    print_execution_profile(profile=profiling_result)


if __name__ == "__main__":
    # test_small()
    test(
        n_qubits=25,
        n_factories=25,
        qubit_layout=(5, 5),
        same_angle=True,
        placement="seperate_region",
        # placement="close",
    )
