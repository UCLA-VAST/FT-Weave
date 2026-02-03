import os
import sys

from itertools import product
import random


random.seed(1234)

# Ensure repository root is on sys.path so `src` is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analog_rotation_execution import factory_angle_execution
from src.ds.device_state import FactoryPool
from src.util import analyze_execution_log, print_execution_profile
from src.animator.rus_round_visualization import plot_all_rus_rounds
from src.animator.circuit_execution_visualization import (
    plot_circuit_execution,
    plot_circuit_execution_vertical,
)


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

    # Visualize
    plot_circuit_execution(log, n_factories)


def test(
    n_qubits: int,
    n_factories: int,
    qubit_layout: tuple,
    same_angle: bool = True,
    visualize_rus: bool = False,
    placement: str = "seperate_region",
    prefix: str = "",
    n_aods: int = 1,
    consider_skip_rus: bool = False,
):
    target_qubits_angles = {}
    if same_angle:
        angle = 0.001
        for i in range(n_qubits):
            target_qubits_angles[i] = angle
    else:
        for i in range(n_qubits):
            target_qubits_angles[i] = random.uniform(0.0001, 0.001)

    n_columns, n_rows = qubit_layout
    if placement == "seperate_region_row":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x, n_rows + y) for x, y in product(range(n_columns), range(n_rows))
        ]
        column_based_placement = False
    elif placement == "seperate_region_col":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        magic_state_locations = [
            (x + n_columns, y) for x, y in product(range(n_columns), range(n_rows))
        ]
        column_based_placement = True
    elif placement == "row_based":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(n_columns), range(0, 2 * n_rows, 2))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(n_columns), range(1, 2 * n_rows + 1, 2))
        ]
        column_based_placement = False
    elif placement == "col_based":
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(0, 2 * n_columns, 2), range(n_rows))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(1, 2 * n_columns + 1, 2), range(n_rows))
        ]
        column_based_placement = True
    elif placement == "checkerboard":
        logic_qubit_locations = []
        magic_state_locations = []
        for x, y in product(range(n_columns * 2), range(n_rows)):
            if (x + y) % 2 == 0:
                logic_qubit_locations.append((x, y))
            else:
                magic_state_locations.append((x, y))
        column_based_placement = False
    else:
        raise ValueError(f"Unknown placement strategy: {placement}")

    print("=" * 70)
    print("MAGIC STATE FACTORY ANGLE EXECUTION SIMULATION")
    print("=" * 70)
    print(f"Number of factories: {n_factories}")
    print(f"Target qubit angles: {target_qubits_angles}")
    print("=" * 70 + "\n")

    # Run simulation
    factory_pool = FactoryPool(num_factories=n_factories)
    total_time, log = factory_angle_execution(
        factory_pool,
        target_qubits_angles,
        logic_qubit_locations,
        magic_state_locations,
        column_based_placement=column_based_placement,
        n_aods=n_aods,
        consider_skip_rus=consider_skip_rus,
    )

    profiling_result = analyze_execution_log(log, n_factories=n_factories)
    print_execution_profile(profile=profiling_result)

    pdf_path = f"output/circuit_execution_vertical/{prefix}{placement}.pdf"
    plot_circuit_execution_vertical(
        log, n_factories, figure_height=50, save_path=pdf_path
    )
    pdf_path = f"output/circuit_execution/{prefix}{placement}.pdf"
    plot_circuit_execution(log, n_factories, figure_width=50, save_path=pdf_path)

    if visualize_rus:
        # Generate RUS round visualizations
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        pdf_path = os.path.join(output_dir, f"rus_rounds_detailed/{prefix}{placement}")
        plot_all_rus_rounds(
            execution_log=log,
            logic_qubit_locations=logic_qubit_locations,
            magic_state_locations=magic_state_locations,
            base_path=pdf_path,
        )

        print("\nVisualization complete!")
        print(f"  Detailed PDF: {pdf_path}")


if __name__ == "__main__":
    # test_small()
    test(
        n_qubits=25,
        n_factories=25,
        qubit_layout=(5, 5),
        # n_qubits=9,
        # n_factories=9,
        # qubit_layout=(3, 3),
        same_angle=True,
        # placement="seperate_region_row",
        # placement="seperate_region_col",
        # placement="row_based",
        placement="col_based",
        # placement="checkerboard",
        visualize_rus=True,
        prefix="no_skip_tmr_matching_aod_2_",
        # prefix="checkerboard_aod_2_",
        n_aods=2,
        consider_skip_rus=True,
    )

    # test(
    #     n_qubits=25,
    #     n_factories=25,
    #     qubit_layout=(5, 5),
    #     # n_qubits=9,
    #     # n_factories=9,
    #     # qubit_layout=(3, 3),
    #     same_angle=True,
    #     # placement="seperate_region_row",
    #     placement="seperate_region_col",
    #     # placement="row_based",
    #     # placement="col_based",
    #     visualize_rus=True,
    # )
