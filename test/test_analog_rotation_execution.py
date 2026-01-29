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
    visualize_rus: bool = False,
    placement: str = "seperate_region",
    prefix: str = "",
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
    else:
        # Generate logic qubit locations as the Cartesian product of columns and rows
        logic_qubit_locations = [
            (x, y) for x, y in product(range(0, 2 * n_columns, 2), range(n_rows))
        ]
        magic_state_locations = [
            (x, y) for x, y in product(range(1, 2 * n_columns + 1, 2), range(n_rows))
        ]
        column_based_placement = True

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
    )

    # reference_log = "[(0, 1, 0, 'SE', 4), (1, 2, 0, 'SE', 4), (2, 3, 0, 'SE', 4), (3, 4, 0, 'Rz', 0.005), (4, 5, 0, 'SE', 4), (5, 6, 0, 'SE', 4), (0, 1, 1, 'SE', 3), (1, 2, 1, 'SE', 3), (2, 3, 1, 'SE', 3), (3, 4, 1, 'Rz', 0.004), (4, 5, 1, 'SE', 3), (5, 6, 1, 'SE', 3), (0, 1, 2, 'SE', 2), (1, 2, 2, 'SE', 2), (2, 3, 2, 'SE', 2), (3, 4, 2, 'Rz', 0.003), (4, 5, 2, 'SE', 2), (5, 6, 2, 'SE', 2), (0, 1, 3, 'SE', 1), (1, 2, 3, 'SE', 1), (2, 3, 3, 'SE', 1), (3, 4, 3, 'Rz', 0.002), (4, 5, 3, 'SE', 1), (5, 6, 3, 'SE', 1), (0, 1, 4, 'SE', 0), (1, 2, 4, 'SE', 0), (2, 3, 4, 'SE', 0), (3, 4, 4, 'Rz', 0.001), (4, 5, 4, 'SE', 0), (5, 6, 4, 'SE', 0), (6, 6, -1, 'Barrier', None), (6, 6, 4, 'TMR_fail', 0), (6, 7, 3, 'CNOT', 1), (7, 8, 3, 'SE', 1), (8, 8, 3, 'RUS_succsss', 1), (6, 7, 2, 'CNOT', 2), (7, 8, 2, 'SE', 2), (8, 8, 2, 'RUS_fail', 2), (6, 6, 1, 'TMR_fail', 3), (6, 6, 0, 'TMR_fail', 4), (8, 8, -1, 'Barrier', None), (8, 9, 0, 'SE', 4), (9, 10, 0, 'SE', 4), (10, 11, 0, 'SE', 4), (11, 12, 0, 'Rz', 0.005), (12, 13, 0, 'SE', 4), (13, 14, 0, 'SE', 4), (8, 9, 1, 'SE', 3), (9, 10, 1, 'SE', 3), (10, 11, 1, 'SE', 3), (11, 12, 1, 'Rz', 0.004), (12, 13, 1, 'SE', 3), (13, 14, 1, 'SE', 3), (8, 9, 2, 'SE', 2), (9, 10, 2, 'SE', 2), (10, 11, 2, 'SE', 2), (11, 12, 2, 'Rz', 0.006), (12, 13, 2, 'SE', 2), (13, 14, 2, 'SE', 2), (8, 9, 3, 'SE', 2), (9, 10, 3, 'SE', 2), (10, 11, 3, 'SE', 2), (11, 12, 3, 'Rz', 0.006), (12, 13, 3, 'SE', 2), (13, 14, 3, 'SE', 2), (8, 9, 4, 'SE', 0), (9, 10, 4, 'SE', 0), (10, 11, 4, 'SE', 0), (11, 12, 4, 'Rz', 0.001), (12, 13, 4, 'SE', 0), (13, 14, 4, 'SE', 0), (14, 14, -1, 'Barrier', None), (14, 15, 4, 'CNOT', 0), (15, 16, 4, 'SE', 0), (16, 16, 4, 'RUS_succsss', 0), (14, 15, 2, 'CNOT', 2), (15, 16, 2, 'SE', 2), (16, 16, 2, 'RUS_succsss', 2), (14, 15, 1, 'CNOT', 3), (15, 16, 1, 'SE', 3), (16, 16, 1, 'RUS_succsss', 3), (14, 15, 0, 'CNOT', 4), (15, 16, 0, 'SE', 4), (16, 16, 0, 'RUS_fail', 4), (16, 16, -1, 'Barrier', None), (16, 17, 0, 'SE', 4), (17, 18, 0, 'SE', 4), (18, 19, 0, 'SE', 4), (19, 20, 0, 'Rz', 0.01), (20, 21, 0, 'SE', 4), (21, 22, 0, 'SE', 4), (16, 17, 1, 'SE', 4), (17, 18, 1, 'SE', 4), (18, 19, 1, 'SE', 4), (19, 20, 1, 'Rz', 0.01), (20, 21, 1, 'SE', 4), (21, 22, 1, 'SE', 4), (16, 17, 2, 'SE', 4), (17, 18, 2, 'SE', 4), (18, 19, 2, 'SE', 4), (19, 20, 2, 'Rz', 0.02), (20, 21, 2, 'SE', 4), (21, 22, 2, 'SE', 4), (16, 17, 3, 'SE', 4), (17, 18, 3, 'SE', 4), (18, 19, 3, 'SE', 4), (19, 20, 3, 'Rz', 0.02), (20, 21, 3, 'SE', 4), (21, 22, 3, 'SE', 4), (16, 17, 4, 'SE', 4), (17, 18, 4, 'SE', 4), (18, 19, 4, 'SE', 4), (19, 20, 4, 'Rz', 0.01), (20, 21, 4, 'SE', 4), (21, 22, 4, 'SE', 4), (22, 22, -1, 'Barrier', None), (22, 23, 0, 'CNOT', 4), (23, 24, 0, 'SE', 4), (24, 24, 0, 'RUS_succsss', 4), (24, 24, -1, 'Barrier', None)]"
    # assert str(log) == reference_log, "Execution log does not match reference log."
    print("\n" + "=" * 70)
    print(f"TOTAL CIRCUIT EXECUTION TIME: {total_time} moments")
    print("=" * 70)

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
        visualize_rus=True,
        # prefix="trivial_return_",
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
