import os
import sys
import logging

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

np.random.seed(1234)

# Ensure repository root is on sys.path so `src` is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

from src.analog_rotation_execution import factory_angle_execution
from src.analog_rotation_execution_parallel import factory_angle_execution_parallel
from src.ds import FactoryPool, get_microarchitecture
from src.util import analyze_execution_log, print_execution_profile
from src.animator import (
    plot_all_rus_rounds,
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

    # build facoty_qubit_map
    facoty_qubit_map = []
    max_x = 0
    max_y = 0
    for x, y in logic_qubit_locations:
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    for x, y in magic_state_locations:
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    for x in range(max_x + 1):
        facoty_qubit_map.append([0 for _ in range(max_y + 1)])
    for x, y in logic_qubit_locations:
        facoty_qubit_map[x][y] = 1
    for x, y in magic_state_locations:
        facoty_qubit_map[x][y] = 2

    logger.info("=" * 70)
    logger.info("MAGIC STATE FACTORY ANGLE EXECUTION SIMULATION")
    logger.info("=" * 70)
    logger.info(f"Number of factories: {n_factories}")
    logger.info(f"Target qubit angles: {target_qubits_angles}")
    logger.info("=" * 70 + "\n")

    # Run simulation
    factory_pool = FactoryPool(num_factories=n_factories)
    total_time, log = factory_angle_execution(
        factory_pool,
        target_qubits_angles,
        logic_qubit_locations,
        magic_state_locations,
        code_distance=7,
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
    n_aods_se: int = 1,
    consider_skip_rus: bool = False,
    tmr_assignment_method: str = "matching",
    trivial_return: bool = False,
    parallel_execution: bool = False,
):
    target_qubits_angles = {}
    if same_angle:
        angle = 0.001
        for i in range(n_qubits):
            target_qubits_angles[i] = angle
    else:
        for i in range(n_qubits):
            target_qubits_angles[i] = np.random.uniform(0.0001, 0.001)

    n_columns, n_rows = qubit_layout
    column_based_placement = False
    if placement == "seperate_region_col" or placement == "col_based":
        column_based_placement = True

    logic_qubit_locations, magic_state_locations = get_microarchitecture(
        n_qubits,
        n_factories,
        qubit_layout,
        placement,
    )

    logger.info("=" * 70)
    logger.info("MAGIC STATE FACTORY ANGLE EXECUTION SIMULATION")
    logger.info("=" * 70)
    logger.info(f"Number of factories: {n_factories}")
    logger.info(f"Target qubit angles: {target_qubits_angles}")
    logger.info("=" * 70 + "\n")

    # Run simulation
    factory_pool = FactoryPool(num_factories=n_factories)
    if parallel_execution:
        total_time, log = factory_angle_execution_parallel(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            code_distance=7,
            column_based_placement=column_based_placement,
            n_aods=n_aods,
            n_aods_se=n_aods_se,
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method=tmr_assignment_method,
            trivial_return=trivial_return,
            rng=np.random.default_rng(42),
        )
    else:
        total_time, log = factory_angle_execution(
            factory_pool,
            target_qubits_angles,
            logic_qubit_locations,
            magic_state_locations,
            code_distance=7,
            column_based_placement=column_based_placement,
            n_aods=n_aods,
            consider_skip_rus=consider_skip_rus,
            tmr_assignment_method=tmr_assignment_method,
            trivial_return=trivial_return,
            rng=np.random.default_rng(42),
        )

    profiling_result = analyze_execution_log(log, n_factories=n_factories)
    print_execution_profile(profile=profiling_result)

    # pdf_path based on the arguments
    base_path = f"{prefix}_n{n_qubits}f{n_factories}_{placement}_naod_{n_aods}_tmr_{tmr_assignment_method}_skipRUS_{consider_skip_rus}_trivial-return{trivial_return}.pdf"
    pdf_path = f"output/circuit_execution_vertical/{base_path}"
    plot_circuit_execution_vertical(
        log, n_factories, figure_height=50, save_path=pdf_path
    )
    pdf_path = f"output/circuit_execution/{base_path}"
    plot_circuit_execution(log, n_factories, figure_width=50, save_path=pdf_path)

    if not parallel_execution and visualize_rus:
        # Generate RUS round visualizations
        pdf_path = os.path.join("output", f"rus_rounds_detailed/{base_path}")
        plot_all_rus_rounds(
            execution_log=log,
            logic_qubit_locations=logic_qubit_locations,
            magic_state_locations=magic_state_locations,
            base_path=pdf_path,
        )

        logger.info("Visualization complete!")
        logger.info(f"  Detailed PDF: {pdf_path}")


if __name__ == "__main__":
    # test_small()
    # test(
    #     n_qubits=25,
    #     n_factories=25,
    #     qubit_layout=(5, 5),
    #     # n_qubits=9,
    #     # n_factories=9,
    #     # qubit_layout=(3, 3),
    #     same_angle=True,
    #     # placement="seperate_region_row",
    #     # placement="seperate_region_col",
    #     # placement="row_based",
    #     placement="col_based",
    #     # placement="checkerboard",
    #     visualize_rus=True,
    #     # prefix="no_skip_tmr_matching_aod_2_",
    #     prefix="reassign_aod_2_",
    #     # prefix="checkerboard_aod_2_",
    #     n_aods=2,
    #     # consider_skip_rus=False,
    #     consider_skip_rus=True,
    # )

    # test(
    #     n_qubits=25,
    #     n_factories=25,
    #     qubit_layout=(5, 5),
    #     # n_qubits=9,
    #     # n_factories=9,
    #     # qubit_layout=(3, 3),
    #     same_angle=True,
    #     # placement="seperate_region_row",
    #     # placement="seperate_region_col",
    #     # placement="row_based",
    #     # placement="col_based",
    #     placement="checkerboard",
    #     visualize_rus=True,
    #     prefix="",
    #     n_aods=1,
    #     consider_skip_rus=False,
    #     tmr_assignment_method="matching",
    #     trivial_return=False,
    # )

    test(
        n_qubits=25,
        n_factories=25,
        qubit_layout=(5, 5),
        same_angle=True,
        placement="col_based",
        visualize_rus=False,
        prefix="",
        n_aods=3,
        n_aods_se=3,
        consider_skip_rus=False,
        tmr_assignment_method="matching",
        trivial_return=False,
        parallel_execution=True,
    )
