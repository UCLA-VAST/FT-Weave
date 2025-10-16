import csv
from typing import List
from src.simulation_surface_code import SimulationResult, SurfaceCodeResourceState
import math
import numpy as np


def simulate(code_distance: int, angle: float, physical_angle: float, p_ph: float, k: int) -> SimulationResult:
    # Create simulator
    simulator = SurfaceCodeResourceState(
        code_distance=code_distance, theta=angle, physical_theta=physical_angle, p_ph=p_ph
    )

    # Run simulation
    result = simulator.run_simulation(n_shots=100_000)

    return result


def run_and_save_csv(
    distance_angles_pairs: List[tuple[int, float, float]],
    p_ph: float,
    k: int,
    csv_filename: str = "simulation_results.csv",
):
    # CSV Header
    fieldnames = ["angle", "physical angle", "distance", "fidelity", "success rate", "space time"]

    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for d, physical_angle, logical_angle in distance_angles_pairs:
            result = simulate(d, logical_angle, physical_angle, p_ph, k)

            # Derived quantities
            fidelity = 1 - result.infidelity
            success_rate = result.success_rate
            space_time = 0

            writer.writerow(
                {
                    "angle": logical_angle,
                    "physical angle": physical_angle,
                    "distance": d,
                    "fidelity": fidelity,
                    "success rate": success_rate,
                    "space time": space_time,
                }
            )

    print(f"Results saved to {csv_filename}")


def logical_rotation(code_distance: int, physical_theta: float) -> float:
    k = math.ceil(code_distance / 2)
    p_ideal = np.sin(physical_theta) ** (2 * k) + np.cos(physical_theta) ** (2 * k)
    theta = np.arcsin(np.sin(physical_theta) ** (k) / np.sqrt(p_ideal))
    return theta


def collect_angles(
    code_distance: int, target_logical_angles: list[float]
) -> list[tuple[int, float, float]]:
    angle_pairs = []
    physical_rotation_candidates = np.linspace(0.001, 0.1, 1000) + np.linspace(0.1, 0.5, 1000)
    for angle in physical_rotation_candidates:
        logical_angle = logical_rotation(code_distance, angle)
        for target_logical_angle in target_logical_angles:
            if np.isclose(logical_angle, target_logical_angle):
                angle_pairs.append((code_distance, angle, logical_angle))
        if len(angle_pairs) == target_logical_angles:
            break
    if not angle_pairs:
        for angle in [0.1, 0.01, 0.001]:
            logical_angle = logical_rotation(code_distance, angle)
            angle_pairs.append((code_distance, angle, logical_angle))
    return angle_pairs


# Example usage
if __name__ == "__main__":
    # target_logical_angles = [math.pi / (2**i) for i in range(2, 11)]
    target_logical_angles = [math.pi / (2**i) for i in range(8, 11)]
    distance_angles_pairs = []
    code_distances = [3, 5, 7]
    for d in code_distances:
        distance_angles_pairs += collect_angles(d, target_logical_angles)
    run_and_save_csv(distance_angles_pairs=distance_angles_pairs, p_ph=0.001, k=2)
