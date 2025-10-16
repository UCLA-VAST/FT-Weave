import csv
from typing import List
from src.simulation_surface_code import SimulationResult, SurfaceCodeResourceState
import math


def simulate(code_distance: int, theta: float, p_ph: float, k: int) -> SimulationResult:
    # Create simulator
    simulator = SurfaceCodeResourceState(
        code_distance=code_distance, theta=theta, p_ph=p_ph
    )

    # Run simulation
    result = simulator.run_simulation(n_shots=1)

    return result


def run_and_save_csv(
    distances: List[int],
    angles: List[float],
    p_ph: float,
    k: int,
    csv_filename: str = "simulation_results.csv",
):
    # CSV Header
    fieldnames = ["angle", "distance", "fidelity", "success rate", "space time"]

    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for theta in angles:
            for d in distances:
                result = simulate(d, theta, p_ph, k)

                # Derived quantities
                fidelity = 1 - result.infidelity
                success_rate = result.success_rate
                # Example: compute space-time volume as total passes
                space_time = sum(result.passes_per_weight.values())

                writer.writerow(
                    {
                        "angle": theta,
                        "distance": d,
                        "fidelity": fidelity,
                        "success rate": success_rate,
                        "space time": space_time,
                    }
                )

    print(f"Results saved to {csv_filename}")


# Example usage
if __name__ == "__main__":
    angles = [math.pi / (2**i) for i in range(2, 11)]
    run_and_save_csv(distances=[3, 5, 7], angles=[0.05, 0.1, 0.15], p_ph=0.001, k=2)
