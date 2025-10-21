import csv
from typing import List
from src.simulation_surface_code import SimulationResult, SurfaceCodeResourceState
import math
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
import pickle
import os 

def simulate(code_distance: int, angle: float, physical_angle: float, p_ph: float, pauli_weight: int) -> SimulationResult:
    # Create simulator
    simulator = SurfaceCodeResourceState(
        code_distance=code_distance, theta=angle, physical_theta=physical_angle, p_ph=p_ph, pauli_weight=pauli_weight
    )

    # Run simulation
    result = simulator.run_simulation(n_shots=5000)

    return result


def run_and_save_csv(
    distance_angles_pairs: List[tuple[int, float, float]],
    p_ph: float,
    pauli_weight: int,
    csv_filename: str = "simulation_results.csv",
):
    # CSV Header
    fieldnames = ["angle", "physical angle", "distance", "fidelity", "success rate", "space time"]

    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for d, physical_angle, logical_angle in distance_angles_pairs:
            result = simulate(d, logical_angle, physical_angle, p_ph, pauli_weight)

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


def logical_rotation(physical_theta: float, k: int) -> float:
    p_ideal = np.sin(physical_theta) ** (2 * k) + np.cos(physical_theta) ** (2 * k)
    theta = np.arcsin(np.sin(physical_theta) ** (k) / np.sqrt(p_ideal))
    return theta


def collect_angles(
    code_distance: int, target_logical_angles: list[float], pauli_weight: int
) -> list[tuple[int, float, float]]:
    angle_pairs = []
    k = math.ceil(code_distance / pauli_weight)
    for logical_angle in target_logical_angles:
        physical_angle = find_physical_angle_by_sympy(logical_angle, k)
        if physical_angle:
            obtain_logical_rotation = logical_rotation(physical_angle, k)
            if np.isclose(logical_angle, obtain_logical_rotation):
                angle_pairs.append((code_distance, physical_angle, logical_angle))
            else:
                print(f"logical rotation is not close to the target rotation: {obtain_logical_rotation}, {logical_angle}")
    if not angle_pairs:
        for angle in [0.1, 0.01, 0.001]:
            logical_angle = logical_rotation(angle, k)
            angle_pairs.append((code_distance, angle, logical_angle))
    return angle_pairs

def find_physical_angle_by_sympy(logical_angle: float, k: int) -> float | None:
    # Define the variable
    x = sp.Symbol('x', real=True)
    # Define the equation
    # lhs = sp.asin(sp.sin(x)**2 / (sp.sin(x)**(2*k) + sp.cos(x)**(2*k)))
    # rhs = logical_angle
    lhs = sp.sin(x)**k / sp.sqrt((sp.sin(x)**(2*k) + sp.cos(x)**(2*k)))
    rhs = sp.sin(logical_angle)

    equation = sp.Eq(lhs, rhs)

    print("Equation to solve:")
    print(equation)
    print("\n" + "="*50 + "\n")

    try:
        solutions = sp.solve(equation, x)
        solutions.sort()
        print(f"Solutions: {solutions}")
        for sol in solutions:
            if sol > 0:
                return float(sol)
    except Exception as e:
        print(f"Error: {e}")
        solutions = []

    print("\n" + "="*50 + "\n")

    # ! numerical method does not work
    # Numerical approach - find solutions in [-pi, pi]
    # print("Using numerical methods to find solutions in [-π, π]:")

    # def equation_func(x_val):
    #     sin_x = np.sin(x_val)
    #     cos_x = np.cos(x_val)
    #     arg_val = sin_x**2 / (sin_x**4 + cos_x**4)
    #     # Clamp to [-1, 1] to avoid domain errors
    #     arg_val = np.clip(arg_val, -1, 1)
    #     return np.arcsin(arg_val) - np.pi/4

    # # Try multiple initial guesses
    # initial_guesses = np.linspace(-np.pi, np.pi, 20)
    # numerical_solutions = []

    # for guess in initial_guesses:
    #     try:
    #         sol = fsolve(equation_func, guess, full_output=True)
    #         x_sol = sol[0][0]
    #         info = sol[1]
            
    #         # Check if it's a valid solution in range
    #         if -np.pi <= x_sol <= np.pi and info['fvec'][0]**2 < 1e-10:
    #             # Check if this solution is new (not already found)
    #             is_new = True
    #             for existing_sol in numerical_solutions:
    #                 if abs(x_sol - existing_sol) < 1e-6:
    #                     is_new = False
    #                     break
    #             if is_new:
    #                 numerical_solutions.append(x_sol)
    #     except:
    #         pass

    # numerical_solutions.sort()

    # print(f"\nNumerical solutions found: {numerical_solutions}")
    # sol = numerical_solutions[0]
    # for sol in numerical_solutions:
    #     if sol > 0:
    #         return sol
    return None

# Example usage
if __name__ == "__main__":
    # target_logical_angles = [math.pi / (2**i) for i in range(2, 11)]
    # clifford_levels = [4, 5, 6, 7, 8, 9, 10]
    clifford_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_logical_angles = [math.pi / (2**i) for i in clifford_levels]
    pauli_weight = 1
    distance_angles_pairs = []
    code_distances = [7, 9, 11, 13]
    for d in code_distances:
        distance_angles_pairs += collect_angles(d, target_logical_angles, pauli_weight)
    run_and_save_csv(distance_angles_pairs=distance_angles_pairs, p_ph=0.001, pauli_weight=1)
