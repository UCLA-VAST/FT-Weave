import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.optimize import brentq

x_value = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
x_value_array = np.array(x_value)


def power_law(x, a, b, c):
    """Power law: ax^b + c"""
    return a * x**b + c


def fit_data_series(data_dict, file_name):
    """
    Fit multiple data series from a data dictionary.

    Args:
        data_dict: Dictionary containing data arrays keyed by name
        file_name: Name of the file being processed (for display)
    """
    print("=" * 80)
    print(f"FUNCTION FITTING FOR {file_name}")
    print("=" * 80)
    print()

    fitted_params = {}

    # Extract and display raw data
    print("Raw data points:")
    for key in data_dict.keys():
        print(f"{key}: {np.array(data_dict[key])}")
    print()

    # Fit functions for each success rate series
    for name, y_data in data_dict.items():
        y_data = np.array(y_data)
        print("=" * 80)
        print(f"Fitting results for {name}:")
        print("=" * 80)

        fitted_params[name] = {}

        # Power law fit
        try:
            popt_power, pcov_power = curve_fit(
                # power_law, x_value_array, y_data, p0=[7.3e-4, 2, 0], maxfev=10000
                power_law,
                x_value_array,
                y_data,
                p0=[-2, 1, 1],
                maxfev=10000,
            )
            pred_power = power_law(x_value_array, *popt_power)
            residual_power = np.sum((y_data - pred_power) ** 2)
            r2_power = 1 - (
                np.sum((y_data - pred_power) ** 2)
                / np.sum((y_data - np.mean(y_data)) ** 2)
            )
            print(f"\nPower law (ax^b + c):")
            print(
                f"  Parameters: a={popt_power[0]:.6e}, b={popt_power[1]:.6f}, c={popt_power[2]:.6f}"
            )
            print(f"  Residual: {residual_power:.6e}, R²: {r2_power:.6f}")
            fitted_params[name]["power_law"] = {
                "params": popt_power,
                "residual": residual_power,
                "r2": r2_power,
            }
        except Exception as e:
            print(f"\nPower law fit failed: {e}")

        print()

    # Print summary
    print("=" * 80)
    print(f"SUMMARY - Best fits by R² value ({file_name}):")
    print("=" * 80)
    for name, fits in fitted_params.items():
        print(f"\n{name}:")
        for fit_type, data in sorted(
            fits.items(), key=lambda x: x[1]["r2"], reverse=True
        ):
            print(
                f"  {fit_type:15s}: R² = {data['r2']:.6f}, Residual = {data['residual']:.6e}"
            )

    print("\n")
    return fitted_params


# ============================================================================
# Process fig_6_d
# ============================================================================
with open("data_star/fig_6_d", "rb") as f:
    data_6d = pickle.load(f)

success_data_6d = {
    "suc3": data_6d["suc3"],
    "suc5": data_6d["suc5"],
    "suc7": data_6d["suc7"],
}

# fitted_params_6d = fit_data_series(success_data_6d, "fig_6_d (suc3, suc5, suc7)")

# # ============================================================================
# # Process fig_6_b
# # ============================================================================
# with open("data_star/fig_6_b", "rb") as f:
#     data_6b = pickle.load(f)

# success_data_6b = {
#     "inf3": data_6b["inf3"],
#     "inf5": data_6b["inf5"],
#     "inf7": data_6b["inf7"],
# }
# print(data_6b)
# fitted_params_6b = fit_data_series(success_data_6b, "fig_6_b (inf3, inf5, inf7)")


def logical_from_physical(physical_angle: float, k: int) -> float:
    s = np.sin(physical_angle / 2)
    c = np.cos(physical_angle / 2)

    num = s**k
    den = np.sqrt(s ** (2 * k) + c ** (2 * k))

    x = num / den

    # numerical safety (VERY important)
    x = np.clip(x, -1.0, 1.0)

    return 2 * np.arcsin(x)


# -------------------------------------------------------
# Inverse mapping: logical -> physical
# -------------------------------------------------------
def convert_logical_angle_to_physical_angle(
    logical_angle: float, d: int, tol: float = 1e-12
) -> float:
    """
    Numerically solve for physical angle θ*
    given logical angle θ.
    """
    if not (0 <= logical_angle <= np.pi):
        raise ValueError(f"Logical angle must lie in [0, π] but got {logical_angle}")

    # Root function:
    k = d // 2

    def root_fn(physical_angle):
        return logical_from_physical(physical_angle, k) - logical_angle

    # avoid exact endpoints (numerical issue)
    eps = 1e-12
    a = eps
    b = np.pi - eps

    fa = root_fn(a)
    fb = root_fn(b)

    # sanity check
    if fa * fb > 0:
        raise RuntimeError(
            "Root not bracketed — logical angle may be numerically unreachable."
        )

    # Physical angle lies in [0, π]
    physical_angle, result = brentq(root_fn, 0.0, np.pi, xtol=tol, full_output=True)
    recovered_logical_angle = logical_from_physical(physical_angle, d // 2)
    assert np.isclose(
        logical_angle, recovered_logical_angle, atol=tol
    ), f"Failed to recover logical angle: expected {logical_angle}, got {recovered_logical_angle}"
    return physical_angle


def get_p_ideal(physical_theta, d):
    """Calculate the ideal success probability for a given physical angle and code distance."""
    k = d // 2
    p_ideal = np.sin(physical_theta) ** (2 * k) + np.cos(physical_theta) ** (2 * k)
    return p_ideal


initialization_errors_distance = []
for key, value in success_data_6d.items():
    d = int(key[-1])  # Extract code distance from key (e.g., "suc3" -> 3)
    print(f"\nCalculating initialization error for {key} with code distance {d}:")
    initialization_errors = []
    for i in range(len(value)):
        physical_theta = convert_logical_angle_to_physical_angle(x_value_array[i], d)
        p_ideal = get_p_ideal(physical_theta, d)
        initialization_error = value[i] / p_ideal
        print(
            f"initialization error for {key} at physical theta {physical_theta:.3e}: {initialization_error:.3e}"
        )
        initialization_errors.append(initialization_error)
    initialization_errors_distance.append(
        sum(initialization_errors) / len(initialization_errors)
    )
    print(
        f"Avg initialization error for {key}: {sum(initialization_errors) / len(initialization_errors):.3e}"
    )
    if len(initialization_errors_distance) > 1:
        ratio = initialization_errors_distance[-1] / initialization_errors_distance[-2]
        print(
            f"Initialization error ratio between distance {d} and previous distance: {ratio:.3f}"
        )
