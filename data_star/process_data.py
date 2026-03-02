import numpy as np
import pickle
from scipy.optimize import curve_fit

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

fitted_params_6d = fit_data_series(success_data_6d, "fig_6_d (suc3, suc5, suc7)")

# ============================================================================
# Process fig_6_b
# ============================================================================
with open("data_star/fig_6_b", "rb") as f:
    data_6b = pickle.load(f)

success_data_6b = {
    "inf3": data_6b["inf3"],
    "inf5": data_6b["inf5"],
    "inf7": data_6b["inf7"],
}

# fitted_params_6b = fit_data_series(success_data_6b, "fig_6_b (inf3, inf5, inf7)")
