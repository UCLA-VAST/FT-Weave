"""
Configuration module for T cultivation compilation.

Manages all configuration constants and provides functions to update them.
"""

import numpy as np

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# SE_STAGE_1 = 10
SE_STAGE_1 = 8  # for MSC-3
SE_STAGE_2 = 4
CNOT_TIME = 1
SE_TIME = 1
TELEPORTATION_SUCCESS_RATE = 0.5
STAGE_1_SUCCESS_RATE = 0.25  # will be fixed
STAGE_2_SUCCESS_RATE = (
    0.40  # derived by the overall post-selection success rate / stage 1 success rate
)
ANGLE_S = np.pi / 2
SYNCHRONIZE_FACTORY_EXECUTION = True

# Physical factory holds `FACTORY_PHYSICAL_SIZE` units; stage 1 uses `STAGE_1_RESOURCE_UNITS`
# per sub-line. Number of parallel subfactories is max(1, FACTORY_PHYSICAL_SIZE // STAGE_1_RESOURCE_UNITS).
FACTORY_PHYSICAL_SIZE = 4
STAGE_1_RESOURCE_UNITS = 1

# Weight for (longest-path-to-sink) term in RUS qubit–factory assignment cost:
# cost = move_duration + weight * (batch_max_lp - lp[node]). Higher lp => lower extra cost.
RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT = 1.0


def compute_num_subfactories(physical_size: int, stage1_resource_units: int) -> int:
    """Return k = max(1, physical_size // stage1_resource_units) with a safe denominator."""
    if stage1_resource_units <= 0:
        return 1
    return max(1, physical_size // stage1_resource_units)


def update_config(**kwargs):
    """
    Update configuration constants.

    Parameters:
        **kwargs: Configuration key-value pairs to update.
                 Valid keys include: SE_STAGE_1, SE_STAGE_2, CNOT_TIME, SE_TIME,
                 TELEPORTATION_SUCCESS_RATE, STAGE_1_SUCCESS_RATE, STAGE_2_SUCCESS_RATE,
                 ANGLE_S, FACTORY_PHYSICAL_SIZE, STAGE_1_RESOURCE_UNITS,
                 SYNCHRONIZE_FACTORY_EXECUTION, RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT.

    Example:
        update_config(SE_STAGE_1=8, TELEPORTATION_SUCCESS_RATE=0.7)
    """
    global SE_STAGE_1, SE_STAGE_2, CNOT_TIME, SE_TIME
    global TELEPORTATION_SUCCESS_RATE, STAGE_1_SUCCESS_RATE, STAGE_2_SUCCESS_RATE, ANGLE_S
    global FACTORY_PHYSICAL_SIZE, STAGE_1_RESOURCE_UNITS, SYNCHRONIZE_FACTORY_EXECUTION
    global RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT

    valid_keys = {
        "SE_STAGE_1",
        "SE_STAGE_2",
        "CNOT_TIME",
        "SE_TIME",
        "TELEPORTATION_SUCCESS_RATE",
        "STAGE_1_SUCCESS_RATE",
        "STAGE_2_SUCCESS_RATE",
        "ANGLE_S",
        "FACTORY_PHYSICAL_SIZE",
        "STAGE_1_RESOURCE_UNITS",
        "SYNCHRONIZE_FACTORY_EXECUTION",
        "RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT",
    }

    # Validate keys
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(f"Invalid configuration keys: {invalid_keys}")

    # Update the provided values
    if "SE_STAGE_1" in kwargs:
        SE_STAGE_1 = kwargs["SE_STAGE_1"]
    if "SE_STAGE_2" in kwargs:
        SE_STAGE_2 = kwargs["SE_STAGE_2"]
    if "CNOT_TIME" in kwargs:
        CNOT_TIME = kwargs["CNOT_TIME"]
    if "SE_TIME" in kwargs:
        SE_TIME = kwargs["SE_TIME"]
    if "TELEPORTATION_SUCCESS_RATE" in kwargs:
        TELEPORTATION_SUCCESS_RATE = kwargs["TELEPORTATION_SUCCESS_RATE"]
    if "STAGE_1_SUCCESS_RATE" in kwargs:
        STAGE_1_SUCCESS_RATE = kwargs["STAGE_1_SUCCESS_RATE"]
    if "STAGE_2_SUCCESS_RATE" in kwargs:
        STAGE_2_SUCCESS_RATE = kwargs["STAGE_2_SUCCESS_RATE"]
    if "ANGLE_S" in kwargs:
        ANGLE_S = kwargs["ANGLE_S"]
    if "FACTORY_PHYSICAL_SIZE" in kwargs:
        FACTORY_PHYSICAL_SIZE = kwargs["FACTORY_PHYSICAL_SIZE"]
    if "STAGE_1_RESOURCE_UNITS" in kwargs:
        STAGE_1_RESOURCE_UNITS = kwargs["STAGE_1_RESOURCE_UNITS"]
    if "SYNCHRONIZE_FACTORY_EXECUTION" in kwargs:
        SYNCHRONIZE_FACTORY_EXECUTION = kwargs["SYNCHRONIZE_FACTORY_EXECUTION"]
    if "RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT" in kwargs:
        RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT = kwargs["RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT"]


def get_config():
    """
    Get current configuration as a dictionary.

    Returns:
        dict: All current configuration values
    """
    return {
        "SE_STAGE_1": SE_STAGE_1,
        "SE_STAGE_2": SE_STAGE_2,
        "CNOT_TIME": CNOT_TIME,
        "SE_TIME": SE_TIME,
        "TELEPORTATION_SUCCESS_RATE": TELEPORTATION_SUCCESS_RATE,
        "STAGE_1_SUCCESS_RATE": STAGE_1_SUCCESS_RATE,
        "STAGE_2_SUCCESS_RATE": STAGE_2_SUCCESS_RATE,
        "ANGLE_S": ANGLE_S,
        "FACTORY_PHYSICAL_SIZE": FACTORY_PHYSICAL_SIZE,
        "STAGE_1_RESOURCE_UNITS": STAGE_1_RESOURCE_UNITS,
        "SYNCHRONIZE_FACTORY_EXECUTION": SYNCHRONIZE_FACTORY_EXECUTION,
        "RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT": RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT,
    }


def reset_config():
    """Reset all configuration constants to their default values."""
    update_config(
        SE_STAGE_1=6,
        SE_STAGE_2=6,
        CNOT_TIME=1,
        SE_TIME=1,
        TELEPORTATION_SUCCESS_RATE=0.5,
        STAGE_1_SUCCESS_RATE=0.25,
        STAGE_2_SUCCESS_RATE=0.40,
        ANGLE_S=np.pi / 2,
        FACTORY_PHYSICAL_SIZE=1,
        STAGE_1_RESOURCE_UNITS=1,
        SYNCHRONIZE_FACTORY_EXECUTION=True,
        RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT=1.0,
    )
