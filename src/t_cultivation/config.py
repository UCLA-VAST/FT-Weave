"""
Configuration module for T cultivation compilation.

Manages all configuration constants and provides functions to update them.
"""

import numpy as np

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
SE_STAGE_1 = 6
SE_STAGE_2 = 6
CNOT_TIME = 1
SE_TIME = 1
TELEPORTATION_SUCCESS_RATE = 0.5
STAGE_1_SUCCESS_RATE = 0.5
STAGE_2_SUCCESS_RATE = 0.5
ANGLE_S = np.pi / 2


def update_config(**kwargs):
    """
    Update configuration constants.

    Parameters:
        **kwargs: Configuration key-value pairs to update.
                 Valid keys: SE_STAGE_1, SE_STAGE_2, CNOT_TIME, SE_TIME,
                            TELEPORTATION_SUCCESS_RATE, ANGLE_S

    Example:
        update_config(SE_STAGE_1=8, TELEPORTATION_SUCCESS_RATE=0.7)
    """
    global SE_STAGE_1, SE_STAGE_2, CNOT_TIME, SE_TIME
    global TELEPORTATION_SUCCESS_RATE, ANGLE_S

    valid_keys = {
        "SE_STAGE_1",
        "SE_STAGE_2",
        "CNOT_TIME",
        "SE_TIME",
        "TELEPORTATION_SUCCESS_RATE",
        "ANGLE_S",
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
    if "ANGLE_S" in kwargs:
        ANGLE_S = kwargs["ANGLE_S"]


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
        "ANGLE_S": ANGLE_S,
    }


def reset_config():
    """Reset all configuration constants to their default values."""
    update_config(
        SE_STAGE_1=6,
        SE_STAGE_2=6,
        CNOT_TIME=1,
        SE_TIME=1,
        TELEPORTATION_SUCCESS_RATE=0.5,
        ANGLE_S=np.pi / 2,
    )
