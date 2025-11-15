"""
Configuration module for transversal star compilation.

Manages all configuration constants and provides functions to update them.
"""

import numpy as np

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
TMR_P = 3
TMR_Q = 2
TMR_PREPARATION_TIME = TMR_P + TMR_Q + 1  # 2 SE + Rz + 3 SE
CNOT_TIME = 1
SE_TIME = 1
INJECTION_SUCCESS_RATE = 0.5
LOOKAHEAD_THRESHOLD = 2  # Max factories working on same angle
LOOKAHEAD_LEVEL = 2
PRECISION = 8
LARGE_ANGLE_LIMIT = round(np.pi / 2, PRECISION)  # time to inject S gate


def update_config(**kwargs):
    """
    Update configuration constants.

    Parameters:
        **kwargs: Configuration key-value pairs to update.
                 Valid keys: TMR_P, TMR_Q, CNOT_TIME, SE_TIME,
                            INJECTION_SUCCESS_RATE, LOOKAHEAD_THRESHOLD,
                            LOOKAHEAD_LEVEL, PRECISION

    Example:
        update_config(TMR_P=4, INJECTION_SUCCESS_RATE=0.7)

    Note:
        - TMR_PREPARATION_TIME is automatically recalculated if TMR_P or TMR_Q change
        - LARGE_ANGLE_LIMIT is automatically recalculated if PRECISION changes
    """
    global TMR_P, TMR_Q, TMR_PREPARATION_TIME, CNOT_TIME, SE_TIME
    global INJECTION_SUCCESS_RATE, LOOKAHEAD_THRESHOLD, LOOKAHEAD_LEVEL
    global PRECISION, LARGE_ANGLE_LIMIT

    valid_keys = {
        "TMR_P",
        "TMR_Q",
        "CNOT_TIME",
        "SE_TIME",
        "INJECTION_SUCCESS_RATE",
        "LOOKAHEAD_THRESHOLD",
        "LOOKAHEAD_LEVEL",
        "PRECISION",
    }

    # Validate keys
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(f"Invalid configuration keys: {invalid_keys}")

    # Update the provided values
    if "TMR_P" in kwargs:
        TMR_P = kwargs["TMR_P"]
    if "TMR_Q" in kwargs:
        TMR_Q = kwargs["TMR_Q"]
    if "CNOT_TIME" in kwargs:
        CNOT_TIME = kwargs["CNOT_TIME"]
    if "SE_TIME" in kwargs:
        SE_TIME = kwargs["SE_TIME"]
    if "INJECTION_SUCCESS_RATE" in kwargs:
        INJECTION_SUCCESS_RATE = kwargs["INJECTION_SUCCESS_RATE"]
    if "LOOKAHEAD_THRESHOLD" in kwargs:
        LOOKAHEAD_THRESHOLD = kwargs["LOOKAHEAD_THRESHOLD"]
    if "LOOKAHEAD_LEVEL" in kwargs:
        LOOKAHEAD_LEVEL = kwargs["LOOKAHEAD_LEVEL"]
    if "PRECISION" in kwargs:
        PRECISION = kwargs["PRECISION"]

    # Recalculate dependent constants
    TMR_PREPARATION_TIME = TMR_P + TMR_Q + 1
    LARGE_ANGLE_LIMIT = round(np.pi / 2, PRECISION)


def get_config():
    """
    Get current configuration as a dictionary.

    Returns:
        dict: All current configuration values
    """
    return {
        "TMR_P": TMR_P,
        "TMR_Q": TMR_Q,
        "TMR_PREPARATION_TIME": TMR_PREPARATION_TIME,
        "CNOT_TIME": CNOT_TIME,
        "SE_TIME": SE_TIME,
        "INJECTION_SUCCESS_RATE": INJECTION_SUCCESS_RATE,
        "LOOKAHEAD_THRESHOLD": LOOKAHEAD_THRESHOLD,
        "LOOKAHEAD_LEVEL": LOOKAHEAD_LEVEL,
        "PRECISION": PRECISION,
        "LARGE_ANGLE_LIMIT": LARGE_ANGLE_LIMIT,
    }


def reset_config():
    """Reset all configuration constants to their default values."""
    update_config(
        TMR_P=3,
        TMR_Q=2,
        CNOT_TIME=1,
        SE_TIME=1,
        INJECTION_SUCCESS_RATE=0.5,
        LOOKAHEAD_THRESHOLD=2,
        LOOKAHEAD_LEVEL=2,
        PRECISION=8,
    )
