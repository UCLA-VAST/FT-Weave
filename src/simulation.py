import random
from .config import (
    INJECTION_SUCCESS_RATE,
)

random.seed(42)


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def simulate_angle_preparation(success_rate):
    """Simulate angle preparation success based on success rate."""
    return random.random() < success_rate


def simulate_injection():
    """Simulate injection success (50% probability)."""
    return random.random() < INJECTION_SUCCESS_RATE


def simulate_TMR_preparation(
    factory_assignments: list[tuple[float, int, float]],
) -> list[bool]:
    """
    Simulate TMR preparation success for each factory.

    Args:
        factory_assignments: List of (angle, qubit, success_rate) tuples for each factory

    Returns:
        List of boolean values indicating whether each factory's TMR preparation was successful.
    """
    return [
        simulate_angle_preparation(success_rate)
        for _, _, success_rate in factory_assignments
    ]


def simulate_RUS_injection(
    qubit_factory_pairs: list[tuple[int, int]],
) -> list[bool]:
    """
    Simulate TMR preparation success for each factory.

    Args:
        qubit_factory_pairs: List of (qubit, factory_ids)

    Returns:
        List of boolean values indicating whether each factory's TMR preparation was successful.
    """
    return [simulate_injection() for _, _ in qubit_factory_pairs]
