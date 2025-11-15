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
