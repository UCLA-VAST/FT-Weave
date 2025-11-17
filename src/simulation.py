import random
from .config import (
    INJECTION_SUCCESS_RATE,
)

from .device_state import FactoryPool

random.seed(42)


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def calculate_success_rate(angle):
    """Calculate success rate for angle preparation (decreases with angle)."""
    return 0.6 - angle // 10


def simulate_angle_preparation(success_rate):
    """Simulate angle preparation success based on success rate."""
    return random.random() < success_rate


def simulate_injection():
    """Simulate injection success (50% probability)."""
    return random.random() < INJECTION_SUCCESS_RATE


def simulate_TMR_preparation(factory_pool: FactoryPool):
    """
    Simulate TMR preparation success for each factory.

    Args:
        factory_pool: FactoryPool object containing Factory objects

    """
    results = []
    for factory in factory_pool.get_tmr_factories():
        if factory.success_rate is not None:
            results.append(simulate_angle_preparation(factory.success_rate))
            factory.set_tmr_state(results[-1])
        else:
            factory.set_tmr_state(False)
            results.append(False)


def simulate_RUS_injection(factory_pool: FactoryPool):
    """
    Simulate TMR preparation success for each factory.

    Args:
        qubit_factory_pairs: List of (qubit, factory_ids)

    """
    for factory in factory_pool.get_rus_factories():
        injection_result = simulate_injection()
        factory.set_rus_state(injection_result)
