import numpy as np
from .config import (
    INJECTION_SUCCESS_RATE,
)

from .ds.device_state import FactoryPool


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def calculate_success_rate(angle, code_distance):
    """Calculate success rate for angle preparation (decreases with angle)."""

    s = np.sin(angle / 2)
    c = np.cos(angle / 2)
    basic_success_rate = s ** (2 * code_distance) + c ** (2 * code_distance)
    initialization_success_rate = 1  # ! to be extracted from the figure
    return basic_success_rate


def simulate_angle_preparation(success_rate, rng: np.random.Generator):
    """Simulate angle preparation success based on success rate."""
    return rng.random() < success_rate


def simulate_injection(rng: np.random.Generator):
    """Simulate injection success (50% probability)."""
    return rng.random() < INJECTION_SUCCESS_RATE


def simulate_TMR_preparation(
    factory_pool: FactoryPool,
    rng: np.random.Generator,
    factory_id_list: list[int] | None = None,
):
    """
    Simulate TMR preparation success for each factory.

    Args:
        factory_pool: FactoryPool object containing Factory objects

    """
    # count = 0
    if not factory_id_list:
        factory_id_list = [
            factory.id for factory in factory_pool.get_tmr_after_rz_factories()
        ]
    for factory_id in factory_id_list:
        factory = factory_pool.get_factory_by_id(factory_id)
        if factory.success_rate is not None:
            outcome = simulate_angle_preparation(factory.success_rate, rng)
            factory.set_tmr_state(outcome)
            # if outcome:
            #     count += 1
        else:
            factory.set_tmr_state(False)
    # n_total = len(factory_id_list)
    # print(f"TMR preparation success count: {count} out of {n_total}")


def simulate_RUS_injection(
    qubit_factory_pairs: list[tuple[int, int]],
    factory_pool: FactoryPool,
    rng: np.random.Generator,
) -> list[bool]:
    """
    Simulate TMR preparation success for each factory.

    Args:
        qubit_factory_pairs: list[tuple(qubit, factory_ids)]
        factory_pool: FactoryPool object containing Factory objects

    Returns:
        rus_simulation: List of bool indicating RUS injection success for each qubit

    """
    rus_simulation = []
    for qubit, factory_id in qubit_factory_pairs:
        injection_result = simulate_injection(rng)
        factory = factory_pool.get_factory_by_id(factory_id)
        factory.set_rus_state(injection_result)
        rus_simulation.append(injection_result)
    return rus_simulation
