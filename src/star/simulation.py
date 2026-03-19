import numpy as np
from .config import (
    TELEPORTATION_SUCCESS_RATE,
)

from src.ds.device_state_star import (
    FactoryPool,
    convert_logical_angle_to_physical_angle,
)


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================
def calculate_success_rate(angle, code_distance):
    """Calculate success rate for angle preparation (decreases with angle)."""
    angle = abs(angle)  # !
    if code_distance == 3:
        a = -8.937874e-01
        b = 0.469195
        c = 0.931988
    elif code_distance == 5:
        a = -6.750004e-01
        b = 0.469195
        c = 0.703842
    elif code_distance == 7:
        a = -4.743114e-01
        b = 0.469195
        c = 0.494559
    elif code_distance == 9:
        initialization_error = 0.7872
        k = code_distance // 2
        physical_theta = convert_logical_angle_to_physical_angle(angle, code_distance)
        p_ideal = np.sin(physical_theta) ** (2 * k) + np.cos(physical_theta) ** (2 * k)
        return initialization_error * p_ideal
    else:
        raise ValueError(f"Unsupported code distance: {code_distance}")

    success_rate = a * angle**b + c
    # assert angle <= 1e-1, f"Angle {angle} is out of expected range (should be <= 0.1)"
    assert (
        0 <= success_rate <= 1
    ), f"Calculated success rate {success_rate} is out of bounds for angle {angle} and code distance {code_distance}"
    return success_rate


def simulate_angle_preparation(success_rate, rng: np.random.Generator):
    """Simulate angle preparation success based on success rate."""
    return rng.random() < success_rate


def simulate_injection(rng: np.random.Generator):
    """Simulate injection success (50% probability)."""
    return rng.random() < TELEPORTATION_SUCCESS_RATE


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
