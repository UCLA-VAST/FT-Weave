import numpy as np

from . import config as tcfg

from src.ds import TFactoryPool


def simulate_stage1_preparation(
    factory_pool: TFactoryPool,
    rng: np.random.Generator,
    factory_id_list: list[int] | None = None,
) -> None:
    """Draw independent Bernoulli outcomes for each subfactory (stage 1 only)."""

    if not factory_id_list:
        factory_id_list = [
            factory.id for factory in factory_pool.get_stage_1_factories()
        ]
    # print(f"Simulating stage 1 preparation for {len(factory_id_list)} factories")
    for factory_id in factory_id_list:
        factory = factory_pool.get_factory_by_id(factory_id)
        # print(f"Factory {factory_id} has {len(factory.subfactories)} subfactories")
        for sf in factory.subfactories:
            sf.stage_1_success = rng.random() < tcfg.STAGE_1_SUCCESS_RATE
            # print(f"Subfactory {sf.index} has stage 1 success: {sf.stage_1_success}")

    # input()


def simulate_stage2_preparation(
    factory_pool: TFactoryPool,
    rng: np.random.Generator,
    factory_id_list: list[int] | None = None,
) -> list[int]:
    """Simulate stage 2 success based on success rate."""
    if not factory_id_list:
        factory_id_list = [
            factory.id for factory in factory_pool.get_stage_2_factories()
        ]
    success_factory_ids = []
    for factory_id in factory_id_list:
        factory = factory_pool.get_factory_by_id(factory_id)
        outcome = rng.random() < tcfg.STAGE_2_SUCCESS_RATE
        factory.set_stage_2_state_outcome(outcome)
        if outcome:
            factory.set_to_wait_for_rus()  # Only proceed to RUS if stage 2 is successful
            success_factory_ids.append(factory_id)
        else:
            factory.free()
    return success_factory_ids


def simulate_teleportation(rng: np.random.Generator):
    """Simulate injection success (50% probability)."""
    return rng.random() < tcfg.TELEPORTATION_SUCCESS_RATE


def simulate_RUS_teleportation(
    qubit_factory_pairs: list[tuple[int, int]],
    factory_pool: TFactoryPool,
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
        injection_result = simulate_teleportation(rng)
        factory = factory_pool.get_factory_by_id(factory_id)
        factory.set_rus_state(injection_result)
        rus_simulation.append(injection_result)
    return rus_simulation
