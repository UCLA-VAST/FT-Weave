import argparse
import os
import sys

import logging

import numpy as np

# Ensure repository root is on sys.path so `src` is importable when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator import plot_t_cultivation_execution
from src.t_cultivation.config import get_config, update_config
from src.t_cultivation.t_cultivation import t_cultivation_execution


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Fig1 failure repro (evaluation_t_cultivation_throughput.py, Type 1)
# Log example:
#   Fig1 failed [distance-7, LER=1e-08] nf=2: No schedulable events remain...
# Matches first TSetting (distance 7, LER 1e-8, factory_physical_size 2),
# n_factories=2, K=10 sequential T on qubit 0.
# ---------------------------------------------------------------------------

FIG1_K_PER_QUBIT = 10
# evaluation: seed = 1000000 * settings.index(setting) + 1000 * nf + trial
FIG1_SETTING_INDEX = 0
FIG1_N_FACTORIES = 2


def fig1_evaluation_seed(
    nf: int, trial: int, setting_index: int = FIG1_SETTING_INDEX
) -> int:
    return 1_000_000 * setting_index + 1_000 * nf + trial


def build_fig1_sequential_t_circuit(k: int = FIG1_K_PER_QUBIT) -> list[dict]:
    """Same as evaluation SweepSpec(n_qubits=1, k_t_per_qubit=k, layers_parallel=False)."""
    return [{"gate": "T", "targets": [0], "params": {}} for _ in range(k)]


def build_grid_locations(
    n_qubits: int, n_factories: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    logic_qubit_locations = [(x, 0) for x in range(n_qubits)]
    magic_state_locations = [(x, 1) for x in range(n_factories)]
    return logic_qubit_locations, magic_state_locations


def apply_fig1_fail_setting() -> None:
    """Same as evaluation TSetting(distance=7, LER=1e-8, factory_physical_size=2)."""
    update_config(
        STAGE_2_FIDELITY_TARGET=1e-8,
        FACTORY_PHYSICAL_SIZE=2,
        STAGE_1_RESOURCE_UNITS=1,
    )


def run_fig1_evaluator_case(
    *,
    seed: int,
    n_factories: int = FIG1_N_FACTORIES,
    n_aods: int = 2,
    to_decompose: bool = False,
    epsilon: float = 1e-4,
) -> list:
    """
    Run the exact Fig1 configuration used in evaluation_t_cultivation_throughput.py.
    Raises RuntimeError when the scheduler cannot make progress (the bug under debug).
    """
    circuit = build_fig1_sequential_t_circuit()
    logic_qubit_locations, magic_state_locations = build_grid_locations(1, n_factories)
    return t_cultivation_execution(
        circuit=circuit,
        n_factories=n_factories,
        logic_qubit_locations=logic_qubit_locations,
        magic_state_locations=magic_state_locations,
        rng=np.random.default_rng(seed),
        n_aods=n_aods,
        to_decompose=to_decompose,
        epsilon=epsilon,
    )


def find_fig1_failing_seed(
    *,
    nf: int = FIG1_N_FACTORIES,
    max_trials: int = 32,
) -> int | None:
    """Return the first trial index seed (same formula as evaluation) that raises RuntimeError."""
    for trial in range(max_trials):
        seed = fig1_evaluation_seed(nf, trial)
        try:
            run_fig1_evaluator_case(seed=seed, n_factories=nf)
        except RuntimeError:
            return seed
    return None


def run_fig1_fail_case_debug(
    *,
    seed: int | None = None,
    plot: bool = False,
) -> None:
    """
    Reproduce the Fig1 [distance-7, LER=1e-08] nf=2 failure for debugging.

    If ``seed`` is None, scans trial indices until the first RuntimeError (same as evaluation).
    Set env ``FIG1_DEBUG_SEED`` to force a specific seed.
    """
    env_seed = os.environ.get("FIG1_DEBUG_SEED")
    if env_seed is not None:
        seed = int(env_seed)

    original_cfg = get_config().copy()
    try:
        apply_fig1_fail_setting()
        if seed is None:
            found = find_fig1_failing_seed()
            if found is None:
                raise RuntimeError(
                    f"No RuntimeError in first {32} trials for nf={FIG1_N_FACTORIES}; "
                    "increase max_trials or set FIG1_DEBUG_SEED / --seed."
                )
            seed = found
            logging.info("Using first failing seed from scan: %s", seed)

        assert seed is not None
        logging.info(
            "Fig1 fail repro: nf=%s seed=%s (trial index = %s)",
            FIG1_N_FACTORIES,
            seed,
            seed - fig1_evaluation_seed(FIG1_N_FACTORIES, 0),
        )
        log = run_fig1_evaluator_case(seed=seed)
        logging.warning(
            "Expected RuntimeError but execution finished (log length=%d).", len(log)
        )
        if plot and log:
            os.makedirs("output/circuit_execution", exist_ok=True)
            plot_t_cultivation_execution(
                execution_log=log,
                n_qubits=1,
                n_factories=FIG1_N_FACTORIES,
                save_path="output/circuit_execution/t_cultivation_fig1_debug_success.pdf",
            )
    except RuntimeError:
        logging.exception(
            "Reproduced Fig1 failure (set breakpoint in t_cultivation scheduler). seed=%s",
            seed,
        )
        raise
    finally:
        update_config(**original_cfg)


def test_t_cultivation_small():
    circuit = [
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
        {"gate": "T", "targets": [0], "params": {}},
    ]

    logic_qubit_locations = [(0, 0)]
    magic_state_locations = [(0, 1)]
    n_factories = len(magic_state_locations)
    update_config(FACTORY_PHYSICAL_SIZE=2)
    for i in range(10):
        execution_log = t_cultivation_execution(
            circuit=circuit,
            n_factories=n_factories,
            logic_qubit_locations=logic_qubit_locations,
            magic_state_locations=magic_state_locations,
            rng=np.random.default_rng(i),
            n_aods=2,
            to_decompose=False,
        )
        assert execution_log, "Execution log should not be empty."
    update_config(FACTORY_PHYSICAL_SIZE=4)
    for i in range(10):
        execution_log = t_cultivation_execution(
            circuit=circuit,
            n_factories=n_factories,
            logic_qubit_locations=logic_qubit_locations,
            magic_state_locations=magic_state_locations,
            rng=np.random.default_rng(i),
            n_aods=2,
            to_decompose=False,
        )
        assert execution_log, "Execution log should not be empty."
    return
    # execution_log = t_cultivation_execution(
    #     circuit=circuit,
    #     n_factories=n_factories,
    #     logic_qubit_locations=logic_qubit_locations,
    #     magic_state_locations=magic_state_locations,
    #     rng=np.random.default_rng(42),
    #     n_aods=2,
    #     to_decompose=False,
    # )

    # assert execution_log, "Execution log should not be empty."

    plot_t_cultivation_execution(
        execution_log=execution_log,
        n_qubits=len(logic_qubit_locations),
        n_factories=n_factories,
        save_path=f"output/circuit_execution/t_cultivation_{len(logic_qubit_locations)}q_with_factories.pdf",
    )


def test_fig1_fail_case_scheduler_deadlock_reproduces() -> None:
    """
    Regression: Fig1 (Type 1: nf=2, distance-7, LER=1e-8, physical_size=2)
    should not deadlock anymore.
    """
    original_cfg = get_config().copy()
    try:
        apply_fig1_fail_setting()
        # Scan a small deterministic window (same seed formula as the evaluator).
        fail_seed = find_fig1_failing_seed(max_trials=32)
        if fail_seed is not None:
            raise AssertionError(
                "Fig1 nf=2 still deadlocks (scheduler) in first trials. "
                f"First failing seed={fail_seed}"
            )
    finally:
        update_config(**original_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="T-cultivation tests / Fig1 failure debug"
    )
    parser.add_argument(
        "--fig1-fail",
        action="store_true",
        help="Reproduce Fig1 [distance-7, LER=1e-08] nf=2 scheduler failure (raises for debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (same formula as evaluation). Default: scan trials or FIG1_DEBUG_SEED",
    )
    parser.add_argument(
        "--plot-on-success",
        action="store_true",
        help="If execution unexpectedly succeeds, write a timeline PDF under output/circuit_execution/",
    )
    args = parser.parse_args()

    if args.fig1_fail:
        run_fig1_fail_case_debug(seed=args.seed, plot=args.plot_on_success)
    else:
        test_t_cultivation_small()
