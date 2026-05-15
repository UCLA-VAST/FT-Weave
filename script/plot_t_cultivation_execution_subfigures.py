import os
import sys
import logging
import argparse

# Ensure repository root is on sys.path so `src` and sibling scripts are importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.animator.circuit_execution_visualization import (
    plot_t_cultivation_execution_subfigures,
)
from src.t_cultivation.config import get_config, update_config
from evaluation_t_cultivation_throughput import (
    SweepSpec,
    TSetting,
    apply_setting,
    run_t_cultivation_metrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main(
    *,
    n_factories_list: list[int],
    output_pdf: str,
    print_profile: bool = True,
) -> None:
    """Generate one stacked timeline figure for fixed d13, ler=1e-08, nq=1."""
    settings = [
        TSetting(fidelity_target=1e-8, factory_physical_size=2, distance=7),
        TSetting(fidelity_target=1e-8, factory_physical_size=4, distance=13),
        # TSetting(fidelity_target=1e-10, factory_physical_size=4, distance=13),
    ]

    target_setting = next(
        (s for s in settings if s.distance == 13 and s.fidelity_target == 1e-8),
        None,
    )
    if target_setting is None:
        raise RuntimeError("Could not find target setting d13, ler=1e-08")

    target_setting_idx = settings.index(target_setting)
    spec = SweepSpec(n_qubits=1, k_t_per_qubit=10, layers_parallel=False)

    original_cfg = get_config().copy()
    try:
        apply_setting(target_setting)
        row_plots = []

        for nf in n_factories_list:
            seed = 1000000 * target_setting_idx + 1000 * nf
            try:
                _, _, exec_log = run_t_cultivation_metrics(
                    spec=spec,
                    n_factories=nf,
                    seed=seed,
                    print_profile=print_profile,
                )
            except RuntimeError as e:
                logging.warning("Row skipped (run failed) nq=1 nf=%d: %s", nf, e)
                continue

            if not exec_log:
                logging.warning("Row skipped (empty log) nq=1 nf=%d", nf)
                continue

            row_plots.append(
                (
                    f"Number of Factory = {nf}",
                    exec_log,
                    1,
                    nf,
                )
            )

        if not row_plots:
            raise RuntimeError("No valid execution logs were produced for row plots")

        out_dir = os.path.dirname(output_pdf)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plot_t_cultivation_execution_subfigures(
            row_plots=row_plots,
            save_path=output_pdf,
        )
        logging.info("Wrote stacked subfigure timeline: %s", output_pdf)
    finally:
        # Restore original config so this script does not leak global settings.
        update_config(
            SE_STAGE_1=original_cfg["SE_STAGE_1"],
            SE_STAGE_2=original_cfg["SE_STAGE_2"],
            CNOT_TIME=original_cfg["CNOT_TIME"],
            SE_TIME=original_cfg["SE_TIME"],
            TELEPORTATION_SUCCESS_RATE=original_cfg["TELEPORTATION_SUCCESS_RATE"],
            STAGE_1_SUCCESS_RATE=original_cfg["STAGE_1_SUCCESS_RATE"],
            STAGE_2_SUCCESS_RATE=original_cfg["STAGE_2_SUCCESS_RATE"],
            STAGE_2_FIDELITY_TARGET=original_cfg["STAGE_2_FIDELITY_TARGET"],
            ANGLE_S=original_cfg["ANGLE_S"],
            FACTORY_PHYSICAL_SIZE=original_cfg["FACTORY_PHYSICAL_SIZE"],
            STAGE_1_RESOURCE_UNITS=original_cfg["STAGE_1_RESOURCE_UNITS"],
            SYNCHRONIZE_FACTORY_EXECUTION=original_cfg["SYNCHRONIZE_FACTORY_EXECUTION"],
            RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT=original_cfg[
                "RUS_ASSIGNMENT_CRITICAL_PATH_WEIGHT"
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate stacked T-cultivation execution timelines for d13, ler=1e-08 "
            "with nq=1 and selected factory counts."
        )
    )
    parser.add_argument(
        "--factories",
        type=int,
        nargs="+",
        default=[1, 5, 10, 16],
        help="Factory counts to render as rows (default: 1 5 10 16)",
    )
    parser.add_argument(
        "--output-pdf",
        default=(
            "output/t_cultivation/factory_throughput/first_trial_timelines/"
            "d13_ler1e-08_nq1_nf_1_5_10_16_rows.pdf"
        ),
        help="Output path for the stacked timeline PDF",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable profile output during simulation",
    )
    args = parser.parse_args()

    main(
        n_factories_list=args.factories,
        output_pdf=args.output_pdf,
        print_profile=not args.no_profile,
    )
