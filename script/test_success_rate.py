from src.star.simulation import calculate_success_rate


def calculate_success_rate_schedule(
    base_angle: float = 0.0015,
    max_power: int = 6,
    code_distances: tuple[int, int] = (7, 9),
) -> dict[str, list[float]]:
    """Calculate success rates for angles base_angle * 2^k, k in [0, max_power].

    Returns a dictionary with the evaluated angles and per-distance success rates.
    """
    angles = [base_angle * (2**k) for k in range(max_power + 1)]
    angles += [0.00584]

    results: dict[str, list[float]] = {
        "angles": angles,
    }
    for d in code_distances:
        results[f"distance_{d}"] = [
            calculate_success_rate(angle, d) for angle in angles
        ]

    return results


def format_success_rate_schedule_table(
    base_angle: float = 0.0015,
    max_power: int = 6,
) -> str:
    """Return a text table for success rates at distances 7 and 9."""
    schedule = calculate_success_rate_schedule(
        base_angle=base_angle,
        max_power=max_power,
        code_distances=(7, 9),
    )

    header = f"{'k':>2} | {'angle':>12} | {'distance_7':>12} | {'distance_9':>12}"
    sep = "-" * len(header)
    lines = [header, sep]

    for k, angle in enumerate(schedule["angles"]):
        rate_7 = schedule["distance_7"][k]
        rate_9 = schedule["distance_9"][k]
        lines.append(f"{k:>2} | {angle:12.6g} | {rate_7:12.6f} | {rate_9:12.6f}")

    return "\n".join(lines)


def print_success_rate_schedule_table(
    base_angle: float = 0.0015,
    max_power: int = 6,
) -> None:
    """Print success-rate schedule table for quick inspection."""
    print(
        format_success_rate_schedule_table(base_angle=base_angle, max_power=max_power)
    )


def main() -> None:
    print_success_rate_schedule_table(base_angle=0.0015, max_power=6)


if __name__ == "__main__":
    main()
