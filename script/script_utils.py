import csv
import os
import sys


def reset_csv_files(*csv_paths: str) -> None:
    """Delete *csv_paths* so a sweep rewrites them from scratch.

    The evaluation writers open their CSVs in append mode, which is what you
    want when stitching several sweeps into one file. Reproduction runs start
    from an empty file instead, so re-running a script twice yields the same
    data rather than duplicated rows.
    """
    for path in csv_paths:
        if os.path.exists(path):
            os.remove(path)


def add_repo_root_to_syspath(script_path: str) -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(script_path), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def ensure_csv_writer(csv_path: str, fieldnames: list[str]):
    header_needed = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if header_needed:
        writer.writeheader()
    return csv_file, writer
