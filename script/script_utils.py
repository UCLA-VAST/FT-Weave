import csv
import os
import sys


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
