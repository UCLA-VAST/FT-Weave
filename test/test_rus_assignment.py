import os
import sys

# Ensure repository root is on sys.path so `src` is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rus.rus_assignment import iterative_greedy_groups

# ---------------------------
# Example usage:
# ---------------------------

if __name__ == "__main__":
    M = [
        [1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3],
        [1, 2, 3, 3, 3],
    ]

    assign, rows = iterative_greedy_groups(M)
    print("Assignment:", assign)
    print("Remaining rows:", rows)
