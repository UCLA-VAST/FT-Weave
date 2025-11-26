import os
import sys

# Ensure repository root is on sys.path so `src` is importable when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tmr.tmr_assignment import run_tmr_row_assignment
from src.util import print_tmr_assignment_results


# Example usage
if __name__ == "__main__":
    # Example 1
    print("EXAMPLE 1")
    M1 = [4, 3, 2, 5, 1]
    A1 = [5, 4, 4, 2]

    result = run_tmr_row_assignment(M1, A1)
    print_tmr_assignment_results(result)

    print("\n" + "=" * 80)
    print("\n")

    # Example 2
    # print("EXAMPLE 2")
    M2 = [10, 5, 4, 6, 8, 1]
    A2 = [10, 5, 3, 7, 9]

    result = run_tmr_row_assignment(M2, A2)
    print_tmr_assignment_results(result)
