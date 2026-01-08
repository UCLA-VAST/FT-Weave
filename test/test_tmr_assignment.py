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
    M1 = {0: 4, 1: 3, 2: 2, 3: 5, 4: 1}
    A1 = {0: 5, 1: 4, 2: 4, 3: 2}

    result = run_tmr_row_assignment(M1, A1)
    print_tmr_assignment_results(result)

    print("\n" + "=" * 80)
    print("\n")

    # Example 2
    # print("EXAMPLE 2")
    M2 = {0: 10, 1: 5, 2: 4, 3: 6, 4: 8}
    A2 = {0: 10, 1: 5, 2: 3, 3: 8, 5: 9}

    result = run_tmr_row_assignment(M2, A2)
    print_tmr_assignment_results(result)
