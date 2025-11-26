from collections import defaultdict
from typing import Any


def greedy_label_column_assignment(
    matrix: list[list[Any]], rows: list[int]
) -> tuple[dict[Any, int], list[int]]:
    """
    Run the greedy selection process on a given set of rows.
    Returns:
        assign: dict[label] = chosen_column
        surviving_rows: list of rows that remain consistent
    """
    if not rows:
        return {}, []

    C = len(matrix[0])

    labels: list[Any] = sorted({matrix[r][c] for r in rows for c in range(C)})
    remaining: set[int] = set(rows)
    assign: dict[Any, int] = {}

    for label in labels:
        # Count occurrences of label across remaining rows
        freq: defaultdict[int, int] = defaultdict(int)
        for r in remaining:
            for c in range(C):
                if matrix[r][c] == label:
                    freq[c] += 1

        if not freq:
            continue

        best_col: int = max(freq, key=lambda c: freq[c])

        assign[label] = best_col

        # Keep only rows consistent with this label assignment
        new_remaining = {r for r in remaining if matrix[r][best_col] == label}
        if not new_remaining:
            # No consistent rows remain; this assignment fails
            return assign, []  # no rows survive

        remaining = new_remaining

    return assign, sorted(remaining)


def iterative_greedy_groups(matrix: list[list[Any]]) -> list[tuple[list, dict]]:
    """
    Repeatedly extract consistent row-groups until no rows remain.
    Returns:
        list of (rows, assignment)
    """
    R = len(matrix)
    leftover = list(range(R))
    result = []

    while leftover:
        assign, row_group = greedy_label_column_assignment(matrix, leftover)

        if not row_group:
            # No consistent subset can be formed from leftover rows;
            # treat each row as its own (degenerate) group with empty assignment.
            # Or you can break. For now we peel them individually.
            for r in leftover:
                result.append(([r], {}))
            break

        result.append((row_group, assign))

        # Remove used rows from leftover
        used = set(row_group)
        leftover = [r for r in leftover if r not in used]
    return result
