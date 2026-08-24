"""Qubit layout helpers for general circuits."""

from __future__ import annotations

import math


def resolve_qubit_layout(
    n_qubits: int,
    layout_override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return (cols, rows) for *n_qubits* logical qubits.

    Args:
          n_qubits: Number of logical qubits in the circuit.
          layout_override: Optional ``(cols, rows)`` from the CLI.

      Returns:
          ``(cols, rows)`` with ``cols * rows >= n_qubits``.

      Raises:
          ValueError: invalid override or too-small grid.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    if layout_override is not None:
        cols, rows = layout_override
        if cols < 1 or rows < 1:
            raise ValueError(
                f"layout dimensions must be positive, got ({cols}, {rows})"
            )
        if cols * rows < n_qubits:
            raise ValueError(
                f"layout ({cols}, {rows}) fits {cols * rows} qubits "
                f"but circuit needs {n_qubits}"
            )
        return cols, rows

    cols = int(math.ceil(math.sqrt(n_qubits)))
    rows = int(math.ceil(n_qubits / cols))
    return cols, rows
