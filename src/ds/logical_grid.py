"""Logical-grid utilities separated from `Architecture`.

This module provides a LogicalGridManager which is responsible for mapping
logical grid indices (row, col) to physical SLM sites and blocks. The
`Architecture` class remains purely physical and is passed to this manager
for geometry information.

The logical block layout follows the rotated-surface-code convention: within
an n x n physical block (n = 2*d - 1) the data/active qubits are placed on a
checkerboard parity (i.e. every other site). This file provides utilities
to enumerate the physical sites that belong to a rotated patch anchored at
the block's upper-left physical coordinate.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, cast

from .architecture import Architecture


class LogicalGridManager:
    """Manage logical-grid generation and conversions.

    Attributes
    ----------
    arch: Architecture
        Reference to the device architecture (physical info only).
    logical_grid_map: dict
        Cached mapping: slm_idx -> 2D list [row][col] -> (topleft_r, topleft_c).
    """

    def __init__(self, arch: Architecture, code_distance: int):
        self.arch = arch
        self.logical_grid_map: Dict[int, List[List[Tuple[int, int]]]] = {}
        self.code_distance = code_distance
        self.block_length = 2 * self.code_distance + 1
        self.num_data_qubit = code_distance**2
        self.num_check_qubit = (code_distance - 1) ** 2 + (code_distance - 1) * 4
        self.num_total_qubit = self.num_data_qubit + self.num_check_qubit

    def is_valid_logical_position(
        self, slm_idx: int, topleft_r: int, topleft_c: int
    ) -> bool:
        """Check whether an n x n logical block with upper-left at
        `(topleft_r, topleft_c)` fits on the given SLM.
        """
        if slm_idx not in self.arch.dict_SLM:
            return False
        slm = self.arch.dict_SLM[slm_idx]
        r_min = topleft_r
        c_min = topleft_c
        r_max = topleft_r + self.block_length - 1
        c_max = topleft_c + self.block_length - 1
        return r_min >= 0 and c_min >= 0 and r_max < slm.n_r and c_max < slm.n_c

    def logical_data_block_from_topleft(
        self, slm_idx: int, topleft_r: int, topleft_c: int
    ) -> List[Tuple[int, int, int]]:
        """Return list of physical site tuples `(slm_idx, r, c)` for rotated
        surface-code logical block.

        The rotated surface code places data (or active) qubits on a
        checkerboard pattern inside the n x n block. We follow the simple
        parity rule: include a site iff ``(r - topleft_r + c - topleft_c) % 2 == 0``.
        This ensures the block follows the rotated layout and that the block
        anchors at the upper-left physical coordinate.
        """
        assert self.is_valid_logical_position(
            slm_idx, topleft_r, topleft_c
        ), "logical block does not fit inside given SLM"
        block: List[Tuple[int, int, int]] = []
        for r in range(topleft_r + 1, topleft_r + self.block_length, 2):
            for c in range(topleft_c + 1, topleft_c + self.block_length, 2):
                block.append((slm_idx, r, c))
        return block

    def logical_x_check_block_from_topleft(
        self, slm_idx: int, topleft_r: int, topleft_c: int
    ) -> List[Tuple[int, int, int]]:
        """Return list of physical site tuples `(slm_idx, r, c)` for rotated
        surface-code logical block.

        The rotated surface code places data (or active) qubits on a
        checkerboard pattern inside the n x n block. We follow the simple
        parity rule: include a site iff ``(r - topleft_r + c - topleft_c) % 2 == 0``.
        This ensures the block follows the rotated layout and that the block
        anchors at the upper-left physical coordinate.
        """
        assert self.is_valid_logical_position(
            slm_idx, topleft_r, topleft_c
        ), "logical block does not fit inside given SLM"
        block: List[Tuple[int, int, int]] = []
        for r in range(topleft_r, topleft_r + self.block_length):
            if r % 2 == 0:
                begin = 4
                end = self.block_length - 3
            else:
                begin = 2
                end = self.block_length - 5
            for c in range(begin, end, 2):
                block.append((slm_idx, r, c))
        return block

    def logical_z_check_block_from_topleft(
        self, slm_idx: int, topleft_r: int, topleft_c: int
    ) -> List[Tuple[int, int, int]]:
        """Return list of physical site tuples `(slm_idx, r, c)` for rotated
        surface-code logical block.

        The rotated surface code places data (or active) qubits on a
        checkerboard pattern inside the n x n block. We follow the simple
        parity rule: include a site iff ``(r - topleft_r + c - topleft_c) % 2 == 0``.
        This ensures the block follows the rotated layout and that the block
        anchors at the upper-left physical coordinate.
        """
        assert self.is_valid_logical_position(
            slm_idx, topleft_r, topleft_c
        ), "logical block does not fit inside given SLM"
        block: List[Tuple[int, int, int]] = []
        for c in range(topleft_c, topleft_c + self.block_length, 2):
            if c % 2 == 0:
                begin = 4
                end = self.block_length - 3
            else:
                begin = 2
                end = self.block_length - 5
            for r in range(begin, end, 2):
                block.append((slm_idx, r, c))
        return block

    def generate_logical_locations(
        self, stride: int | None = None
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Generate list of upper-left coordinates for logical blocks per SLM.

        If `stride` is None the blocks are non-overlapping and use a stride
        equal to the block side length. Returns mapping slm_idx -> list of
        `(topleft_r, topleft_c)`.
        """
        if stride is None:
            stride = self.block_length
        logical_map: Dict[int, List[Tuple[int, int]]] = {}
        for slm_idx, slm in self.arch.dict_SLM.items():
            locs: List[Tuple[int, int]] = []
            max_r = slm.n_r - self.block_length
            max_c = slm.n_c - self.block_length
            if max_r < 0 or max_c < 0:
                logical_map[slm_idx] = locs
                continue
            r = 0
            while r <= max_r:
                c = 0
                while c <= max_c:
                    locs.append((r, c))
                    c += stride
                r += stride
            logical_map[slm_idx] = locs
        return logical_map

    def logical_index_to_topleft(
        self,
        slm_idx: int,
        logical_r: int,
        logical_c: int,
        stride: int | None = None,
    ) -> Tuple[int, int]:
        """Map logical index to upper-left physical coordinate using `stride`."""
        if stride is None:
            stride = self.block_length
        rows, cols = self.block_length, self.block_length
        assert (
            0 <= logical_r < rows and 0 <= logical_c < cols
        ), "logical index out of range for this SLM and code distance"
        topleft_r = logical_r * stride
        topleft_c = logical_c * stride
        return (topleft_r, topleft_c)

    def generate_logical_grid(self, stride: int | None = None):
        """Generate and cache 2D grid mapping slm_idx -> [row][col] -> (topleft_r,topleft_c).

        `stride` controls how logical blocks are tiled; if None uses `block_length`.
        """
        grid_map: Dict[int, List[List[Tuple[int, int]]]] = {}
        for slm_idx in self.arch.dict_SLM:
            rows, cols = self.block_length, self.block_length
            mapping: List[List[Tuple[int, int]]] = [
                [(0, 0) for _ in range(cols)] for _ in range(rows)
            ]
            for lr in range(rows):
                for lc in range(cols):
                    mapping[lr][lc] = self.logical_index_to_topleft(
                        slm_idx, lr, lc, stride=stride
                    )
            grid_map[slm_idx] = mapping
        self.logical_grid_map = grid_map

    def physical_data_sites_from_logical_index(
        self,
        slm_idx: int,
        logical_r: int,
        logical_c: int,
        as_xy: bool = False,
    ) -> List[Tuple[int, int, int]] | List[Tuple[int, int]]:
        """Return the physical sites for logical grid cell; optionally as (x,y).

        Returns list of `(slm_idx, r, c)` or list of `(x,y)` when `as_xy=True`.
        """

        if not self.logical_grid_map:
            self.generate_logical_grid()
        assert slm_idx in self.logical_grid_map, "SLM has no logical grid mapping"
        grid = self.logical_grid_map[slm_idx]
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        assert (
            0 <= logical_r < rows and 0 <= logical_c < cols
        ), "logical index out of range"
        topleft_r, topleft_c = grid[logical_r][logical_c]
        phys_sites = self.logical_data_block_from_topleft(slm_idx, topleft_r, topleft_c)
        if as_xy:
            # `phys_sites` are tuples (slm_idx, r, c)
            return [
                self.arch.exact_SLM_location(site[0], site[1], site[2])
                for site in phys_sites
            ]
        return phys_sites

    def physical_indices_location_for_logical_qubit(
        self,
        logical_idx: int,
        logical_qubit: Tuple[int, int, int],
        as_xy: bool = False,
    ) -> (
        Tuple[List[int], List[Tuple[int, int, int, int]]]
        | Tuple[List[int], List[Tuple[int, int, int]]]
    ):
        """Compute the set of physical qubit indices for a list of logical qubits.

        Args:
            logical_qubits: list of tuples `(slm_idx, logical_r, logical_c)`
            as_xy: if True, return set of `(x,y)` positions instead of indices

        Returns:
            A set of physical qubit indices: either `(slm_idx, r, c)` tuples or
            `(x,y)` coordinate tuples when `as_xy=True`.
        """
        begin_index = logical_idx * self.num_total_qubit

        list_indices = [
            i for i in range(begin_index, begin_index + self.num_data_qubit)
        ]
        list_phys: List[Tuple[int, int, int, int]] = []
        list_xy: List[Tuple[int, int, int]] = []
        slm_idx, lr, lc = logical_qubit
        if as_xy:
            coords = cast(
                List[Tuple[int, int]],
                self.physical_data_sites_from_logical_index(
                    slm_idx, lr, lc, as_xy=True
                ),
            )
            assert len(coords) == len(list_indices)
            for qubit, (x, y) in zip(list_indices, coords):
                list_xy.append((qubit, x, y))
        else:
            sites = cast(
                List[Tuple[int, int, int]],
                self.physical_data_sites_from_logical_index(
                    slm_idx, lr, lc, as_xy=False
                ),
            )
            assert len(sites) == len(list_indices)
            for qubit, (slm_idx, r, c) in zip(list_indices, sites):
                list_phys.append((qubit, slm_idx, r, c))

        return (list_indices, list_xy) if as_xy else (list_indices, list_phys)
