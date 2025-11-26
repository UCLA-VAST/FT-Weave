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
import math

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
        self.num_check_qubit = (code_distance - 1) ** 2 + 2 * (code_distance - 1)
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
        for r in range(1, self.block_length, 2):
            for c in range(1, self.block_length, 2):
                block.append((slm_idx, r + topleft_r, c + topleft_c))
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
        for r in range(0, self.block_length, 2):
            if r % 4 == 0:
                begin = 4
                end = self.block_length - 2
            else:
                begin = 2
                end = self.block_length - 4
            for c in range(begin, end, 2):
                block.append((slm_idx, topleft_r + r, topleft_c + c))
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
        for c in range(0, self.block_length, 2):
            if c % 4 == 2:
                begin = 4
                end = self.block_length - 2
            else:
                begin = 2
                end = self.block_length - 4
            for r in range(begin, end, 2):
                block.append((slm_idx, topleft_r + r, topleft_c + c))
        return block

    def logical_index_to_topleft(
        self,
        logical_r: int,
        logical_c: int,
        stride: int | None = None,
    ) -> Tuple[int, int]:
        """Map logical index to upper-left physical coordinate using `stride`."""
        if stride is None:
            stride = self.block_length
        topleft_r = logical_r * stride
        topleft_c = logical_c * stride
        return (topleft_r, topleft_c)

    def generate_logical_grid(self, stride: int | None = None):
        """Generate and cache 2D grid mapping slm_idx -> [row][col] -> (topleft_r,topleft_c).

        `stride` controls how logical blocks are tiled; if None uses `block_length`.
        """
        grid_map: Dict[int, List[List[Tuple[int, int]]]] = {}
        for slm_idx, slm in self.arch.dict_SLM.items():
            rows = math.floor(slm.n_r / self.block_length)
            cols = math.floor(slm.n_c / self.block_length)
            mapping: List[List[Tuple[int, int]]] = [
                [(0, 0) for _ in range(cols)] for _ in range(rows)
            ]
            for lr in range(rows):
                for lc in range(cols):
                    mapping[lr][lc] = self.logical_index_to_topleft(
                        lr, lc, stride=stride
                    )
            grid_map[slm_idx] = mapping
        self.logical_grid_map = grid_map

    def _physical_sites_from_logical_index(
        self,
        slm_idx: int,
        logical_r: int,
        logical_c: int,
        block_getter,
    ):
        """Internal helper to fetch physical sites for a logical cell using the
        provided block_getter (`logical_data_block_from_topleft`,
        `logical_x_check_block_from_topleft`, or `logical_z_check_block_from_topleft`).
        """
        if not self.logical_grid_map:
            self.generate_logical_grid()
        assert slm_idx in self.logical_grid_map, "SLM has no logical grid mapping"
        grid = self.logical_grid_map[slm_idx]
        topleft_r, topleft_c = grid[logical_r][logical_c]
        phys_sites = block_getter(slm_idx, topleft_r, topleft_c)
        # print(grid[logical_r][logical_c])
        # print(phys_sites)
        # input()
        return phys_sites

    def physical_data_sites_from_logical_index(
        self,
        slm_idx: int,
        logical_r: int,
        logical_c: int,
    ) -> List[Tuple[int, int, int]]:
        """Return the physical sites for logical grid cell; optionally as (x,y).

        Returns list of `(slm_idx, r, c)` or list of `(x,y)` when `as_xy=True`.
        """

        return self._physical_sites_from_logical_index(
            slm_idx,
            logical_r,
            logical_c,
            self.logical_data_block_from_topleft,
        )

    def physical_z_check_sites_from_logical_index(
        self,
        slm_idx: int,
        logical_r: int,
        logical_c: int,
    ) -> List[Tuple[int, int, int]]:
        """Return the physical sites for logical grid cell; optionally as (x,y).

        Returns list of `(slm_idx, r, c)` or list of `(x,y)` when `as_xy=True`.
        """

        return self._physical_sites_from_logical_index(
            slm_idx,
            logical_r,
            logical_c,
            self.logical_z_check_block_from_topleft,
        )

    def physical_x_check_sites_from_logical_index(
        self,
        slm_idx: int,
        logical_r: int,
        logical_c: int,
    ) -> List[Tuple[int, int, int]]:
        """Return the physical sites for logical grid cell; optionally as (x,y).

        Returns list of `(slm_idx, r, c)` or list of `(x,y)` when `as_xy=True`.
        """

        return self._physical_sites_from_logical_index(
            slm_idx,
            logical_r,
            logical_c,
            self.logical_x_check_block_from_topleft,
        )

    def physical_indices_location_for_logical_qubit(
        self,
        logical_idx: int,
        logical_qubit: Tuple[int, int, int],
        include_check_qubit: bool = False,
    ) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
        """Compute the set of physical qubit indices for a list of logical qubits.

        Args:
            logical_qubits: list of tuples `(slm_idx, logical_r, logical_c)`
            as_xy: if True, return set of `(x,y)` positions instead of indices

        Returns:
            A set of physical qubit indices: either `(slm_idx, r, c)` tuples or
            `(x,y)` coordinate tuples when `as_xy=True`.
        """
        begin_index = logical_idx * self.num_total_qubit
        if include_check_qubit:
            bound = self.num_total_qubit
        else:
            bound = self.num_data_qubit
        list_indices = [i for i in range(begin_index, begin_index + bound)]
        list_phys: List[Tuple[int, int, int, int]] = []
        slm_idx, lr, lc = logical_qubit

        sites = self.physical_data_sites_from_logical_index(slm_idx, lr, lc)
        if include_check_qubit:
            # print(sites)
            # print(self.physical_x_check_sites_from_logical_index(slm_idx, lr, lc))
            # print(self.physical_z_check_sites_from_logical_index(slm_idx, lr, lc))
            sites += self.physical_x_check_sites_from_logical_index(slm_idx, lr, lc)
            sites += self.physical_z_check_sites_from_logical_index(slm_idx, lr, lc)
        assert len(sites) == len(list_indices)
        for qubit, (sidx, r, c) in zip(list_indices, sites):
            list_phys.append((qubit, sidx, r, c))
        return list_indices, list_phys
