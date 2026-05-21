from __future__ import annotations

import time
from typing import Any, Sequence

from .expansions import (
    DEFAULT_EXPANSIONS,
    rearrange_rowbyrow,
    coalesce_same_angle,
    flatten_aod_qubits,
    flatten_locs,
)
from .timing import get_begin_time, get_duration


def _rearrange_locs_row_major(locs: list) -> list:
    """Ensure begin/end locs are ``[[row0...], [row1...], ...]`` with each loc ``[q, slm, r, c]``.

    Legacy prompts sometimes pass a *flat* list ``[[q, slm, r, c], ...]`` (one implicit row).
    Row-major lists also have ``len(first_row) == 4`` when that row has four qubits; those must
    **not** be wrapped again (otherwise each "loc" becomes a list of locs and expansion breaks).
    """
    if not locs:
        return locs
    first_row = locs[0]
    if not isinstance(first_row, (list, tuple)):
        return locs
    if not first_row:
        return locs
    # Row-major: first cell is itself a qubit loc [q, slm, r, c] (sequence of scalars).
    cell0 = first_row[0]
    if isinstance(cell0, (list, tuple)) and len(cell0) == 4:
        return [[list(loc) for loc in row] for row in locs]
    # Flat list of locs: wrap as a single row.
    if len(first_row) == 4:
        return [[list(loc) for loc in locs]]
    return locs


class ZAIRWriter:
    PARKING_DIST: float = 1.0

    def __init__(
        self,
        architecture,
        custom_expansions: dict | None = None,
        qubit_mapping: list | None = None,
        n_q: int | None = None,
    ) -> None:
        self.architecture = architecture
        self.parking_dist = self.PARKING_DIST
        self.qubit_mapping = qubit_mapping
        self.n_q = n_q
        self.expansion_strategies = DEFAULT_EXPANSIONS.copy()
        if custom_expansions:
            self.expansion_strategies.update(custom_expansions)
        self._instructions: list[dict] = []

    def build(self, prompts: Sequence[dict], apply_timing: bool = True) -> tuple[list[dict], float]:
        t_start = time.time()
        self._instructions.clear()
        for prompt in prompts:
            inst_dict = self.write_instruction(prompt)
            inst_dict["layer"] = prompt.get("layer")
            self._instructions.append(inst_dict)
        if apply_timing:
            self._apply_timing()
        return self._instructions, time.time() - t_start

    def write_instruction(self, prompt: dict) -> dict:
        itype = prompt["type"]
        if itype == "init":
            return self._write_init(prompt)
        if itype == "rydberg":
            return self._write_rydberg(prompt)
        if itype == "1qGate":
            return self._write_1q_gate(prompt)
        if itype == "rearrangeJob":
            return self._write_rearrange_job(prompt)
        raise ValueError(f"Unknown instruction type: {itype}")

    def _write_init(self, prompt: dict) -> dict:
        init_locs = prompt.get("init_locs")
        if init_locs is None and self.qubit_mapping is not None and self.n_q is not None:
            # Legacy mapping can be either:
            # - [[(slm,row,col), ...], ...] or
            # - [(qid, slm, row, col), ...]
            if self.qubit_mapping and isinstance(self.qubit_mapping[0], tuple) and len(self.qubit_mapping[0]) == 4:
                init_locs = [list(loc) for loc in self.qubit_mapping[: self.n_q]]
            else:
                init_locs = [
                    [
                        i,
                        self.qubit_mapping[0][i][0],
                        self.qubit_mapping[0][i][1],
                        self.qubit_mapping[0][i][2],
                    ]
                    for i in range(self.n_q)
                ]
        elif init_locs is None:
            init_locs = []
        return {
            "type": "init",
            "id": prompt.get("id", 0),
            "init_locs": init_locs,
            "dependency": prompt.get("dependency", {}),
            "begin_time": 0.0,
            "end_time": 0.0,
        }

    def _write_rydberg(self, prompt: dict) -> dict:
        return {
            "type": "rydberg",
            "id": prompt.get("id", 0),
            "zone_id": prompt.get("zone_id", 0),
            "gates": prompt.get("gates", []),
            "dependency": prompt.get("dependency", {}),
            "begin_time": 0.0,
            "end_time": 0.0,
        }

    def _write_1q_gate(self, prompt: dict) -> dict:
        if "inst" in prompt:
            inst_list = prompt.get("inst", [])
        else:
            gates = prompt.get("gates", [])
            mapping = prompt.get("mapping", {})
            inst_list = []
            for gate in gates:
                q = gate["q"]
                angle = gate.get("angle", 0.0)
                loc = mapping.get(q, [0, 0, 0])
                inst_list.append({"angle": angle, "locs": [[q, loc[0], loc[1], loc[2]]]})

        expand_fn = self.expansion_strategies.get("1qGate", coalesce_same_angle)
        if expand_fn and callable(expand_fn):
            inst_list = expand_fn(inst_list)
        return {
            "type": "1qGate",
            "id": prompt.get("id", 0),
            "unitary": prompt.get("unitary", ""),
            "inst": inst_list,
            "dependency": prompt.get("dependency", {}),
            "begin_time": 0.0,
            "end_time": 0.0,
        }

    def _write_rearrange_job(self, prompt: dict) -> dict:
        begin_locs = _rearrange_locs_row_major(prompt.get("begin_locs", []))
        end_locs = _rearrange_locs_row_major(prompt.get("end_locs", []))
        aod_qubits = prompt.get("aod_qubits", [])
        if aod_qubits and isinstance(aod_qubits[0], int):
            aod_qubits = [aod_qubits]
        normalized_prompt = dict(prompt)
        normalized_prompt["begin_locs"] = begin_locs
        normalized_prompt["end_locs"] = end_locs
        normalized_prompt["aod_qubits"] = aod_qubits

        job = _RearrangeJobProxy(normalized_prompt, self.architecture, self.parking_dist)
        expand_fn = self.expansion_strategies.get("rearrangeJob", rearrange_rowbyrow)
        if expand_fn and callable(expand_fn):
            detail_insts = expand_fn(job, self.architecture, self.parking_dist)
        else:
            raise ValueError("Missing expansion strategy for rearrangeJob")
        return {
            "type": "rearrangeJob",
            "id": normalized_prompt.get("id", 0),
            "aod_id": normalized_prompt.get("aod_id", 0),
            "aod_qubits": flatten_aod_qubits(normalized_prompt.get("aod_qubits", [])),
            "begin_locs": flatten_locs(normalized_prompt.get("begin_locs", [])),
            "end_locs": flatten_locs(normalized_prompt.get("end_locs", [])),
            "insts": detail_insts,
            "dependency": normalized_prompt.get("dependency", {}),
            "begin_time": 0.0,
            "end_time": 0.0,
        }

    def _apply_timing(self) -> None:
        for idx, inst in enumerate(self._instructions):
            begin_time = get_begin_time(self._instructions, idx, inst.get("dependency", {}))
            inst["begin_time"] = begin_time
            itype = inst.get("type", "")
            if itype == "rearrangeJob" and "insts" in inst:
                inst["end_time"] = begin_time + get_duration(self.architecture, inst)
            elif itype == "rydberg":
                inst["end_time"] = begin_time + getattr(self.architecture, "time_rydberg", 0.0)
            elif itype == "1qGate":
                inst["end_time"] = begin_time + getattr(self.architecture, "time_1qGate", 0.0)
            elif itype == "init":
                inst["end_time"] = begin_time
            elif "end_time" not in inst:
                inst["end_time"] = begin_time


class _RearrangeJobProxy:
    def __init__(self, prompt: dict, architecture, parking_dist: float) -> None:
        self.prompt = prompt
        self.architecture = architecture
        self.PARKING_DIST = parking_dist
        self.aod_id = prompt.get("aod_id", 0)
        self.aod_qubits = prompt.get("aod_qubits", [])
        self.begin_locs = prompt.get("begin_locs", [])
        self.end_locs = prompt.get("end_locs", [])
        self.dependency = prompt.get("dependency", {})
        self.code: dict = {}
        self.detail_insts: list[dict] = []

