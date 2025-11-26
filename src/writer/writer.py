"""
Writer utilities for ZAC.

Implements writer classes and helpers that serialize scheduling results and
device instructions into output formats (CSV, JSON, or custom command streams)
consumable by downstream tools or experiment runners.

Public API
----------
- Writer (base class) and concrete writer implementations that provide a
  write(...) method to emit schedules/instructions.

Example
-------
from zac.writer.writer import Writer
w = Writer(...)
w.write(schedule)
"""

import time
from typing import Sequence, Any

from src.writer.inst import INSTRUCTIONS, ARGUMENTS
from src.writer.inst import ComboInst
from src.writer.op_expansions import DEFAULT_EXPANSION_STRATEGIES


class Writer:
    """Base writer interface for ZAC scheduling output.

    Provide serialization for schedules, instruction streams, or other data
    produced by the scheduler. Concrete subclasses should implement the
    write(...) method to persist or emit the data in a specific format
    (for example JSON, CSV, or a device command stream).

    Methods
    -------
    write(data, *args, **kwargs)
        Serialize and emit `data`. Must be implemented by subclasses.

    Notes
    -----
    - Implementations should document supported arguments and any returned
      values or side effects (e.g. files written).
    - If a writer may be used concurrently, ensure thread-safety in the
      subclass implementation.
    """

    def __init__(
        self,
        data: dict,
        custom_expansions: dict | None = None,
    ):
        """Initialize the class by pre-processing a data dictionary into a cache of keyword arguments for different instruction types.

        Parameters:
        ----------
            - data (dict): A dictionary containing various parameters needed for instruction generation.
            - custom_expansions (dict, optional): A dictionary of custom expansion strategies for combo instructions.
                Defaults to None.
        """
        self.PARKING_DIST = 1  # constant, the distance of AOD row and col to some trap. We use 1um here.
        self.insts = []

        self.expansion_strategies = DEFAULT_EXPANSION_STRATEGIES.copy()
        if custom_expansions:
            self.expansion_strategies.update(custom_expansions)

        self.data = {}
        data["parking_dist"] = self.PARKING_DIST

        for itype in ARGUMENTS:
            arg_keys = ARGUMENTS[itype]
            inst_kwargs = {key: data[key] for key in arg_keys}
            self.data[itype] = inst_kwargs

    def write_zair(self, itype: str, prompt: dict, *args: Any, **kwargs: Any) -> dict:
        """Generate a ZAIR instruction based on the provided type and prompt.

        Parameters
        ----------
            - itype : str
                The type of instruction to generate.
            - prompt : dict
                A dictionary containing the details of the instruction.
            - *args :
                Additional positional arguments for the instruction.

        Returns
        -------
            - dict: a dictionary representing the generated ZAIR instruction.
        """
        InstructionClass = INSTRUCTIONS.get(itype)

        if InstructionClass:
            inst = InstructionClass(prompt, *args, **kwargs)
            if isinstance(inst, ComboInst):
                func_expand = self.expansion_strategies.get(itype)
                if func_expand:
                    func_expand(inst)
                else:
                    raise ValueError(f"Missing Expansion: '{itype}'")
            return inst.code
        else:
            raise ValueError(f"Unknown Instructions type: '{itype}'")

    def build(self, prompts: Sequence) -> tuple[list, float]:
        """Construct a sequence of ZAIR instructions from a list of prompts.

        Parameters
        ----------
            - prompts: Sequence


        Returns
        -------
            - tuple: A tuple containing a list of ZAIR instructions and the time
                taken to generate them.
        """
        t_s = time.time()
        for _, prompt in enumerate(prompts):
            itype = prompt["type"]
            inst = self.write_zair(itype, prompt, **self.data[itype])
            inst["layer"] = prompt.get("layer")
            self.insts.append(inst)
        time_write = time.time() - t_s

        return self.insts, time_write
