from __future__ import annotations

from typing import Any

from src.writer.zair_writer import ZAIRWriter


class FTZAIRWriter:
    """Writer for FT-ZAIR logical instructions.

    FT-ZAIR keeps one instruction per logical operation while optionally storing
    ZAIR-compatible realization payloads (e.g. ``rearrange_job``/``gate_inst``).
    """

    _SUPPORTED_TYPES = {
        "init",
        "move",
        "return_move",
        "H",
        "S",
        "Rz",
        "CNOT",
        "SE",
        "SE_q",
        "SE_stage_1",
        "SE_stage_2",
        "RUS_fail",
        "TMR_fail",
    }

    def __init__(self, architecture) -> None:
        self.architecture = architecture
        self._zair_writer = ZAIRWriter(architecture=architecture)

    def write_instruction(self, prompt: dict[str, Any]) -> dict[str, Any]:
        itype = str(prompt.get("type", ""))
        if itype not in self._SUPPORTED_TYPES:
            raise ValueError(f"Unsupported FT-ZAIR instruction type: {itype}")
        # FT-ZAIR is a stable, explicit logical-op IR: preserve payload as-is.
        return dict(prompt)

    def write_zair_instruction(self, prompt: dict[str, Any]) -> dict[str, Any]:
        """Bridge helper: render one ZAIR instruction using existing ZAIR writer."""
        return self._zair_writer.write_instruction(prompt)

