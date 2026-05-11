from .zair_writer import ZAIRWriter
from .execution_log_to_zair import (
    default_init_locs_for_logic_grid,
    default_init_locs_with_factories,
    execution_log_to_animator_code,
    execution_log_to_ft_animator_code,
    execution_log_to_ft_zair_instructions,
    execution_log_to_zair_instructions,
    pack_timeline_for_matplotlib_animator,
)
from .ft_to_zair import build_rotated_surface_code_layout, ft_zair_to_zair_instructions
from .ft_zair_writer import FTZAIRWriter

class Writer:
    """Backward-compatible wrapper around `ZAIRWriter`.

    Supports legacy construction style:
      Writer(data={"architecture": ..., "n_q": ..., "qubit_mapping": ...})
    """

    def __init__(self, data: dict, custom_expansions: dict | None = None):
        self._writer = ZAIRWriter(
            architecture=data["architecture"],
            custom_expansions=custom_expansions,
            qubit_mapping=data.get("qubit_mapping"),
            n_q=data.get("n_q"),
        )

    def build(self, prompts):
        return self._writer.build(prompts, apply_timing=False)

    def write_zair(self, itype: str, prompt: dict):
        if itype != prompt.get("type"):
            prompt = dict(prompt)
            prompt["type"] = itype
        return self._writer.write_instruction(prompt)
