"""Star package.

Keep package initialization lightweight to avoid circular imports.
Import execution entrypoints from their modules directly when needed:
    - src.star.analog_rotation_execution
    - src.star.analog_rotation_execution_parallel
"""

__all__: list[str] = []
