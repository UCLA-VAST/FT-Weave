# Examples

Minimal scripts for compiling **general circuits** (not just TFIM) on the T-cultivation architecture.

## Files


| File                                                                   | Description                                                    |
| ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| `[minimal_clifford_t.qasm](minimal_clifford_t.qasm)`                   | Small Clifford+T circuit on 4 qubits                           |
| `[compile_t_cultivation_circuit.py](compile_t_cultivation_circuit.py)` | Load QASM (or build IR by hand), compile, print execution logs |


## Quick start

Run from the **repository root**:

```bash
# Default: bundled QASM, split-layer compile
uv run script/examples/compile_t_cultivation_circuit.py

# Full scheduler (CNOT through real T-cultivation scheduling)
uv run script/examples/compile_t_cultivation_circuit.py --no-split-layers

# Save execution diagram (data qubits q* + factory qubits f*)
uv run script/examples/compile_t_cultivation_circuit.py --no-split-layers --plot

# Programmatic circuit (no QASM file)
uv run script/examples/compile_t_cultivation_circuit.py --hand-built --plot

# Your own QASM file
uv run script/examples/compile_t_cultivation_circuit.py --qasm path/to/circuit.qasm --plot
```

Expected output: the layered circuit and a per-log summary (event counts, wall time).

With `--plot`, writes a timeline PDF under `output/circuit_execution/examples/` showing:
- **q0, q1, …** — logical (data) qubit lanes
- **f0, f1, …** — T-factory lanes

`stage_2_success` / `stage_2_fail` entries are written to the execution log (visible in the console op summary) but omitted from the timeline figure.

Both a labeled version and a `_no_text` variant are saved (see console for paths).

## Compile modes


| Mode                       | Flag                | Behavior                                                                                                                                      |
| -------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Split layers** (default) | `--split-layers`    | Each `T` / `Rz` layer runs `t_cultivation_execution`. `H` / `CNOT` get lightweight stub logs (same as TFIM evaluation).                       |
| **Full scheduler**         | `--no-split-layers` | Entire circuit goes through one `t_cultivation_execution` call. Use this when **CNOT** gates must be scheduled together with **T** injection. |


For circuits with interleaved Clifford and non-Clifford gates, prefer `--no-split-layers`.

## Circuit input formats

### OpenQASM 2.0

Supported native gates include `h`, `t`, `tdg`, `cx`, `rz`, `x`, `y`, `z`, `s`, `sdg`, and common one-qubit rotations (`rx`, `ry`, `u`, …) which are normalized to the internal IR.

### Internal IR (`list[dict]`)

Same format as TFIM generators:

```python
{"gate": "T",    "targets": [0],      "params": {}}
{"gate": "CNOT", "targets": [(0, 1)], "params": {}}
{"gate": "Rz",   "targets": [0, 1],   "params": {"angles": {0: 0.5, 1: 0.3}}}
```

For `Rz`, consecutive rotations between Clifford blocks are merged into one layer. Per-qubit angles live in `params["angles"]`; a single shared angle can use `params["theta"]`.

## Programmatic compile (minimal)

```python
import numpy as np
from src.circuit.circuit_info import count_qubits
from src.circuit.layer_partition import partition_into_layers
from src.circuit.placement import resolve_qubit_layout
from src.circuit.qasm_loader import load_qasm_file
from src.t_cultivation.config import update_config
from src.t_cultivation.general_circuit_t import compile_circuit_t_cultivation

flat = load_qasm_file("script/examples/minimal_clifford_t.qasm")
n_qubits = count_qubits(flat)
layout = resolve_qubit_layout(n_qubits)

update_config(STAGE_2_FIDELITY_TARGET=1e-8, FACTORY_PHYSICAL_SIZE=2)
config = {
    "n_aods": 2,
    "rng": np.random.default_rng(0),
    "epsilon": 1e-4,
    "to_decompose": False,
}

circuit, execution_logs, _ = compile_circuit_t_cultivation(
    partition_into_layers(flat),
    n_qubits=n_qubits,
    qubit_layout=layout,
    placement="col_based",
    code_distance=7,
    config=config,
    split_layers=True,  # or False for full scheduler
)
```

## Fidelity evaluation

For compile + logical fidelity + CSV output, use the evaluation CLI:

```bash
uv run script/evaluation_fidelity_qasm.py \
  --qasm script/examples/minimal_clifford_t.qasm \
  --backend t_cultivation \
  --no-split-layers \
  --code-distance 7 \
  --n-aods 2
```

Results are appended to `output/evaluation/qasm/qasm_t_cultivation_fidelity_results.csv`.

For **Clifford+Rz** circuits (no bare `T`/`Tdg`), use `--backend star` or `--backend auto`.

## Related code


| Module                                   | Role                                                  |
| ---------------------------------------- | ----------------------------------------------------- |
| `src/circuit/qasm_loader.py`             | QASM → internal IR                                    |
| `src/circuit/layer_partition.py`         | Merge consecutive `Rz` layers between Clifford blocks |
| `src/t_cultivation/general_circuit_t.py` | `compile_circuit_t_cultivation()`                     |
| `script/evaluation_fidelity_qasm.py`     | Full evaluation harness                               |


