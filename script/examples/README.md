# Examples

Minimal scripts for compiling **general circuits** (not just TFIM) on the two
magic-state architectures. Start here to see the full path from a circuit to an
execution log.

## Files

| File                                | Description                                                     |
| ----------------------------------- | --------------------------------------------------------------- |
| `minimal_clifford_rz.qasm`          | Small Clifford+Rz circuit on 4 qubits (STAR input)              |
| `minimal_clifford_t.qasm`           | Small Clifford+T circuit on 4 qubits (T-cultivation input)      |
| `compile_star_circuit.py`           | Load QASM (or build IR by hand), compile with STAR              |
| `compile_t_cultivation_circuit.py`  | Same, for the T-cultivation architecture                        |

STAR injects arbitrary-angle rotations directly, so it takes **Clifford+Rz**
circuits and rejects bare `T`/`Tdg`. T-cultivation prepares `T` states, so it takes
**Clifford+T** circuits. `validate_circuit_for_backend()` enforces this up front.

## Quick start

Run from the **repository root**:

```bash
# STAR: bundled Clifford+Rz QASM
uv run script/examples/compile_star_circuit.py

# STAR: save an execution diagram (data qubits q* + factory qubits f*)
uv run script/examples/compile_star_circuit.py --plot

# STAR: asynchronous (per-factory) execution instead of synchronous
uv run script/examples/compile_star_circuit.py --parallel-execution --plot

# T-cultivation: full scheduler (CNOT scheduled together with T injection)
uv run script/examples/compile_t_cultivation_circuit.py --no-split-layers --plot

# Programmatic circuit (no QASM file), either backend
uv run script/examples/compile_star_circuit.py --hand-built --plot

# Your own QASM file
uv run script/examples/compile_star_circuit.py --qasm path/to/circuit.qasm --plot
```

Expected output: the layered circuit and a per-log summary (event counts, wall time).

With `--plot`, a timeline PDF is written under `output/examples/`:

- **q0, q1, …** — logical (data) qubit lanes
- **f0, f1, …** — magic-state factory lanes

Both a labeled version and a `_no_text` variant are saved (see console for paths).
For T-cultivation, `stage_2_success` / `stage_2_fail` entries appear in the
execution log (visible in the console op summary) but are omitted from the figure.

## Compile modes

`compile_star_circuit.py` always partitions the circuit into layers: consecutive
`Rz` rotations between Clifford blocks are merged, so a whole layer of rotations is
injected in one STAR round.

`compile_t_cultivation_circuit.py` offers two modes:

| Mode                       | Flag                | Behavior                                                                                                                                      |
| -------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Split layers** (default) | `--split-layers`    | Each `T` / `Rz` layer runs `t_cultivation_execution`. `H` / `CNOT` get lightweight stub logs (same as the TFIM evaluation).                    |
| **Full scheduler**         | `--no-split-layers` | The entire circuit goes through one `t_cultivation_execution` call. Use this when **CNOT** gates must be scheduled together with **T** injection. |

For circuits with interleaved Clifford and non-Clifford gates, prefer
`--no-split-layers`.

## Circuit input formats

### OpenQASM 2.0

Supported native gates include `h`, `t`, `tdg`, `cx`, `rz`, `x`, `y`, `z`, `s`,
`sdg`, and common one-qubit rotations (`rx`, `ry`, `u`, …), which are normalized to
the internal IR.

### Internal IR (`list[dict]`)

Same format as the TFIM generators:

```python
{"gate": "T",    "targets": [0],      "params": {}}
{"gate": "CNOT", "targets": [(0, 1)], "params": {}}
{"gate": "Rz",   "targets": [0, 1],   "params": {"angles": {0: 0.5, 1: 0.3}}}
```

For `Rz`, consecutive rotations between Clifford blocks are merged into one layer.
Per-qubit angles live in `params["angles"]`; a single shared angle can use
`params["theta"]`.

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
    split_layers=True,  # or False for the full scheduler
)
```

See the top-level [README](../../README.md) for the STAR equivalent and for the
meaning of each compile-strategy flag.

## Fidelity evaluation

These examples stop at the execution log. To go all the way to logical fidelity and
CSV output, use the evaluation sweeps, which compile a 2D transverse-field Ising
model Trotter layer across strategies, AOD counts, and code distances:

```bash
uv run script/evaluation_fidelity_star.py
uv run script/evaluation_fidelity_t_cultivation.py
```

Results land in `output/evaluation/fidelity/`.

## Related code

| Module                                     | Role                                                  |
| ------------------------------------------ | ----------------------------------------------------- |
| `src/circuit/qasm_loader.py`               | QASM → internal IR                                    |
| `src/circuit/gate_set.py`                  | Gate-set classification and per-backend validation    |
| `src/circuit/layer_partition.py`           | Merge consecutive `Rz` layers between Clifford blocks |
| `src/star/general_circuit_star.py`         | `compile_circuit_star()`                              |
| `src/t_cultivation/general_circuit_t.py`   | `compile_circuit_t_cultivation()`                     |
