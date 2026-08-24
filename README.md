# Transversal STAR Compilation

A compiler and architectural simulator for **magic-state injection on neutral-atom
quantum computers**. It takes a logical Clifford+rotation circuit, schedules the
magic-state preparation and teleportation that implement its non-Clifford gates on
a concrete atom-array microarchitecture, and reports both the **execution time**
and the **end-to-end logical fidelity** of the result.

---

## 1. Design and goals

### The problem

On a fault-tolerant neutral-atom machine, Clifford gates are cheap but non-Clifford
rotations are not: each one consumes a magic state that has to be *prepared* in a
factory, *transported* to the data qubit by an AOD (acousto-optic deflector) move,
and *teleported* into the logical qubit — with all three steps subject to failure and
retry. The cost of a rotation is therefore an *architectural* question, not a gate-count
question. It depends on where factories sit relative to data qubits, how many atoms can
be moved in parallel, and how aggressively the scheduler is allowed to speculate.

This repository exists to answer that question quantitatively for two magic-state
architectures, under one shared model:

| Architecture      | Non-Clifford primitive           | Preparation                                                            |
| ----------------- | -------------------------------- | ---------------------------------------------------------------------- |
| **STAR**          | arbitrary-angle `Rz(θ)` directly | Transversal-injection Magic-state Rotation (TMR), repeat-until-success  |
| **T-cultivation** | `T` gates (`Rz` via gridsynth)   | two-stage cultivation: a *check* stage and an *escape* stage            |

Because both are driven through the same scheduler, movement model, and error model,
their execution-time and fidelity numbers are directly comparable.

### Design principles

**Discrete-event execution, not gate counting.** Compilation produces an *execution
log*: a flat list of timestamped events (`TMR`, `move`, `CNOT`, `return_move`,
`RUS_success`, `SE_q`, …), each annotated with the factories and logical qubits it
touches. The log is the single intermediate representation. Everything downstream —
runtime profiling, fidelity simulation, and every figure in the paper — is a pure
function of it. Nothing is estimated by formula after the fact.

**Randomness is first-class.** TMR preparation, cultivation stages, and teleportation
all succeed probabilistically, so a compiled circuit is a *sample*, not a fixed
schedule. Every entry point takes an explicit `numpy.random.Generator`, and the
evaluations average over seeded trials (`seed = 42 + trial`), which makes every
reported number reproducible.

**Compilation strategies are toggles, not forks.** Routing, microarchitecture,
speculation, and synchronization are independent flags on one code path rather than
separate implementations. That is what makes the ablation studies meaningful: two
strategies differ only in the flags named in their setting tuple.

**Physical parameters live in one place.** Error rates are in
`src/error_model/`; timing and success-rate constants are in `src/star/config.py`
and `src/t_cultivation/config.py`, each with an `update_config()` entry point.

### Repository layout

```
src/
  circuit/            OpenQASM loading, gate-set validation, layer partitioning
  ds/                 device state: factory pools, atom grid, microarchitecture placement
  star/               STAR compilation
    tmr/                magic-state (angle) preparation and factory assignment
    rus/                repeat-until-success teleportation, routing, return moves
  t_cultivation/      T-cultivation compilation (check stage / escape stage)
  execution_log/      execution-log construction, movement, logical-SE scheduling
  error_model/        physical error rates and the surface-code logical error model
  fidelity_simulation/  execution log -> end-to-end logical fidelity
  animator/           execution timeline and RUS trap-grid figures
  util.py             execution-log analysis and profiling

script/
  examples/           minimal end-to-end compile examples (start here)
  figures/            paper figure generators
  evaluation_fidelity_star.py            STAR evaluation sweep -> CSV
  evaluation_fidelity_t_cultivation.py   T-cultivation evaluation sweep -> CSV
  process_csv_paper.py                   runtime figures from the profiling CSVs

test/                 pytest suite
output/               generated data and figures
```

---

## 2. Usage

### Install

The project uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

All commands below are run from the repository root, and all output paths are
relative to it.

Run the test suite with:

```bash
uv run pytest test -q
```

### Compile a circuit

Two minimal, self-contained examples show the full path from a circuit to an
execution log. STAR takes **Clifford+Rz** circuits; T-cultivation takes
**Clifford+T** circuits.

```bash
uv run script/examples/compile_star_circuit.py --plot
```

```bash
uv run script/examples/compile_t_cultivation_circuit.py --no-split-layers --plot
```

Each prints the layered circuit and a per-layer event summary, and with `--plot`
writes a timeline PDF to `output/circuit_execution/examples/`. Both accept
`--qasm <file>` for your own OpenQASM 2.0 circuit and `--hand-built` to use the
programmatic IR instead. See [`script/examples/README.md`](script/examples/README.md)
for the circuit input formats and the compile modes.

### Compile from Python

```python
import numpy as np
from src.circuit.circuit_info import count_qubits
from src.circuit.layer_partition import partition_into_layers
from src.circuit.placement import resolve_qubit_layout
from src.circuit.qasm_loader import load_qasm_file
from src.star.general_circuit_star import compile_circuit_star

flat = load_qasm_file("script/examples/minimal_clifford_rz.qasm")
n_qubits = count_qubits(flat)

circuit, execution_logs, profiling = compile_circuit_star(
    partition_into_layers(flat),
    n_qubits=n_qubits,
    qubit_layout=resolve_qubit_layout(n_qubits),
    placement="col_based",
    code_distance=7,
    config={
        "n_aods": 2,
        "rng": np.random.default_rng(0),
        "consider_skip_rus": 2,
        "trivial_return": False,
        "decompose_move": True,
        "tmr_assignment_method": "matching",
        "prepare_lookahead_angles": True,
        "save_log": False,
    },
    parallel_execution=False,
    analyze_result=True,
)
```

`execution_logs` is one timestamped event list per circuit layer;
`profiling` breaks each rotation round into movement, preparation, and
teleportation time.

### Compilation strategies

A STAR strategy is the tuple
`(placement, prepare_lookahead_angles, trivial_return, consider_skip_rus, decompose_move, parallel_execution)`:

| Knob                       | Effect                                                                |
| -------------------------- | --------------------------------------------------------------------- |
| `placement`                | magic-state microarchitecture: `col_based` interleaves factory columns with data columns; `seperate_region_row` puts all factories in one block |
| `prepare_lookahead_angles` | speculatively prepare the next rotation's angles during the current one |
| `trivial_return`           | return magic states along the reverse of the forward path (vs. an optimized return route) |
| `consider_skip_rus`        | operation rematerialization: `2` re-prepares a magic state instead of paying a whole repeat-until-success round; `0` disables it |
| `decompose_move`           | split one transfer into independently scheduled AOD legs               |
| `parallel_execution`       | asynchronous per-factory execution instead of a global synchronous barrier |

A T-cultivation strategy is `(trivial_return, decompose_move, redistribute_stage1_success)`,
where the last flag patches check-stage outcomes across factories.

The strategies used in the paper are defined — with comments — as `SETTINGS` in
[`script/evaluation_fidelity_star.py`](script/evaluation_fidelity_star.py) and
`ABLATION_GRID` in
[`script/evaluation_fidelity_t_cultivation.py`](script/evaluation_fidelity_t_cultivation.py).

### Run an evaluation sweep

Both evaluation scripts sweep compile strategies × AOD count × code distance ×
qubit count over a 2D transverse-field Ising model Trotter layer, and write CSVs to
`output/evaluation/fidelity/`:

```bash
uv run script/evaluation_fidelity_star.py
```

```bash
uv run script/evaluation_fidelity_t_cultivation.py
```

By default each run **rewrites** its CSVs, so repeating it reproduces the same data.
Pass `--append` to stitch several sweeps into one file, and `--trials N` to change
the number of seeded trials per configuration.

---

## 3. Reproducing the paper

> **TODO — citation.** Fill in the title, author list, and BibTeX entry.
> Preprint: <https://arxiv.org/pdf/2509.18294>

### Figure map

| Figure     | Script                                                                            | Output                                                             |
| ---------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 3          | `script/figures/plot_star_t_cultivation_execution_subfigures.py`                  | `output/circuit_execution/star_t_cultivation_1q2f_subfigures.pdf`   |
| 5, 6       | `script/figures/plot_star_analog_rotation_execution_subfigures_strategy_compare.py` | `output/circuit_execution/star_n9f9_optimized_vs_unoptimized_subfigures.pdf`, `output/rus_rounds_detailed/strategy_compare/` |
| 7          | `script/figures/plot_star_analog_rotation_execution_subfigures_sync_async.py`     | `output/circuit_execution/star_n9f9_col_based_naod3_synchronous_vs_asynchronous_timeline_movement.pdf`, `output/rus_rounds_detailed/sync_async/` |
| 8, 9       | `script/process_csv_paper.py`                                                     | `output/prx_quantum/`                                               |
| Appendix   | `script/figures/compare_fidelity.py`                                              | `output/evaluation/fidelity/comparison/`                            |

Figures 3, 5, 6, and 7 are self-contained: they compile their own small examples on
the fly. Figures 8, 9, and the appendix read the evaluation CSVs, so the two sweeps
must run first.

### Full reproduction

```bash
uv sync
uv run script/evaluation_fidelity_star.py
uv run script/evaluation_fidelity_t_cultivation.py
uv run script/figures/plot_star_t_cultivation_execution_subfigures.py
uv run script/figures/plot_star_analog_rotation_execution_subfigures_strategy_compare.py
uv run script/figures/plot_star_analog_rotation_execution_subfigures_sync_async.py
uv run script/process_csv_paper.py
uv run script/figures/compare_fidelity.py
```

The two sweeps take roughly 5–15 minutes each on a laptop; the figure scripts take
seconds to a couple of minutes.

The execution-timeline scripts (figures 3, 5, 6, 7) each write two PDFs: a labeled
one and a `_no_text` sibling with the in-box labels stripped, for use in a paper
layout where the labels are set in LaTeX. `process_csv_paper.py` and
`compare_fidelity.py` write a single PDF per figure.

### What each figure script does

**Figure 3** — `plot_star_t_cultivation_execution_subfigures.py`. A minimal
side-by-side execution example: one logical qubit and two factories, STAR on top and
T-cultivation below, showing how a single rotation is realized in each architecture.

**Figures 5 and 6** — `plot_star_analog_rotation_execution_subfigures_strategy_compare.py`.
Compares three movement strategies on a 25-qubit / 25-factory instance: the
unoptimized separate-region baseline, the optimized column-based strategy, and the
optimized strategy with operation rematerialization. Writes the stacked execution
timeline (figure 5) and, for each strategy, a per-RUS-round trap-grid movement
visualization under `output/rus_rounds_detailed/strategy_compare/` (figure 6).

**Figure 7** — `plot_star_analog_rotation_execution_subfigures_sync_async.py`.
Compares synchronous against asynchronous execution in a single panel: the timeline
on the left, the corresponding atom movement over the highlighted window on the
right, with the realtime-control events each mode requires marked on the timeline.

**Figures 8 and 9** — `process_csv_paper.py`. Reads
`star_full_trotter_profiling_results.csv` and
`t_cultivation_fidelity_profiling_results.csv` and plots execution time against
qubit count for STAR and T-cultivation: the compilation-strategy ablation and the
AOD sweep (figure 8), and the stacked runtime profile that attributes time to
movement, preparation, and teleportation (figure 9).

**Appendix** — `compare_fidelity.py`. Reads the three fidelity CSVs and plots
end-to-end logical infidelity against qubit count for the uncorrected physical
baseline, STAR, and T-cultivation, plus a per-error-source stacked breakdown for
each. Also prints the per-case infidelity table and the pairwise improvement
factors quoted in the text.

### Reproducibility notes

- Every trial is seeded (`seed = 42 + trial`), so a rerun reproduces the committed
  CSVs bit for bit.
- `script/process_csv_paper.py` selects strategies by exact setting tuple. Its
  `SETTINGS` list mirrors the first four entries of `SETTINGS` in
  `evaluation_fidelity_star.py`, and `_T_SETTING_ABLATION_GRID` mirrors
  `ABLATION_GRID` in `evaluation_fidelity_t_cultivation.py`. If you change a sweep,
  change both.
- `script/figures/compare_fidelity.py` imports `MAIN_SETTINGS` directly from
  `evaluation_fidelity_star.py`, so the fidelity figures always track the sweep.
