# FT-Weave

A real-time compilation framework for fault-tolerant quantum architectures.

FT-Weave compiles a logical circuit onto an early fault-tolerant (EFTQC) neutral-atom
architecture by jointly orchestrating **resource preparation**, **teleportation
routing**, and **teleportation-correction handling** — deferring the decisions that
depend on probabilistic hardware outcomes to runtime instead of baking them into a
static schedule. It is instantiated here on two representative EFTQC architectures,
**transversal STAR** and a **T-cultivation-based** architecture, and reports both
end-to-end execution time and logical fidelity.

This repository is the reference implementation for the paper *FT-Weave: Real-Time
Compilation Framework for Fault Tolerant Quantum Architectures* — see
[§3, Reproducing the paper](#3-reproducing-the-paper).

---

## 1. Design and goals

### The problem

Magic-state preparation and consumption are fundamentally probabilistic. A factory
attempt may fail; a teleportation may need correction. Compilers that schedule
execution offline must therefore assume a *fixed* factory throughput — and when the
real throughput deviates, the static schedule stalls and hardware sits idle.

The cost of a non-Clifford gate is consequently an *architectural and runtime*
question, not a gate-count question. It depends on where factories sit relative to
data qubits, how many atoms an AOD (acousto-optic deflector) can move in parallel,
and how the runtime reacts when a preparation fails. FT-Weave exists to make those
decisions at runtime, and to make their consequences measurable.

### The framework

FT-Weave organizes execution around **four recurring stages** — resource
preparation, resource assignment, routing, and correction handling — and partitions
decisions across them into a static and a dynamic phase:

| Phase       | Decisions                                                                                                                  | Determined by                    |
| ----------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **Static**  | logical circuit scheduling → microarchitecture design → static routing → initial factory assignment                         | the circuit and the architecture |
| **Dynamic** | teleportation assignment → routing for teleportation → factory-assignment update, closing the loop on hardware feedback     | probabilistic runtime outcomes   |

The stages are *interfaces*, not implementations: each architecture plugs its own
preparation protocol and optimization policy into the same compilation flow. That is
what lets STAR and T-cultivation be compared under one movement model, one error
model, and one scheduler.

### The two architectures

| Architecture      | Non-Clifford primitive           | Preparation                                                                   |
| ----------------- | -------------------------------- | ----------------------------------------------------------------------------- |
| **STAR**          | arbitrary-angle `Rz(θ)` directly | transversal multi-rotation (TMR), 6 QEC cycles per attempt, then RUS teleportation; a failed teleportation creates a *corrective* rotation demand that feeds back into the next round |
| **T-cultivation** | `T` gates (`Rz` via Gridsynth)   | two-stage cultivation: a small-footprint *check* stage (12.5 QEC cycles) feeding an *escape* stage (0.5 cycles at d = 7, 9; 6 at d = 13) |

### Compilation techniques

Each technique targets one stage of the flow. All are independent flags on a single
code path rather than separate implementations, which is what makes the ablation in
figure 8 meaningful — two strategies differ only in the flags named in their setting
tuple.

| Technique                            | Stage        | Where it lives                                     |
| ------------------------------------ | ------------ | -------------------------------------------------- |
| Dynamic angle collection             | preparation  | `src/star/tmr/angle_collection.py`, `factory_angle_assignment/` |
| Check-stage factory redistribution   | preparation  | `complete_stage1_preparation` in `src/t_cultivation/util.py` |
| Routing-aware factory assignment     | assignment   | `src/star/tmr/tmr_assignment.py`, `src/star/rus/rus_assignment/` |
| Movement optimization                | routing      | `src/star/rus/rus_routing/` (forward + return moves) |
| Resource rematerialization           | prep/routing | `consider_skip_rus` in `src/star/analog_rotation_execution.py` |
| Synchronous / asynchronous policy    | execution    | `analog_rotation_execution.py` vs `analog_rotation_execution_parallel.py` |

> Note: the code and figure labels call resource rematerialization *operation
> rematerialization*; the paper uses the former.

### Design principles

**Discrete-event execution, not gate counting.** Compilation produces an *execution
log*: a flat list of timestamped events (`TMR`, `move`, `CNOT`, `return_move`,
`RUS_success`, `SE_q`, …), each annotated with the factories and logical qubits it
touches. The log is the single intermediate representation. Runtime profiling,
fidelity simulation, and every figure in the paper are pure functions of it. Nothing
is estimated by formula after the fact.

**Randomness is first-class.** Preparation, cultivation, and teleportation all
succeed probabilistically, so a compiled circuit is a *sample*, not a fixed schedule.
Every entry point takes an explicit `numpy.random.Generator`, and the evaluations
average over seeded trials (`seed = 42 + trial`).

**The hardware model is configurable.** The reported numbers instantiate one
representative neutral-atom execution model, not a universal hardware configuration.
Error rates live in `src/error_model/`; operation latencies and success rates live in
`src/star/config.py` and `src/t_cultivation/config.py`, each with an `update_config()`
entry point.

### Repository layout

```
src/
  circuit/            OpenQASM loading, gate-set validation, layer partitioning (static stage 1)
  ds/                 device state: factory pools, atom grid, microarchitecture (static stage 2)
  star/               STAR instantiation
    tmr/                angle collection and routing-aware factory assignment
    rus/                RUS teleportation, routing, return-move optimization
  t_cultivation/      T-cultivation instantiation (check stage / escape stage)
  execution_log/      execution-log construction, movement, logical-SE scheduling
  error_model/        physical error rates and the surface-code logical error model
  fidelity_simulation/  execution log -> end-to-end logical fidelity
  animator/           execution timeline and movement-schedule figures
  util.py             execution-log analysis and profiling

script/
  examples/           minimal end-to-end compile examples (start here)
  figures/            paper figure generators, one per file: plot_figNN_*.py
  evaluation_fidelity_star.py            STAR evaluation sweep -> CSV
  evaluation_fidelity_t_cultivation.py   T-cultivation evaluation sweep -> CSV

test/                 pytest suite

output/
  evaluation/fidelity/  evaluation sweep CSVs
  figures/              paper figures, named figNN_<caption>; supplementary/ for the rest
  examples/             output of the example scripts (not tracked)
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
writes a timeline PDF to `output/examples/`. Both accept
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
    partition_into_layers(flat),          # static stage 1: logical circuit scheduling
    n_qubits=n_qubits,
    qubit_layout=resolve_qubit_layout(n_qubits),
    placement="col_based",                # static stage 2: microarchitecture design
    code_distance=7,
    config={
        "n_aods": 2,
        "rng": np.random.default_rng(0),
        "consider_skip_rus": 2,           # resource rematerialization
        "trivial_return": False,          # optimized return movement
        "decompose_move": True,
        "tmr_assignment_method": "matching",   # routing-aware factory assignment
        "prepare_lookahead_angles": True,      # dynamic angle collection
        "save_log": False,
    },
    parallel_execution=False,             # synchronous execution policy
    analyze_result=True,
)
```

`execution_logs` is one timestamped event list per circuit layer; `profiling` breaks
each rotation round into movement, preparation, and teleportation time.

### Compilation strategies

A STAR strategy is the tuple
`(placement, prepare_lookahead_angles, trivial_return, consider_skip_rus, decompose_move, parallel_execution)`:

| Knob                       | Effect                                                                |
| -------------------------- | --------------------------------------------------------------------- |
| `placement`                | microarchitecture: `col_based` is the alternating-column layout that interleaves factories with data qubits; `seperate_region_row` puts factories in a separate region on the opposite side of the lattice |
| `prepare_lookahead_angles` | dynamic angle collection — speculatively allocate factory budget to upcoming rotation demands |
| `trivial_return`           | return magic states along the reverse of the forward path, instead of an optimized return route |
| `consider_skip_rus`        | resource rematerialization: `2` regenerates a resource when that beats routing an existing one; `0` disables it |
| `decompose_move`           | split one transfer into independently scheduled AOD legs               |
| `parallel_execution`       | asynchronous per-factory execution instead of synchronous round boundaries |

A T-cultivation strategy is `(trivial_return, decompose_move, redistribute_stage1_success)`,
where the last flag enables check-stage factory redistribution.

The four strategies compared in figure 8 are defined — with the paper's labels as
comments — as `SETTINGS` in
[`script/evaluation_fidelity_star.py`](script/evaluation_fidelity_star.py) and
`ABLATION_GRID` in
[`script/evaluation_fidelity_t_cultivation.py`](script/evaluation_fidelity_t_cultivation.py):

| Figure 8 label (STAR)    | Setting                                            |
| ------------------------ | -------------------------------------------------- |
| Baseline                 | `("seperate_region_row", False, True,  0, False, False)` |
| + Routing opt.           | `("seperate_region_row", False, False, 0, True,  False)` |
| Greedy exec.             | `("col_based", True,  False, 2, False, True)`      |
| High parallelism exec.   | `("col_based", False, False, 2, True,  False)`     |

### Run an evaluation sweep

Both evaluation scripts compile a second-order Trotter layer of the 2D
transverse-field Ising model — the paper's workload — across compile strategies ×
AOD count × code distance × problem size, and write CSVs to
`output/evaluation/fidelity/`:

```bash
uv run script/evaluation_fidelity_star.py
```

```bash
uv run script/evaluation_fidelity_t_cultivation.py
```

The sweep covers 4×4 to 10×10 logical qubits with one factory per logical qubit,
1–5 AODs, surface-code distances d = 7, 9, 13, and 10 seeded trials per
configuration. By default each run **rewrites** its CSVs, so repeating it reproduces
the same data; pass `--append` to stitch several sweeps into one file and
`--trials N` to change the trial count.

---

## 3. Reproducing the paper

**FT-Weave: Real-Time Compilation Framework for Fault Tolerant Quantum
Architectures.**

> **TODO — citation.** The manuscript version in hand carries no author list
> (anonymized for review). Fill in the authors, venue, and BibTeX entry before
> release. Preprint: <https://arxiv.org/pdf/2509.18294>

### Figure map

Every generated figure lands in `output/figures/`, named for its number in the
paper and its caption:

| Figure | Script                                                                              | Output under `output/figures/`                              |
| ------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 3      | `script/figures/plot_fig03_execution_abstraction.py`                    | `fig03_execution_abstraction_star_vs_t_cultivation.pdf`       |
| 5      | `script/figures/plot_fig05_06_compilation_strategies.py` | `fig05_execution_timeline_compilation_strategies.pdf`         |
| 6      | *(same script)*                                                                     | `fig06_movement_schedules/{baseline,optimized,optimized_with_rematerialization}/round{N}_{move,return}.pdf` |
| 7      | `script/figures/plot_fig07_synchronous_vs_asynchronous.py`        | `fig07_synchronous_vs_asynchronous_execution.pdf`, plus the right-hand panels alone in `fig07_movement_schedules/{synchronous,asynchronous}/` |
| 8      | `script/figures/plot_fig08_09_execution_time.py`                                                       | `fig08_compilation_strategies_and_aod_parallelism_d9.pdf`     |
| 9      | *(same script)*                                                                     | `fig09_runtime_profiles_d9.pdf`                               |
| 10     | `script/figures/plot_fig10_overall_infidelity.py`                                                | `fig10_overall_execution_infidelity.pdf`                      |

Figures 1, 2, and 4 are hand-drawn conceptual diagrams with no generating script.

Figures 3, 5, 6, and 7 are self-contained: they compile their own examples on the
fly. Figures 8, 9, and 10 read the evaluation CSVs, so the two sweeps must run first.

`output/figures/supplementary/` holds variants that are useful for inspection but do
not appear in the paper: per-error-source infidelity breakdowns, linear-scale
fidelity curves, and single-architecture versions of figure 8.

Figure 6 in the paper is assembled from three of the per-round PDFs — `baseline/`
round 1, `optimized/` round 1, and `optimized_with_rematerialization/` round 5,
matching the rounds highlighted in figure 5. The scripts write every round, so you
can pick different ones.

### Full reproduction

```bash
uv sync
uv run script/evaluation_fidelity_star.py
uv run script/evaluation_fidelity_t_cultivation.py
uv run script/figures/plot_fig03_execution_abstraction.py
uv run script/figures/plot_fig05_06_compilation_strategies.py
uv run script/figures/plot_fig07_synchronous_vs_asynchronous.py
uv run script/figures/plot_fig08_09_execution_time.py
uv run script/figures/plot_fig10_overall_infidelity.py
```

The two sweeps take roughly 5–15 minutes each on a laptop; the figure scripts take
seconds to a couple of minutes.

The execution-timeline scripts (figures 3, 5, 6, 7) each write two PDFs: a labeled
one and a `_no_text` sibling with the in-box labels stripped, for a layout where the
labels are set in LaTeX. `plot_fig08_09_execution_time.py` and `plot_fig10_overall_infidelity.py` write a
single PDF per figure.

### What each figure script does

**Figure 3** — `plot_fig03_execution_abstraction.py`. The execution
abstraction shared by both architectures, at minimal scale (one logical qubit, two
factories): TMR preparation then RUS teleportation for STAR on top, check stage then
escape stage then teleportation for T-cultivation below.

**Figures 5 and 6** — `plot_fig05_06_compilation_strategies.py`.
Identical-angle `Rz` gates on 25 logical qubits with 25 factories and **one AOD**,
compiled three ways: the unoptimized baseline, the coordinated assignment and
movement optimizations of §IV, and the same plus resource rematerialization. Writes
the stacked execution timeline (figure 5) and, per strategy, the
per-teleportation-round movement schedules on a distance-3 surface-code toy layout
(figure 6). Which rounds are highlighted on the timeline — and therefore which
movement schedules the paper shows — is set by `--row-rus-shade-rounds`
(default `1|1,5|5`).

**Figure 7** — `plot_fig07_synchronous_vs_asynchronous.py`. The same 25-qubit /
25-factory workload with **four AODs**, under synchronous and asynchronous
execution: the timeline on the left, the movement schedule for the highlighted
window on the right, with the realtime-control events each policy requires marked
on the timeline. The window is set by `--trap-window-start` /
`--trap-window-end`.

**Figures 8 and 9** — `plot_fig08_09_execution_time.py`. Reads
`star_full_trotter_profiling_results.csv` and
`t_cultivation_fidelity_profiling_results.csv` at d = 9. Figure 8 plots the
cumulative effect of the compilation strategies and the AOD sweep; figure 9
decomposes end-to-end execution time into preparation and movement components.

**Figure 10** — `plot_fig10_overall_infidelity.py`. Overall execution infidelity against qubit
count for the physical baseline, STAR at d = 7 and 9, and T-cultivation at
d = 7, 9, and 13, with shaded min/max bands across the 10 trials. The script also
writes the supplementary breakdowns described above, and prints the per-case
infidelity table and pairwise improvement factors.

### Reproducibility notes

- Every trial is seeded (`seed = 42 + trial`), so a rerun reproduces the committed
  CSVs byte for byte. This was verified by running a sweep twice and diffing.
- The committed data reproduces the §VII A headline numbers. Routing optimization vs
  baseline: **15%** at one AOD and **4%** at five (paper: 15% and 4%). Greedy
  execution vs routing-optimized: **61%** and **46%** (paper: 60% and 44%). The
  remaining step, high-parallelism vs greedy execution, is the smallest effect and
  sits within trial noise.
- `script/figures/plot_fig08_09_execution_time.py` selects strategies by exact setting tuple. Its
  `SETTINGS` list mirrors the first four entries of `SETTINGS` in
  `evaluation_fidelity_star.py`, and `_T_SETTING_ABLATION_GRID` mirrors
  `ABLATION_GRID` in `evaluation_fidelity_t_cultivation.py`. If you change a sweep,
  change both.
- `script/figures/plot_fig10_overall_infidelity.py` imports `MAIN_SETTINGS` directly from
  `evaluation_fidelity_star.py`, so the fidelity figures always track the sweep.
- `qiskit` is a hard dependency: OpenQASM parsing and the Gridsynth `Rz`
  decomposition that sets the T-gate count per rotation. The committed T-cultivation
  data is byte-identical under qiskit 2.5.0 and 2.5.2.
