import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_FIG_FONT_SIZE = 18

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": _FIG_FONT_SIZE,
        "axes.titlesize": _FIG_FONT_SIZE,
        "axes.labelsize": _FIG_FONT_SIZE,
        "xtick.labelsize": _FIG_FONT_SIZE,
        "ytick.labelsize": _FIG_FONT_SIZE,
        "legend.fontsize": _FIG_FONT_SIZE,
    }
)

_TARGET_T_FIG_WIDTH = 20.0
_RAW_STACKED_FIG_WIDTH = _TARGET_T_FIG_WIDTH * 0.39
_STAR_STACKED_FIG_WIDTH = _TARGET_T_FIG_WIDTH * 0.59
_UPPER_ROW_FIG_HEIGHT = 4.8
_T_FIG_HEIGHT = 5.6

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def get_n_qubit(layout_str):
    if isinstance(layout_str, str):
        rows, cols = eval(layout_str)
        return rows * cols
    return layout_str[0] * layout_str[1]


def get_code_distances(star_df):
    if "code_distance" not in star_df.columns:
        return [None]
    return sorted(star_df["code_distance"].dropna().astype(int).unique())


def normalize_star_df(star_df):
    star_df = star_df.copy()
    if "skip_rus" not in star_df.columns and "consider_skip_rus" in star_df.columns:
        star_df["skip_rus"] = star_df["consider_skip_rus"]
    return star_df


def format_t_cultivation_setting_label(
    code_distance, fidelity_target, factory_physical_size
):
    return f"{int(code_distance)}/{int(factory_physical_size)}/{fidelity_target:g}"


def _sort_t_cultivation_bar_settings(
    triples: list[tuple],
) -> list[tuple]:
    """Order bar clusters: code distance, factory size, then LER decreasing (1e-8 before 1e-10)."""

    def _key(t: tuple) -> tuple:
        code_distance, fidelity_target, factory_physical_size = t
        return (
            int(code_distance),
            int(factory_physical_size),
            -np.log10(float(fidelity_target)),
        )

    return sorted(triples, key=_key)


def _t_cultivation_setting_key(row, group_cols: list[str]) -> tuple:
    """Normalize ``qubit_layout`` and numeric columns for duplicate checks."""
    parts: list = []
    for c in group_cols:
        v = row[c]
        if c == "qubit_layout":
            if isinstance(v, str):
                nr, nc = eval(v)
            else:
                nr, nc = int(v[0]), int(v[1])
            parts.append((nr, nc))
        elif c in ("factory_physical_size", "n_aods", "n_trotter_steps"):
            parts.append(int(v))
        elif c in ("J", "h", "dt"):
            parts.append(float(v))
        else:
            parts.append(str(v))
    return tuple(parts)


def _resolve_t_cultivation_fidelity_csv_path() -> str | None:
    """Prefer ``evaluation_results.csv`` when it contains T-cultivation fidelity columns.

    You can copy or symlink ``t_cultivation_fidelity_results.csv`` to
    ``output/evaluation/evaluation_results.csv`` so all tooling reads one file.
    """
    current_required = {
        "fidelity_total",
        "code_distance",
        "fidelity_target",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_of_rz_h",
        "fidelity_of_t_gate",
        "fidelity_cnot",
        "fidelity_h",
    }
    candidates = [
        os.path.join("output", "evaluation", "evaluation_results.csv"),
        os.path.join(
            "output", "evaluation", "fidelity", "t_cultivation_fidelity_results.csv"
        ),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        header_columns = set(header.columns)
        if current_required.issubset(header_columns):
            return path
    return None


def _count_cnot_h_tfim_layer(
    qc_one_layer: list,
) -> tuple[int, int]:
    n_cnot = 0
    n_h = 0
    for inst in qc_one_layer:
        g = inst.get("gate")
        if g == "CNOT":
            n_cnot += len(inst.get("targets", []))
        elif g == "H":
            n_h += len(inst.get("targets", []))
    return n_cnot, n_h


def _count_rz_tfim_layer(qc_one_layer: list) -> int:
    """Count logical Rz gates in one TFIM layer."""
    n_rz = 0
    for inst in qc_one_layer:
        if inst.get("gate") == "Rz":
            n_rz += len(inst.get("targets", []))
    return n_rz


def _fidelity_component_label(method: str, column: str) -> str:
    """Return the display label for a fidelity component column."""
    labels = {
        "star": {
            "fidelity_of_rz_injection": "RZ Injection",
            "fidelity_of_rz_teleportaion": "Teleportation-CNOT",
            "fidelity_of_rz_s": "Rz Correction",
            "fidelity_cnot": "CNOT",
            "fidelity_1q": "H",
            "fidelity_of_rz_layer": "RZ Layer",
        },
        "t_cultivation": {
            "fidelity_of_rz_teleportaion": "Teleportation-CNOT",
            "fidelity_of_rz_s": "Rz decomposition-S",
            "fidelity_of_rz_h": "Rz decomposition-H",
            "fidelity_of_t_gate": "T",
            "fidelity_cnot": "CNOT",
            "fidelity_h": "H",
            "fidelity_approximation": "Rz approx. error",
            "fidelity_total_gate_only": "Gate-only total",
        },
    }
    return labels.get(method, {}).get(column, column)


def _fidelity_component_color(column: str) -> str:
    """Return a stable color per semantic component across stacked plots."""
    color_map = {
        # Keep CZ/CNOT visually identical across methods.
        "fidelity_cz": "#1f77b4",
        "fidelity_cnot": "#1f77b4",
        # Teleportation channel.
        "fidelity_of_rz_teleportaion": "#ff7f0e",
        # S-channel should match between STAR and T-cultivation.
        "fidelity_of_rz_s": "#2ca02c",
        # H-channel should match between STAR and T-cultivation.
        "fidelity_1q": "#8c564b",
        "fidelity_h": "#8c564b",
        "fidelity_of_rz_h": "#bcbd22",
        # Other components.
        "fidelity_of_rz_injection": "#d62728",
        "fidelity_of_t_gate": "#9467bd",
        "fidelity_approximation": "#17becf",
        "fidelity_move": "#bcbd22",
        "fidelity_idle": "#7f7f7f",
        "fidelity_init": "#e377c2",
        "fidelity_measurement": "#aec7e8",
    }
    return color_map.get(column, "#7f7f7f")


def _append_component_means(
    row: dict,
    group_df: pd.DataFrame,
    method: str,
    columns: list[str],
) -> None:
    for column in columns:
        if column not in group_df.columns:
            continue
        row[column] = group_df[column].mean()
        row[f"{column}_label"] = _fidelity_component_label(method, column)


def add_t_cultivation_approximation_fidelity(
    t_df: pd.DataFrame,
    *,
    epsilon: float = 1e-4,
) -> pd.DataFrame:
    """Multiply T-cultivation fidelity by Rz approximation fidelity.

    For each row, the approximation error per Rz is ``epsilon**2`` and
    approximation fidelity is ``(1 - epsilon**2)**n_rz_total`` where
    ``n_rz_total`` is derived from the original logical TFIM circuit.
    """
    from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz

    out = t_df.copy()
    if out.empty:
        return out

    required = {"qubit_layout", "J", "h", "dt", "n_trotter_steps", "fidelity_total"}
    if not required.issubset(set(out.columns)):
        return out

    out["fidelity_total"] = pd.to_numeric(out["fidelity_total"], errors="coerce")
    out = out.dropna(subset=["fidelity_total"]).copy()
    if out.empty:
        return out

    approx_error_per_rz = float(epsilon) ** 2
    approx_fidelity_per_rz = 1.0 - approx_error_per_rz
    n_rz_total_list: list[int] = []
    fidelity_approx_list: list[float] = []
    rz_count_cache: dict[tuple, int] = {}

    for _, row in out.iterrows():
        layout_val = row["qubit_layout"]
        if isinstance(layout_val, str):
            n_rows, n_cols = eval(layout_val)
        else:
            n_rows, n_cols = int(layout_val[0]), int(layout_val[1])

        n_qubits = n_rows * n_cols
        J = float(row["J"])
        h = float(row["h"])
        dt = float(row["dt"])
        n_trotter_steps = max(1, int(row["n_trotter_steps"]))

        key = (n_rows, n_cols, J, h, dt)
        if key not in rz_count_cache:
            qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
                n_qubits=n_qubits,
                qubit_layout=(n_rows, n_cols),
                J=J,
                h=h,
                dt=dt,
                logical=True,
                order=2,
            )
            rz_count_cache[key] = _count_rz_tfim_layer(qc_one_layer)

        n_rz_total = rz_count_cache[key] * n_trotter_steps
        fidelity_approx = float(approx_fidelity_per_rz**n_rz_total)
        n_rz_total_list.append(n_rz_total)
        fidelity_approx_list.append(fidelity_approx)

    out["n_rz_total"] = n_rz_total_list
    out["epsilon_approx"] = float(epsilon)
    out["approx_error_per_rz"] = approx_error_per_rz
    out["fidelity_approximation"] = fidelity_approx_list
    out["fidelity_total_gate_only"] = out["fidelity_total"]
    out["fidelity_total"] = out["fidelity_total"] * out["fidelity_approximation"]

    return out


def augment_t_cultivation_df_with_distance9_extrapolation(
    t_df: pd.DataFrame,
    *,
    reference_distance: int = 7,
    extrapolated_distance: int = 9,
    fidelity_target: float = 1e-8,
) -> pd.DataFrame:
    """Append distance-9 rows by analytic scaling from distance-7 CSV means.

    Uses the same logical TFIM layer as STAR/T-cultivation (``generate_one_layer_2d_tfim_circuit_cz``)
    only to count CNOT and H applications for the Clifford factor — no T-cultivation
    execution or ``generate_one_layer_2d_tfim_circuit_t_cultivation``.

    RUS teleportation infidelity is scaled as if ``fidelity_of_rz_teleportaion = F_CNOT^n``
    with ``n`` inferred from the distance-7 mean and the distance-7 logical CNOT
    fidelity. The factor ``f_inner = f_total / (f_inj * f_tel * f_cliff)`` from the
    measured distance-7 means captures remaining Rz-path (S, T, …) contributions and
    is held fixed when forming ``fidelity_total`` at distance 9.
    """
    from src.error_model import LogicalErrorModel, PhysicalErrorModel
    from src.tfim_logical import generate_one_layer_2d_tfim_circuit_cz

    current_required = {
        "fidelity_total",
        "code_distance",
        "fidelity_target",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_of_rz_h",
        "fidelity_of_t_gate",
        "fidelity_cnot",
        "fidelity_h",
    }
    out = t_df.copy()
    assert current_required.issubset(set(out.columns))
    cd_arr = np.asarray(
        pd.to_numeric(out["code_distance"], errors="coerce"), dtype=float
    )
    ft_arr = np.asarray(
        pd.to_numeric(out["fidelity_target"], errors="coerce"), dtype=float
    )
    ref_mask = (cd_arr == reference_distance) & np.isclose(
        ft_arr, fidelity_target, rtol=0.0, atol=1e-15
    )
    if not ref_mask.any():
        return out

    group_cols = [
        "factory_physical_size",
        "qubit_layout",
        "placement",
        "n_aods",
        "J",
        "h",
        "dt",
        "n_trotter_steps",
    ]
    measure_cols = [
        "fidelity_total",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_of_rz_h",
        "fidelity_of_t_gate",
        "fidelity_cnot",
        "fidelity_h",
    ]
    ref_agg = (
        out.loc[ref_mask, group_cols + measure_cols]
        .groupby(group_cols, as_index=False)[measure_cols]
        .mean()
    )
    ref_templates = (
        out.loc[ref_mask]
        .groupby(group_cols, as_index=False)
        .first()
        .reset_index(drop=True)
    )
    template_by_key = {
        _t_cultivation_setting_key(row, group_cols): row.to_dict()
        for _, row in ref_templates.iterrows()
    }

    existing_keys = set()
    for i in range(len(out)):
        if np.isnan(cd_arr[i]) or np.isnan(ft_arr[i]):
            continue
        if int(cd_arr[i]) == extrapolated_distance and np.isclose(
            ft_arr[i], fidelity_target, rtol=0.0, atol=1e-15
        ):
            existing_keys.add(_t_cultivation_setting_key(out.iloc[i], group_cols))

    physical_error_model = PhysicalErrorModel("lookahead")
    lm_ref = LogicalErrorModel(
        physical_model=physical_error_model, code_distance=reference_distance
    )
    lm_ref.integrate_t_cultivation_fidelity_target(fidelity_target)
    lm_tgt = LogicalErrorModel(
        physical_model=physical_error_model, code_distance=extrapolated_distance
    )
    lm_tgt.integrate_t_cultivation_fidelity_target(fidelity_target)

    f_cnot_ref = lm_ref.get_logical_fidelity("CNOT")
    f_cnot_target = lm_tgt.get_logical_fidelity("CNOT")
    f_h_ref = lm_ref.get_logical_fidelity("H")
    f_h_target = lm_tgt.get_logical_fidelity("H")
    f_s_ref = lm_ref.get_logical_fidelity("S")
    f_s_target = lm_tgt.get_logical_fidelity("S")
    f_t_ref = lm_ref.get_logical_fidelity("T")
    f_t_target = lm_tgt.get_logical_fidelity("T")

    new_rows: list[dict] = []

    for _, row in ref_agg.iterrows():
        cfg_row = row[group_cols]
        key = _t_cultivation_setting_key(cfg_row, group_cols)
        if key in existing_keys:
            continue

        factory_physical_size = int(cfg_row["factory_physical_size"])
        n_aods = int(cfg_row["n_aods"])
        placement = str(cfg_row["placement"])
        layout_val = cfg_row["qubit_layout"]
        if isinstance(layout_val, str):
            n_rows, n_cols = eval(layout_val)
        else:
            n_rows, n_cols = int(layout_val[0]), int(layout_val[1])
        n_qubits = n_rows * n_cols
        n_factories = n_qubits
        J = float(cfg_row["J"])
        h = float(cfg_row["h"])
        dt = float(cfg_row["dt"])
        n_trotter_steps = max(1, int(cfg_row["n_trotter_steps"]))

        mean_total = float(row["fidelity_total"])
        mean_tel = float(row["fidelity_of_rz_teleportaion"])
        mean_s = float(row["fidelity_of_rz_s"])
        mean_rz_h = float(row["fidelity_of_rz_h"])
        mean_t = float(row["fidelity_of_t_gate"])
        mean_cnot = float(row["fidelity_cnot"])
        mean_h = float(row["fidelity_h"])

        if (
            mean_total <= 0
            or mean_tel <= 0
            or mean_s <= 0
            or mean_rz_h <= 0
            or mean_t <= 0
            or mean_cnot <= 0
            or mean_h <= 0
            or f_cnot_ref <= 0
            or f_cnot_ref >= 1.0
            or f_h_ref <= 0
            or f_h_ref >= 1.0
            or f_s_ref <= 0
            or f_s_ref >= 1.0
            or f_t_ref <= 0
            or f_t_ref >= 1.0
        ):
            continue

        n_tel = np.log(mean_tel) / np.log(f_cnot_ref)
        n_s = np.log(mean_s) / np.log(f_s_ref)
        n_rz_h = np.log(mean_rz_h) / np.log(f_h_ref)
        n_t = np.log(mean_t) / np.log(f_t_ref)

        f_tel_tgt = float(f_cnot_target**n_tel)
        f_s_gate_tgt = float(f_s_target**n_s)
        f_rz_h_tgt = float(f_h_target**n_rz_h)
        f_t_gate_tgt = float(f_t_target**n_t)

        qc_one_layer = generate_one_layer_2d_tfim_circuit_cz(
            n_qubits=n_qubits,
            qubit_layout=(n_rows, n_cols),
            J=J,
            h=h,
            dt=dt,
            logical=True,
            order=2,
        )
        n_cnot_layer, n_h_layer = _count_cnot_h_tfim_layer(qc_one_layer)
        f_cliff_tgt = (f_cnot_target ** (n_cnot_layer * n_trotter_steps)) * (
            f_h_target ** (n_h_layer * n_trotter_steps)
        )

        f_total_tgt = f_tel_tgt * f_s_gate_tgt * f_rz_h_tgt * f_t_gate_tgt * f_cliff_tgt

        template = template_by_key.get(key, {})
        new_row = dict(template)
        new_row.update(
            {
                "trial": 0,
                "code_distance": extrapolated_distance,
                "fidelity_target": fidelity_target,
                "factory_physical_size": factory_physical_size,
                "qubit_layout": f"({n_rows}, {n_cols})",
                "placement": placement,
                "n_qubits": n_qubits,
                "n_factories": n_factories,
                "n_aods": n_aods,
                "J": J,
                "h": h,
                "dt": dt,
                "n_trotter_steps": n_trotter_steps,
                "fidelity_total": f_total_tgt,
                "fidelity_of_rz_teleportaion": f_tel_tgt,
                "fidelity_of_rz_s": f_s_gate_tgt,
                "fidelity_of_rz_h": f_rz_h_tgt,
                "fidelity_of_t_gate": f_t_gate_tgt,
                "fidelity_cnot": f_cliff_tgt,
                "fidelity_h": f_h_target,
            }
        )
        new_rows.append(new_row)
        existing_keys.add(key)

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    return out


def load_and_process_data():
    """Load raw, star, and optional t-cultivation fidelity results."""
    raw_df = pd.read_csv("output/evaluation/fidelity/raw_fidelity_results.csv")
    star_df = pd.read_csv("output/evaluation/fidelity/star_fidelity_results.csv")
    t_path = _resolve_t_cultivation_fidelity_csv_path()
    t_cultivation_df = pd.read_csv(t_path) if t_path else None
    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_cultivation_df = augment_t_cultivation_df_with_distance9_extrapolation(
            t_cultivation_df
        )
        t_cultivation_df = add_t_cultivation_approximation_fidelity(
            t_cultivation_df, epsilon=1e-4
        )

    return raw_df, star_df, t_cultivation_df


def generate_comparison_csv(raw_df, star_df, output_path, t_cultivation_df=None):
    """Generate a comparison CSV with raw, STAR, and optional T-cultivation rows."""

    star_df = normalize_star_df(star_df)

    # Prepare comparison data
    comparison_data = []

    # Add raw results (aggregate by n_qubit)
    for n_qubit in sorted(raw_df["n_qubit"].unique()):
        raw_subset = raw_df[raw_df["n_qubit"] == n_qubit]
        avg_fidelity = raw_subset["fidelity"].mean()

        comparison_data.append(
            {
                "method": "raw",
                "n_qubit": n_qubit,
                "qubit_layout": f"({int(np.sqrt(n_qubit))}, {int(np.sqrt(n_qubit))})",
                "placement": "N/A",
                "n_aods": "N/A",
                "skip_rus": "N/A",
                "trivial_return": "N/A",
                "decompose_move": "N/A",
                "parallel_execution": "N/A",
                "mean_fidelity": avg_fidelity,
                "std_fidelity": raw_subset["fidelity"].std(),
                "n_samples": len(raw_subset),
                "infidelity": 1 - avg_fidelity,
            }
        )

    # Add STAR results (aggregate by configuration)
    # Group by all configuration parameters
    star_grouped = star_df.groupby(
        [
            "qubit_layout",
            "placement",
            "n_aods",
            "skip_rus",
            "trivial_return",
            "decompose_move",
            "parallel_execution",
        ]
    )

    for config, group_df in star_grouped:
        (
            qubit_layout,
            placement,
            n_aods,
            skip_rus,
            trivial_return,
            decompose_move,
            parallel_execution,
        ) = config

        # Parse qubit layout to get n_qubit
        if isinstance(qubit_layout, str):
            rows, cols = eval(qubit_layout)
            n_qubit = rows * cols
        else:
            n_qubit = qubit_layout[0] * qubit_layout[1]

        row = {
            "method": "star",
            "n_qubit": n_qubit,
            "qubit_layout": str(qubit_layout),
            "placement": placement,
            "n_aods": n_aods,
            "skip_rus": skip_rus,
            "trivial_return": trivial_return,
            "decompose_move": decompose_move,
            "parallel_execution": parallel_execution,
            "mean_fidelity": group_df["fidelity"].mean(),
            "std_fidelity": group_df["fidelity"].std(),
            "n_samples": len(group_df),
            "infidelity": 1 - group_df["fidelity"].mean(),
        }
        _append_component_means(
            row,
            group_df,
            "star",
            [
                "fidelity_of_rz_injection",
                "fidelity_of_rz_teleportaion",
                "fidelity_of_rz_s",
                "fidelity_of_rz_layer",
                "fidelity_cnot",
                "fidelity_1q",
            ],
        )
        comparison_data.append(row)

    # Add T-cultivation results (aggregate by configuration + three settings)
    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = t_cultivation_df.copy()
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        t_df["fidelity_total"] = pd.to_numeric(
            t_df.get("fidelity_total"), errors="coerce"
        )
        t_df = t_df.dropna(subset=["fidelity_total"])
        if not t_df.empty:
            t_grouped = t_df.groupby(
                [
                    "n_qubit",
                    "qubit_layout",
                    "placement",
                    "n_aods",
                    "code_distance",
                    "fidelity_target",
                    "factory_physical_size",
                ]
            )
            for config, group_df in t_grouped:
                (
                    n_qubit,
                    qubit_layout,
                    placement,
                    n_aods,
                    code_distance,
                    fidelity_target,
                    factory_physical_size,
                ) = config
                mean_fidelity = group_df["fidelity_total"].mean()
                row = {
                    "method": "t_cultivation",
                    "n_qubit": n_qubit,
                    "qubit_layout": str(qubit_layout),
                    "placement": placement,
                    "n_aods": n_aods,
                    "skip_rus": "N/A",
                    "trivial_return": "N/A",
                    "decompose_move": "N/A",
                    "parallel_execution": "N/A",
                    "code_distance": code_distance,
                    "fidelity_target": fidelity_target,
                    "factory_physical_size": factory_physical_size,
                    "mean_fidelity": mean_fidelity,
                    "std_fidelity": group_df["fidelity_total"].std(),
                    "n_samples": len(group_df),
                    "infidelity": 1 - mean_fidelity,
                }
                _append_component_means(
                    row,
                    group_df,
                    "t_cultivation",
                    [
                        "fidelity_of_rz_teleportaion",
                        "fidelity_of_rz_s",
                        "fidelity_of_rz_h",
                        "fidelity_of_t_gate",
                        "fidelity_cnot",
                        "fidelity_h",
                        "fidelity_approximation",
                        "fidelity_total_gate_only",
                    ],
                )
                comparison_data.append(row)

    # Create DataFrame and save
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(["n_qubit", "method", "mean_fidelity"])
    comparison_df.to_csv(output_path, index=False)
    print(f"Comparison CSV saved to: {output_path}")

    return comparison_df


def plot_raw_infidelity_breakdown(raw_df, output_dir):
    """Plot stacked infidelity breakdown for raw results by error term."""

    # Calculate infidelity for each error source
    error_terms = [
        "fidelity_cz",
        "fidelity_1q",
        "fidelity_move",
        "fidelity_idle",
        "fidelity_init",
        "fidelity_measurement",
    ]

    # Create stacked bar plot showing contribution of each error term
    fig, ax = plt.subplots(figsize=(_RAW_STACKED_FIG_WIDTH, _UPPER_ROW_FIG_HEIGHT))

    n_qubits = sorted(raw_df["n_qubit"].unique())
    infidelity_data = {term: [] for term in error_terms}

    for n in n_qubits:
        subset = raw_df[raw_df["n_qubit"] == n]
        for term in error_terms:
            # Calculate contribution to total infidelity
            infidelity_data[term].append(1 - subset[term].mean())

    # Create stacked bar chart
    x = np.arange(len(n_qubits))
    width = 0.30

    bottom = np.zeros(len(n_qubits))
    colors = [_fidelity_component_color(term) for term in error_terms]
    label_map = {
        "fidelity_idle": "Idle",
        "fidelity_move": "Move",
        "fidelity_init": "Init",
        "fidelity_measurement": "Measurement",
    }

    for idx, (term, color) in enumerate(zip(error_terms, colors)):
        values = infidelity_data[term]
        ax.bar(
            x,
            values,
            width,
            label=label_map.get(term, term.replace("fidelity_", "").upper()),
            bottom=bottom,
            color=color,
            alpha=0.8,
        )
        bottom += values

    ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
    ax.set_ylabel("Total Infidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title("Raw Infidelity Breakdown by Error Source", fontsize=_FIG_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(n_qubits, fontsize=_FIG_FONT_SIZE)
    ax.legend(loc="upper left", fontsize=_FIG_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax.grid(True, alpha=0.3, axis="y")

    output_path = os.path.join(output_dir, "raw_infidelity_stacked.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Raw infidelity stacked plot saved to: {output_path}")
    plt.close()


def plot_star_infidelity_breakdown(star_df, output_dir):
    """Plot infidelity breakdown for STAR results: 2 rows (placements) × 4 cols (error terms)."""

    # Error terms in STAR data (exclude CNOT and 1Q as they are constant)
    error_terms = [
        "fidelity_of_rz_injection",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
    ]

    star_df = normalize_star_df(star_df)
    star_df["n_qubit"] = star_df["qubit_layout"].apply(get_n_qubit)

    # Create merged figure for all placements
    placements = sorted(star_df["placement"].unique())
    fig, axes = plt.subplots(
        len(placements), len(error_terms), figsize=(16, 5.0 * len(placements))
    )

    for placement_idx, placement in enumerate(placements):
        placement_df = star_df[star_df["placement"] == placement]

        for term_idx, term in enumerate(error_terms):
            ax = axes[placement_idx, term_idx]

            # For each n_aods configuration, plot the trend
            for n_aods in sorted(placement_df["n_aods"].unique()):
                aod_subset = placement_df[placement_df["n_aods"] == n_aods]
                grouped = aod_subset.groupby("n_qubit")

                n_qubits = sorted(aod_subset["n_qubit"].unique())
                infidelities = [1 - grouped.get_group(n)[term].mean() for n in n_qubits]

                ax.plot(
                    n_qubits,
                    infidelities,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=f"n_aods={n_aods}",
                    alpha=0.7,
                )

            ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
            ax.set_ylabel("Infidelity (1 - Fidelity)", fontsize=_FIG_FONT_SIZE)
            term_name = _fidelity_component_label("star", term)
            ax.set_title(term_name, fontsize=_FIG_FONT_SIZE)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            ax.tick_params(axis="both", labelsize=_FIG_FONT_SIZE)
            if placement_idx == 0:
                ax.legend(fontsize=_FIG_FONT_SIZE)

    # Set row labels
    for placement_idx, placement in enumerate(placements):
        axes[placement_idx, 0].set_ylabel(
            f"{placement.upper()}\nInfidelity",
            fontsize=_FIG_FONT_SIZE,
            fontweight="bold",
        )

    fig.suptitle(
        "STAR Infidelity Breakdown by Error Type and Placement",
        fontsize=_FIG_FONT_SIZE,
    )
    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_infidelity_breakdown.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"STAR infidelity breakdown plot saved to: {output_path}")
    plt.close(fig)

    # Create one clustered stacked chart by distance, averaged over STAR configurations
    stacked_terms = [
        "fidelity_cnot",
        "fidelity_1q",
        "fidelity_of_rz_injection",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
    ]
    colors = [_fidelity_component_color(term) for term in stacked_terms]

    if "code_distance" in star_df.columns:
        available_distances = sorted(
            star_df["code_distance"].dropna().astype(int).unique()
        )
        distances = [d for d in [7, 9] if d in available_distances]
        if len(distances) == 0:
            distances = available_distances
    else:
        distances = []

    if len(distances) == 0:
        print("Skipping STAR stacked distance cluster plot: no code_distance available")
        return

    fig, ax = plt.subplots(
        figsize=(_STAR_STACKED_FIG_WIDTH, _UPPER_ROW_FIG_HEIGHT * 1.18)
    )

    grouped = (
        star_df[star_df["code_distance"].isin(distances)]
        .groupby(["n_qubit", "code_distance"], as_index=False)[stacked_terms]
        .mean()
    )

    n_qubits = sorted(grouped["n_qubit"].unique())
    x_group = np.arange(len(n_qubits))
    group_width = 0.45
    bar_width = group_width / len(distances)

    xtick_positions = []
    xtick_labels = []

    for dist_idx, code_distance in enumerate(distances):
        offset = (dist_idx - (len(distances) - 1) / 2) * bar_width
        x = x_group + offset

        subset = (
            grouped[grouped["code_distance"] == code_distance]
            .set_index("n_qubit")
            .reindex(n_qubits)
        )

        bottom = np.zeros(len(n_qubits))
        for term, color in zip(stacked_terms, colors):
            values = (1 - subset[term]).fillna(0).values
            label = _fidelity_component_label("star", term)
            ax.bar(
                x,
                values,
                bar_width * 0.9,
                label=label if dist_idx == 0 else "_nolegend_",
                bottom=bottom,
                color=color,
                alpha=0.85,
            )
            bottom += values

        xtick_positions.extend(x.tolist())
        xtick_labels.extend([str(code_distance)] * len(x))

    # Set y-axis ticks for better readability
    yticks = np.arange(0, 0.0061, 0.001)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.3f}" for y in yticks])

    ax.set_ylabel("Total Infidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title(
        "STAR Infidelity Breakdown (Distance 7 and 9)", fontsize=_FIG_FONT_SIZE
    )
    ax.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=_FIG_FONT_SIZE)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=_FIG_FONT_SIZE)
    ax.set_xlabel("Code Distance", fontsize=_FIG_FONT_SIZE, labelpad=2)

    secax = ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    secax.set_xticks(x_group)
    secax.set_xticklabels([str(n) for n in n_qubits], fontsize=_FIG_FONT_SIZE)
    secax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=10)
    secax.spines["bottom"].set_position(("outward", 42))

    fig.tight_layout()
    output_path = os.path.join(output_dir, "star_infidelity_stacked_best.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"STAR infidelity stacked plot saved to: {output_path}")
    plt.close(fig)


def plot_t_cultivation_infidelity_breakdown(t_cultivation_df, output_dir):
    """Plot stacked infidelity breakdown for T-cultivation results by setting."""

    current_terms = [
        "fidelity_cnot",
        "fidelity_h",
        "fidelity_of_rz_teleportaion",
        "fidelity_of_rz_s",
        "fidelity_of_rz_h",
        "fidelity_of_t_gate",
        "fidelity_approximation",
    ]

    t_df = t_cultivation_df.copy()
    t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)

    if not all(column in t_df.columns for column in current_terms):
        missing = [column for column in current_terms if column not in t_df.columns]
        print(
            "Skipping T-cultivation stacked infidelity plot: "
            f"missing required columns {missing}"
        )
        return

    stacked_terms = current_terms
    for column in ["fidelity_total", *stacked_terms]:
        if column in t_df.columns:
            t_df[column] = pd.to_numeric(t_df[column], errors="coerce")

    t_df = t_df.dropna(subset=["n_qubit", *stacked_terms])
    if t_df.empty:
        print("Skipping T-cultivation stacked infidelity plot: no usable data")
        return

    grouped = (
        t_df.groupby(
            ["n_qubit", "code_distance", "fidelity_target", "factory_physical_size"],
            as_index=False,
        )[stacked_terms]
        .mean()
        .sort_values(
            ["n_qubit", "code_distance", "factory_physical_size", "fidelity_target"]
        )
    )

    settings = _sort_t_cultivation_bar_settings(
        [
            (code_distance, fidelity_target, factory_physical_size)
            for code_distance, fidelity_target, factory_physical_size in grouped[
                ["code_distance", "fidelity_target", "factory_physical_size"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ]
    )

    if len(settings) == 0:
        print("Skipping T-cultivation stacked infidelity plot: no settings available")
        return

    n_qubits = sorted(grouped["n_qubit"].unique())

    total_infidelity = (1 - grouped[stacked_terms]).sum(axis=1)
    grouped = grouped.assign(total_infidelity=total_infidelity)
    setting_totals = grouped.groupby(
        ["code_distance", "fidelity_target", "factory_physical_size"]
    )["total_infidelity"].max()
    dominant_setting = setting_totals.idxmax()
    dominant_group = grouped[
        (grouped["code_distance"] == dominant_setting[0])
        & (grouped["fidelity_target"] == dominant_setting[1])
        & (grouped["factory_physical_size"] == dominant_setting[2])
    ]
    other_max = setting_totals.drop(dominant_setting).max()
    dominant_min = dominant_group["total_infidelity"].min()

    use_broken_axis = (
        pd.notna(other_max) and pd.notna(dominant_min) and other_max < dominant_min
    )

    # Figure width: scale with qubit groups × settings to avoid x-label overlap.
    _n_set = max(1, len(settings))
    _n_q = max(1, len(n_qubits))
    fig_w = float(max(13.0, min(20.0, 10.0 + 0.32 * _n_q * _n_set)))
    fig_h = 9.5 if use_broken_axis else 10.0

    if use_broken_axis:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
        )
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_bottom.tick_params(top=False)
        bottom_ylim_max = max(other_max * 1.4, other_max + 1e-4, 1e-4)
        top_ylim_min = max(dominant_min * 0.8, bottom_ylim_max * 1.5)
        ax_bottom.set_ylim(0, bottom_ylim_max)
        ax_top.set_ylim(top_ylim_min, grouped["total_infidelity"].max() * 1.08)
    else:
        fig, ax_bottom = plt.subplots(figsize=(fig_w, fig_h))
        ax_top = None
        ax_bottom.set_ylim(0, 0.006)
    x_group = np.arange(len(n_qubits))
    group_width = 0.95
    bar_width = group_width / len(settings)

    xtick_positions = []
    xtick_labels = []

    colors = [_fidelity_component_color(term) for term in stacked_terms]

    def draw_bars(ax):
        for setting_idx, (
            code_distance,
            fidelity_target,
            factory_physical_size,
        ) in enumerate(settings):
            offset = (setting_idx - (len(settings) - 1) / 2) * bar_width
            x = x_group + offset

            subset = (
                grouped[
                    (grouped["code_distance"] == code_distance)
                    & (grouped["fidelity_target"] == fidelity_target)
                    & (grouped["factory_physical_size"] == factory_physical_size)
                ]
                .set_index("n_qubit")
                .reindex(n_qubits)
            )

            bottom = np.zeros(len(n_qubits))
            for term, color in zip(stacked_terms, colors):
                values = (1 - subset[term]).fillna(0).values
                label = _fidelity_component_label("t_cultivation", term)
                ax.bar(
                    x,
                    values,
                    bar_width * 0.9,
                    label=(
                        label if setting_idx == 0 and ax is ax_bottom else "_nolegend_"
                    ),
                    bottom=bottom,
                    color=color,
                    alpha=0.85,
                    edgecolor=None,
                    linewidth=0.0,
                    hatch=None,
                )
                bottom += values

            xtick_positions.extend(x.tolist())
            setting_label = format_t_cultivation_setting_label(
                code_distance, fidelity_target, factory_physical_size
            )
            xtick_labels.extend([setting_label] * len(x))

    draw_bars(ax_bottom)
    if ax_top is not None:
        draw_bars(ax_top)
        # Store top_ylim_min for later use in ticks
        top_ylim_min = ax_top.get_ylim()[0]
    else:
        top_ylim_min = None

    # Set y-axis ticks for better readability on the lower (cropped) axis.
    yticks = np.array([0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.010])
    ax_bottom.set_yticks(yticks)
    ax_bottom.set_yticklabels([f"{y:.3f}" for y in yticks])
    if ax_top is not None:
        # Keep default top-axis ticks and add one explicit 0.010 tick.
        top_ticks = ax_top.get_yticks()
        if ax_top.get_ylim()[0] <= 0.010 <= ax_top.get_ylim()[1]:
            ax_top.set_yticks(np.sort(np.unique(np.append(top_ticks, 0.010))))
        ax_top.grid(True, alpha=0.3, axis="y")
        ax_top.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax_bottom.set_ylabel("Total Infidelity", fontsize=_FIG_FONT_SIZE, labelpad=12)
    ax_bottom.tick_params(axis="y", labelsize=_FIG_FONT_SIZE)
    ax_bottom.grid(True, alpha=0.3, axis="y")
    ax_bottom.set_xticks(xtick_positions)
    ax_bottom.set_xticklabels(
        xtick_labels,
        fontsize=_FIG_FONT_SIZE,
        linespacing=1.1,
        ha="center",
        rotation=35,
    )
    ax_bottom.set_xlabel(
        "T cultivation setting (distance/#Stage 1 Patch/LER)",
        fontsize=_FIG_FONT_SIZE,
        labelpad=-2,
    )
    ax_bottom.tick_params(axis="x", pad=1, length=3)

    if ax_top is not None:
        ax_top.set_title(
            "T-cultivation Infidelity Breakdown by Setting", fontsize=_FIG_FONT_SIZE
        )
        handles, labels = ax_bottom.get_legend_handles_labels()
        filtered = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if not label.startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.18),
                ncol=4,
                fontsize=_FIG_FONT_SIZE,
                frameon=True,
                framealpha=0.95,
            )

        ax_top.text(
            0,
            0,
            "//",
            transform=ax_top.transAxes,
            fontsize=_FIG_FONT_SIZE,
            va="center",
            ha="left",
        )
        ax_bottom.text(
            0,
            1,
            "//",
            transform=ax_bottom.transAxes,
            fontsize=_FIG_FONT_SIZE,
            va="center",
            ha="left",
        )
    else:
        ax_bottom.set_title(
            "T-cultivation Infidelity Breakdown by Setting", fontsize=_FIG_FONT_SIZE
        )
        handles, labels = ax_bottom.get_legend_handles_labels()
        filtered = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if not label.startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.18),
                ncol=4,
                fontsize=_FIG_FONT_SIZE,
                frameon=True,
                framealpha=0.95,
            )

    secax = ax_bottom.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    secax.set_xticks(x_group)
    secax.set_xticklabels([str(n) for n in n_qubits], fontsize=_FIG_FONT_SIZE)
    secax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE, labelpad=4)
    secax.tick_params(axis="x", pad=2)
    secax.spines["bottom"].set_position(("outward", 108))

    if ax_top is not None:
        fig.subplots_adjust(hspace=0.05, top=0.85, bottom=0.53, left=0.08, right=0.97)
    else:
        fig.subplots_adjust(top=0.88, bottom=0.50, left=0.08, right=0.97, wspace=0.2)
    output_path = os.path.join(output_dir, "t_cultivation_infidelity_stacked_best.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"T-cultivation infidelity stacked plot saved to: {output_path}")
    plt.close(fig)


def plot_star_settings_comparison(star_df, output_dir):
    """Plot merged STAR ablation comparison: rows=placements × cols=AOD (1,2,5)."""

    star_df = normalize_star_df(star_df)
    star_df["n_qubit"] = star_df["qubit_layout"].apply(get_n_qubit)

    for col in ["trivial_return", "decompose_move", "parallel_execution"]:
        if star_df[col].dtype == object:
            star_df[col] = (
                star_df[col].astype(str).str.lower().map({"true": True, "false": False})
            )

    ablation_settings_all = [
        ("Vanilla", True, 0, False, False),
        ("Optimized Return", False, 0, False, False),
        ("Optimized Return + Skip Partial RUS", False, 1, False, False),
        ("Optimized Return + Skip Whole RUS", False, 2, False, False),
        ("Optimized Decompose Return + Skip Whole RUS", False, 2, True, False),
        (
            "Optimized Return + Skip Whole RUS + Async RUS",
            False,
            2,
            False,
            True,
        ),
    ]

    aod_values = [1, 2, 5]
    placements = sorted(star_df["placement"].unique())
    n_placements = len(placements)

    for code_distance in get_code_distances(star_df):
        if code_distance is None:
            distance_df = star_df
            distance_suffix = ""
            title_suffix = ""
        else:
            distance_df = star_df[star_df["code_distance"] == code_distance]
            distance_suffix = f"_distance_{code_distance}"
            title_suffix = f" (Distance = {code_distance})"

        if distance_df.empty:
            continue

        fig, axes = plt.subplots(
            n_placements, len(aod_values), figsize=(18, 5.0 * n_placements)
        )
        if n_placements == 1:
            axes = axes.reshape(1, len(aod_values))

        col_ylims: list[tuple[float, float] | None] = [None] * len(aod_values)

        for col_idx, n_aods in enumerate(aod_values):
            aod_df = distance_df[distance_df["n_aods"] == n_aods]
            if aod_df.empty:
                print(f"No data found for n_aods={n_aods}, distance={code_distance}")
                continue

            if n_aods == 1:
                ablation_settings = ablation_settings_all[:-1]
            else:
                ablation_settings = ablation_settings_all

            for row_idx, placement in enumerate(placements):
                ax = axes[row_idx, col_idx]
                placement_df = aod_df[aod_df["placement"] == placement]

                for (
                    label,
                    trivial_return,
                    skip_rus,
                    decompose_move,
                    parallel_execution,
                ) in ablation_settings:
                    subset = placement_df[
                        (placement_df["trivial_return"] == trivial_return)
                        & (placement_df["skip_rus"] == skip_rus)
                        & (placement_df["decompose_move"] == decompose_move)
                        & (placement_df["parallel_execution"] == parallel_execution)
                    ]

                    if subset.empty:
                        continue

                    grouped = subset.groupby("n_qubit")["fidelity"]
                    means = grouped.mean().sort_index()
                    mins = grouped.min().reindex(means.index)
                    maxs = grouped.max().reindex(means.index)
                    lower_err = (means - mins).clip(lower=0)
                    upper_err = (maxs - means).clip(lower=0)
                    yerr = np.vstack([lower_err.values, upper_err.values])

                    ax.errorbar(
                        means.index,
                        means.values,
                        yerr=yerr,
                        marker="o",
                        linewidth=2,
                        markersize=7,
                        capsize=4,
                        label=label,
                    )

                ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
                ax.set_ylabel("Fidelity", fontsize=_FIG_FONT_SIZE)
                ax.set_title(f"{placement}, AOD={n_aods}", fontsize=_FIG_FONT_SIZE)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=_FIG_FONT_SIZE)
                ax.legend(fontsize=_FIG_FONT_SIZE)

                current = ax.get_ylim()
                prev_ylim = col_ylims[col_idx]
                if prev_ylim is None:
                    col_ylims[col_idx] = current
                else:
                    col_ylims[col_idx] = (
                        min(prev_ylim[0], current[0]),
                        max(prev_ylim[1], current[1]),
                    )

        for col_idx in range(len(aod_values)):
            if col_ylims[col_idx] is not None:
                for row_idx in range(n_placements):
                    axes[row_idx, col_idx].set_ylim(col_ylims[col_idx])

        fig.suptitle(
            f"STAR Ablation Study{title_suffix}", fontsize=_FIG_FONT_SIZE, y=0.995
        )
        fig.tight_layout()
        output_path = os.path.join(
            output_dir, f"star_ablation_fidelity{distance_suffix}.pdf"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"STAR ablation plot saved to: {output_path}")


def plot_overall_fidelity_comparison(
    raw_df, star_df, output_dir, t_cultivation_df=None
):
    """Create overall fidelity comparison with STAR and optional T-cultivation lines."""

    fig, ax = plt.subplots(figsize=(10, 5.0))

    # Plot raw fidelity
    raw_data = (
        raw_df.groupby("n_qubit", as_index=False)["fidelity"]
        .mean()
        .sort_values("n_qubit")
    )
    ax.plot(
        raw_data["n_qubit"],
        raw_data["fidelity"],
        marker="s",
        linewidth=3,
        markersize=10,
        label="Raw (Physical)",
        color="red",
        alpha=0.7,
    )

    star_data = normalize_star_df(star_df)
    star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
    colors = {7: "tab:blue", 9: "tab:orange"}

    for code_distance in get_code_distances(star_data):
        if code_distance is None:
            distance_df = star_data
            label = "STAR"
            color = None
        else:
            distance_df = star_data[star_data["code_distance"] == code_distance]
            label = f"STAR (distance={code_distance})"
            color = colors.get(code_distance)

        if distance_df.empty:
            continue

        grouped = distance_df.groupby("n_qubit")["fidelity"]
        means = grouped.mean().sort_index()
        mins = grouped.min().reindex(means.index)
        maxs = grouped.max().reindex(means.index)

        line = ax.plot(
            means.index,
            means.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=label,
            alpha=0.8,
            color=color,
        )[0]
        ax.fill_between(
            means.index,
            mins.values,
            maxs.values,
            color=line.get_color(),
            alpha=0.15,
        )

    # Plot optional T-cultivation lines by setting:
    # (code_distance, factory_physical_size, fidelity_target)
    if t_cultivation_df is not None and not t_cultivation_df.empty:
        t_df = t_cultivation_df.copy()
        t_df["n_qubit"] = t_df["qubit_layout"].apply(get_n_qubit)
        t_df["fidelity_total"] = pd.to_numeric(
            t_df.get("fidelity_total"), errors="coerce"
        )
        t_df = t_df.dropna(subset=["fidelity_total"])
        if not t_df.empty:
            setting_cols = ["code_distance", "factory_physical_size", "fidelity_target"]
            for setting, setting_df in t_df.groupby(setting_cols):
                code_distance, factory_physical_size, fidelity_target = setting
                grouped = setting_df.groupby("n_qubit")["fidelity_total"]
                means = grouped.mean().sort_index()
                mins = grouped.min().reindex(means.index)
                maxs = grouped.max().reindex(means.index)
                label = (
                    "T-cultivation "
                    f"(d={int(code_distance)}, size={int(factory_physical_size)}, "
                    f"target={fidelity_target:g})"
                )
                line = ax.plot(
                    means.index,
                    means.values,
                    marker="^",
                    linewidth=2,
                    markersize=8,
                    linestyle="--",
                    label=label,
                    alpha=0.9,
                )[0]
                ax.fill_between(
                    means.index,
                    mins.values,
                    maxs.values,
                    color=line.get_color(),
                    alpha=0.12,
                )

    ax.set_xlabel("Number of Qubits", fontsize=_FIG_FONT_SIZE)
    ax.set_ylabel("Mean Fidelity", fontsize=_FIG_FONT_SIZE)
    ax.set_title(
        "Overall Fidelity Comparison: Raw vs STAR vs T-cultivation",
        fontsize=_FIG_FONT_SIZE,
    )
    ax.tick_params(axis="both", labelsize=_FIG_FONT_SIZE)
    ax.legend(
        fontsize=_FIG_FONT_SIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
    )
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(right=0.74)

    output_path = os.path.join(output_dir, "overall_fidelity_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Overall fidelity comparison plot saved to: {output_path}")
    plt.close()


def print_star_vs_t_cultivation_improvement_ratio(star_df, t_cultivation_df) -> None:
    """Print STAR/T-cultivation fidelity ratios at matched code distance and qubit count."""
    if t_cultivation_df is None or t_cultivation_df.empty:
        print("Skipping STAR vs T-cultivation ratio: no T-cultivation data loaded")
        return
    if "code_distance" not in star_df.columns:
        print("Skipping STAR vs T-cultivation ratio: STAR has no code_distance column")
        return

    star_data = normalize_star_df(star_df).copy()
    star_data["n_qubit"] = star_data["qubit_layout"].apply(get_n_qubit)
    star_data["fidelity"] = pd.to_numeric(star_data["fidelity"], errors="coerce")
    star_data["code_distance"] = pd.to_numeric(
        star_data["code_distance"], errors="coerce"
    )
    star_data = star_data.dropna(subset=["code_distance", "n_qubit", "fidelity"])
    if star_data.empty:
        print("Skipping STAR vs T-cultivation ratio: STAR data is not usable")
        return

    t_data = t_cultivation_df.copy()
    t_data["n_qubit"] = t_data["qubit_layout"].apply(get_n_qubit)
    t_data["fidelity_total"] = pd.to_numeric(t_data["fidelity_total"], errors="coerce")
    t_data["code_distance"] = pd.to_numeric(t_data["code_distance"], errors="coerce")
    t_data = t_data.dropna(subset=["code_distance", "n_qubit", "fidelity_total"])
    if t_data.empty:
        print("Skipping STAR vs T-cultivation ratio: T-cultivation data is not usable")
        return

    # Average across each method's internal settings, then compare matched distance/qubit points.
    star_grouped = (
        star_data.groupby(["code_distance", "n_qubit"], as_index=False)["fidelity"]
        .mean()
        .rename(columns={"fidelity": "star_fidelity"})
    )
    t_grouped = (
        t_data.groupby(["code_distance", "n_qubit"], as_index=False)["fidelity_total"]
        .mean()
        .rename(columns={"fidelity_total": "t_cultivation_fidelity"})
    )

    merged = (
        star_grouped.merge(t_grouped, on=["code_distance", "n_qubit"], how="inner")
        .sort_values(["code_distance", "n_qubit"])
        .reset_index(drop=True)
    )
    if merged.empty:
        print(
            "Skipping STAR vs T-cultivation ratio: no overlapping "
            "(code_distance, n_qubit) points"
        )
        return

    eps = 1e-15
    merged["star_infidelity"] = 1.0 - merged["star_fidelity"]
    merged["t_cultivation_infidelity"] = 1.0 - merged["t_cultivation_fidelity"]

    merged["infidelity_reduction_abs"] = (
        merged["t_cultivation_infidelity"] - merged["star_infidelity"]
    )
    merged["infidelity_reduction_pct"] = (
        merged["infidelity_reduction_abs"]
        / np.clip(merged["t_cultivation_infidelity"], eps, None)
    ) * 100.0
    merged["infidelity_reduction_factor"] = np.clip(
        merged["t_cultivation_infidelity"], eps, None
    ) / np.clip(merged["star_infidelity"], eps, None)

    print("\nSTAR vs T-cultivation infidelity reduction (same distance):")
    print(
        "  Columns: distance, n_qubit, STAR infidelity, "
        "T-cultivation infidelity, reduction(abs), reduction(%), reduction_factor"
    )
    for _, row in merged.iterrows():
        print(
            "  "
            f"d={int(row['code_distance'])}, "
            f"n={int(row['n_qubit'])}, "
            f"STAR_inf={row['star_infidelity']:.8e}, "
            f"T_inf={row['t_cultivation_infidelity']:.8e}, "
            f"red_abs={row['infidelity_reduction_abs']:.8e}, "
            f"red_pct={row['infidelity_reduction_pct']:.3f}%, "
            f"factor={row['infidelity_reduction_factor']:.6f}"
        )

    # Distance-level aggregate and geometric mean of reduction factor across qubit sizes.
    by_distance = (
        merged.groupby("code_distance", as_index=False)[
            [
                "star_infidelity",
                "t_cultivation_infidelity",
                "infidelity_reduction_abs",
                "infidelity_reduction_pct",
            ]
        ]
        .mean()
        .sort_values("code_distance")
    )

    geom_rows = []
    for code_distance, dist_df in merged.groupby("code_distance"):
        factors = np.clip(dist_df["infidelity_reduction_factor"].to_numpy(), eps, None)
        geomean_factor = float(np.exp(np.mean(np.log(factors))))
        geom_rows.append(
            {
                "code_distance": code_distance,
                "geomean_reduction_factor": geomean_factor,
            }
        )
    geomean_by_distance = pd.DataFrame(geom_rows)
    by_distance = by_distance.merge(
        geomean_by_distance, on="code_distance", how="left"
    ).sort_values("code_distance")

    print("\nDistance-level summary (includes geomean reduction factor):")
    for _, row in by_distance.iterrows():
        print(
            "  "
            f"d={int(row['code_distance'])}: "
            f"mean_STAR_inf={row['star_infidelity']:.8e}, "
            f"mean_T_inf={row['t_cultivation_infidelity']:.8e}, "
            f"mean_red_abs={row['infidelity_reduction_abs']:.8e}, "
            f"mean_red_pct={row['infidelity_reduction_pct']:.3f}%, "
            f"geomean_factor={row['geomean_reduction_factor']:.6f}"
        )


def main():
    """Main function to generate comparison CSV and plots."""
    print("=" * 80)
    print("FIDELITY COMPARISON AND VISUALIZATION")
    print("=" * 80 + "\n")

    # Create output directory
    output_dir = "output/evaluation/fidelity/comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    raw_df, star_df, t_cultivation_df = load_and_process_data()
    t_count = 0 if t_cultivation_df is None else len(t_cultivation_df)
    print(
        f"Loaded {len(raw_df)} raw results, {len(star_df)} STAR results, "
        f"and {t_count} T-cultivation results\n"
    )

    # Generate comparison CSV
    print("Generating comparison CSV...")
    comparison_csv_path = os.path.join(output_dir, "fidelity_comparison.csv")
    comparison_df = generate_comparison_csv(
        raw_df, star_df, comparison_csv_path, t_cultivation_df=t_cultivation_df
    )
    print(f"Generated {len(comparison_df)} comparison rows\n")

    print_star_vs_t_cultivation_improvement_ratio(star_df, t_cultivation_df)
    print()

    # Generate plots
    print("Generating plots...")
    print("\n1. Raw infidelity stacked breakdown...")
    plot_raw_infidelity_breakdown(raw_df, output_dir)

    print("\n2. STAR infidelity breakdown...")
    plot_star_infidelity_breakdown(star_df, output_dir)

    print("\n3. T-cultivation infidelity breakdown...")
    if t_cultivation_df is not None:
        plot_t_cultivation_infidelity_breakdown(t_cultivation_df, output_dir)
    else:
        print("Skipping T-cultivation stacked infidelity plot: no data loaded")

    print("\n4. STAR settings comparison (1 AOD vs 5 AOD)...")
    plot_star_settings_comparison(star_df, output_dir)

    print("\n5. Overall fidelity comparison...")
    plot_overall_fidelity_comparison(
        raw_df, star_df, output_dir, t_cultivation_df=t_cultivation_df
    )

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
