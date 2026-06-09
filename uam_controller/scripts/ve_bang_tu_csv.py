#!/usr/bin/env python3
"""Generate detailed UAM verification tables and plots from a logger CSV.

The default input is the best Case B / combined / x030 trial used for the
graduation report.  The script also accepts --csv so it can be reused for any
other run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in report mode
    yaml = None


DEFAULT_CSV = Path(
    "/home/wicom/uam_results/case_b_combined_ff_narrow_sweep_20260607_150103/"
    "runs/case_b_ff85_x030/combined_amp080/case_b_rbfnn_ff/"
    "rbfnn_residual_arm_trial_004/logger/"
    "20260607_152514_case_b_rbfnn_ff_combined_amp080/flight_timeseries.csv"
)


def finite(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: object, digits: int = 4, unit: str = "") -> str:
    value = finite(value)
    if not math.isfinite(value):
        return "-"
    text = f"{value:.{digits}f}"
    return f"{text} {unit}".strip()


def rms(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(arr))))


def vector_norm(frame: pd.DataFrame, cols: list[str]) -> pd.Series | None:
    if not all(col in frame.columns for col in cols):
        return None
    return np.sqrt(np.square(frame[cols].astype(float)).sum(axis=1))


def norm_stats(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    norm = vector_norm(frame, cols)
    if norm is None:
        return {}
    return {
        "rms": rms(norm),
        "mean": float(norm.mean()),
        "max": float(norm.max()),
        "final": float(norm.iloc[-1]),
    }


def correlation(left: pd.Series, right: pd.Series) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    left_arr = left_arr[mask]
    right_arr = right_arr[mask]
    if left_arr.size < 2:
        return math.nan
    if float(np.std(left_arr)) < 1e-12 or float(np.std(right_arr)) < 1e-12:
        return math.nan
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return math.nan
    return numerator / denominator


def series_stats(frame: pd.DataFrame, col: str) -> dict[str, float]:
    if col not in frame.columns:
        return {}
    values = frame[col].astype(float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "rms": rms(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "max_abs": float(values.abs().max()),
        "final": float(values.iloc[-1]),
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def draw_table_png(
    path: Path,
    title: str,
    headers: list[str],
    rows: list[list[object]],
    *,
    font_size: int = 8,
    width: float = 13.5,
) -> None:
    height = max(2.5, 0.42 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10, weight="bold")
    table = ax.table(
        cellText=[[str(item) for item in row] for row in rows],
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f4858")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7f9")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_series(out_dir: Path, name: str, frame: pd.DataFrame, cols: list[str], ylabel: str) -> None:
    available = [col for col in cols if col in frame.columns]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for col in available:
        ax.plot(frame["t_s"], frame[col], label=col, linewidth=1.4)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=220)
    plt.close(fig)


def plot_ff_disturbance_residual(out_dir: Path, frame: pd.DataFrame) -> None:
    axes = [
        ("x", "dbg_tau_arm_ff_x", "dbg_tau_arm_disturbance_x"),
        ("y", "dbg_tau_arm_ff_y", "dbg_tau_arm_disturbance_y"),
        ("z", "dbg_tau_arm_ff_z", "dbg_tau_arm_disturbance_z"),
    ]
    if not all(ff in frame.columns and dist in frame.columns for _, ff, dist in axes):
        return
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
    for ax, (name, ff_col, dist_col) in zip(axs, axes):
        residual = frame[dist_col].astype(float) - frame[ff_col].astype(float)
        ax.plot(frame["t_s"], frame[dist_col], label=f"dist_{name}", linewidth=1.2)
        ax.plot(frame["t_s"], frame[ff_col], label=f"ff_{name}", linewidth=1.2)
        ax.plot(frame["t_s"], residual, label=f"residual_{name}", linewidth=1.1, linestyle="--")
        ax.grid(True, alpha=0.35)
        ax.set_ylabel("Nm")
        ax.legend(loc="best", fontsize=8)
    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "ff_disturbance_residual_by_axis.png", dpi=220)
    plt.close(fig)


def plot_rbfnn_vs_rate_error(out_dir: Path, frame: pd.DataFrame) -> None:
    axes = [
        ("x / roll", "dbg_n_hat_x", "dbg_e_omega_x"),
        ("y / pitch", "dbg_n_hat_y", "dbg_e_omega_y"),
        ("z / yaw", "dbg_n_hat_z", "dbg_e_omega_z"),
    ]
    if not all(n in frame.columns and e in frame.columns for _, n, e in axes):
        return
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
    for ax, (name, n_col, e_col) in zip(axs, axes):
        ax.plot(frame["t_s"], frame[e_col], label=f"e_omega {name}", linewidth=1.2)
        ax2 = ax.twinx()
        ax2.plot(frame["t_s"], frame[n_col], label=f"n_hat {name}", color="#c44e52", linewidth=1.1)
        ax.grid(True, alpha=0.35)
        ax.set_ylabel("rad/s")
        ax2.set_ylabel("n_hat")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "rbfnn_vs_rate_error_by_axis.png", dpi=220)
    plt.close(fig)


def find_scoreboard(csv_path: Path) -> Path | None:
    for parent in csv_path.parents:
        candidate = parent / "scoreboard.csv"
        if candidate.exists():
            return candidate
        if parent.name.startswith("rbfnn_residual_arm_trial_"):
            case_dir = parent.parent
            candidate = case_dir / "scoreboard.csv"
            if candidate.exists():
                return candidate
    return None


def read_scoreboard_row(scoreboard: Path | None, summary_path: Path | None) -> dict[str, str]:
    if scoreboard is None or not scoreboard.exists():
        return {}
    rows = list(csv.DictReader(scoreboard.open("r", newline="", encoding="utf-8")))
    if summary_path is not None:
        summary_text = str(summary_path)
        for row in rows:
            if row.get("summary_json") == summary_text:
                return row
    return rows[0] if rows else {}


def load_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_controller_params(scoreboard_row: dict[str, str]) -> dict:
    config_path = Path(scoreboard_row.get("config_path", ""))
    if yaml is None or not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    for node_name in ("uam_backstepping_rbfnn_node", "uam_adaptive_controller"):
        node = data.get(node_name, {})
        if isinstance(node, dict) and isinstance(node.get("ros__parameters"), dict):
            params = node["ros__parameters"]
            if "rbfnn_lr" in params or "arm_ff_scale_roll" in params:
                return params
    for node in data.values():
        if not isinstance(node, dict):
            continue
        params = node.get("ros__parameters", {})
        if isinstance(params, dict) and ("rbfnn_lr" in params or "arm_ff_scale_roll" in params):
            return params
    return {}


def summary_section_rows(summary: dict, scoreboard_row: dict[str, str], csv_path: Path, frame: pd.DataFrame) -> list[list[object]]:
    rows = [
        ["Case", summary.get("case_name", frame["case_name"].iloc[0] if "case_name" in frame.columns else "-")],
        ["Verdict", scoreboard_row.get("verdict") or summary.get("verdict", "-")],
        ["CSV", str(csv_path)],
        ["Total duration", fmt(summary.get("duration_s", frame["t_s"].iloc[-1] - frame["t_s"].iloc[0]), 3, "s")],
        ["Total samples", summary.get("samples", len(frame))],
        ["Analysis phase", summary.get("analysis_phase", "-")],
        ["Analysis window", f"{fmt(summary.get('analysis_start_s'), 3, 's')} - {fmt(summary.get('analysis_end_s'), 3, 's')}"],
        ["Analysis duration", fmt(summary.get("analysis_duration_s"), 3, "s")],
        ["Analysis samples", summary.get("analysis_samples", len(frame))],
        ["External enabled duration", fmt(summary.get("external_enabled_duration_s"), 3, "s")],
        ["External enabled fraction", fmt(summary.get("external_enabled_fraction"), 4)],
    ]
    if scoreboard_row:
        rows.extend(
            [
                ["Score", fmt(scoreboard_row.get("score"), 3)],
                ["Stage", scoreboard_row.get("stage", "-")],
                ["Trial ID", scoreboard_row.get("trial_id", "-")],
                ["Config", scoreboard_row.get("config_path", "-")],
            ]
        )
    return rows


def performance_rows(summary: dict, analysis: pd.DataFrame) -> list[list[object]]:
    alt = summary.get("altitude", {})
    xy = summary.get("xy_drift", {})
    vel = summary.get("velocity", {})
    attitude = summary.get("attitude", {})
    rate = summary.get("rate_tracking", {})

    xy_norm = vector_norm(analysis, ["x_m", "y_m"])
    rows = [
        ["Altitude RMSE", fmt(alt.get("rmse_error_m", series_stats(analysis, "alt_error_m").get("rms")), 4, "m")],
        ["Altitude mean", fmt(alt.get("mean_m", series_stats(analysis, "alt_m").get("mean")), 4, "m")],
        ["Altitude std", fmt(alt.get("std_m", series_stats(analysis, "alt_m").get("std")), 4, "m")],
        ["Altitude min / max", f"{fmt(alt.get('min_m', series_stats(analysis, 'alt_m').get('min')), 4, 'm')} / {fmt(alt.get('max_m', series_stats(analysis, 'alt_m').get('max')), 4, 'm')}"],
        ["XY mean drift", fmt(xy.get("mean_m", float(xy_norm.mean()) if xy_norm is not None else math.nan), 4, "m")],
        ["XY max drift", fmt(xy.get("max_m", float(xy_norm.max()) if xy_norm is not None else math.nan), 4, "m")],
        ["XY final drift", fmt(xy.get("final_m", float(xy_norm.iloc[-1]) if xy_norm is not None else math.nan), 4, "m")],
        ["Velocity norm RMS", fmt(norm_stats(analysis, ["vx_mps", "vy_mps", "vz_mps"]).get("rms"), 4, "m/s")],
        ["Velocity norm max", fmt(norm_stats(analysis, ["vx_mps", "vy_mps", "vz_mps"]).get("max"), 4, "m/s")],
        ["Vertical speed mean abs", fmt(vel.get("vertical_speed_abs_mean_mps"), 4, "m/s")],
        ["Vertical speed max abs", fmt(vel.get("vertical_speed_abs_max_mps"), 4, "m/s")],
        ["Roll RMS / max abs", f"{fmt(attitude.get('roll_rms_deg', series_stats(analysis, 'roll_deg').get('rms')), 4, 'deg')} / {fmt(attitude.get('roll_abs_max_deg', series_stats(analysis, 'roll_deg').get('max_abs')), 4, 'deg')}"],
        ["Pitch RMS / max abs", f"{fmt(attitude.get('pitch_rms_deg', series_stats(analysis, 'pitch_deg').get('rms')), 4, 'deg')} / {fmt(attitude.get('pitch_abs_max_deg', series_stats(analysis, 'pitch_deg').get('max_abs')), 4, 'deg')}"],
        ["Roll/pitch max abs", fmt(attitude.get("roll_pitch_abs_max_deg"), 4, "deg")],
        ["Rate error norm RMS", fmt(rate.get("e_omega_norm_rms_radps", norm_stats(analysis, ["dbg_e_omega_x", "dbg_e_omega_y", "dbg_e_omega_z"]).get("rms")), 4, "rad/s")],
        ["Rate error norm max", fmt(rate.get("e_omega_norm_max_radps", norm_stats(analysis, ["dbg_e_omega_x", "dbg_e_omega_y", "dbg_e_omega_z"]).get("max")), 4, "rad/s")],
    ]
    return rows


def controller_rows(summary: dict, analysis: pd.DataFrame) -> list[list[object]]:
    external = summary.get("external_output", {})
    arm = summary.get("arm_motion", {})
    rows = [
        ["Torque norm RMS", fmt(external.get("torque_norm_rms", norm_stats(analysis, ["torque_x", "torque_y", "torque_z"]).get("rms")), 5)],
        ["Torque norm max", fmt(external.get("torque_norm_max", norm_stats(analysis, ["torque_x", "torque_y", "torque_z"]).get("max")), 5)],
        ["Controller tau norm RMS", fmt(norm_stats(analysis, ["dbg_tau_x", "dbg_tau_y", "dbg_tau_z"]).get("rms"), 5, "Nm")],
        ["Controller tau norm max", fmt(norm_stats(analysis, ["dbg_tau_x", "dbg_tau_y", "dbg_tau_z"]).get("max"), 5, "Nm")],
        ["RBFNN n_hat norm RMS", fmt(external.get("n_hat_norm_rms", norm_stats(analysis, ["dbg_n_hat_x", "dbg_n_hat_y", "dbg_n_hat_z"]).get("rms")), 7)],
        ["RBFNN n_hat norm max", fmt(external.get("n_hat_norm_max", norm_stats(analysis, ["dbg_n_hat_x", "dbg_n_hat_y", "dbg_n_hat_z"]).get("max")), 7)],
        ["Arm FF norm RMS", fmt(external.get("tau_arm_ff_norm_rms_nm", norm_stats(analysis, ["dbg_tau_arm_ff_x", "dbg_tau_arm_ff_y", "dbg_tau_arm_ff_z"]).get("rms")), 5, "Nm")],
        ["Arm FF norm max", fmt(external.get("tau_arm_ff_norm_max_nm", norm_stats(analysis, ["dbg_tau_arm_ff_x", "dbg_tau_arm_ff_y", "dbg_tau_arm_ff_z"]).get("max")), 5, "Nm")],
        ["Virtual disturbance norm RMS", fmt(external.get("tau_arm_disturbance_norm_rms_nm", norm_stats(analysis, ["dbg_tau_arm_disturbance_x", "dbg_tau_arm_disturbance_y", "dbg_tau_arm_disturbance_z"]).get("rms")), 5, "Nm")],
        ["Virtual disturbance norm max", fmt(external.get("tau_arm_disturbance_norm_max_nm", norm_stats(analysis, ["dbg_tau_arm_disturbance_x", "dbg_tau_arm_disturbance_y", "dbg_tau_arm_disturbance_z"]).get("max")), 5, "Nm")],
        ["Residual torque norm RMS", fmt(external.get("tau_residual_norm_rms_nm"), 5, "Nm")],
        ["Residual torque norm max", fmt(external.get("tau_residual_norm_max_nm"), 5, "Nm")],
        ["FF/disturbance dot mean", fmt(external.get("ff_disturbance_dot_mean"), 8)],
        ["Arm FF enabled / fresh", f"{fmt(external.get('arm_ff_enabled_fraction'), 4)} / {fmt(external.get('arm_ff_fresh_fraction'), 4)}"],
        ["Virtual disturbance enabled", fmt(external.get("arm_virtual_disturbance_enabled_fraction"), 4)],
        ["Joint command seen", str(arm.get("arm_command_seen", "-"))],
        ["Joint motion detected", str(arm.get("arm_motion_detected", "-"))],
        ["Joint command norm RMS", fmt(arm.get("joint_cmd_norm_rms_rad"), 5, "rad")],
        ["Joint command norm max", fmt(arm.get("joint_cmd_norm_max_rad"), 5, "rad")],
        ["Joint actual norm RMS", fmt(arm.get("joint_pos_norm_rms_rad"), 5, "rad")],
        ["Joint actual norm max", fmt(arm.get("joint_pos_norm_max_rad"), 5, "rad")],
        ["Joint command max span", fmt(arm.get("joint_cmd_span_max_rad"), 5, "rad")],
        ["Joint actual max span", fmt(arm.get("joint_pos_span_max_rad"), 5, "rad")],
    ]
    return rows


def rbfnn_detail_rows(summary: dict, analysis: pd.DataFrame, params: dict) -> list[list[object]]:
    external = summary.get("external_output", {})
    rbfnn_enabled = params.get("rbfnn_output_enable", params.get("rbfnn_enable", "-"))
    rows = [
        ["RBFNN enable/output", str(rbfnn_enabled)],
        ["RBFNN learning enable", str(params.get("rbfnn_learning_enable", "-"))],
        ["RBFNN output gain", fmt(params.get("rbfnn_output_gain"), 6)],
        ["RBFNN learning rate", fmt(params.get("rbfnn_lr"), 6)],
        ["RBFNN e-modification", fmt(params.get("rbfnn_e_modification"), 6)],
        ["RBFNN Gaussian width", fmt(params.get("rbfnn_gaussian_width"), 6)],
        ["RBFNN neurons", params.get("rbfnn_num_neurons", "-")],
        ["n_hat norm RMS", fmt(external.get("n_hat_norm_rms", norm_stats(analysis, ["dbg_n_hat_x", "dbg_n_hat_y", "dbg_n_hat_z"]).get("rms")), 8)],
        ["n_hat norm max", fmt(external.get("n_hat_norm_max", norm_stats(analysis, ["dbg_n_hat_x", "dbg_n_hat_y", "dbg_n_hat_z"]).get("max")), 8)],
    ]

    for axis, n_col, e_col in [
        ("x / roll", "dbg_n_hat_x", "dbg_e_omega_x"),
        ("y / pitch", "dbg_n_hat_y", "dbg_e_omega_y"),
        ("z / yaw", "dbg_n_hat_z", "dbg_e_omega_z"),
    ]:
        n_stats = series_stats(analysis, n_col)
        e_stats = series_stats(analysis, e_col)
        if not n_stats:
            continue
        rows.extend(
            [
                [f"{axis} n_hat RMS", fmt(n_stats["rms"], 8)],
                [f"{axis} n_hat max abs", fmt(n_stats["max_abs"], 8)],
                [f"{axis} n_hat mean/final", f"{fmt(n_stats['mean'], 8)} / {fmt(n_stats['final'], 8)}"],
                [f"{axis} rate-error RMS", fmt(e_stats.get("rms"), 6, "rad/s")],
                [f"{axis} |n_hat| / e_omega RMS", fmt(safe_ratio(n_stats["rms"], e_stats.get("rms", math.nan)), 6)],
            ]
        )
    return rows


def feedforward_detail_rows(summary: dict, analysis: pd.DataFrame, params: dict) -> list[list[object]]:
    external = summary.get("external_output", {})
    rows = [
        ["Arm FF enable", str(params.get("arm_ff_enable", "-"))],
        ["Arm FF scale roll", fmt(params.get("arm_ff_scale_roll"), 6)],
        ["Arm FF scale pitch", fmt(params.get("arm_ff_scale_pitch"), 6)],
        ["Arm FF scale yaw", fmt(params.get("arm_ff_scale_yaw"), 6)],
        ["Arm FF max roll", fmt(params.get("arm_ff_max_roll_nm"), 6, "Nm")],
        ["Arm FF max pitch", fmt(params.get("arm_ff_max_pitch_nm"), 6, "Nm")],
        ["Arm FF max yaw", fmt(params.get("arm_ff_max_yaw_nm"), 6, "Nm")],
        ["Virtual disturbance enable", str(params.get("arm_virtual_disturbance_enable", "-"))],
        ["Virtual scale roll", fmt(params.get("arm_virtual_disturbance_scale_roll"), 6)],
        ["Virtual scale pitch", fmt(params.get("arm_virtual_disturbance_scale_pitch"), 6)],
        ["Virtual scale yaw", fmt(params.get("arm_virtual_disturbance_scale_yaw"), 6)],
        ["Arm FF norm RMS", fmt(external.get("tau_arm_ff_norm_rms_nm", norm_stats(analysis, ["dbg_tau_arm_ff_x", "dbg_tau_arm_ff_y", "dbg_tau_arm_ff_z"]).get("rms")), 6, "Nm")],
        ["Arm FF norm max", fmt(external.get("tau_arm_ff_norm_max_nm", norm_stats(analysis, ["dbg_tau_arm_ff_x", "dbg_tau_arm_ff_y", "dbg_tau_arm_ff_z"]).get("max")), 6, "Nm")],
        ["Virtual disturbance norm RMS", fmt(external.get("tau_arm_disturbance_norm_rms_nm", norm_stats(analysis, ["dbg_tau_arm_disturbance_x", "dbg_tau_arm_disturbance_y", "dbg_tau_arm_disturbance_z"]).get("rms")), 6, "Nm")],
        ["Virtual disturbance norm max", fmt(external.get("tau_arm_disturbance_norm_max_nm", norm_stats(analysis, ["dbg_tau_arm_disturbance_x", "dbg_tau_arm_disturbance_y", "dbg_tau_arm_disturbance_z"]).get("max")), 6, "Nm")],
        ["Residual norm RMS", fmt(external.get("tau_residual_norm_rms_nm"), 6, "Nm")],
        ["Residual norm max", fmt(external.get("tau_residual_norm_max_nm"), 6, "Nm")],
        ["FF/disturbance dot mean", fmt(external.get("ff_disturbance_dot_mean"), 8)],
        ["FF enabled/fresh fraction", f"{fmt(external.get('arm_ff_enabled_fraction'), 4)} / {fmt(external.get('arm_ff_fresh_fraction'), 4)}"],
        ["Virtual disturbance enabled fraction", fmt(external.get("arm_virtual_disturbance_enabled_fraction"), 4)],
    ]

    for axis, ff_col, dist_col in [
        ("x / roll", "dbg_tau_arm_ff_x", "dbg_tau_arm_disturbance_x"),
        ("y / pitch", "dbg_tau_arm_ff_y", "dbg_tau_arm_disturbance_y"),
        ("z / yaw", "dbg_tau_arm_ff_z", "dbg_tau_arm_disturbance_z"),
    ]:
        if ff_col not in analysis.columns or dist_col not in analysis.columns:
            continue
        ff_stats = series_stats(analysis, ff_col)
        dist_stats = series_stats(analysis, dist_col)
        residual = analysis[dist_col].astype(float) - analysis[ff_col].astype(float)
        residual_rms = rms(residual)
        residual_max = float(residual.abs().max())
        rows.extend(
            [
                [f"{axis} FF RMS / max abs", f"{fmt(ff_stats['rms'], 6, 'Nm')} / {fmt(ff_stats['max_abs'], 6, 'Nm')}"],
                [f"{axis} disturbance RMS / max abs", f"{fmt(dist_stats['rms'], 6, 'Nm')} / {fmt(dist_stats['max_abs'], 6, 'Nm')}"],
                [f"{axis} residual RMS / max abs", f"{fmt(residual_rms, 6, 'Nm')} / {fmt(residual_max, 6, 'Nm')}"],
                [f"{axis} FF/disturbance RMS ratio", fmt(safe_ratio(ff_stats["rms"], dist_stats["rms"]), 6)],
                [f"{axis} residual/disturbance RMS ratio", fmt(safe_ratio(residual_rms, dist_stats["rms"]), 6)],
                [f"{axis} FF-disturbance correlation", fmt(correlation(analysis[ff_col], analysis[dist_col]), 6)],
                [f"{axis} FF*disturbance mean", fmt(float((analysis[ff_col] * analysis[dist_col]).mean()), 8)],
            ]
        )
    return rows


def axis_rows(analysis: pd.DataFrame) -> list[list[object]]:
    groups = [
        ("Position", [("x_m", "x", "m"), ("y_m", "y", "m"), ("alt_m", "alt", "m"), ("alt_error_m", "alt err", "m")]),
        ("Velocity", [("vx_mps", "vx", "m/s"), ("vy_mps", "vy", "m/s"), ("vz_mps", "vz", "m/s")]),
        ("Attitude", [("roll_deg", "roll", "deg"), ("pitch_deg", "pitch", "deg"), ("yaw_deg", "yaw", "deg")]),
        ("Body rate", [("p_radps", "p", "rad/s"), ("q_radps", "q", "rad/s"), ("r_radps", "r", "rad/s")]),
        ("Rate setpoint", [("roll_rate_sp", "p sp", "rad/s"), ("pitch_rate_sp", "q sp", "rad/s"), ("yaw_rate_sp", "r sp", "rad/s")]),
        ("Rate error", [("dbg_e_omega_x", "e_p", "rad/s"), ("dbg_e_omega_y", "e_q", "rad/s"), ("dbg_e_omega_z", "e_r", "rad/s")]),
        ("RBFNN", [("dbg_n_hat_x", "n_hat_x", ""), ("dbg_n_hat_y", "n_hat_y", ""), ("dbg_n_hat_z", "n_hat_z", "")]),
        ("Arm FF", [("dbg_tau_arm_ff_x", "ff_x", "Nm"), ("dbg_tau_arm_ff_y", "ff_y", "Nm"), ("dbg_tau_arm_ff_z", "ff_z", "Nm")]),
        ("Arm disturbance", [("dbg_tau_arm_disturbance_x", "dist_x", "Nm"), ("dbg_tau_arm_disturbance_y", "dist_y", "Nm"), ("dbg_tau_arm_disturbance_z", "dist_z", "Nm")]),
    ]
    rows: list[list[object]] = []
    for group, items in groups:
        for col, label, unit in items:
            stats = series_stats(analysis, col)
            if not stats:
                continue
            rows.append(
                [
                    group,
                    label,
                    unit,
                    fmt(stats["mean"], 5),
                    fmt(stats["std"], 5),
                    fmt(stats["rms"], 5),
                    fmt(stats["min"], 5),
                    fmt(stats["max"], 5),
                    fmt(stats["max_abs"], 5),
                    fmt(stats["final"], 5),
                ]
            )
    return rows


def joint_rows(analysis: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for idx in range(1, 7):
        pos = f"joint_pos_{idx}"
        cmd = f"joint_cmd_{idx}"
        vel = f"joint_vel_{idx}"
        if pos not in analysis.columns and cmd not in analysis.columns:
            continue
        pos_stats = series_stats(analysis, pos)
        cmd_stats = series_stats(analysis, cmd)
        vel_stats = series_stats(analysis, vel)
        err_rms = math.nan
        err_max = math.nan
        if pos in analysis.columns and cmd in analysis.columns:
            err = analysis[pos].astype(float) - analysis[cmd].astype(float)
            err_rms = rms(err)
            err_max = float(err.abs().max())
        rows.append(
            [
                f"J{idx}",
                fmt(cmd_stats.get("min"), 5, "rad"),
                fmt(cmd_stats.get("max"), 5, "rad"),
                fmt(cmd_stats.get("max", math.nan) - cmd_stats.get("min", math.nan), 5, "rad"),
                fmt(pos_stats.get("min"), 5, "rad"),
                fmt(pos_stats.get("max"), 5, "rad"),
                fmt(pos_stats.get("max", math.nan) - pos_stats.get("min", math.nan), 5, "rad"),
                fmt(err_rms, 5, "rad"),
                fmt(err_max, 5, "rad"),
                fmt(vel_stats.get("rms"), 5, "rad/s"),
                fmt(vel_stats.get("max_abs"), 5, "rad/s"),
            ]
        )
    return rows


def health_rows(analysis: pd.DataFrame, summary: dict) -> list[list[object]]:
    rows: list[list[object]] = []
    for col in [
        "odom_age_s",
        "rates_sp_age_s",
        "torque_age_s",
        "thrust_age_s",
        "debug_age_s",
        "joint_age_s",
        "arm_cmd_age_s",
    ]:
        stats = series_stats(analysis, col)
        if stats:
            rows.append([col, fmt(stats["mean"], 5, "s"), fmt(stats["max"], 5, "s"), fmt(stats["final"], 5, "s")])
    flags = summary.get("failure_flags", {})
    for key, value in flags.items():
        rows.append([f"flag:{key}", str(value), "-", "-"])
    return rows


def build_report(args: argparse.Namespace) -> tuple[Path, Path]:
    csv_path = args.csv.resolve()
    summary_path = args.summary.resolve() if args.summary else csv_path.with_name("summary.json")
    scoreboard_path = args.scoreboard.resolve() if args.scoreboard else find_scoreboard(csv_path)

    df = pd.read_csv(csv_path)
    summary = load_summary(summary_path)
    scoreboard_row = read_scoreboard_row(scoreboard_path, summary_path)
    params = load_controller_params(scoreboard_row)

    if "t_s" not in df.columns:
        raise ValueError(f"{csv_path} does not contain t_s")

    if args.use_all_samples:
        analysis = df.copy()
    else:
        start = finite(summary.get("analysis_start_s"), float(df["t_s"].iloc[0]))
        end = finite(summary.get("analysis_end_s"), float(df["t_s"].iloc[-1]))
        analysis = df[(df["t_s"] >= start) & (df["t_s"] <= end)].copy()
        if analysis.empty:
            analysis = df.copy()

    out = args.out.resolve() if args.out else csv_path.parent / "review_tables"
    plots = out / "plots"
    tables = out / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for folder in (plots, tables):
        for old_file in folder.iterdir():
            if old_file.is_file() and old_file.suffix.lower() in {".png", ".csv"}:
                old_file.unlink()

    table_defs = [
        ("01_overview", "Overview", ["Field", "Value"], summary_section_rows(summary, scoreboard_row, csv_path, df)),
        ("02_flight_performance", "Flight Performance", ["Metric", "Value"], performance_rows(summary, analysis)),
        ("03_controller_arm_rbfnn", "Controller / Arm / RBFNN", ["Metric", "Value"], controller_rows(summary, analysis)),
        ("04_rbfnn_detail", "RBFNN Detail", ["Metric", "Value"], rbfnn_detail_rows(summary, analysis, params)),
        ("05_feedforward_detail", "Feedforward / Disturbance Detail", ["Metric", "Value"], feedforward_detail_rows(summary, analysis, params)),
        (
            "06_axis_statistics",
            "Axis Statistics",
            ["Group", "Signal", "Unit", "Mean", "Std", "RMS", "Min", "Max", "Max abs", "Final"],
            axis_rows(analysis),
        ),
        (
            "07_joint_statistics",
            "Joint Statistics",
            ["Joint", "Cmd min", "Cmd max", "Cmd span", "Actual min", "Actual max", "Actual span", "Err RMS", "Err max", "Vel RMS", "Vel max abs"],
            joint_rows(analysis),
        ),
        ("08_data_health", "Data Freshness / Health", ["Signal", "Mean", "Max", "Final"], health_rows(analysis, summary)),
    ]

    md_lines = [
        "# UAM Detailed Review Tables",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Summary JSON: `{summary_path if summary_path.exists() else '-'}`",
        f"- Scoreboard: `{scoreboard_path if scoreboard_path else '-'}`",
        f"- Output: `{out}`",
        "",
    ]
    for slug, title, headers, rows in table_defs:
        write_csv(tables / f"{slug}.csv", headers, rows)
        draw_table_png(tables / f"{slug}.png", title, headers, rows, font_size=7 if len(headers) > 8 else 8)
        md_lines.extend([f"## {title}", "", markdown_table(headers, rows), ""])

    plot_series(plots, "position_xy_alt", analysis, ["x_m", "y_m", "alt_m"], "m")
    plot_series(plots, "velocity", analysis, ["vx_mps", "vy_mps", "vz_mps"], "m/s")
    plot_series(plots, "attitude", analysis, ["roll_deg", "pitch_deg", "yaw_deg"], "deg")
    plot_series(plots, "rate_actual_vs_setpoint", analysis, ["p_radps", "q_radps", "r_radps", "roll_rate_sp", "pitch_rate_sp", "yaw_rate_sp"], "rad/s")
    plot_series(plots, "rate_error", analysis, ["dbg_e_omega_x", "dbg_e_omega_y", "dbg_e_omega_z"], "rad/s")
    plot_series(plots, "torque_and_thrust", analysis, ["torque_x", "torque_y", "torque_z", "thrust_z"], "normalized")
    plot_series(plots, "arm_joints_actual", analysis, [f"joint_pos_{idx}" for idx in range(1, 7)], "rad")
    plot_series(plots, "arm_joints_command", analysis, [f"joint_cmd_{idx}" for idx in range(1, 7)], "rad")
    plot_series(plots, "arm_feedforward", analysis, ["dbg_tau_arm_ff_x", "dbg_tau_arm_ff_y", "dbg_tau_arm_ff_z"], "Nm")
    plot_series(plots, "arm_virtual_disturbance", analysis, ["dbg_tau_arm_disturbance_x", "dbg_tau_arm_disturbance_y", "dbg_tau_arm_disturbance_z"], "Nm")
    plot_series(plots, "rbfnn_output", analysis, ["dbg_n_hat_x", "dbg_n_hat_y", "dbg_n_hat_z"], "controller units")
    plot_ff_disturbance_residual(plots, analysis)
    plot_rbfnn_vs_rate_error(plots, analysis)
    plot_series(plots, "data_age", analysis, ["odom_age_s", "rates_sp_age_s", "debug_age_s", "joint_age_s", "arm_cmd_age_s"], "s")

    report_md = out / "detailed_review_report.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    return out, report_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to flight_timeseries.csv")
    parser.add_argument("--summary", type=Path, default=None, help="Path to summary.json. Defaults to CSV sibling.")
    parser.add_argument("--scoreboard", type=Path, default=None, help="Path to scoreboard.csv. Auto-detected by default.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to CSV sibling/review_tables.")
    parser.add_argument("--use-all-samples", action="store_true", help="Use full CSV instead of the settled analysis window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out, report = build_report(args)
    print(f"Review output: {out}")
    print(f"Markdown report: {report}")
    print(f"Table PNGs: {out / 'tables'}")
    print(f"Plots: {out / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
