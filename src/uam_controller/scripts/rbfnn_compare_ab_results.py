#!/usr/bin/env python3
"""Compare A/B/C/D RBFNN experiment summaries.

Expected cases:
  A: RBFNN output off, arm feedforward on
  B: RBFNN output on,  arm feedforward on
  C: RBFNN output on,  arm feedforward off
  D: RBFNN output off, arm feedforward off
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CASES = (
    ("A_no_rbfnn_ff", "case_a_no_rbfnn_ff"),
    ("B_rbfnn_ff", "case_b_rbfnn_ff"),
    ("C_rbfnn_no_ff", "case_c_rbfnn_no_ff"),
    ("D_no_rbfnn_no_ff", "case_d_no_rbfnn_no_ff"),
)


def nested(data: dict[str, Any], *keys: str, default: Any = math.nan) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def latest_summary(root: Path, case_token: str) -> Path | None:
    best_results = [
        p for p in root.rglob("best_result.json")
        if case_token in str(p) or case_token in p.parent.name
    ]
    summary_paths: list[Path] = []
    for result_path in best_results:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary_path = Path(str(result.get("summary_json", "")))
        if summary_path.exists():
            summary_paths.append(summary_path)
    if summary_paths:
        return max(summary_paths, key=lambda p: p.stat().st_mtime)

    candidates = [
        p for p in root.rglob("summary.json")
        if case_token in str(p) or case_token in p.parent.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def row_from_summary(case_label: str, summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    roll_rms = finite_float(nested(summary, "attitude", "roll_rms_deg"))
    pitch_rms = finite_float(nested(summary, "attitude", "pitch_rms_deg"))
    angle_rms = max(v for v in (roll_rms, pitch_rms) if math.isfinite(v)) \
        if any(math.isfinite(v) for v in (roll_rms, pitch_rms)) else math.nan

    return {
        "case": case_label,
        "case_name": summary.get("case_name", ""),
        "verdict": summary.get("verdict", ""),
        "analysis_duration_s": finite_float(summary.get("analysis_duration_s")),
        "external_enabled_duration_s": finite_float(summary.get("external_enabled_duration_s")),
        "xy_mean_m": finite_float(nested(summary, "xy_drift", "mean_m")),
        "xy_max_m": finite_float(nested(summary, "xy_drift", "max_m")),
        "angle_rms_deg": angle_rms,
        "angle_max_deg": finite_float(nested(summary, "attitude", "roll_pitch_abs_max_deg")),
        "rate_err_rms_radps": finite_float(nested(summary, "rate_tracking", "e_omega_norm_rms_radps")),
        "rate_err_max_radps": finite_float(nested(summary, "rate_tracking", "e_omega_norm_max_radps")),
        "n_hat_norm_rms": finite_float(nested(summary, "external_output", "n_hat_norm_rms")),
        "n_hat_norm_max": finite_float(nested(summary, "external_output", "n_hat_norm_max")),
        "tau_arm_ff_norm_rms_nm": finite_float(nested(summary, "external_output", "tau_arm_ff_norm_rms_nm")),
        "tau_arm_ff_norm_max_nm": finite_float(nested(summary, "external_output", "tau_arm_ff_norm_max_nm")),
        "tau_arm_disturbance_norm_rms_nm": finite_float(
            nested(summary, "external_output", "tau_arm_disturbance_norm_rms_nm")
        ),
        "tau_arm_disturbance_norm_max_nm": finite_float(
            nested(summary, "external_output", "tau_arm_disturbance_norm_max_nm")
        ),
        "tau_residual_norm_rms_nm": finite_float(
            nested(summary, "external_output", "tau_residual_norm_rms_nm")
        ),
        "tau_residual_norm_max_nm": finite_float(
            nested(summary, "external_output", "tau_residual_norm_max_nm")
        ),
        "ff_disturbance_dot_mean": finite_float(
            nested(summary, "external_output", "ff_disturbance_dot_mean")
        ),
        "arm_motion_detected": nested(summary, "arm_motion", "arm_motion_detected", default=False),
        "summary_json": str(summary_path),
    }


def add_delta_columns(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    baseline = rows[0]
    for row in rows:
        for key in ("rate_err_rms_radps", "angle_rms_deg", "xy_mean_m", "tau_residual_norm_rms_nm"):
            base_val = finite_float(baseline.get(key))
            val = finite_float(row.get(key))
            row[f"delta_{key}_vs_A"] = val - base_val if math.isfinite(base_val) and math.isfinite(val) else math.nan


def verdict_text(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 3:
        return "Not enough cases for A/B/C conclusion."

    a, b, c = rows[0], rows[1], rows[2]
    rate_improvement = finite_float(a["rate_err_rms_radps"]) - finite_float(b["rate_err_rms_radps"])
    angle_delta = finite_float(b["angle_rms_deg"]) - finite_float(a["angle_rms_deg"])
    xy_delta = finite_float(b["xy_mean_m"]) - finite_float(a["xy_mean_m"])
    residual_delta = finite_float(b["tau_residual_norm_rms_nm"]) - finite_float(a["tau_residual_norm_rms_nm"])
    n_hat = finite_float(b["n_hat_norm_rms"])
    ff_dot_a = finite_float(a.get("ff_disturbance_dot_mean"))
    ff_dot_b = finite_float(b.get("ff_disturbance_dot_mean"))
    ff_max_a = finite_float(a.get("tau_arm_ff_norm_max_nm"))
    ff_max_b = finite_float(b.get("tau_arm_ff_norm_max_nm"))

    if not math.isfinite(ff_max_a) or ff_max_a <= 1.0e-6:
        return "Case A requested arm_ff, but tau_arm_ff max is zero: restore nonzero arm_ff_scale_* before judging RBFNN."
    if not math.isfinite(ff_max_b) or ff_max_b <= 1.0e-6:
        return "Case B requested arm_ff, but tau_arm_ff max is zero: restore nonzero arm_ff_scale_* before judging RBFNN."

    if math.isfinite(ff_dot_a) and ff_dot_a < 0.0:
        return "Case A has negative arm_ff/disturbance dot product: arm_ff sign is likely wrong before RBFNN is evaluated."
    if math.isfinite(ff_dot_b) and ff_dot_b < 0.0:
        return "Case B has negative arm_ff/disturbance dot product: RBFNN is learning on a corrupted residual."

    if not all(math.isfinite(v) for v in (rate_improvement, angle_delta, xy_delta, residual_delta, n_hat)):
        return "Some required metrics are missing; inspect the summary paths manually."

    if rate_improvement > 0.0 and angle_delta <= 0.05 and xy_delta <= 0.02 and residual_delta <= 0.0 and n_hat > 1.0e-4:
        base = "B improves rate tracking against A without meaningful attitude/XY penalty: RBFNN contribution is measurable."
        if len(rows) >= 4:
            return base + " " + no_ff_verdict_text(c, rows[3])
        return base

    if abs(rate_improvement) <= 0.005 and n_hat <= 1.0e-4:
        return "B is nearly identical to A and n_hat is very small: RBFNN contribution is still weak."

    if rate_improvement < 0.0 or angle_delta > 0.05 or xy_delta > 0.02 or residual_delta > 0.0:
        base = "B is worse than A on rate, attitude, XY, or residual torque: fix arm_ff sign/scale first, then retune RBFNN."
        if len(rows) >= 4:
            return base + " " + no_ff_verdict_text(c, rows[3])
        return base

    c_rate = finite_float(c["rate_err_rms_radps"])
    if math.isfinite(c_rate) and c_rate < finite_float(b["rate_err_rms_radps"]):
        base = "C outperforms B: arm_ff may be overcompensating or using the wrong scale/sign."
        if len(rows) >= 4:
            return base + " " + no_ff_verdict_text(c, rows[3])
        return base

    return "RBFNN has some activity, but the improvement is not yet strong enough for a clean claim."


def no_ff_verdict_text(c: dict[str, Any], d: dict[str, Any]) -> str:
    c_rate = finite_float(c.get("rate_err_rms_radps"))
    d_rate = finite_float(d.get("rate_err_rms_radps"))
    c_xy = finite_float(c.get("xy_mean_m"))
    d_xy = finite_float(d.get("xy_mean_m"))
    c_angle = finite_float(c.get("angle_rms_deg"))
    d_angle = finite_float(d.get("angle_rms_deg"))
    c_n_hat = finite_float(c.get("n_hat_norm_rms"))
    if not all(math.isfinite(v) for v in (c_rate, d_rate, c_xy, d_xy, c_angle, d_angle, c_n_hat)):
        return "C/D no-FF metrics are incomplete."
    if d_rate > c_rate and c_xy <= d_xy + 0.02 and c_angle <= d_angle + 0.05 and c_n_hat > 1.0e-4:
        return "C beats D in the no-FF pair, so RBFNN contribution is measurable without arm_ff."
    if c_rate >= d_rate:
        return "C does not beat D in the no-FF pair, so RBFNN-only contribution is weak."
    return "C improves rate against D but adds XY/attitude penalty in the no-FF pair."


def write_markdown(path: Path, rows: list[dict[str, Any]], conclusion: str) -> None:
    headers = [
        "case",
        "rate_err_rms_radps",
        "angle_rms_deg",
        "xy_mean_m",
        "n_hat_norm_rms",
        "tau_arm_disturbance_norm_max_nm",
        "tau_arm_ff_norm_max_nm",
        "tau_residual_norm_rms_nm",
        "ff_disturbance_dot_mean",
        "delta_rate_err_rms_radps_vs_A",
        "delta_tau_residual_norm_rms_nm_vs_A",
    ]

    case_title = "/".join(str(row.get("case", "")).split("_", 1)[0] for row in rows)
    lines = [f"# RBFNN {case_title} Comparison", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}" if math.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Conclusion", "", conclusion, "", "## Sources", ""])
    for row in rows:
        lines.append(f"- {row['case']}: `{row['summary_json']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Experiment output root containing case summaries.")
    parser.add_argument("--case-a", default=DEFAULT_CASES[0][1], help="Token used to locate Case A summary.")
    parser.add_argument("--case-b", default=DEFAULT_CASES[1][1], help="Token used to locate Case B summary.")
    parser.add_argument("--case-c", default=DEFAULT_CASES[2][1], help="Token used to locate Case C summary.")
    parser.add_argument("--case-d", default=DEFAULT_CASES[3][1], help="Token used to locate optional Case D summary.")
    parser.add_argument("--include-d", action="store_true", help="Include Case D if present.")
    parser.add_argument("--output-prefix", default="rbfnn_ab_comparison", help="Output file prefix under --root.")
    args = parser.parse_args()

    case_specs = (
        (DEFAULT_CASES[0][0], args.case_a),
        (DEFAULT_CASES[1][0], args.case_b),
        (DEFAULT_CASES[2][0], args.case_c),
    )
    if args.include_d:
        case_specs = case_specs + ((DEFAULT_CASES[3][0], args.case_d),)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for label, token in case_specs:
        path = latest_summary(args.root, token)
        if path is None:
            missing.append(f"{label} token='{token}'")
            continue
        rows.append(row_from_summary(label, path))

    if missing:
        print("Missing summaries:")
        for item in missing:
            print(f"  - {item}")
        return 2

    add_delta_columns(rows)

    csv_path = args.root / f"{args.output_prefix}.csv"
    md_path = args.root / f"{args.output_prefix}.md"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    conclusion = verdict_text(rows)
    write_markdown(md_path, rows, conclusion)

    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Conclusion: {conclusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
