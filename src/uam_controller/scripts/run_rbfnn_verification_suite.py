#!/usr/bin/env python3
"""Run fixed-parameter RBFNN verification cases automatically."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


DEFAULT_GZ_ROOT = Path("/home/wicom/PX4-Autopilot/Tools/simulation/gz")
DEFAULT_RESULTS_ROOT = Path("/home/wicom/uam_results")

CASES = {
    "A": {
        "label": "case_a_no_rbfnn_ff",
        "stage": "bs_arm_rne_static",
        "description": "RBFNN output off, arm_ff on, virtual disturbance on",
    },
    "B": {
        "label": "case_b_rbfnn_ff",
        "stage": "rbfnn_residual_arm",
        "description": "RBFNN output on, arm_ff on, virtual disturbance on",
    },
    "C": {
        "label": "case_c_rbfnn_no_ff",
        "stage": "rbfnn_arm_no_ff",
        "description": "RBFNN output on, arm_ff off, virtual disturbance on",
    },
}

MEDIAN_METRICS = (
    "score",
    "alt_rmse_m",
    "xy_mean_m",
    "xy_max_m",
    "angle_rms_deg",
    "angle_max_deg",
    "rate_err_rms_radps",
    "rate_err_max_radps",
    "arm_cmd_span_rad",
    "arm_actual_span_rad",
    "arm_span_ratio",
    "external_enabled_fraction",
    "external_enabled_duration_s",
    "analysis_duration_s",
    "n_hat_norm_rms",
    "n_hat_norm_max",
    "arm_ff_max_nm",
    "arm_disturbance_rms_nm",
    "arm_disturbance_max_nm",
    "tau_residual_rms_nm",
    "tau_residual_max_nm",
    "ff_disturbance_dot_mean",
)


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_best_config(root: Path = DEFAULT_RESULTS_ROOT) -> Path | None:
    candidates = [Path(p) for p in glob.glob(str(root / "**" / "final_best_uam_controller_params.yaml"), recursive=True)]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def amp_tag(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def median_metric(rows: list[dict[str, str]], key: str) -> float:
    values = [finite_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return statistics.median(values) if values else math.nan


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> int:
    print("+ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=False).returncode


def build_autotune_cmd(
    args: argparse.Namespace,
    *,
    config: Path,
    output_dir: Path,
    case_key: str,
    pattern: str,
    amplitude: float,
    duration: float,
    rate_hz: int,
) -> list[str]:
    case = CASES[case_key]
    case_name = f"{case['label']}_{pattern}_amp{amp_tag(amplitude)}"
    cmd = [
        sys.executable,
        str(args.gz_root / "tools" / "uam_rerun_autotune.py"),
        "--fixed-config",
        "--stage",
        case["stage"],
        "--trials",
        str(args.repeats),
        "--base-config",
        str(config),
        "--output-dir",
        str(output_dir / case["label"]),
        "--case-name",
        case_name,
        "--arm-pattern",
        pattern,
        "--arm-amplitude",
        f"{amplitude:.6g}",
        "--arm-duration-s",
        f"{duration:.6g}",
        "--arm-rate-hz",
        str(rate_hz),
        "--arm-state-source",
        args.arm_state_source,
        "--px4-wait-s",
        f"{args.px4_wait_s:.6g}",
        "--ros-wait-s",
        f"{args.ros_wait_s:.6g}",
        "--post-ros-settle-s",
        f"{args.post_ros_settle_s:.6g}",
        "--arm-wait-s",
        f"{args.arm_wait_s:.6g}",
        "--takeoff-wait-s",
        f"{args.takeoff_wait_s:.6g}",
        "--handoff-timeout-s",
        f"{args.handoff_timeout_s:.6g}",
        "--handoff-settle-s",
        f"{args.handoff_settle_s:.6g}",
        "--flight-time-s",
        f"{args.flight_time_s:.6g}",
        "--min-external-duration-s",
        f"{args.min_external_duration_s:.6g}",
        "--min-external-fraction",
        f"{args.min_external_fraction:.6g}",
    ]
    if args.use_gazebo_arm_visual:
        cmd.append("--use-gazebo-arm-visual")
    return cmd


def write_suite_summary(path: Path, commands: list[list[str]], args: argparse.Namespace, config: Path) -> None:
    lines = [
        "# RBFNN Verification Suite",
        "",
        f"- Config: `{config}`",
        f"- Output root: `{args.output_root}`",
        f"- Cases: `{','.join(args.cases)}`",
        f"- Repeats per case: `{args.repeats}`",
        f"- Primary trajectory: `{args.pattern}`, amp={args.amplitude}, duration={args.duration_s}s, rate={args.rate_hz}Hz",
    ]
    if args.include_strong:
        lines.append(
            f"- Strong trajectory: `{args.strong_pattern}`, amp={args.strong_amplitude}, "
            f"duration={args.strong_duration_s}s, rate={args.strong_rate_hz}Hz"
        )
    lines.extend(["", "## Commands", ""])
    for cmd in commands:
        lines.append("```bash")
        lines.append(" ".join(cmd))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_scoreboard(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_verdicts(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict", "")).strip() or "UNKNOWN"
        counts[verdict] = counts.get(verdict, 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def median_rows_for_trajectory(traj_dir: Path, cases: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_key in cases:
        case = CASES[case_key]
        scoreboard = traj_dir / case["label"] / "scoreboard.csv"
        trial_rows = read_scoreboard(scoreboard)
        if not trial_rows:
            continue
        row: dict[str, object] = {
            "case": case_key,
            "case_name": case["label"],
            "stage": case["stage"],
            "repeats": len(trial_rows),
            "verdict_counts": count_verdicts(trial_rows),
            "scoreboard": str(scoreboard),
        }
        for key in MEDIAN_METRICS:
            row[f"median_{key}"] = median_metric(trial_rows, key)
        rows.append(row)
    return rows


def comparison_conclusion(rows: list[dict[str, object]]) -> str:
    by_case = {str(row["case"]): row for row in rows}
    if not all(key in by_case for key in ("A", "B", "C")):
        return "Missing one or more A/B/C cases; median conclusion was not computed."

    a = by_case["A"]
    b = by_case["B"]
    c = by_case["C"]
    rate_improvement = finite_float(a.get("median_rate_err_rms_radps")) - finite_float(b.get("median_rate_err_rms_radps"))
    xy_delta = finite_float(b.get("median_xy_mean_m")) - finite_float(a.get("median_xy_mean_m"))
    angle_delta = finite_float(b.get("median_angle_rms_deg")) - finite_float(a.get("median_angle_rms_deg"))
    residual_delta = finite_float(b.get("median_tau_residual_rms_nm")) - finite_float(a.get("median_tau_residual_rms_nm"))
    n_hat = finite_float(b.get("median_n_hat_norm_rms"))
    dot_a = finite_float(a.get("median_ff_disturbance_dot_mean"))
    dot_b = finite_float(b.get("median_ff_disturbance_dot_mean"))
    rate_c = finite_float(c.get("median_rate_err_rms_radps"))
    rate_b = finite_float(b.get("median_rate_err_rms_radps"))

    if math.isfinite(dot_a) and dot_a < 0.0:
        return "FAIL_BAD_ARM_FF_SIGN: Case A median arm_ff/disturbance dot product is negative."
    if math.isfinite(dot_b) and dot_b < 0.0:
        return "FAIL_BAD_ARM_FF_SIGN: Case B median arm_ff/disturbance dot product is negative."
    if not all(math.isfinite(v) for v in (rate_improvement, xy_delta, angle_delta, residual_delta, n_hat)):
        return "Missing median metrics; inspect the per-case scoreboards."
    if (
        rate_improvement > 0.0
        and xy_delta <= 0.02
        and angle_delta <= 0.05
        and residual_delta <= 0.0
        and n_hat > 1.0e-4
    ):
        return "PASS_RBFNN_CONTRIBUTION: B improves rate RMS without meaningful XY/attitude/residual penalty."
    if math.isfinite(rate_c) and math.isfinite(rate_b) and rate_c < rate_b:
        return "ARM_FF_SUSPECT: Case C has lower median rate RMS than B, so arm_ff sign/scale still needs correction."
    return "FAIL_RBFNN_CONTRIBUTION: B does not meet the required improvement gates against A."


def write_median_comparison(traj_dir: Path, cases: list[str], output_prefix: str) -> tuple[Path, Path] | None:
    rows = median_rows_for_trajectory(traj_dir, cases)
    if not rows:
        return None

    conclusion = comparison_conclusion(rows)
    csv_path = traj_dir / f"{output_prefix}_median.csv"
    md_path = traj_dir / f"{output_prefix}_median.md"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_headers = [
        "case",
        "repeats",
        "verdict_counts",
        "median_rate_err_rms_radps",
        "median_angle_rms_deg",
        "median_xy_mean_m",
        "median_xy_max_m",
        "median_n_hat_norm_rms",
        "median_tau_residual_rms_nm",
        "median_ff_disturbance_dot_mean",
    ]
    lines = ["# RBFNN A/B/C Median Comparison", ""]
    lines.append("| " + " | ".join(table_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(table_headers)) + " |")
    for row in rows:
        values = []
        for header in table_headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}" if math.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Conclusion", "", conclusion, "", "## Scoreboards", ""])
    for row in rows:
        lines.append(f"- {row['case']}: `{row['scoreboard']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Best YAML config to verify. Defaults to latest final_best YAML.")
    parser.add_argument("--output-root", type=Path, default=None, help="Verification output root.")
    parser.add_argument("--gz-root", type=Path, default=DEFAULT_GZ_ROOT)
    parser.add_argument("--cases", default="A,B,C", help="Comma-separated subset of A,B,C.")
    parser.add_argument("--pattern", default="slow_step", choices=("slow_step", "sin", "step", "chirp", "combined"))
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--rate-hz", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3, help="Number of fixed-config repeats per A/B/C case.")
    parser.add_argument("--include-strong", action="store_true", help="Also run the stronger combined trajectory.")
    parser.add_argument("--strong-pattern", default="combined", choices=("slow_step", "sin", "step", "chirp", "combined"))
    parser.add_argument("--strong-duration-s", type=float, default=180.0)
    parser.add_argument("--strong-amplitude", type=float, default=0.08)
    parser.add_argument("--strong-rate-hz", type=int, default=10)
    parser.add_argument("--arm-state-source", choices=("commanded", "gazebo"), default="commanded")
    parser.add_argument("--use-gazebo-arm-visual", action="store_true")
    parser.add_argument("--px4-wait-s", type=float, default=30.0)
    parser.add_argument("--ros-wait-s", type=float, default=20.0)
    parser.add_argument("--post-ros-settle-s", type=float, default=60.0)
    parser.add_argument("--arm-wait-s", type=float, default=5.0)
    parser.add_argument("--takeoff-wait-s", type=float, default=30.0)
    parser.add_argument("--handoff-timeout-s", type=float, default=90.0)
    parser.add_argument("--handoff-settle-s", type=float, default=4.0)
    parser.add_argument("--flight-time-s", type=float, default=55.0)
    parser.add_argument("--min-external-duration-s", type=float, default=20.0)
    parser.add_argument("--min-external-fraction", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.cases = [case.strip().upper() for case in args.cases.split(",") if case.strip()]
    unknown = [case for case in args.cases if case not in CASES]
    if unknown:
        print(f"Unknown cases: {unknown}. Valid cases: {sorted(CASES)}", file=sys.stderr)
        return 2

    config = args.config or latest_best_config()
    if config is None or not config.exists():
        print("Best config not found. Pass --config /path/final_best_uam_controller_params.yaml", file=sys.stderr)
        return 2
    config = config.resolve()

    if not (args.gz_root / "tools" / "uam_rerun_autotune.py").exists():
        print(f"Missing uam_rerun_autotune.py under {args.gz_root}", file=sys.stderr)
        return 2

    if args.output_root is None:
        args.output_root = DEFAULT_RESULTS_ROOT / f"rbfnn_verification_suite_{now_stamp()}"
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    commands: list[list[str]] = []
    trajectories = [(args.pattern, args.amplitude, args.duration_s, args.rate_hz)]
    if args.include_strong:
        trajectories.append((args.strong_pattern, args.strong_amplitude, args.strong_duration_s, args.strong_rate_hz))

    print("RBFNN verification suite")
    print(f"  config     : {config}")
    print(f"  output root: {args.output_root}")
    print(f"  cases      : {','.join(args.cases)}")
    print("")

    for pattern, amplitude, duration, rate_hz in trajectories:
        traj_dir = args.output_root / f"{pattern}_amp{amp_tag(amplitude)}"
        output_prefix = f"rbfnn_ab_comparison_{pattern}_amp{amp_tag(amplitude)}"
        for case_key in args.cases:
            print(f"[run] case {case_key}: {CASES[case_key]['description']} | {pattern}, amp={amplitude}")
            cmd = build_autotune_cmd(
                args,
                config=config,
                output_dir=traj_dir,
                case_key=case_key,
                pattern=pattern,
                amplitude=amplitude,
                duration=duration,
                rate_hz=rate_hz,
            )
            commands.append(cmd)
            rc = run_command(cmd, cwd=args.gz_root, env=env, dry_run=args.dry_run)
            if rc != 0:
                print(f"[error] case {case_key} failed with exit code {rc}", file=sys.stderr)
                write_suite_summary(args.output_root / "suite_summary.md", commands, args, config)
                return rc

        compare_script = Path(__file__).resolve().with_name("rbfnn_compare_ab_results.py")
        compare_cmd = [
            sys.executable,
            str(compare_script),
            "--root",
            str(traj_dir),
            "--output-prefix",
            output_prefix,
        ]
        commands.append(compare_cmd)
        rc = run_command(compare_cmd, cwd=args.gz_root, env=env, dry_run=args.dry_run)
        if rc != 0:
            print(f"[warn] comparison failed for {traj_dir} with exit code {rc}", file=sys.stderr)
        if not args.dry_run:
            median_outputs = write_median_comparison(traj_dir, args.cases, output_prefix)
            if median_outputs:
                csv_path, md_path = median_outputs
                print(f"Median CSV: {csv_path}")
                print(f"Median Markdown: {md_path}")

    write_suite_summary(args.output_root / "suite_summary.md", commands, args, config)
    print("")
    print(f"Done. Suite summary: {args.output_root / 'suite_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
