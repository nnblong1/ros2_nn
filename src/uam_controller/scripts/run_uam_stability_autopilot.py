#!/usr/bin/env python3
"""Automatic UAM stability candidate generation, verification, ranking, and reporting."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_GZ_ROOT = Path("/home/wicom/PX4-Autopilot/Tools/simulation/gz")
DEFAULT_RESULTS_ROOT = Path("/home/wicom/uam_results")
DEFAULT_BASE_CONFIG = DEFAULT_RESULTS_ROOT / "rbfnn_best_param_search_fffix_20260529_181119/yaml_conservation_ff85.yaml"

PARAM_SENTINELS = (
    "rate_Kp_roll",
    "rate_Kp_pitch",
    "rate_Kp_yaw",
    "arm_ff_scale_roll",
    "rbfnn_output_gain",
)

RATE_STABILITY_KEYS = (
    "rate_Kp_roll",
    "rate_Kp_pitch",
    "rate_Kp_yaw",
    "rate_Ki_roll",
    "rate_Ki_pitch",
    "rate_Ki_yaw",
    "rate_Kd_roll",
    "rate_Kd_pitch",
    "rate_Kd_yaw",
    "base_roll_offset",
    "base_pitch_offset",
)

ARM_FF_KEYS = (
    "arm_ff_enable",
    "arm_ff_start_delay_s",
    "arm_ff_ramp_s",
    "arm_ff_rate_limit_nm_s",
    "arm_ff_lpf_alpha",
    "arm_ff_timeout_s",
    "arm_ff_max_roll_nm",
    "arm_ff_max_pitch_nm",
    "arm_ff_max_yaw_nm",
    "arm_ff_input_frame",
    "arm_ff_reaction_sign",
    "arm_ff_scale_roll",
    "arm_ff_scale_pitch",
    "arm_ff_scale_yaw",
)

VIRTUAL_ARM_KEYS = (
    "arm_virtual_disturbance_enable",
    "arm_virtual_disturbance_max_roll_nm",
    "arm_virtual_disturbance_max_pitch_nm",
    "arm_virtual_disturbance_max_yaw_nm",
    "arm_virtual_disturbance_scale_roll",
    "arm_virtual_disturbance_scale_pitch",
    "arm_virtual_disturbance_scale_yaw",
    "arm_virtual_disturbance_reaction_sign",
)

GOOD_VERDICTS = {"GOOD", "OK"}


@dataclass
class Candidate:
    name: str
    path: Path
    description: str


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def amp_tag(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any) -> str:
    number = finite(value)
    return f"{number:.6g}" if math.isfinite(number) else "nan"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping YAML")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def controller_params(data: dict[str, Any]) -> dict[str, Any]:
    for node_data in data.values():
        if not isinstance(node_data, dict):
            continue
        params = node_data.get("ros__parameters")
        if not isinstance(params, dict):
            continue
        if any(key in params for key in PARAM_SENTINELS):
            return params
    raise KeyError("Could not find UAM controller ros__parameters in YAML")


def latest_robust_config() -> Path | None:
    roots = [p for p in DEFAULT_RESULTS_ROOT.glob("case_d_robust_select_*") if p.is_dir()]
    configs = [root / "robust_case_d_uam_controller_params.yaml" for root in roots]
    configs = [path for path in configs if path.exists()]
    return max(configs, key=lambda p: p.stat().st_mtime) if configs else None


def restore_keys(dst_data: dict[str, Any], src_data: dict[str, Any], keys: tuple[str, ...]) -> None:
    dst_params = controller_params(dst_data)
    src_params = controller_params(src_data)
    for key in keys:
        if key in src_params:
            dst_params[key] = copy.deepcopy(src_params[key])


def merge_robust_rate_with_base_ff(base_data: dict[str, Any], robust_data: dict[str, Any]) -> dict[str, Any]:
    """Use robust-D rate/offset values, but preserve ff85 feedforward and virtual-arm setup."""
    merged = copy.deepcopy(base_data)
    merged_params = controller_params(merged)
    robust_params = controller_params(robust_data)
    for key in RATE_STABILITY_KEYS:
        if key in robust_params:
            merged_params[key] = copy.deepcopy(robust_params[key])
    restore_keys(merged, base_data, ARM_FF_KEYS)
    restore_keys(merged, base_data, VIRTUAL_ARM_KEYS)
    if sum(abs(finite(merged_params.get(key), 0.0)) for key in ("arm_ff_scale_roll", "arm_ff_scale_pitch", "arm_ff_scale_yaw")) > 0.0:
        merged_params["arm_ff_enable"] = True
    return merged


def ensure_real_arm_ff(data: dict[str, Any], base_data: dict[str, Any]) -> None:
    restore_keys(data, base_data, ARM_FF_KEYS)
    params = controller_params(data)
    if sum(abs(finite(params.get(key), 0.0)) for key in ("arm_ff_scale_roll", "arm_ff_scale_pitch", "arm_ff_scale_yaw")) > 0.0:
        params["arm_ff_enable"] = True


def set_rbfnn_safe_params(data: dict[str, Any], *, output_gain: float, e_mod: float, lr: float, width: float) -> None:
    params = controller_params(data)
    params["rbfnn_output_gain"] = float(output_gain)
    params["rbfnn_e_modification"] = float(e_mod)
    params["rbfnn_lr"] = float(lr)
    params["rbfnn_gaussian_width"] = float(width)


def scale_feedforward_from_base(data: dict[str, Any], base_data: dict[str, Any], multiplier: float) -> None:
    ensure_real_arm_ff(data, base_data)
    params = controller_params(data)
    base_params = controller_params(base_data)
    for key in ("arm_ff_scale_roll", "arm_ff_scale_pitch", "arm_ff_scale_yaw"):
        params[key] = finite(base_params.get(key), 0.0) * multiplier
    params["arm_ff_enable"] = True


def write_candidate(
    candidates: list[Candidate],
    candidates_dir: Path,
    index: int,
    name: str,
    data: dict[str, Any],
    description: str,
) -> None:
    path = candidates_dir / f"{index:02d}_{name}.yaml"
    write_yaml(path, data)
    candidates.append(Candidate(name=f"{index:02d}_{name}", path=path, description=description))


def generate_candidates(args: argparse.Namespace) -> list[Candidate]:
    base_data = load_yaml(args.base_config)
    candidates_dir = args.output_root / "candidates"
    candidates: list[Candidate] = []
    index = 1

    base_candidate = copy.deepcopy(base_data)
    ensure_real_arm_ff(base_candidate, base_data)
    write_candidate(candidates, candidates_dir, index, "ff85_original", base_candidate, "Original ff85/base YAML")
    index += 1

    variant_base = copy.deepcopy(base_candidate)
    robust_config = args.robust_d_config
    if robust_config and robust_config.exists():
        robust_data = load_yaml(robust_config)
        variant_base = merge_robust_rate_with_base_ff(base_data, robust_data)
        write_candidate(
            candidates,
            candidates_dir,
            index,
            "robust_d_rate_ff85_ff",
            variant_base,
            f"Rate/offset from {robust_config}, feedforward restored from ff85",
        )
        index += 1

    for output_gain, e_mod, lr, width in (
        (0.10, 0.065, 0.0020, 4.0),
        (0.18, 0.060, 0.0025, 4.0),
        (0.25, 0.055, 0.0030, 3.5),
    ):
        data = copy.deepcopy(variant_base)
        ensure_real_arm_ff(data, base_data)
        set_rbfnn_safe_params(data, output_gain=output_gain, e_mod=e_mod, lr=lr, width=width)
        write_candidate(
            candidates,
            candidates_dir,
            index,
            f"rbfnn_safe_gain{int(output_gain * 100):02d}",
            data,
            f"Safe-limited RBFNN output_gain={output_gain}, e_mod={e_mod}, lr={lr}, width={width}",
        )
        index += 1

    for multiplier in (0.60, 0.75, 0.85, 1.00):
        data = copy.deepcopy(variant_base)
        scale_feedforward_from_base(data, base_data, multiplier)
        write_candidate(
            candidates,
            candidates_dir,
            index,
            f"ff_scale_{int(multiplier * 100):03d}",
            data,
            f"arm_ff_scale_* = ff85 * {multiplier}",
        )
        index += 1

    return candidates[: args.max_candidates]


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def build_suite_cmd(args: argparse.Namespace, candidate: Candidate, run_root: Path) -> list[str]:
    suite_script = Path(__file__).resolve().with_name("run_rbfnn_verification_suite.py")
    return [
        sys.executable,
        str(suite_script),
        "--config",
        str(candidate.path),
        "--output-root",
        str(run_root),
        "--cases",
        "A,B,C,D",
        "--pattern",
        "slow_step",
        "--duration-s",
        f"{args.slow_duration_s:.6g}",
        "--amplitude",
        f"{args.slow_amplitude:.6g}",
        "--rate-hz",
        str(args.slow_rate_hz),
        "--include-strong",
        "--strong-pattern",
        "combined",
        "--strong-duration-s",
        f"{args.combined_duration_s:.6g}",
        "--strong-amplitude",
        f"{args.combined_amplitude:.6g}",
        "--strong-rate-hz",
        str(args.combined_rate_hz),
        "--repeats",
        str(args.repeats),
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
    ] + (["--dry-run"] if args.dry_run else [])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verdict_counts(text: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in str(text or "").split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        try:
            out[key.strip()] = int(value)
        except ValueError:
            continue
    return out


def fail_count(row: dict[str, Any]) -> int:
    repeats = int(finite(row.get("repeats"), 0.0))
    counts = verdict_counts(row.get("verdict_counts"))
    return max(0, repeats - counts.get("GOOD", 0) - counts.get("OK", 0))


def trajectory_tag(pattern: str, amplitude: float) -> str:
    return f"{pattern}_amp{amp_tag(amplitude)}"


def read_median_table(run_root: Path, tag: str) -> dict[str, dict[str, str]]:
    path = run_root / tag / f"rbfnn_ab_comparison_{tag}_median.csv"
    rows = read_csv(path)
    return {row.get("case", ""): row for row in rows}


def metric(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
    return finite(row.get(f"median_{key}"), default)


def summarize_candidate(candidate: Candidate, run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    slow_tag = trajectory_tag("slow_step", args.slow_amplitude)
    combined_tag = trajectory_tag("combined", args.combined_amplitude)
    tables = {
        slow_tag: read_median_table(run_root, slow_tag),
        combined_tag: read_median_table(run_root, combined_tag),
    }

    row: dict[str, Any] = {
        "candidate": candidate.name,
        "description": candidate.description,
        "config_path": str(candidate.path),
        "run_root": str(run_root),
        "score": 0.0,
        "rejected": False,
        "reject_reasons": "",
    }
    reasons: list[str] = []
    score = 0.0
    total_failures = 0

    for tag, cases in tables.items():
        if not cases:
            reasons.append(f"{tag}:missing_median_csv")
            score += 2_000_000.0
            continue

        combined_weight = 1.45 if tag.startswith("combined") else 1.0
        for case_key in ("A", "B", "C", "D"):
            case_row = cases.get(case_key)
            if case_row is None:
                reasons.append(f"{tag}:{case_key}:missing")
                score += 500_000.0
                continue

            failures = fail_count(case_row)
            total_failures += failures
            case_weight = {"A": 0.9, "B": 1.45, "C": 0.75, "D": 1.0}[case_key]
            score += 100_000.0 * failures * combined_weight * case_weight
            score += combined_weight * case_weight * (
                520.0 * metric(case_row, "alt_rmse_m", 2.0)
                + 520.0 * metric(case_row, "xy_mean_m", 2.0)
                + 360.0 * metric(case_row, "xy_max_m", 2.0)
                + 80.0 * metric(case_row, "angle_rms_deg", 20.0)
                + 230.0 * metric(case_row, "rate_err_rms_radps", 2.0)
            )

        for case_key in ("A", "B"):
            ff_max = metric(cases.get(case_key), "arm_ff_max_nm", 0.0)
            if ff_max <= 1.0e-6:
                reasons.append(f"{tag}:{case_key}:arm_ff_zero")
                score += 1_000_000.0

        if tag.startswith("combined"):
            for case_key in ("A", "B", "C", "D"):
                case_row = cases.get(case_key)
                case_fail_xy = verdict_counts(case_row.get("verdict_counts") if case_row else "").get("FAIL_XY", 0)
                row[f"combined_{case_key.lower()}_fail_xy"] = case_fail_xy
                if case_fail_xy > 1:
                    reasons.append(f"{tag}:{case_key}_FAIL_XY>{1}")
                    score += 1_500_000.0 * (case_fail_xy - 1)

            b_row = cases.get("B")
            a_row = cases.get("A")
            row["combined_a_rate"] = metric(a_row, "rate_err_rms_radps")
            row["combined_b_rate"] = metric(b_row, "rate_err_rms_radps")
            row["combined_a_xy_mean"] = metric(a_row, "xy_mean_m")
            row["combined_b_xy_mean"] = metric(b_row, "xy_mean_m")
            row["combined_b_xy_max"] = metric(b_row, "xy_max_m")
            row["combined_b_angle"] = metric(b_row, "angle_rms_deg")
            row["combined_b_n_hat"] = metric(b_row, "n_hat_norm_rms")
            row["combined_b_residual"] = metric(b_row, "tau_residual_rms_nm")
            row["combined_a_residual"] = metric(a_row, "tau_residual_rms_nm")

    row["total_failures"] = total_failures
    row["score"] = score
    row["rejected"] = bool(reasons)
    row["reject_reasons"] = ";".join(reasons)
    return row


def classify_winner(row: dict[str, Any]) -> str:
    if row.get("rejected"):
        return "NEEDS_OUTER_LOOP_TUNE"

    a_rate = finite(row.get("combined_a_rate"))
    b_rate = finite(row.get("combined_b_rate"))
    a_xy = finite(row.get("combined_a_xy_mean"))
    b_xy = finite(row.get("combined_b_xy_mean"))
    b_xy_max = finite(row.get("combined_b_xy_max"))
    b_angle = finite(row.get("combined_b_angle"))
    a_res = finite(row.get("combined_a_residual"))
    b_res = finite(row.get("combined_b_residual"))
    n_hat = finite(row.get("combined_b_n_hat"))

    stable = (
        finite(row.get("combined_b_fail_xy"), 99.0) <= 1.0
        and math.isfinite(b_xy_max)
        and b_xy_max <= 0.50
        and math.isfinite(b_angle)
        and b_angle <= 1.35
    )
    if not stable:
        return "NEEDS_OUTER_LOOP_TUNE"

    rbfnn_pass = (
        math.isfinite(a_rate)
        and math.isfinite(b_rate)
        and math.isfinite(a_xy)
        and math.isfinite(b_xy)
        and math.isfinite(a_res)
        and math.isfinite(b_res)
        and b_rate < a_rate
        and b_xy <= a_xy + 0.02
        and b_res <= a_res
        and n_hat > 1.0e-4
    )
    if rbfnn_pass:
        return "RBFNN_PASS"
    return "RBFNN_LIMITED"


def write_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "candidate",
        "score",
        "rejected",
        "reject_reasons",
        "total_failures",
        "combined_a_fail_xy",
        "combined_b_fail_xy",
        "combined_c_fail_xy",
        "combined_d_fail_xy",
        "combined_a_rate",
        "combined_b_rate",
        "combined_a_xy_mean",
        "combined_b_xy_mean",
        "combined_b_xy_max",
        "combined_b_angle",
        "combined_b_n_hat",
        "combined_a_residual",
        "combined_b_residual",
        "config_path",
        "run_root",
        "description",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = index
            writer.writerow(out)


def median_markdown_for_winner(run_root: Path, tag: str) -> list[str]:
    cases = read_median_table(run_root, tag)
    headers = (
        "case",
        "verdict_counts",
        "median_xy_mean_m",
        "median_xy_max_m",
        "median_alt_rmse_m",
        "median_angle_rms_deg",
        "median_rate_err_rms_radps",
        "median_arm_ff_max_nm",
        "median_n_hat_norm_rms",
        "median_tau_residual_rms_nm",
    )
    lines = [f"### {tag}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for case_key in ("A", "B", "C", "D"):
        row = cases.get(case_key, {})
        values = [case_key]
        for header in headers[1:]:
            value = row.get(header, "")
            values.append(fmt(value) if header.startswith("median_") else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    candidates: list[Candidate],
    rows: list[dict[str, Any]],
    winner: dict[str, Any],
    final_config: Path,
    final_status: str,
    commands: list[list[str]],
) -> None:
    lines = [
        "# UAM Stability Autopilot Report",
        "",
        f"- Final status: `{final_status}`",
        f"- Final YAML: `{final_config}`",
        f"- Winner: `{winner.get('candidate')}`",
        f"- Base config: `{args.base_config}`",
        f"- Robust-D source: `{args.robust_d_config}`",
        f"- Repeats: `{args.repeats}`",
        "",
        "## Candidate Ranking",
        "",
        "| rank | candidate | score | rejected | combined B FAIL_XY | B xy_max | B rate | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(rows[:12], start=1):
        lines.append(
            "| {rank} | {candidate} | {score} | {rejected} | {fail_xy} | {xy_max} | {rate} | {reason} |".format(
                rank=rank,
                candidate=row.get("candidate", ""),
                score=fmt(row.get("score")),
                rejected=row.get("rejected", ""),
                fail_xy=fmt(row.get("combined_b_fail_xy")),
                xy_max=fmt(row.get("combined_b_xy_max")),
                rate=fmt(row.get("combined_b_rate")),
                reason=str(row.get("reject_reasons", "")) or "-",
            )
        )

    lines.extend(["", "## Winner A/B/C/D Medians", ""])
    winner_root = Path(str(winner.get("run_root", "")))
    lines.extend(median_markdown_for_winner(winner_root, trajectory_tag("slow_step", args.slow_amplitude)))
    lines.extend(median_markdown_for_winner(winner_root, trajectory_tag("combined", args.combined_amplitude)))

    lines.extend(["## Generated Candidates", ""])
    for candidate in candidates:
        lines.append(f"- `{candidate.name}`: `{candidate.path}` - {candidate.description}")

    lines.extend(["", "## Commands", ""])
    for cmd in commands:
        lines.append("```bash")
        lines.append(" ".join(str(part) for part in cmd))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--robust-d-config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT / f"uam_stability_autopilot_{now_stamp()}")
    parser.add_argument("--gz-root", type=Path, default=DEFAULT_GZ_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Smoke mode: 1 repeat, first 2 candidates.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm-state-source", choices=("commanded", "gazebo"), default="commanded")
    parser.add_argument("--slow-amplitude", type=float, default=0.08)
    parser.add_argument("--slow-duration-s", type=float, default=300.0)
    parser.add_argument("--slow-rate-hz", type=int, default=5)
    parser.add_argument("--combined-amplitude", type=float, default=0.08)
    parser.add_argument("--combined-duration-s", type=float, default=180.0)
    parser.add_argument("--combined-rate-hz", type=int, default=10)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.repeats = 1
        args.max_candidates = min(args.max_candidates, 2)

    if args.robust_d_config is None:
        args.robust_d_config = latest_robust_config()

    if not args.base_config.exists():
        print(f"Base config not found: {args.base_config}", file=sys.stderr)
        return 2
    if not (args.gz_root / "tools/uam_rerun_autotune.py").exists():
        print(
            "Missing custom autotune tool: "
            f"{args.gz_root / 'tools/uam_rerun_autotune.py'}\n"
            "The current PX4 v1.16.2-rbfnn Gazebo submodule does not include this thesis-specific script. "
            "Use the stable QGC/PX4 v1.16.2-rbfnn launch path, or restore/port the legacy autotune tool "
            "before running this autopilot search.",
            file=sys.stderr,
        )
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidates(args)
    if not candidates:
        print("No candidates generated.", file=sys.stderr)
        return 2

    print("UAM stability autopilot")
    print(f"  output root: {args.output_root}")
    print(f"  base config: {args.base_config}")
    print(f"  robust-D   : {args.robust_d_config}")
    print(f"  candidates : {len(candidates)}")
    print(f"  repeats    : {args.repeats}")
    print("")

    rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    runs_dir = args.output_root / "runs"
    for index, candidate in enumerate(candidates, start=1):
        run_root = runs_dir / candidate.name
        print(f"[candidate {index}/{len(candidates)}] {candidate.name}: {candidate.description}")
        cmd = build_suite_cmd(args, candidate, run_root)
        commands.append(cmd)
        rc = run_command(cmd, args.gz_root, args.dry_run)
        if rc != 0:
            print(f"[error] verification failed for {candidate.name} with exit code {rc}", file=sys.stderr)
            return rc
        if args.dry_run:
            row = {
                "candidate": candidate.name,
                "description": candidate.description,
                "config_path": str(candidate.path),
                "run_root": str(run_root),
                "score": 0.0,
                "rejected": False,
                "reject_reasons": "dry_run",
            }
        else:
            row = summarize_candidate(candidate, run_root, args)
        rows.append(row)
        rows.sort(key=lambda item: (bool(item.get("rejected")), finite(item.get("score"), 1e12)))
        write_ranking(args.output_root / "candidate_ranking.csv", rows)

    rows.sort(key=lambda item: (bool(item.get("rejected")), finite(item.get("score"), 1e12)))
    winner = rows[0]
    final_status = "DRY_RUN" if args.dry_run else classify_winner(winner)
    final_config = args.output_root / "final_stable_uam_controller_params.yaml"
    shutil.copy2(Path(str(winner["config_path"])), final_config)
    (args.output_root / "final_status.txt").write_text(final_status + "\n", encoding="utf-8")

    write_ranking(args.output_root / "candidate_ranking.csv", rows)
    write_report(
        args.output_root / "final_stability_report.md",
        args=args,
        candidates=candidates,
        rows=rows,
        winner=winner,
        final_config=final_config,
        final_status=final_status,
        commands=commands,
    )

    print("")
    print("Done.")
    print(f"  status : {final_status}")
    print(f"  final  : {final_config}")
    print(f"  ranking: {args.output_root / 'candidate_ranking.csv'}")
    print(f"  report : {args.output_root / 'final_stability_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
