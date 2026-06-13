#!/usr/bin/env python3
"""Validate Case D YAML candidates with repeated fixed-config trials."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_GZ_ROOT = Path("/home/wicom/PX4-Autopilot/Tools/simulation/gz")
DEFAULT_RESULTS_ROOT = Path("/home/wicom/uam_results")
DEFAULT_BASE_CONFIG = DEFAULT_RESULTS_ROOT / "rbfnn_best_param_search_fffix_20260529_181119/yaml_conservation_ff85.yaml"

TRAJECTORIES = (
    ("slow_step_amp080", "slow_step", 0.08, 300.0, 5),
    ("combined_amp080", "combined", 0.08, 180.0, 10),
)

GOOD_VERDICTS = {"GOOD", "OK"}

PARAM_SENTINELS = (
    "rate_Kp_roll",
    "rate_Kp_pitch",
    "rate_Kp_yaw",
    "arm_ff_scale_roll",
    "rbfnn_output_gain",
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


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def finite(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def median(values: list[float]) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping YAML")
    return data


def controller_params(data: dict) -> dict:
    for node_data in data.values():
        if not isinstance(node_data, dict):
            continue
        params = node_data.get("ros__parameters")
        if not isinstance(params, dict):
            continue
        if any(key in params for key in PARAM_SENTINELS):
            return params
    raise KeyError("Could not find controller ros__parameters in YAML")


def merge_stability_yaml(base_ff_config: Path, robust_config: Path, output_path: Path) -> None:
    """Keep robust Case-D gains, but restore real arm feedforward from ff85/base YAML."""
    robust_data = copy.deepcopy(load_yaml(robust_config))
    base_data = load_yaml(base_ff_config)
    robust_params = controller_params(robust_data)
    base_params = controller_params(base_data)

    restored = []
    for key in ARM_FF_KEYS:
        if key in base_params:
            robust_params[key] = copy.deepcopy(base_params[key])
            restored.append(key)

    if not restored:
        raise KeyError(f"No arm_ff_* keys were restored from {base_ff_config}")

    scale_norm = sum(abs(finite(robust_params.get(key), 0.0)) for key in (
        "arm_ff_scale_roll",
        "arm_ff_scale_pitch",
        "arm_ff_scale_yaw",
    ))
    if scale_norm > 0.0:
        robust_params["arm_ff_enable"] = bool(base_params.get("arm_ff_enable", True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(robust_data, f, sort_keys=False)


def latest_tune_root() -> Path | None:
    roots = [p for p in DEFAULT_RESULTS_ROOT.glob("case_d_tune_ff85_*") if p.is_dir()]
    return max(roots, key=lambda p: p.stat().st_mtime) if roots else None


def safe_name(path: Path, index: int) -> str:
    parent = path.parent.parent.name if path.parent.name == "configs" else path.parent.name
    stem = path.stem.replace("bs_arm_disturbance_no_ff_", "")
    return f"cand_{index:02d}_{parent}_{stem}".replace("-", "_")


def verdict_rank(verdict: str) -> int:
    if verdict == "GOOD":
        return 0
    if verdict == "OK":
        return 1
    return 2


def candidate_paths(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.base_config.exists():
        candidates.append(args.base_config)

    for path in args.candidate_yaml:
        if path.exists():
            candidates.append(path)

    tune_root = args.tune_root
    if tune_root is None:
        tune_root = latest_tune_root()
    if tune_root:
        for scoreboard in sorted(tune_root.glob("*/scoreboard.csv")):
            rows = read_csv(scoreboard)
            rows = [row for row in rows if Path(row.get("config_path", "")).exists()]
            rows.sort(
                key=lambda row: (
                    verdict_rank(str(row.get("verdict", ""))),
                    finite(row.get("score"), 1e12),
                    finite(row.get("xy_max_m"), 1e6),
                    finite(row.get("rate_err_rms_radps"), 1e6),
                )
            )
            for row in rows[: args.top_per_phase]:
                candidates.append(Path(row["config_path"]))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out[: args.max_candidates]


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def run_candidate(args: argparse.Namespace, yaml_path: Path, case_dir: Path, case_name: str) -> int:
    for traj_name, pattern, amplitude, duration, rate_hz in TRAJECTORIES:
        out_dir = case_dir / traj_name
        cmd = [
            sys.executable,
            str(args.gz_root / "tools/uam_rerun_autotune.py"),
            "--fixed-config",
            "--stage",
            "bs_arm_disturbance_no_ff",
            "--trials",
            str(args.repeats),
            "--base-config",
            str(yaml_path),
            "--output-dir",
            str(out_dir),
            "--case-name",
            f"{case_name}_{traj_name}",
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
            "--fail-angle-deg",
            f"{args.fail_angle_deg:.6g}",
            "--fail-xy-m",
            f"{args.fail_xy_m:.6g}",
            "--fail-rate-rms-radps",
            f"{args.fail_rate_rms_radps:.6g}",
            "--good-alt-rmse-m",
            f"{args.good_alt_rmse_m:.6g}",
            "--good-xy-mean-m",
            f"{args.good_xy_mean_m:.6g}",
            "--good-angle-rms-deg",
            f"{args.good_angle_rms_deg:.6g}",
            "--good-angle-max-deg",
            f"{args.good_angle_max_deg:.6g}",
            "--good-rate-rms-radps",
            f"{args.good_rate_rms_radps:.6g}",
        ]
        rc = run_command(cmd, args.gz_root, args.dry_run)
        if rc != 0:
            return rc
    return 0


def summarize_candidate(name: str, yaml_path: Path, case_dir: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "candidate": name,
        "config_path": str(yaml_path),
        "fail_count": 0,
        "good_count": 0,
        "ok_count": 0,
        "trial_count": 0,
    }
    robust_score = 0.0
    for traj_name, *_ in TRAJECTORIES:
        rows = read_csv(case_dir / traj_name / "scoreboard.csv")
        summary[f"{traj_name}_verdicts"] = ";".join(
            f"{verdict}:{sum(1 for row in rows if row.get('verdict') == verdict)}"
            for verdict in sorted({row.get("verdict", "UNKNOWN") for row in rows})
        )
        fail_count = sum(1 for row in rows if row.get("verdict") not in GOOD_VERDICTS)
        good_count = sum(1 for row in rows if row.get("verdict") == "GOOD")
        ok_count = sum(1 for row in rows if row.get("verdict") == "OK")
        summary["fail_count"] = int(summary["fail_count"]) + fail_count
        summary["good_count"] = int(summary["good_count"]) + good_count
        summary["ok_count"] = int(summary["ok_count"]) + ok_count
        summary["trial_count"] = int(summary["trial_count"]) + len(rows)

        metrics = {}
        for key in (
            "alt_rmse_m",
            "xy_mean_m",
            "xy_max_m",
            "angle_rms_deg",
            "angle_max_deg",
            "rate_err_rms_radps",
            "rate_err_max_radps",
        ):
            value = median([finite(row.get(key)) for row in rows])
            metrics[key] = value
            summary[f"{traj_name}_{key}_median"] = value

        robust_score += (
            100000.0 * fail_count
            + 260.0 * metrics["alt_rmse_m"]
            + 230.0 * metrics["xy_mean_m"]
            + 95.0 * metrics["xy_max_m"]
            + 35.0 * metrics["angle_rms_deg"]
            + 12.0 * metrics["angle_max_deg"]
            + 420.0 * metrics["rate_err_rms_radps"]
            + 35.0 * metrics["rate_err_max_radps"]
        )
    summary["robust_score"] = robust_score
    return summary


def write_outputs(output_root: Path, rows: list[dict[str, object]]) -> None:
    rows.sort(key=lambda row: finite(row.get("robust_score"), 1e12))
    fields = [
        "candidate",
        "robust_score",
        "fail_count",
        "good_count",
        "ok_count",
        "trial_count",
        "slow_step_amp080_verdicts",
        "slow_step_amp080_xy_mean_m_median",
        "slow_step_amp080_xy_max_m_median",
        "slow_step_amp080_angle_rms_deg_median",
        "slow_step_amp080_rate_err_rms_radps_median",
        "combined_amp080_verdicts",
        "combined_amp080_xy_mean_m_median",
        "combined_amp080_xy_max_m_median",
        "combined_amp080_angle_rms_deg_median",
        "combined_amp080_rate_err_rms_radps_median",
        "config_path",
    ]
    csv_path = output_root / "candidate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Case D Robust Candidate Selection", ""]
    md_lines.append("| candidate | score | fails | slow xy/rate | combined xy/rate | config |")
    md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        md_lines.append(
            "| {candidate} | {score:.3f} | {fails} | {sxy:.3f}/{sr:.3f} | {cxy:.3f}/{cr:.3f} | `{config}` |".format(
                candidate=row["candidate"],
                score=finite(row["robust_score"], 1e12),
                fails=row["fail_count"],
                sxy=finite(row.get("slow_step_amp080_xy_mean_m_median"), 0.0),
                sr=finite(row.get("slow_step_amp080_rate_err_rms_radps_median"), 0.0),
                cxy=finite(row.get("combined_amp080_xy_mean_m_median"), 0.0),
                cr=finite(row.get("combined_amp080_rate_err_rms_radps_median"), 0.0),
                config=row["config_path"],
            )
        )
    (output_root / "candidate_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustly select a Case D YAML candidate.")
    parser.add_argument("--tune-root", type=Path, default=None)
    parser.add_argument("--candidate-yaml", type=Path, action="append", default=[])
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT / f"case_d_robust_select_{now_stamp()}")
    parser.add_argument("--gz-root", type=Path, default=DEFAULT_GZ_ROOT)
    parser.add_argument("--top-per-phase", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arm-state-source", choices=("commanded", "gazebo"), default="commanded")
    parser.add_argument("--px4-wait-s", type=float, default=30.0)
    parser.add_argument("--ros-wait-s", type=float, default=20.0)
    parser.add_argument("--post-ros-settle-s", type=float, default=60.0)
    parser.add_argument("--arm-wait-s", type=float, default=5.0)
    parser.add_argument("--takeoff-wait-s", type=float, default=30.0)
    parser.add_argument("--handoff-timeout-s", type=float, default=90.0)
    parser.add_argument("--handoff-settle-s", type=float, default=4.0)
    parser.add_argument("--flight-time-s", type=float, default=55.0)
    parser.add_argument("--fail-angle-deg", type=float, default=6.0)
    parser.add_argument("--fail-xy-m", type=float, default=0.35)
    parser.add_argument("--fail-rate-rms-radps", type=float, default=0.24)
    parser.add_argument("--good-alt-rmse-m", type=float, default=0.045)
    parser.add_argument("--good-xy-mean-m", type=float, default=0.09)
    parser.add_argument("--good-angle-rms-deg", type=float, default=1.05)
    parser.add_argument("--good-angle-max-deg", type=float, default=3.5)
    parser.add_argument("--good-rate-rms-radps", type=float, default=0.145)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.gz_root / "tools/uam_rerun_autotune.py").exists():
        print(
            "Missing custom autotune tool: "
            f"{args.gz_root / 'tools/uam_rerun_autotune.py'}\n"
            "The current PX4 v1.16.2-rbfnn Gazebo submodule does not include this thesis-specific script. "
            "Use the stable QGC/PX4 v1.16.2-rbfnn launch path, or restore/port the legacy autotune tool "
            "before running robust Case D selection.",
            file=sys.stderr,
        )
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = candidate_paths(args)
    if not candidates:
        print("No candidates found.", file=sys.stderr)
        return 2

    print("Case D robust selection")
    print(f"  output root: {args.output_root}")
    print(f"  repeats    : {args.repeats}")
    print("  candidates :")
    for path in candidates:
        print(f"    - {path}")
    print("")

    rows: list[dict[str, object]] = []
    for index, yaml_path in enumerate(candidates, start=1):
        name = safe_name(yaml_path, index)
        case_dir = args.output_root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "candidate_config_path.txt").write_text(str(yaml_path) + "\n", encoding="utf-8")
        print(f"[candidate {index}/{len(candidates)}] {name}")
        rc = run_candidate(args, yaml_path, case_dir, name)
        if rc != 0:
            return rc
        rows.append(summarize_candidate(name, yaml_path, case_dir))
        write_outputs(args.output_root, rows)

    rows.sort(key=lambda row: finite(row.get("robust_score"), 1e12))
    best = rows[0]
    best_src = Path(str(best["config_path"]))
    best_dst = args.output_root / "robust_case_d_uam_controller_params.yaml"
    source_dst = args.output_root / "robust_case_d_source_params.yaml"
    shutil.copy2(best_src, source_dst)
    merge_stability_yaml(args.base_config, best_src, best_dst)
    (args.output_root / "best_candidate.txt").write_text(
        f"{best['candidate']}\nsource={best_src}\nmerged={best_dst}\nscore={best['robust_score']}\n",
        encoding="utf-8",
    )
    write_outputs(args.output_root, rows)
    print("")
    print("Best robust Case D candidate:")
    print(f"  candidate: {best['candidate']}")
    print(f"  score    : {best['robust_score']}")
    print(f"  config   : {best_dst}")
    print(f"  summary  : {args.output_root / 'candidate_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
