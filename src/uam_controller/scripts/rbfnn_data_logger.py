#!/usr/bin/env python3
"""
Experiment logger for UAM QGC/external-rate-controller validation.

The node records PX4 telemetry, external torque/thrust setpoints, controller
debug data, enable state, and arm command/feedback into one CSV.  When the
node stops, it writes summary JSON/Markdown files that can be used directly in
the thesis report.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from px4_msgs.msg import (
    VehicleLandDetected,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
    VehicleThrustSetpoint,
    VehicleTorqueSetpoint,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray


PX4_RESULTS_ROOT = Path.home() / "uam_verification_logs"
N_JOINTS = 6
N_AXES = 3


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite_list(values: Any, size: int) -> list[float]:
    out = [math.nan] * size
    if values is None:
        return out

    for idx, value in enumerate(list(values)[:size]):
        out[idx] = finite(value)
    return out


def quat_to_rpy_deg(q: list[float]) -> tuple[float, float, float]:
    q0, q1, q2, q3 = q

    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def safe_case_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._-") or "manual_case"


def clean_values(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: list[float]) -> float:
    vals = clean_values(values)
    return sum(vals) / len(vals) if vals else math.nan


def maximum(values: list[float]) -> float:
    vals = clean_values(values)
    return max(vals) if vals else math.nan


def minimum(values: list[float]) -> float:
    vals = clean_values(values)
    return min(vals) if vals else math.nan


def stddev(values: list[float]) -> float:
    vals = clean_values(values)
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else math.nan
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def rms(values: list[float]) -> float:
    vals = clean_values(values)
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else math.nan


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    return value


class RBFNNDataLogger(Node):
    def __init__(self) -> None:
        super().__init__("rbfnn_data_logger")

        self.declare_parameter("case_name", "manual_case")
        self.declare_parameter("output_root", str(PX4_RESULTS_ROOT))
        self.declare_parameter("log_rate_hz", 20.0)
        self.declare_parameter("target_alt_m", 2.0)
        self.declare_parameter("analysis_settle_after_enable_s", 2.0)
        self.declare_parameter("notes", "")

        self.case_name = safe_case_name(self.get_parameter("case_name").value)
        self.output_root = Path(str(self.get_parameter("output_root").value)).expanduser()
        self.log_rate_hz = max(1.0, float(self.get_parameter("log_rate_hz").value))
        self.target_alt_m = float(self.get_parameter("target_alt_m").value)
        self.analysis_settle_after_enable_s = max(
            0.0,
            float(self.get_parameter("analysis_settle_after_enable_s").value),
        )
        self.notes = str(self.get_parameter("notes").value)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / f"{stamp}_{self.case_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "flight_timeseries.csv"
        self.metadata_path = self.run_dir / "metadata.json"
        self.summary_json_path = self.run_dir / "summary.json"
        self.summary_md_path = self.run_dir / "summary.md"

        self.start_wall = time.time()
        self.start_iso = datetime.now().isoformat(timespec="seconds")
        self._closed = False
        self.samples: list[dict[str, Any]] = []
        self.last_odom_wall = math.nan
        self.last_rates_sp_wall = math.nan
        self.last_torque_wall = math.nan
        self.last_thrust_wall = math.nan
        self.last_debug_wall = math.nan
        self.last_joint_wall = math.nan
        self.last_arm_cmd_wall = math.nan

        self.state: dict[str, Any] = {
            "x_m": math.nan,
            "y_m": math.nan,
            "z_ned_m": math.nan,
            "alt_m": math.nan,
            "vx_mps": math.nan,
            "vy_mps": math.nan,
            "vz_mps": math.nan,
            "roll_deg": math.nan,
            "pitch_deg": math.nan,
            "yaw_deg": math.nan,
            "p_radps": math.nan,
            "q_radps": math.nan,
            "r_radps": math.nan,
            "roll_rate_sp": math.nan,
            "pitch_rate_sp": math.nan,
            "yaw_rate_sp": math.nan,
            "rates_sp_thrust_x": math.nan,
            "rates_sp_thrust_y": math.nan,
            "rates_sp_thrust_z": math.nan,
            "torque_x": math.nan,
            "torque_y": math.nan,
            "torque_z": math.nan,
            "thrust_x": math.nan,
            "thrust_y": math.nan,
            "thrust_z": math.nan,
            "dbg_omega_x": math.nan,
            "dbg_omega_y": math.nan,
            "dbg_omega_z": math.nan,
            "dbg_omega_des_x": math.nan,
            "dbg_omega_des_y": math.nan,
            "dbg_omega_des_z": math.nan,
            "dbg_e_omega_x": math.nan,
            "dbg_e_omega_y": math.nan,
            "dbg_e_omega_z": math.nan,
            "dbg_n_hat_x": math.nan,
            "dbg_n_hat_y": math.nan,
            "dbg_n_hat_z": math.nan,
            "dbg_tau_x": math.nan,
            "dbg_tau_y": math.nan,
            "dbg_tau_z": math.nan,
            "dbg_tau_arm_ff_x": math.nan,
            "dbg_tau_arm_ff_y": math.nan,
            "dbg_tau_arm_ff_z": math.nan,
            "dbg_tau_norm_x": math.nan,
            "dbg_tau_norm_y": math.nan,
            "dbg_tau_norm_z": math.nan,
            "dbg_arm_ff_enabled": False,
            "dbg_arm_ff_fresh": False,
            "dbg_tau_arm_disturbance_x": math.nan,
            "dbg_tau_arm_disturbance_y": math.nan,
            "dbg_tau_arm_disturbance_z": math.nan,
            "dbg_arm_virtual_disturbance_enabled": False,
            "controller_enabled": False,
            "landed": True,
            "armed": False,
            "arming_state": -1,
            "nav_state": -1,
        }
        for idx in range(N_JOINTS):
            self.state[f"joint_pos_{idx + 1}"] = math.nan
            self.state[f"joint_vel_{idx + 1}"] = math.nan
            self.state[f"joint_cmd_{idx + 1}"] = math.nan

        self.headers = [
            "t_s",
            "wall_time_s",
            "case_name",
            "armed",
            "arming_state",
            "nav_state",
            "landed",
            "controller_enabled",
            "odom_age_s",
            "rates_sp_age_s",
            "torque_age_s",
            "thrust_age_s",
            "debug_age_s",
            "joint_age_s",
            "arm_cmd_age_s",
            "x_m",
            "y_m",
            "z_ned_m",
            "alt_m",
            "alt_error_m",
            "vx_mps",
            "vy_mps",
            "vz_mps",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "p_radps",
            "q_radps",
            "r_radps",
            "roll_rate_sp",
            "pitch_rate_sp",
            "yaw_rate_sp",
            "rates_sp_thrust_x",
            "rates_sp_thrust_y",
            "rates_sp_thrust_z",
            "torque_x",
            "torque_y",
            "torque_z",
            "thrust_x",
            "thrust_y",
            "thrust_z",
            "dbg_omega_x",
            "dbg_omega_y",
            "dbg_omega_z",
            "dbg_omega_des_x",
            "dbg_omega_des_y",
            "dbg_omega_des_z",
            "dbg_e_omega_x",
            "dbg_e_omega_y",
            "dbg_e_omega_z",
            "dbg_n_hat_x",
            "dbg_n_hat_y",
            "dbg_n_hat_z",
            "dbg_tau_x",
            "dbg_tau_y",
            "dbg_tau_z",
            "dbg_tau_arm_ff_x",
            "dbg_tau_arm_ff_y",
            "dbg_tau_arm_ff_z",
            "dbg_tau_norm_x",
            "dbg_tau_norm_y",
            "dbg_tau_norm_z",
            "dbg_arm_ff_enabled",
            "dbg_arm_ff_fresh",
            "dbg_tau_arm_disturbance_x",
            "dbg_tau_arm_disturbance_y",
            "dbg_tau_arm_disturbance_z",
            "dbg_arm_virtual_disturbance_enabled",
        ]
        self.headers.extend([f"joint_pos_{idx + 1}" for idx in range(N_JOINTS)])
        self.headers.extend([f"joint_vel_{idx + 1}" for idx in range(N_JOINTS)])
        self.headers.extend([f"joint_cmd_{idx + 1}" for idx in range(N_JOINTS)])

        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.headers, extrasaction="ignore")
        self.csv_writer.writeheader()

        self._write_metadata()

        self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.odom_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleRatesSetpoint,
            "/fmu/out/vehicle_rates_setpoint",
            self.rates_sp_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleTorqueSetpoint,
            "/fmu/in/vehicle_torque_setpoint",
            self.torque_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleThrustSetpoint,
            "/fmu/in/vehicle_thrust_setpoint",
            self.thrust_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self.land_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self.status_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(Bool, "/uam/controller_enable", self.enable_cb, 10)
        self.create_subscription(Float64MultiArray, "/uam/debug_state", self.debug_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        self.create_subscription(
            JointState,
            "/arm_controller/joint_trajectory_plan",
            self.arm_command_cb,
            10,
        )

        self.timer = self.create_timer(1.0 / self.log_rate_hz, self.timer_cb)
        self.get_logger().info(
            "Experiment logger started | case=%s | rate=%.1fHz | dir=%s"
            % (self.case_name, self.log_rate_hz, self.run_dir)
        )

    def _write_metadata(self) -> None:
        metadata = {
            "case_name": self.case_name,
            "created_at": self.start_iso,
            "output_directory": str(self.run_dir),
            "target_alt_m": self.target_alt_m,
            "log_rate_hz": self.log_rate_hz,
            "analysis_settle_after_enable_s": self.analysis_settle_after_enable_s,
            "notes": self.notes,
            "files": {
                "timeseries_csv": str(self.csv_path),
                "summary_json": str(self.summary_json_path),
                "summary_markdown": str(self.summary_md_path),
            },
            "topics": {
                "odometry": "/fmu/out/vehicle_odometry",
                "rates_setpoint": "/fmu/out/vehicle_rates_setpoint",
                "torque_setpoint": "/fmu/in/vehicle_torque_setpoint",
                "thrust_setpoint": "/fmu/in/vehicle_thrust_setpoint",
                "controller_enable": "/uam/controller_enable",
                "debug_state": "/uam/debug_state",
                "joint_states": "/joint_states",
                "arm_command": "/arm_controller/joint_trajectory_plan",
                "land_detected": "/fmu/out/vehicle_land_detected",
                "vehicle_status": "/fmu/out/vehicle_status_v1",
            },
            "debug_state_layout": {
                "0_2": "omega body rates",
                "3_5": "omega_des desired body rates",
                "6_8": "e_omega rate error",
                "9_11": "n_hat RBFNN compensation estimate",
                "12_14": "tau commanded torque before normalization [Nm]",
                "15_17": "tau_arm_ff compensation torque [Nm]",
                "18_20": "tau_norm sent to PX4 before saturation",
                "21": "arm_ff_enabled",
                "22": "arm_ff_fresh",
                "23_25": "tau_arm_virtual_disturbance injected into simulated plant [Nm]",
                "26": "arm_virtual_disturbance_enabled",
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _message_age(self, wall: float, now: float) -> float:
        return now - wall if math.isfinite(wall) else math.nan

    def odom_cb(self, msg: VehicleOdometry) -> None:
        self.last_odom_wall = time.time()
        pos = finite_list(msg.position, N_AXES)
        vel = finite_list(msg.velocity, N_AXES)
        omega = finite_list(msg.angular_velocity, N_AXES)
        quat = finite_list(msg.q, 4)
        roll, pitch, yaw = quat_to_rpy_deg(quat) if all(math.isfinite(v) for v in quat) else (math.nan, math.nan, math.nan)

        self.state.update(
            {
                "x_m": pos[0],
                "y_m": pos[1],
                "z_ned_m": pos[2],
                "alt_m": -pos[2] if math.isfinite(pos[2]) else math.nan,
                "vx_mps": vel[0],
                "vy_mps": vel[1],
                "vz_mps": vel[2],
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "p_radps": omega[0],
                "q_radps": omega[1],
                "r_radps": omega[2],
            }
        )

    def rates_sp_cb(self, msg: VehicleRatesSetpoint) -> None:
        self.last_rates_sp_wall = time.time()
        thrust = finite_list(msg.thrust_body, N_AXES)
        self.state.update(
            {
                "roll_rate_sp": finite(msg.roll),
                "pitch_rate_sp": finite(msg.pitch),
                "yaw_rate_sp": finite(msg.yaw),
                "rates_sp_thrust_x": thrust[0],
                "rates_sp_thrust_y": thrust[1],
                "rates_sp_thrust_z": thrust[2],
            }
        )

    def torque_cb(self, msg: VehicleTorqueSetpoint) -> None:
        self.last_torque_wall = time.time()
        torque = finite_list(msg.xyz, N_AXES)
        self.state.update({"torque_x": torque[0], "torque_y": torque[1], "torque_z": torque[2]})

    def thrust_cb(self, msg: VehicleThrustSetpoint) -> None:
        self.last_thrust_wall = time.time()
        thrust = finite_list(msg.xyz, N_AXES)
        self.state.update({"thrust_x": thrust[0], "thrust_y": thrust[1], "thrust_z": thrust[2]})

    def land_cb(self, msg: VehicleLandDetected) -> None:
        self.state["landed"] = bool(msg.landed)

    def status_cb(self, msg: VehicleStatus) -> None:
        arming_state = int(getattr(msg, "arming_state", -1))
        self.state["arming_state"] = arming_state
        self.state["armed"] = arming_state == 2
        self.state["nav_state"] = int(getattr(msg, "nav_state", -1))

    def enable_cb(self, msg: Bool) -> None:
        self.state["controller_enabled"] = bool(msg.data)

    def debug_cb(self, msg: Float64MultiArray) -> None:
        self.last_debug_wall = time.time()
        data = list(msg.data)
        if len(data) >= 15:
            self.state.update(
                {
                    "dbg_omega_x": finite(data[0]),
                    "dbg_omega_y": finite(data[1]),
                    "dbg_omega_z": finite(data[2]),
                    "dbg_omega_des_x": finite(data[3]),
                    "dbg_omega_des_y": finite(data[4]),
                    "dbg_omega_des_z": finite(data[5]),
                    "dbg_e_omega_x": finite(data[6]),
                    "dbg_e_omega_y": finite(data[7]),
                    "dbg_e_omega_z": finite(data[8]),
                    "dbg_n_hat_x": finite(data[9]),
                    "dbg_n_hat_y": finite(data[10]),
                    "dbg_n_hat_z": finite(data[11]),
                    "dbg_tau_x": finite(data[12]),
                    "dbg_tau_y": finite(data[13]),
                    "dbg_tau_z": finite(data[14]),
                }
            )
        if len(data) >= 23:
            self.state.update(
                {
                    "dbg_tau_arm_ff_x": finite(data[15]),
                    "dbg_tau_arm_ff_y": finite(data[16]),
                    "dbg_tau_arm_ff_z": finite(data[17]),
                    "dbg_tau_norm_x": finite(data[18]),
                    "dbg_tau_norm_y": finite(data[19]),
                    "dbg_tau_norm_z": finite(data[20]),
                    "dbg_arm_ff_enabled": bool(finite(data[21]) > 0.5),
                    "dbg_arm_ff_fresh": bool(finite(data[22]) > 0.5),
                }
            )
        if len(data) >= 27:
            self.state.update(
                {
                    "dbg_tau_arm_disturbance_x": finite(data[23]),
                    "dbg_tau_arm_disturbance_y": finite(data[24]),
                    "dbg_tau_arm_disturbance_z": finite(data[25]),
                    "dbg_arm_virtual_disturbance_enabled": bool(finite(data[26]) > 0.5),
                }
            )

    def joint_state_cb(self, msg: JointState) -> None:
        self.last_joint_wall = time.time()
        positions = finite_list(msg.position, N_JOINTS)
        velocities = finite_list(msg.velocity, N_JOINTS)
        for idx in range(N_JOINTS):
            self.state[f"joint_pos_{idx + 1}"] = positions[idx]
            self.state[f"joint_vel_{idx + 1}"] = velocities[idx]

    def arm_command_cb(self, msg: JointState) -> None:
        self.last_arm_cmd_wall = time.time()
        commands = finite_list(msg.position, N_JOINTS)
        for idx in range(N_JOINTS):
            self.state[f"joint_cmd_{idx + 1}"] = commands[idx]

    def timer_cb(self) -> None:
        now_wall = time.time()
        t_s = now_wall - self.start_wall
        sample = dict(self.state)
        sample.update(
            {
                "t_s": t_s,
                "wall_time_s": now_wall,
                "case_name": self.case_name,
                "alt_error_m": sample["alt_m"] - self.target_alt_m
                if math.isfinite(sample["alt_m"])
                else math.nan,
                "odom_age_s": self._message_age(self.last_odom_wall, now_wall),
                "rates_sp_age_s": self._message_age(self.last_rates_sp_wall, now_wall),
                "torque_age_s": self._message_age(self.last_torque_wall, now_wall),
                "thrust_age_s": self._message_age(self.last_thrust_wall, now_wall),
                "debug_age_s": self._message_age(self.last_debug_wall, now_wall),
                "joint_age_s": self._message_age(self.last_joint_wall, now_wall),
                "arm_cmd_age_s": self._message_age(self.last_arm_cmd_wall, now_wall),
            }
        )
        self.samples.append(sample)
        self.csv_writer.writerow({key: self._csv_value(sample.get(key, math.nan)) for key in self.headers})

        if int(t_s * self.log_rate_hz) % max(1, int(self.log_rate_hz * 2.0)) == 0:
            self.csv_file.flush()

    def _csv_value(self, value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            return f"{value:.6f}" if math.isfinite(value) else ""
        return value

    def _select_analysis_samples(self) -> tuple[str, list[dict[str, Any]]]:
        external_samples = [s for s in self.samples if bool(s.get("controller_enabled", False))]
        if external_samples:
            first_enable_t = finite(external_samples[0].get("t_s", math.nan))
            settled_start_t = first_enable_t + self.analysis_settle_after_enable_s
            settled_samples = [
                s
                for s in external_samples
                if finite(s.get("t_s", math.nan)) >= settled_start_t
            ]
            if len(settled_samples) >= max(10, int(self.log_rate_hz)):
                return "external_enabled_settled", settled_samples
            return "external_enabled", external_samples

        flight_samples = [
            s
            for s in self.samples
            if bool(s.get("armed", False)) or finite(s.get("alt_m", math.nan)) > 0.5
        ]
        if flight_samples:
            return "armed_or_airborne", flight_samples

        return "all_samples", list(self.samples)

    def _sample_duration(self, samples: list[dict[str, Any]]) -> float:
        if len(samples) < 2:
            return 0.0
        first_t = finite(samples[0].get("t_s", math.nan))
        last_t = finite(samples[-1].get("t_s", math.nan))
        if not (math.isfinite(first_t) and math.isfinite(last_t)):
            return 0.0
        return max(0.0, last_t - first_t)

    def _values(self, samples: list[dict[str, Any]], key: str) -> list[float]:
        return [finite(s.get(key, math.nan)) for s in samples]

    def _vector_norms(self, samples: list[dict[str, Any]], keys: tuple[str, str, str]) -> list[float]:
        norms: list[float] = []
        for sample in samples:
            vals = [finite(sample.get(key, math.nan)) for key in keys]
            if all(math.isfinite(v) for v in vals):
                norms.append(math.sqrt(sum(v * v for v in vals)))
        return norms

    def _xy_drift(self, samples: list[dict[str, Any]]) -> list[float]:
        base_x = math.nan
        base_y = math.nan
        drifts: list[float] = []
        for sample in samples:
            x = finite(sample.get("x_m", math.nan))
            y = finite(sample.get("y_m", math.nan))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not (math.isfinite(base_x) and math.isfinite(base_y)):
                base_x = x
                base_y = y
            drifts.append(math.hypot(x - base_x, y - base_y))
        return drifts

    def _max_joint_span(self, samples: list[dict[str, Any]], prefix: str) -> float:
        spans = []
        for idx in range(1, N_JOINTS + 1):
            values = clean_values(self._values(samples, f"{prefix}_{idx}"))
            if values:
                spans.append(max(values) - min(values))
        return maximum(spans)

    def _write_summary(self) -> None:
        if not self.samples:
            summary = {
                "case_name": self.case_name,
                "samples": 0,
                "verdict": "NO_SAMPLES",
            }
            self.summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self.summary_md_path.write_text("# UAM Verification Summary\n\nNo samples were recorded.\n", encoding="utf-8")
            return

        phase_name, analysis = self._select_analysis_samples()
        first_t = finite(self.samples[0].get("t_s", 0.0))
        last_t = finite(self.samples[-1].get("t_s", 0.0))
        duration_s = max(0.0, last_t - first_t)
        enabled_samples = [s for s in self.samples if bool(s.get("controller_enabled", False))]
        first_enable_t = finite(enabled_samples[0].get("t_s", math.nan)) if enabled_samples else math.nan
        last_enable_t = finite(enabled_samples[-1].get("t_s", math.nan)) if enabled_samples else math.nan
        external_enabled_duration_s = self._sample_duration(enabled_samples)
        analysis_start_t = finite(analysis[0].get("t_s", math.nan)) if analysis else math.nan
        analysis_end_t = finite(analysis[-1].get("t_s", math.nan)) if analysis else math.nan
        analysis_duration_s = self._sample_duration(analysis)
        enabled_fraction = len(enabled_samples) / len(self.samples)

        alt = self._values(analysis, "alt_m")
        alt_err = [v - self.target_alt_m for v in clean_values(alt)]
        xy_drift = self._xy_drift(analysis)
        roll_abs = [abs(v) for v in self._values(analysis, "roll_deg")]
        pitch_abs = [abs(v) for v in self._values(analysis, "pitch_deg")]
        roll_pitch_abs = [max(r, p) for r, p in zip(roll_abs, pitch_abs) if math.isfinite(r) and math.isfinite(p)]
        vz_abs = [abs(v) for v in self._values(analysis, "vz_mps")]
        e_omega_norm = self._vector_norms(
            analysis, ("dbg_e_omega_x", "dbg_e_omega_y", "dbg_e_omega_z")
        )
        torque_norm = self._vector_norms(analysis, ("torque_x", "torque_y", "torque_z"))
        n_hat_norm = self._vector_norms(
            analysis, ("dbg_n_hat_x", "dbg_n_hat_y", "dbg_n_hat_z")
        )
        tau_arm_ff_norm = self._vector_norms(
            analysis, ("dbg_tau_arm_ff_x", "dbg_tau_arm_ff_y", "dbg_tau_arm_ff_z")
        )
        tau_arm_disturbance_norm = self._vector_norms(
            analysis,
            (
                "dbg_tau_arm_disturbance_x",
                "dbg_tau_arm_disturbance_y",
                "dbg_tau_arm_disturbance_z",
            ),
        )
        tau_residual_norm: list[float] = []
        ff_disturbance_dot: list[float] = []
        for sample in analysis:
            ff = [
                finite(sample.get("dbg_tau_arm_ff_x", math.nan)),
                finite(sample.get("dbg_tau_arm_ff_y", math.nan)),
                finite(sample.get("dbg_tau_arm_ff_z", math.nan)),
            ]
            disturbance = [
                finite(sample.get("dbg_tau_arm_disturbance_x", math.nan)),
                finite(sample.get("dbg_tau_arm_disturbance_y", math.nan)),
                finite(sample.get("dbg_tau_arm_disturbance_z", math.nan)),
            ]
            if all(math.isfinite(v) for v in ff + disturbance):
                residual = [disturbance[idx] - ff[idx] for idx in range(N_AXES)]
                tau_residual_norm.append(math.sqrt(sum(v * v for v in residual)))
                ff_disturbance_dot.append(sum(disturbance[idx] * ff[idx] for idx in range(N_AXES)))
        joint_cmd_norm = self._vector_norms(
            analysis, tuple(f"joint_cmd_{idx}" for idx in range(1, N_JOINTS + 1))
        )
        joint_pos_norm = self._vector_norms(
            analysis, tuple(f"joint_pos_{idx}" for idx in range(1, N_JOINTS + 1))
        )
        joint_cmd_max = maximum(joint_cmd_norm)
        joint_pos_max = maximum(joint_pos_norm)
        joint_cmd_span_max = self._max_joint_span(analysis, "joint_cmd")
        joint_pos_span_max = self._max_joint_span(analysis, "joint_pos")
        arm_command_seen = math.isfinite(joint_cmd_max) and joint_cmd_max > 0.05
        arm_motion_detected = (
            math.isfinite(joint_pos_max)
            and joint_pos_max > 0.03
            and math.isfinite(joint_pos_span_max)
            and joint_pos_span_max > 0.03
        )

        summary = {
            "case_name": self.case_name,
            "created_at": self.start_iso,
            "output_directory": str(self.run_dir),
            "analysis_phase": phase_name,
            "samples": len(self.samples),
            "analysis_samples": len(analysis),
            "duration_s": duration_s,
            "target_alt_m": self.target_alt_m,
            "analysis_settle_after_enable_s": self.analysis_settle_after_enable_s,
            "analysis_start_s": analysis_start_t,
            "analysis_end_s": analysis_end_t,
            "analysis_duration_s": analysis_duration_s,
            "external_enabled_fraction": enabled_fraction,
            "external_enabled_duration_s": external_enabled_duration_s,
            "time_first_external_enable_s": first_enable_t,
            "time_last_external_enable_s": last_enable_t,
            "altitude": {
                "mean_m": mean(alt),
                "min_m": minimum(alt),
                "max_m": maximum(alt),
                "std_m": stddev(alt),
                "rmse_error_m": rms(alt_err),
            },
            "xy_drift": {
                "mean_m": mean(xy_drift),
                "max_m": maximum(xy_drift),
                "final_m": xy_drift[-1] if xy_drift else math.nan,
            },
            "attitude": {
                "roll_rms_deg": rms(self._values(analysis, "roll_deg")),
                "pitch_rms_deg": rms(self._values(analysis, "pitch_deg")),
                "roll_abs_max_deg": maximum(roll_abs),
                "pitch_abs_max_deg": maximum(pitch_abs),
                "roll_pitch_abs_max_deg": maximum(roll_pitch_abs),
            },
            "velocity": {
                "vertical_speed_abs_mean_mps": mean(vz_abs),
                "vertical_speed_abs_max_mps": maximum(vz_abs),
            },
            "rate_tracking": {
                "e_omega_norm_rms_radps": rms(e_omega_norm),
                "e_omega_norm_max_radps": maximum(e_omega_norm),
            },
            "external_output": {
                "torque_norm_rms": rms(torque_norm),
                "torque_norm_max": maximum(torque_norm),
                "n_hat_norm_rms": rms(n_hat_norm),
                "n_hat_norm_max": maximum(n_hat_norm),
                "tau_arm_ff_norm_rms_nm": rms(tau_arm_ff_norm),
                "tau_arm_ff_norm_max_nm": maximum(tau_arm_ff_norm),
                "tau_arm_disturbance_norm_rms_nm": rms(tau_arm_disturbance_norm),
                "tau_arm_disturbance_norm_max_nm": maximum(tau_arm_disturbance_norm),
                "tau_residual_norm_rms_nm": rms(tau_residual_norm),
                "tau_residual_norm_max_nm": maximum(tau_residual_norm),
                "ff_disturbance_dot_mean": mean(ff_disturbance_dot),
                "arm_ff_enabled_fraction": (
                    sum(1 for s in analysis if bool(s.get("dbg_arm_ff_enabled", False))) / len(analysis)
                    if analysis else math.nan
                ),
                "arm_ff_fresh_fraction": (
                    sum(1 for s in analysis if bool(s.get("dbg_arm_ff_fresh", False))) / len(analysis)
                    if analysis else math.nan
                ),
                "arm_virtual_disturbance_enabled_fraction": (
                    sum(1 for s in analysis if bool(s.get("dbg_arm_virtual_disturbance_enabled", False))) / len(analysis)
                    if analysis else math.nan
                ),
            },
            "arm_motion": {
                "joint_cmd_norm_rms_rad": rms(joint_cmd_norm),
                "joint_cmd_norm_max_rad": joint_cmd_max,
                "joint_cmd_span_max_rad": joint_cmd_span_max,
                "joint_pos_norm_rms_rad": rms(joint_pos_norm),
                "joint_pos_norm_max_rad": joint_pos_max,
                "joint_pos_span_max_rad": joint_pos_span_max,
                "arm_command_seen": arm_command_seen,
                "arm_motion_detected": arm_motion_detected,
            },
            "failure_flags": {
                "roll_or_pitch_gt_35_deg": maximum(roll_pitch_abs) > 35.0
                if math.isfinite(maximum(roll_pitch_abs))
                else False,
                "altitude_outside_1_to_3m": (
                    minimum(alt) < 1.0 or maximum(alt) > 3.0
                    if math.isfinite(minimum(alt)) and math.isfinite(maximum(alt))
                    else False
                ),
                "xy_drift_gt_1m": maximum(xy_drift) > 1.0
                if math.isfinite(maximum(xy_drift))
                else False,
                "arm_command_without_motion": arm_command_seen and not arm_motion_detected,
            },
        }
        summary["verdict"] = (
            "FAIL"
            if any(summary["failure_flags"].values()) or len(analysis) < max(10, int(self.log_rate_hz))
            else "PASS_CANDIDATE"
        )

        self.summary_json_path.write_text(json.dumps(json_clean(summary), indent=2), encoding="utf-8")
        self.summary_md_path.write_text(self._summary_markdown(summary), encoding="utf-8")
        self.get_logger().info(
            "Experiment summary written | verdict=%s | summary=%s"
            % (summary["verdict"], self.summary_md_path)
        )

    def _summary_markdown(self, summary: dict[str, Any]) -> str:
        altitude = summary["altitude"]
        xy = summary["xy_drift"]
        attitude = summary["attitude"]
        rate = summary["rate_tracking"]
        ext = summary["external_output"]
        arm = summary["arm_motion"]
        flags = summary["failure_flags"]

        lines = [
            "# UAM Verification Summary",
            "",
            f"- Case: `{summary['case_name']}`",
            f"- Verdict: `{summary['verdict']}`",
            f"- Analysis phase: `{summary['analysis_phase']}`",
            f"- Duration: {fmt(summary['duration_s'])} s",
            f"- Samples: {summary['samples']} total, {summary['analysis_samples']} analyzed",
            f"- Analysis window: {fmt(summary['analysis_start_s'])}-{fmt(summary['analysis_end_s'])} s ({fmt(summary['analysis_duration_s'])} s)",
            f"- External enabled fraction: {fmt(summary['external_enabled_fraction'], 4)}",
            f"- External enabled duration: {fmt(summary['external_enabled_duration_s'])} s",
            f"- First external enable: {fmt(summary['time_first_external_enable_s'])} s",
            "",
            "## Hover Metrics",
            "",
            f"- Altitude mean/std: {fmt(altitude['mean_m'])} / {fmt(altitude['std_m'])} m",
            f"- Altitude min/max: {fmt(altitude['min_m'])} / {fmt(altitude['max_m'])} m",
            f"- Altitude RMSE vs target: {fmt(altitude['rmse_error_m'])} m",
            f"- XY drift mean/max/final: {fmt(xy['mean_m'])} / {fmt(xy['max_m'])} / {fmt(xy['final_m'])} m",
            f"- Roll RMS/max abs: {fmt(attitude['roll_rms_deg'])} / {fmt(attitude['roll_abs_max_deg'])} deg",
            f"- Pitch RMS/max abs: {fmt(attitude['pitch_rms_deg'])} / {fmt(attitude['pitch_abs_max_deg'])} deg",
            "",
            "## Controller Metrics",
            "",
            f"- Rate error norm RMS/max: {fmt(rate['e_omega_norm_rms_radps'])} / {fmt(rate['e_omega_norm_max_radps'])} rad/s",
            f"- Torque norm RMS/max: {fmt(ext['torque_norm_rms'])} / {fmt(ext['torque_norm_max'])}",
            f"- RBFNN n_hat norm RMS/max: {fmt(ext['n_hat_norm_rms'])} / {fmt(ext['n_hat_norm_max'])}",
            f"- Arm FF torque norm RMS/max: {fmt(ext['tau_arm_ff_norm_rms_nm'])} / {fmt(ext['tau_arm_ff_norm_max_nm'])} Nm",
            f"- Virtual arm disturbance norm RMS/max: {fmt(ext['tau_arm_disturbance_norm_rms_nm'])} / {fmt(ext['tau_arm_disturbance_norm_max_nm'])} Nm",
            f"- Arm residual torque norm RMS/max: {fmt(ext['tau_residual_norm_rms_nm'])} / {fmt(ext['tau_residual_norm_max_nm'])} Nm",
            f"- Arm FF/disturbance dot mean: {fmt(ext['ff_disturbance_dot_mean'], 6)}",
            f"- Arm FF enabled/fresh fraction: {fmt(ext['arm_ff_enabled_fraction'], 4)} / {fmt(ext['arm_ff_fresh_fraction'], 4)}",
            f"- Virtual disturbance enabled fraction: {fmt(ext['arm_virtual_disturbance_enabled_fraction'], 4)}",
            "",
            "## Arm Motion",
            "",
            f"- Motion detected: `{arm['arm_motion_detected']}`",
            f"- Command seen: `{arm['arm_command_seen']}`",
            f"- Joint command norm RMS/max: {fmt(arm['joint_cmd_norm_rms_rad'])} / {fmt(arm['joint_cmd_norm_max_rad'])} rad",
            f"- Joint command max span: {fmt(arm['joint_cmd_span_max_rad'])} rad",
            f"- Joint actual norm RMS/max: {fmt(arm['joint_pos_norm_rms_rad'])} / {fmt(arm['joint_pos_norm_max_rad'])} rad",
            f"- Joint actual max span: {fmt(arm['joint_pos_span_max_rad'])} rad",
            "",
            "## Failure Flags",
            "",
        ]
        lines.extend([f"- {key}: `{value}`" for key, value in flags.items()])
        lines.append("")
        lines.append(f"Timeseries CSV: `{self.csv_path}`")
        return "\n".join(lines) + "\n"

    def destroy_node(self) -> bool:
        if not self._closed:
            self._closed = True
            try:
                self.csv_file.flush()
                self._write_summary()
            finally:
                self.csv_file.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RBFNNDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
