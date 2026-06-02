#!/usr/bin/env python3
"""
arm_trajectory_generator.py
----------------------------
Tạo các chuyển động cánh tay đa dạng để thu thập dữ liệu huấn luyện LSTM.

Các pattern chuyển động:
  1. Sin wave  — dao động sin trên từng khớp với tần số/biên độ khác nhau
  2. Step      — nhảy bậc giữa các vị trí ngẫu nhiên
  3. Chirp     — tần số tăng dần (quét tần số)
  4. Random    — vị trí ngẫu nhiên mượt (spline nội suy)
  5. Slow step — chuyển mượt rồi giữ lâu để PX4 có thời gian kéo UAV về vị trí
  6. Combined  — kết hợp tất cả

Cách dùng:
  # Chạy tất cả pattern, mỗi cái 60s (tổng ~5 phút)
  ros2 run uam_controller arm_trajectory_generator.py

  # Chỉ chạy 1 pattern cụ thể
    

  # Biên độ nhỏ hơn (an toàn hơn cho UAV đang bay)
  ros2 run uam_controller arm_trajectory_generator.py --amplitude 0.3
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import argparse
import sys
import time


class ArmTrajectoryGenerator(Node):
    """Tạo trajectory đa dạng cho cánh tay robot."""

    N_JOINTS = 6

    # Giới hạn an toàn cho từng khớp (radian) — giảm so với SDF để an toàn khi bay
    JOINT_LIMITS = [
        (-1.5, 1.5),    # Joint 1 (base rotation)
        (-1.2, 1.2),    # Joint 2 (shoulder)
        (-1.2, 1.2),    # Joint 3 (elbow)
        (-1.5, 1.5),    # Joint 4 (wrist 1)
        (-1.2, 1.2),    # Joint 5 (wrist 2)
        (-1.5, 1.5),    # Joint 6 (wrist 3)
    ]

    def __init__(self, pattern, duration, amplitude, rate, step_hold_time, transition_time):
        super().__init__('arm_trajectory_generator')

        self.pattern = pattern
        self.duration = duration
        self.amplitude = amplitude
        self.rate_hz = rate
        self.dt = 1.0 / rate
        self._step_hold_time = max(float(step_hold_time), self.dt)
        self._transition_time = max(float(transition_time), self.dt)

        # Publisher
        self.pub = self.create_publisher(
            JointState,
            '/arm_controller/joint_trajectory_plan',
            10,
        )

        self.t0 = time.time()
        self.timer = self.create_timer(self.dt, self._tick)
        self._command_count = 0
        self._last_command_log = 0.0

        # Cho pattern step: thời gian giữ mỗi bước
        self._step_targets = self._random_positions()
        self._step_last_switch = 0.0

        # Cho pattern slow_step: chuyển mượt rồi giữ lâu
        self._slow_segment_idx = -1
        self._slow_start_target = np.zeros(self.N_JOINTS)
        self._slow_end_target = np.zeros(self.N_JOINTS)

        # Cho pattern random smooth: tạo waypoints
        self._rand_waypoints = [self._random_positions() for _ in range(20)]
        self._rand_segment_duration = duration / len(self._rand_waypoints)

        # Cho pattern combined: danh sách pattern con
        self._combined_patterns = ['sin', 'step', 'chirp', 'random']
        self._combined_segment = duration / len(self._combined_patterns)

        self.get_logger().info(
            f'═══ Arm Trajectory Generator ═══\n'
            f'  Pattern   : {pattern}\n'
            f'  Duration  : {duration}s\n'
            f'  Amplitude : {amplitude}\n'
            f'  Rate      : {rate} Hz\n'
            f'  Step hold : {self._step_hold_time}s\n'
            f'  Transition: {self._transition_time}s\n'
            f'  Bắt đầu tạo chuyển động...'
        )

    # ─────────────────────────────────────────
    #  Các pattern chuyển động
    # ─────────────────────────────────────────

    def _gen_sin(self, t):
        """Sin wave với tần số và pha khác nhau cho mỗi khớp."""
        pos = np.zeros(self.N_JOINTS)
        # Mỗi khớp có tần số khác nhau để tạo chuyển động đa dạng
        freqs = [0.3, 0.5, 0.7, 0.4, 0.6, 0.35]
        phases = [0, np.pi/3, np.pi/2, np.pi/4, 2*np.pi/3, np.pi/6]
        for i in range(self.N_JOINTS):
            pos[i] = self.amplitude * np.sin(2 * np.pi * freqs[i] * t + phases[i])
        return pos

    def _gen_step(self, t):
        """Nhảy bậc giữa các vị trí ngẫu nhiên, giữ mỗi vị trí vài giây."""
        if t - self._step_last_switch >= self._step_hold_time:
            self._step_targets = self._random_positions()
            self._step_last_switch = t
            self.get_logger().info(
                f'  [Step] → [{", ".join(f"{v:.2f}" for v in self._step_targets)}]'
            )
        return self._step_targets

    def _gen_slow_step(self, t):
        """Chuyển mượt sang tư thế mới rồi giữ lâu để PX4 hồi vị trí."""
        cycle = max(self._step_hold_time, self.dt)
        transition = min(max(self._transition_time, self.dt), cycle)
        seg_idx = int(t / cycle)
        local_t = t - seg_idx * cycle

        if seg_idx != self._slow_segment_idx:
            self._slow_segment_idx = seg_idx
            self._slow_start_target = self._slow_end_target.copy()
            self._slow_end_target = self._random_positions()
            self.get_logger().info(
                f'  [SlowStep] transition={transition:.1f}s hold={cycle-transition:.1f}s '
                f'→ [{", ".join(f"{v:.2f}" for v in self._slow_end_target)}]'
            )

        if local_t < transition:
            alpha = np.clip(local_t / transition, 0.0, 1.0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            return self._slow_start_target + alpha * (self._slow_end_target - self._slow_start_target)

        return self._slow_end_target

    def _gen_chirp(self, t):
        """Chirp — tần số tăng dần từ 0.1 Hz đến 2 Hz."""
        pos = np.zeros(self.N_JOINTS)
        f0, f1 = 0.1, 2.0  # tần số bắt đầu và kết thúc
        T = self.duration
        # Tần số tức thời: f(t) = f0 + (f1-f0) * t / T
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / T)
        for i in range(self.N_JOINTS):
            amp = self.amplitude * (0.5 + 0.5 * np.sin(0.3 * i))  # biên độ khác nhau
            offset = i * np.pi / self.N_JOINTS  # pha lệch
            pos[i] = amp * np.sin(phase + offset)
        return pos

    def _gen_random(self, t):
        """Random smooth — nội suy tuyến tính giữa các waypoint ngẫu nhiên."""
        seg_idx = int(t / self._rand_segment_duration)
        seg_idx = min(seg_idx, len(self._rand_waypoints) - 2)
        alpha = (t - seg_idx * self._rand_segment_duration) / self._rand_segment_duration
        alpha = np.clip(alpha, 0, 1)
        # Nội suy mượt (smoothstep)
        alpha = alpha * alpha * (3 - 2 * alpha)
        p0 = self._rand_waypoints[seg_idx]
        p1 = self._rand_waypoints[seg_idx + 1]
        return p0 + alpha * (p1 - p0)

    def _gen_combined(self, t):
        """Kết hợp tất cả pattern theo thứ tự."""
        seg_idx = int(t / self._combined_segment)
        seg_idx = min(seg_idx, len(self._combined_patterns) - 1)
        local_t = t - seg_idx * self._combined_segment
        pat = self._combined_patterns[seg_idx]
        return self._dispatch(pat, local_t)

    # ─────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────

    def _random_positions(self):
        """Tạo vị trí ngẫu nhiên trong giới hạn an toàn."""
        pos = np.zeros(self.N_JOINTS)
        for i in range(self.N_JOINTS):
            lo, hi = self.JOINT_LIMITS[i]
            # Thu hẹp thêm theo amplitude
            lo_s = lo * self.amplitude / 1.5
            hi_s = hi * self.amplitude / 1.5
            pos[i] = np.random.uniform(lo_s, hi_s)
        return pos

    def _clamp(self, pos):
        """Giới hạn vị trí trong phạm vi an toàn."""
        for i in range(self.N_JOINTS):
            lo, hi = self.JOINT_LIMITS[i]
            pos[i] = np.clip(pos[i], lo, hi)
        return pos

    def _dispatch(self, pattern, t):
        """Chọn hàm tạo trajectory theo tên pattern."""
        generators = {
            'sin': self._gen_sin,
            'step': self._gen_step,
            'slow_step': self._gen_slow_step,
            'chirp': self._gen_chirp,
            'random': self._gen_random,
            'combined': self._gen_combined,
        }
        return generators[pattern](t)

    # ─────────────────────────────────────────
    #  Timer callback
    # ─────────────────────────────────────────

    def _tick(self):
        t = time.time() - self.t0

        if t >= self.duration:
            self.get_logger().info(
                f'═══ Hoàn thành pattern "{self.pattern}" sau {self.duration}s ═══'
            )
            # Về vị trí home (0, 0, 0, 0, 0, 0)
            self._publish_positions(np.zeros(self.N_JOINTS))
            self.timer.cancel()
            raise SystemExit(0)

        pos = self._dispatch(self.pattern, t)
        pos = self._clamp(pos)
        self._publish_positions(pos)
        self._log_command_periodically(t, pos)

    def _publish_positions(self, pos):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'Joint_{i+1}' for i in range(self.N_JOINTS)]
        msg.position = [float(p) for p in pos]
        self.pub.publish(msg)
        self._command_count += 1

    def _log_command_periodically(self, t, pos):
        now = time.time()
        if self._command_count == 1 or now - self._last_command_log >= 2.0:
            self._last_command_log = now
            max_abs = float(np.max(np.abs(pos))) if len(pos) else 0.0
            self.get_logger().info(
                f'  [Cmd #{self._command_count:04d}] '
                f't={t:6.1f}s pattern={self.pattern} max|q|={max_abs:.3f} rad '
                f'→ [{", ".join(f"{v:.2f}" for v in pos)}]'
            )


def main():
    parser = argparse.ArgumentParser(
        description='Tạo chuyển động cánh tay cho thu thập dữ liệu LSTM'
    )
    parser.add_argument(
        '--pattern', type=str, default='combined',
        choices=['sin', 'step', 'slow_step', 'chirp', 'random', 'combined'],
        help='Loại chuyển động (default: combined = tất cả)'
    )
    parser.add_argument(
        '--duration', type=float, default=300.0,
        help='Thời gian chạy mỗi pattern (giây, default: 300 = 5 phút)'
    )
    parser.add_argument(
        '--amplitude', type=float, default=0.6,
        help='Biên độ chuyển động (0.1-1.5, default: 0.6 — an toàn cho UAV bay)'
    )
    parser.add_argument(
        '--rate', type=int, default=10,
        help='Tần số publish (Hz, default: 10)'
    )
    parser.add_argument(
        '--step-hold-time', type=float, default=3.0,
        help='Thời gian mỗi chu kỳ step/slow_step (giây, default: 3)'
    )
    parser.add_argument(
        '--transition-time', type=float, default=1.0,
        help='Thời gian chuyển mượt cho slow_step (giây, default: 1)'
    )

    # Parse riêng để tránh xung đột với ROS2 args
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = ArmTrajectoryGenerator(
        pattern=args.pattern,
        duration=args.duration,
        amplitude=args.amplitude,
        rate=args.rate,
        step_hold_time=args.step_hold_time,
        transition_time=args.transition_time,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            try:
                node.get_logger().info('Trajectory generator dừng.')
            except Exception:
                pass
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
