#!/usr/bin/env python3
"""
arm_dynamics_node.py
--------------------
Node Python tính toán lực và mô-men xoắn tương tác từ cánh tay 6-DoF
lên thân UAV bằng thuật toán đệ quy Newton-Euler (RNE).

Kết quả được xuất sang /arm/interaction_wrench để node C++ tham khảo
và so sánh với đầu ra RBFNN (giám sát chất lượng học thích nghi).

Tần số: 50 Hz
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import WrenchStamped
import numpy as np
from typing import List, Optional


def _finite_array(values, size, default=0.0) -> np.ndarray:
    arr = np.full(size, default, dtype=float)
    for idx, value in enumerate(list(values)[:size]):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if not np.isfinite(value):
            value = default
        arr[idx] = value
    return arr


def _quat_to_rot_matrix(q) -> np.ndarray:
    """Return body-to-world rotation for PX4 quaternion [w, x, y, z]."""
    qw, qx, qy, qz = _finite_array(q, 4)
    norm = np.linalg.norm([qw, qx, qy, qz])
    if norm < 1e-9:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ])


def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(3)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# ============================================================
#  Cấu trúc thông số hình học một Link (Denavit-Hartenberg)
# ============================================================
class DHLink:
    """
    Thông số Denavit-Hartenberg chuẩn cho một khớp quay.
    """
    def __init__(self,
                 alpha: float, a: float,
                 d: float, mass: float,
                 com: np.ndarray,
                 inertia: np.ndarray):
        """
        alpha : góc xoắn DH [rad]
        a     : khoảng cách DH [m]
        d     : dịch chuyển DH [m]
        mass  : khối lượng link [kg]
        com   : trọng tâm link trong frame link [m], shape (3,)
        inertia: tensor quán tính 3×3 tại CoM [kg·m²]
        """
        self.alpha   = alpha
        self.a       = a
        self.d       = d
        self.mass    = mass
        self.com     = np.array(com).reshape(3)
        self.inertia = np.array(inertia).reshape(3, 3)


# ============================================================
#  Lớp tính toán đệ quy Newton-Euler
# ============================================================
class RecursiveNewtonEuler:
    """
    Thuật toán đệ quy Newton-Euler tính lực/mô-men tương tác.

    Tham chiếu: Siciliano et al., "Robotics: Modelling, Planning and Control"
    """

    GRAVITY = np.array([0.0, 0.0, -9.81])  # Trọng lực trong inertial frame

    def __init__(self, links: List[DHLink]):
        self.links  = links
        self.n_dof  = len(links)

    def dh_rotation(self, alpha: float, theta: float) -> np.ndarray:
        """Ma trận xoay DH: R_i-1_i"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct,        -st,       0.0],
            [st * ca,    ct * ca,  -sa],
            [st * sa,    ct * sa,   ca]
        ])

    def dh_transform(self, alpha: float, a: float,
                      d: float, theta: float) -> np.ndarray:
        """Ma trận biến đổi đồng nhất 4×4 DH"""
        R = self.dh_rotation(alpha, theta)
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3]   = a
        T[1, 3]   = -d * np.sin(alpha)
        T[2, 3]   =  d * np.cos(alpha)
        return T

    def compute_interaction_wrench(self,
                                    q:  np.ndarray,
                                    dq: np.ndarray,
                                    ddq: np.ndarray,
                                    base_acc: Optional[np.ndarray] = None,
                                    base_omega: Optional[np.ndarray] = None,
                                    base_alpha: Optional[np.ndarray] = None
                                    ) -> np.ndarray:
        """
        Tính lực và mô-men tại khớp 0 (điểm gắn vào UAV).

        Tham số:
            q    : góc khớp [rad]   shape (n,)
            dq   : vận tốc khớp     shape (n,)
            ddq  : gia tốc khớp     shape (n,)
            base_acc : gia tốc thân UAV [m/s²], mặc định = trọng lực
            base_omega : vận tốc góc thân UAV [rad/s]
            base_alpha : gia tốc góc thân UAV [rad/s²]

        Trả về:
            wrench_0 : [fx, fy, fz, nx, ny, nz] tại khớp 0
        """
        n = self.n_dof
        if base_acc is None:
            base_acc = -self.GRAVITY  # gia tốc trọng trường lên trên
        if base_omega is None:
            base_omega = np.zeros(3)
        if base_alpha is None:
            base_alpha = np.zeros(3)

        # ── Đệ quy thuận (Forward Recursion) ──
        # Vận tốc góc ω, gia tốc góc α, gia tốc tuyến tính tại từng link
        omega_prev  = np.array(base_omega, dtype=float).reshape(3)
        alpha_prev  = np.array(base_alpha, dtype=float).reshape(3)
        a_prev      = base_acc.copy()    # gia tốc tại gốc

        omegas  = []
        alphas  = []
        a_links = []
        a_coms  = []
        R_mats  = []

        for i in range(n):
            lnk = self.links[i]
            R   = self.dh_rotation(lnk.alpha, q[i])   # R_i-1_i
            R_T = R.T                                   # R_i_i-1

            z_prev = np.array([0.0, 0.0, 1.0])          # trục z trong frame i-1

            # Vận tốc góc: ω_i = R^T * ω_i-1 + dq_i * z_i
            omega_i = R_T @ omega_prev + dq[i] * np.array([0, 0, 1])

            # Gia tốc góc: α_i = R^T * α_i-1 + ddq_i * z_i + ω_i × (dq_i * z_i)
            alpha_i = (R_T @ alpha_prev
                       + ddq[i] * np.array([0, 0, 1])
                       + np.cross(omega_i, dq[i] * np.array([0, 0, 1])))

            # Vector vị trí link trong frame i
            p_i = np.array([lnk.a, -lnk.d * np.sin(lnk.alpha),
                             lnk.d * np.cos(lnk.alpha)])

            # Gia tốc tuyến tính origin of frame i
            a_i = (R_T @ a_prev
                   + np.cross(alpha_i, p_i)
                   + np.cross(omega_i, np.cross(omega_i, p_i)))

            # Gia tốc tuyến tính tại CoM
            a_com_i = (a_i
                       + np.cross(alpha_i, lnk.com)
                       + np.cross(omega_i, np.cross(omega_i, lnk.com)))

            omegas.append(omega_i)
            alphas.append(alpha_i)
            a_links.append(a_i)
            a_coms.append(a_com_i)
            R_mats.append(R)

            omega_prev = omega_i
            alpha_prev = alpha_i
            a_prev     = a_i

        # ── Đệ quy lùi (Backward Recursion) ──
        # Khởi tạo lực và mô-men tại end-effector = 0 (không tải trọng)
        f_next = np.zeros(3)
        n_next = np.zeros(3)

        for i in range(n - 1, -1, -1):
            lnk = self.links[i]
            R   = R_mats[i]

            # Lực quán tính Newton
            F_i = lnk.mass * a_coms[i]

            # Mô-men quán tính Euler
            N_i = (lnk.inertia @ alphas[i]
                   + np.cross(omegas[i], lnk.inertia @ omegas[i]))

            # Vector vị trí link trong frame i
            p_i = np.array([lnk.a, -lnk.d * np.sin(lnk.alpha),
                             lnk.d * np.cos(lnk.alpha)])

            # Đệ quy lùi lực: f_i = R_i+1 * f_i+1 + F_i
            if i < n - 1:
                R_next = R_mats[i + 1]
                f_i    = R_next @ f_next + F_i
            else:
                f_i    = f_next + F_i

            # Đệ quy lùi mô-men: n_i = R_i+1*n_i+1 + p_i × f_i + com_i × F_i + N_i
            if i < n - 1:
                R_next = R_mats[i + 1]
                n_i    = (R_next @ n_next
                          + np.cross(p_i, f_i)
                          + np.cross(lnk.com, F_i)
                          + N_i)
            else:
                n_i    = (np.cross(p_i, f_i)
                          + np.cross(lnk.com, F_i)
                          + N_i)

            f_next = f_i
            n_next = n_i

        # f_next, n_next tại khớp 0 = tương tác lên UAV
        return np.concatenate([f_next, n_next])


class SDFSerialArmDynamics:
    """
    Lightweight serial-arm wrench model using joint origins, joint axes, CoM,
    mass, and inertia extracted from the Gazebo SDF at the zero pose.

    This avoids forcing the imported CAD arm into a simplified DH convention
    when the SDF joint axes are mixed Y/Z axes.
    """

    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(self,
                 joint_origins: np.ndarray,
                 joint_axes: np.ndarray,
                 link_masses: np.ndarray,
                 link_coms: np.ndarray,
                 link_inertias: np.ndarray):
        self.joint_origins = np.array(joint_origins, dtype=float).reshape(-1, 3)
        self.joint_axes = np.array(joint_axes, dtype=float).reshape(-1, 3)
        self.link_masses = np.array(link_masses, dtype=float).reshape(-1)
        self.link_coms = np.array(link_coms, dtype=float).reshape(-1, 3)
        self.link_inertias = np.array(link_inertias, dtype=float).reshape(-1, 3, 3)
        self.n_dof = len(self.link_masses)

        if not (
            len(self.joint_origins) == len(self.joint_axes) == len(self.link_coms) == self.n_dof
        ):
            raise ValueError("SDF serial arm parameter lengths must all match")

        for i in range(self.n_dof):
            norm = np.linalg.norm(self.joint_axes[i])
            if norm < 1e-9:
                raise ValueError(f"joint_axis_xyz[{i}] is zero")
            self.joint_axes[i] = self.joint_axes[i] / norm

    def compute_interaction_wrench(self,
                                    q: np.ndarray,
                                    dq: np.ndarray,
                                    ddq: np.ndarray,
                                    base_acc: Optional[np.ndarray] = None,
                                    base_omega: Optional[np.ndarray] = None,
                                    base_alpha: Optional[np.ndarray] = None
                                    ) -> np.ndarray:
        if base_acc is None:
            base_acc = -self.GRAVITY
        if base_omega is None:
            base_omega = np.zeros(3)
        if base_alpha is None:
            base_alpha = np.zeros(3)

        parent_R = np.eye(3)
        parent_omega = np.array(base_omega, dtype=float).reshape(3)
        parent_alpha = np.array(base_alpha, dtype=float).reshape(3)

        joint_origin = self.joint_origins[0].copy()
        joint_acc = (
            np.array(base_acc, dtype=float).reshape(3)
            + np.cross(parent_alpha, joint_origin)
            + np.cross(parent_omega, np.cross(parent_omega, joint_origin))
        )

        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        for i in range(self.n_dof):
            axis = parent_R @ self.joint_axes[i]
            child_R = parent_R @ _axis_angle_to_rot(self.joint_axes[i], q[i])

            omega = parent_omega + axis * dq[i]
            alpha = (
                parent_alpha
                + axis * ddq[i]
                + np.cross(parent_omega, axis * dq[i])
            )

            r_com = child_R @ self.link_coms[i]
            com_acc = (
                joint_acc
                + np.cross(alpha, r_com)
                + np.cross(omega, np.cross(omega, r_com))
            )
            inertia_world = child_R @ self.link_inertias[i] @ child_R.T

            force = self.link_masses[i] * com_acc
            torque = (
                np.cross(joint_origin + r_com, force)
                + inertia_world @ alpha
                + np.cross(omega, inertia_world @ omega)
            )

            total_force += force
            total_torque += torque

            if i < self.n_dof - 1:
                r_next = child_R @ (self.joint_origins[i + 1] - self.joint_origins[i])
                joint_acc = (
                    joint_acc
                    + np.cross(alpha, r_next)
                    + np.cross(omega, np.cross(omega, r_next))
                )
                joint_origin = joint_origin + r_next
                parent_R = child_R
                parent_omega = omega
                parent_alpha = alpha

        return np.concatenate([total_force, total_torque])


# ============================================================
#  Node ROS2
# ============================================================
class ArmDynamicsNode(Node):

    def __init__(self):
        super().__init__('arm_dynamics_node')

        self._declare_model_parameters()
        self.use_sdf_kinematics = bool(self.get_parameter('use_sdf_kinematics').value)

        if self.use_sdf_kinematics:
            self.arm_model = self._load_sdf_serial_model_from_parameters()
            self.get_logger().info(
                "SDF serial arm model loaded from ROS parameters | total_moving_mass=%.3f kg"
                % float(np.sum(self.arm_model.link_masses))
            )
        else:
            links = self._load_links_from_parameters()
            self.arm_model = RecursiveNewtonEuler(links)
            self.get_logger().info(
                "DH RNE arm model loaded from ROS parameters | total_mass=%.3f kg"
                % sum(link.mass for link in links)
            )

        # Trạng thái khớp hiện tại
        self.q   = np.zeros(6)
        self.dq  = np.zeros(6)
        self.ddq = np.zeros(6)
        self._prev_dq   = np.zeros(6)
        self._prev_q    = np.zeros(6)
        self._prev_time = None

        # Trạng thái base UAV. PX4 VehicleOdometry dùng frame FRD/NED; RNE này
        # chỉ dùng thành phần động học base như một hiệu chỉnh bậc nhỏ, có clamp
        # và LPF để tránh inject nhiễu vào feedforward.
        self.use_base_motion = bool(self.get_parameter('use_base_motion').value)
        self.use_base_linear_acc = bool(self.get_parameter('use_base_linear_acc').value)
        self.base_lpf_alpha = float(self.get_parameter('base_motion_lpf_alpha').value)
        self.base_acc_limit = float(self.get_parameter('base_acc_limit_ms2').value)
        self.base_alpha_limit = float(self.get_parameter('base_alpha_limit_radps2').value)
        self.joint_acc_limit = float(self.get_parameter('joint_acc_limit_radps2').value)
        self.base_omega = np.zeros(3)
        self.base_alpha = np.zeros(3)
        self.base_acc_body = np.zeros(3)
        self._prev_base_omega = np.zeros(3)
        self._prev_base_velocity_world = np.zeros(3)
        self._prev_odom_time = None

        # ── Publisher ──
        self.wrench_pub = self.create_publisher(
            WrenchStamped,
            '/arm/interaction_wrench',
            10
        )

        # ── Subscriber ──
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            10
        )

        self.get_logger().info("Arm Dynamics (Newton-Euler) Node khởi động.")

    def _declare_model_parameters(self):
        self.declare_parameter('dh_a', [0.0, 0.0, 0.155, 0.034, 0.0, 0.0])
        self.declare_parameter('dh_alpha', [0.0, 1.5708, 0.0, 0.0, 1.5708, -1.5708])
        self.declare_parameter('dh_d', [0.034, 0.0, 0.0, 0.043, 0.075, 0.035])
        self.declare_parameter('link_lengths', [0.049, 0.155, 0.034, 0.043, 0.075, 0.035])
        self.declare_parameter('link_masses', [0.1432, 0.0742, 0.1122, 0.0298, 0.0448, 0.0149])
        self.declare_parameter('link_com_xyz', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('link_inertia_diag', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('link_inertia_full', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('use_sdf_kinematics', False)
        self.declare_parameter('joint_origin_xyz', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('joint_axis_xyz', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('use_base_motion', True)
        self.declare_parameter('use_base_linear_acc', False)
        self.declare_parameter('base_motion_lpf_alpha', 0.15)
        self.declare_parameter('base_acc_limit_ms2', 4.0)
        self.declare_parameter('base_alpha_limit_radps2', 8.0)
        self.declare_parameter('joint_acc_limit_radps2', 25.0)

    def _param_array(self, name: str, size: int, default: list[float]) -> np.ndarray:
        values = list(self.get_parameter(name).value)
        if len(values) < size:
            values = list(default)
        return _finite_array(values, size)

    def _load_links_from_parameters(self) -> List[DHLink]:
        n = 6
        dh_a = self._param_array('dh_a', n, [0.0] * n)
        dh_alpha = self._param_array('dh_alpha', n, [0.0] * n)
        dh_d = self._param_array('dh_d', n, [0.0] * n)
        lengths = self._param_array('link_lengths', n, [0.05] * n)
        masses = self._param_array('link_masses', n, [0.05] * n)

        com_values = list(self.get_parameter('link_com_xyz').value)
        inertia_values = list(self.get_parameter('link_inertia_diag').value)

        links: List[DHLink] = []
        for i in range(n):
            mass = max(float(masses[i]), 1e-4)
            length = max(abs(float(lengths[i])), 1e-3)

            if len(com_values) >= 3 * (i + 1):
                com = _finite_array(com_values[3 * i:3 * i + 3], 3)
            else:
                # Conservative default for the small printed arm: CoM is close
                # to the middle of the current DH displacement. This keeps the
                # RNE feedforward in the same order as the SDF instead of the
                # previous 17 kg UR5-like model.
                p_i = np.array([
                    dh_a[i],
                    -dh_d[i] * np.sin(dh_alpha[i]),
                    dh_d[i] * np.cos(dh_alpha[i]),
                ])
                if np.linalg.norm(p_i) < 1e-4:
                    p_i = np.array([0.0, 0.0, -length])
                com = 0.5 * p_i

            if len(inertia_values) >= 3 * (i + 1):
                inertia_diag = np.maximum(
                    _finite_array(inertia_values[3 * i:3 * i + 3], 3),
                    1e-7,
                )
            else:
                radius = max(0.006, 0.08 * length)
                i_transverse = max(mass * length * length / 12.0, 1e-7)
                i_axis = max(0.5 * mass * radius * radius, 1e-7)
                inertia_diag = np.array([i_transverse, i_transverse, i_axis])

            links.append(
                DHLink(
                    alpha=float(dh_alpha[i]),
                    a=float(dh_a[i]),
                    d=float(dh_d[i]),
                    mass=mass,
                    com=com,
                    inertia=np.diag(inertia_diag),
                )
            )
        return links

    def _load_sdf_serial_model_from_parameters(self) -> SDFSerialArmDynamics:
        n = 6
        origins = self._param_array('joint_origin_xyz', 3 * n, [0.0] * (3 * n)).reshape(n, 3)
        axes = self._param_array(
            'joint_axis_xyz',
            3 * n,
            [
                0.0, 0.0, -1.0,
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 1.0, 0.0,
                0.0, -1.0, 0.0,
            ],
        ).reshape(n, 3)
        masses = self._param_array('link_masses', n, [0.05] * n)
        coms = self._param_array('link_com_xyz', 3 * n, [0.0] * (3 * n)).reshape(n, 3)
        inertia_diag = self._param_array(
            'link_inertia_diag',
            3 * n,
            [1e-5, 1e-5, 1e-5] * n,
        ).reshape(n, 3)
        full_values = list(self.get_parameter('link_inertia_full').value)
        if len(full_values) >= 9 * n:
            inertias = _finite_array(full_values, 9 * n).reshape(n, 3, 3)
            inertias = 0.5 * (inertias + np.swapaxes(inertias, 1, 2))
            for idx in range(n):
                if np.any(np.linalg.eigvalsh(inertias[idx]) <= 1e-10):
                    inertias[idx] = np.diag(np.maximum(inertia_diag[idx], 1e-9))
        else:
            inertias = np.array([np.diag(np.maximum(row, 1e-9)) for row in inertia_diag])

        return SDFSerialArmDynamics(
            joint_origins=origins,
            joint_axes=axes,
            link_masses=np.maximum(masses, 1e-5),
            link_coms=coms,
            link_inertias=inertias,
        )

    def joint_callback(self, msg: JointState):
        if len(msg.position) < 6:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        self.q[:] = list(msg.position[:6])

        if msg.velocity and len(msg.velocity) >= 6:
            self.dq[:] = list(msg.velocity[:6])
        elif self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                self.dq[:] = (self.q - self._prev_q) / dt

        if self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                raw_ddq = (self.dq - self._prev_dq) / dt
                raw_ddq = np.clip(raw_ddq, -self.joint_acc_limit, self.joint_acc_limit)
                self.ddq[:] = 0.25 * raw_ddq + 0.75 * self.ddq

        self._prev_dq   = self.dq.copy()
        self._prev_q    = self.q.copy()
        self._prev_time = now

        # Tính wrench tương tác
        base_acc = None
        base_omega = None
        base_alpha = None
        if self.use_base_motion:
            base_omega = self.base_omega
            base_alpha = self.base_alpha
            if self.use_base_linear_acc:
                base_acc = -self.arm_model.GRAVITY + self.base_acc_body

        wrench = self.arm_model.compute_interaction_wrench(
            self.q,
            self.dq,
            self.ddq,
            base_acc=base_acc,
            base_omega=base_omega,
            base_alpha=base_alpha,
        )

        msg_out = WrenchStamped()
        msg_out.header.stamp    = self.get_clock().now().to_msg()
        msg_out.header.frame_id = "base_link"
        msg_out.wrench.force.x  = float(wrench[0])
        msg_out.wrench.force.y  = float(wrench[1])
        msg_out.wrench.force.z  = float(wrench[2])
        msg_out.wrench.torque.x = float(wrench[3])
        msg_out.wrench.torque.y = float(wrench[4])
        msg_out.wrench.torque.z = float(wrench[5])
        self.wrench_pub.publish(msg_out)

    def odom_callback(self, msg: VehicleOdometry):
        now = self.get_clock().now().nanoseconds / 1e9
        omega = _finite_array(msg.angular_velocity, 3)
        velocity_world = _finite_array(msg.velocity, 3)

        if self._prev_odom_time is not None:
            dt = now - self._prev_odom_time
            if dt > 1e-4:
                raw_alpha = np.clip(
                    (omega - self._prev_base_omega) / dt,
                    -self.base_alpha_limit,
                    self.base_alpha_limit,
                )
                raw_acc_world = np.clip(
                    (velocity_world - self._prev_base_velocity_world) / dt,
                    -self.base_acc_limit,
                    self.base_acc_limit,
                )
                rot_body_to_world = _quat_to_rot_matrix(msg.q)
                raw_acc_body = rot_body_to_world.T @ raw_acc_world

                alpha = np.clip(self.base_lpf_alpha, 0.0, 1.0)
                self.base_alpha = alpha * raw_alpha + (1.0 - alpha) * self.base_alpha
                self.base_acc_body = alpha * raw_acc_body + (1.0 - alpha) * self.base_acc_body

        self.base_omega = omega
        self._prev_base_omega = omega.copy()
        self._prev_base_velocity_world = velocity_world.copy()
        self._prev_odom_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ArmDynamicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
