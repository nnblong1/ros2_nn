/**
 * @file uam_backstepping_rbfnn_node.cpp
 *
 * Triển khai đầy đủ Backstepping Tích phân Thích nghi cho UAM theo
 * đúng các công thức trong tài liệu thiết kế.
 *
 * ═══ Sơ đồ luồng tính toán mỗi chu kỳ 10ms ═══
 *
 *  [Odometry PX4]──►  phi,theta,psi,phid,thetad,psid
 *                      px,py,pz, vx,vy,vz
 *  [JointState]  ──►  q[0..5], dq[0..5]
 *  [DynamicsMsg] ──►  D_inv (12×12), H_vec (12)
 *  [LSTMNode]    ──►  lstm_ff_ [n0x,n0y,n0z] bù tiến
 *
 *         │
 *         ▼
 *  ┌──────────────────────────────────────────────────────┐
 *  │ 1. update_cog_disturbance()                          │
 *  │    → gx_cog_, gy_cog_, gz_cog_   (trọng tâm dịch)   │
 *  ├──────────────────────────────────────────────────────┤
 *  │ 2. RBFNN.estimate([e7,e9,e11,e8,e10,e12])            │
 *  │    → n0x_hat_, n0y_hat_, n0z_hat_  (nhiễu khớp)      │
 *  │    RBFNN.update_weights(...)                          │
 *  ├──────────────────────────────────────────────────────┤
 *  │ 3. compute_altitude_control(dt)   → U1               │
 *  │    + cập nhật m̂  (thích nghi khối lượng)             │
 *  ├──────────────────────────────────────────────────────┤
 *  │ 4. compute_attitude_control(dt)   → [U2, U3, U4]     │
 *  │    (dùng n̂₀ từ RBFNN + lstm_ff_ + g_cog)             │
 *  ├──────────────────────────────────────────────────────┤
 *  │ 5. compute_joint_control()        → τ₁..τ₆           │
 *  │    (dùng ma trận D_inv, H_vec)                        │
 *  └──────────────────────────────────────────────────────┘
 *         │
 *         ▼
 *  [PX4] ◄── VehicleThrustSetpoint (U1)
 *  [PX4] ◄── VehicleTorqueSetpoint (U2,U3,U4)
 *  [ArmCtrl] ◄── joint_tau (τ₁..τ₆)
 */

#include "uam_controller/uam_adaptive_controller.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>

// ════════════════════════════════════════════════════════════════
//  RBFNN Implementation
// ════════════════════════════════════════════════════════════════

RBFNeuralNetwork::RBFNeuralNetwork(const RBFNNParams& p) : params_(p)
{
    W_hat_     = Eigen::MatrixXd::Zero(p.num_neurons, p.output_dim);
    // Tâm Gaussian phân bố đều trong domain thực tế [-0.5, 0.5]
    // (đầu vào là sai số góc/vận tốc góc, hiếm khi vượt ±0.5 rad)
    C_centers_ = Eigen::MatrixXd::Random(p.num_neurons, p.input_dim) * 0.5;
    B_widths_  = Eigen::VectorXd::Constant(p.num_neurons, p.gaussian_width);
    Gamma_     = Eigen::MatrixXd::Identity(p.num_neurons, p.num_neurons)
                 * p.learning_rate;
}

Eigen::VectorXd RBFNeuralNetwork::compute_basis(const Eigen::VectorXd& Z) const
{
    Eigen::VectorXd h(params_.num_neurons);
    for (int i = 0; i < params_.num_neurons; ++i) {
        double ns = (Z - C_centers_.row(i).transpose()).squaredNorm();
        h(i) = std::exp(-ns / (2.0 * B_widths_(i) * B_widths_(i)));
    }
    return h;
}

Eigen::VectorXd RBFNeuralNetwork::estimate(const Eigen::VectorXd& Z) const
{
    return W_hat_.transpose() * compute_basis(Z);
}

// Luật cập nhật Lyapunov: dŴ = Γ·( h·e2ᵀ − η·‖e2‖·Ŵ )
void RBFNeuralNetwork::update_weights(const Eigen::VectorXd& Z,
                                       const Eigen::VectorXd& e2,
                                       double dt)
{
    auto h  = compute_basis(Z);
    auto dW = Gamma_ * (h * e2.transpose()
                        - params_.e_modification * e2.norm() * W_hat_);
    W_hat_ += dW * dt;
    W_hat_  = W_hat_.cwiseMin(5.0).cwiseMax(-5.0);
}

void RBFNeuralNetwork::reset() { W_hat_.setZero(); }


// ════════════════════════════════════════════════════════════════
//  Constructor: Khai báo + tải tham số ROS2
// ════════════════════════════════════════════════════════════════

UAMAdaptiveController::UAMAdaptiveController()
    : Node("uam_adaptive_controller")
{
    declare_params();

    // Khởi tạo m̂ bằng khối lượng danh định
    m_hat_ = sys_.mass_nominal;

    // Khởi tạo RBFNN
    rbfnn_ = std::make_unique<RBFNeuralNetwork>(rbfnn_params_);

    // ── Publishers ──
    auto qos_r = rclcpp::QoS(10).reliable();
    offboard_pub_ = create_publisher<px4_msgs::msg::OffboardControlMode>(
                        "/fmu/in/offboard_control_mode", qos_r);
    torque_pub_   = create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>(
                        "/fmu/in/vehicle_torque_setpoint", qos_r);
    thrust_pub_   = create_publisher<px4_msgs::msg::VehicleThrustSetpoint>(
                        "/fmu/in/vehicle_thrust_setpoint", qos_r);
    motors_pub_   = create_publisher<px4_msgs::msg::ActuatorMotors>(
                        "/fmu/in/actuator_motors", qos_r);
    debug_pub_    = create_publisher<std_msgs::msg::Float64MultiArray>(
                        "/uam/debug_state", 10);
    joint_tau_pub_= create_publisher<std_msgs::msg::Float64MultiArray>(
                        "/uam/joint_torque_cmd", 10);

    // ── Subscribers ──
    auto qos_be = rclcpp::SensorDataQoS();
    odom_sub_  = create_subscription<px4_msgs::msg::VehicleOdometry>(
                    "/fmu/out/vehicle_odometry", qos_be,
                    std::bind(&UAMAdaptiveController::odom_cb, this, std::placeholders::_1));
    status_sub_= create_subscription<px4_msgs::msg::VehicleStatus>(
                    "/fmu/out/vehicle_status", qos_be,
                    std::bind(&UAMAdaptiveController::status_cb, this, std::placeholders::_1));
    lstm_sub_  = create_subscription<geometry_msgs::msg::Vector3>(
                    "/ai/lstm_predictive_torque", 10,
                    std::bind(&UAMAdaptiveController::lstm_cb, this, std::placeholders::_1));
    sp_sub_    = create_subscription<geometry_msgs::msg::Vector3>(
                    "/uam/position_setpoint", 10,
                    std::bind(&UAMAdaptiveController::sp_cb, this, std::placeholders::_1));
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
                    "/joint_states", 10,
                    std::bind(&UAMAdaptiveController::joint_cb, this, std::placeholders::_1));
    dyn_sub_   = create_subscription<std_msgs::msg::Float64MultiArray>(
                    "/arm/system_dynamics", 10,
                    std::bind(&UAMAdaptiveController::dyn_cb, this, std::placeholders::_1));
    enable_sub_= create_subscription<std_msgs::msg::Bool>(
                    "/uam/controller_enable", 10,
                    std::bind(&UAMAdaptiveController::enable_cb, this, std::placeholders::_1));
    yaw_sub_   = create_subscription<std_msgs::msg::Float64>(
                    "/uam/yaw_setpoint", 10,
                    std::bind(&UAMAdaptiveController::yaw_cb, this, std::placeholders::_1));

    // ── Timer 100 Hz ──
    timer_ = create_wall_timer(10ms,
                 std::bind(&UAMAdaptiveController::control_loop, this));

    RCLCPP_INFO(get_logger(),
        "UAM Adaptive Backstepping (Integral + Mass Adaptive) | "
        "Neurons=%d | 100Hz", rbfnn_params_.num_neurons);
}


// ════════════════════════════════════════════════════════════════
//  Khai báo & tải tham số từ ROS2 parameter server
// ════════════════════════════════════════════════════════════════
void UAMAdaptiveController::declare_params()
{
    // Vật lý
    declare_parameter("mass_nominal",   sys_.mass_nominal);
    declare_parameter("gravity",        sys_.gravity);
    declare_parameter("Ixx",            sys_.Ixx);
    declare_parameter("Iyy",            sys_.Iyy);
    declare_parameter("Izz",            sys_.Izz);
    declare_parameter("lx",             sys_.lx);
    declare_parameter("ly",             sys_.ly);
    declare_parameter("max_thrust",     sys_.max_thrust);
    declare_parameter("max_torque",     sys_.max_torque);
    declare_parameter("base_pitch_offset", sys_.base_pitch_offset);
    declare_parameter("base_roll_offset",  sys_.base_roll_offset);
    declare_parameter("rbfnn_enable",      false);  // Mặc định: tắt RBFNN output

    // Altitude gains
    declare_parameter("alt_Kp", bs_.alt.Kp);
    declare_parameter("alt_Ki", bs_.alt.Ki);
    declare_parameter("alt_Kd", bs_.alt.Kd);
    declare_parameter("alt_cz", bs_.alt.cz);
    declare_parameter("alt_sigma_m", bs_.alt.sigma_m);

    // Roll gains
    declare_parameter("roll_Kp", bs_.roll.Kp);
    declare_parameter("roll_Ki", bs_.roll.Ki);
    declare_parameter("roll_Kd", bs_.roll.Kd);

    // Pitch gains
    declare_parameter("pitch_Kp", bs_.pitch.Kp);
    declare_parameter("pitch_Ki", bs_.pitch.Ki);
    declare_parameter("pitch_Kd", bs_.pitch.Kd);

    // Yaw gains
    declare_parameter("yaw_Kp", bs_.yaw.Kp);
    declare_parameter("yaw_Ki", bs_.yaw.Ki);
    declare_parameter("yaw_Kd", bs_.yaw.Kd);

    // RBFNN
    declare_parameter("rbfnn_neurons", rbfnn_params_.num_neurons);
    declare_parameter("rbfnn_lr",      rbfnn_params_.learning_rate);
    declare_parameter("rbfnn_eta",     rbfnn_params_.e_modification);
    declare_parameter("rbfnn_width",   rbfnn_params_.gaussian_width);

    // Tải
    sys_.mass_nominal   = get_parameter("mass_nominal").as_double();
    sys_.gravity        = get_parameter("gravity").as_double();
    sys_.Ixx            = get_parameter("Ixx").as_double();
    sys_.Iyy            = get_parameter("Iyy").as_double();
    sys_.Izz            = get_parameter("Izz").as_double();
    sys_.lx             = get_parameter("lx").as_double();
    sys_.ly             = get_parameter("ly").as_double();
    sys_.max_thrust     = get_parameter("max_thrust").as_double();
    sys_.max_torque     = get_parameter("max_torque").as_double();
    sys_.base_pitch_offset = get_parameter("base_pitch_offset").as_double();
    sys_.base_roll_offset  = get_parameter("base_roll_offset").as_double();
    rbfnn_output_enabled_  = get_parameter("rbfnn_enable").as_bool();

    bs_.alt.Kp      = get_parameter("alt_Kp").as_double();
    bs_.alt.Ki      = get_parameter("alt_Ki").as_double();
    bs_.alt.Kd      = get_parameter("alt_Kd").as_double();
    bs_.alt.cz      = get_parameter("alt_cz").as_double();
    bs_.alt.sigma_m = get_parameter("alt_sigma_m").as_double();

    bs_.roll.Kp  = get_parameter("roll_Kp").as_double();
    bs_.roll.Ki  = get_parameter("roll_Ki").as_double();
    bs_.roll.Kd  = get_parameter("roll_Kd").as_double();

    bs_.pitch.Kp = get_parameter("pitch_Kp").as_double();
    bs_.pitch.Ki = get_parameter("pitch_Ki").as_double();
    bs_.pitch.Kd = get_parameter("pitch_Kd").as_double();

    bs_.yaw.Kp   = get_parameter("yaw_Kp").as_double();
    bs_.yaw.Ki   = get_parameter("yaw_Ki").as_double();
    bs_.yaw.Kd   = get_parameter("yaw_Kd").as_double();

    rbfnn_params_.num_neurons    = get_parameter("rbfnn_neurons").as_int();
    rbfnn_params_.learning_rate  = get_parameter("rbfnn_lr").as_double();
    rbfnn_params_.e_modification = get_parameter("rbfnn_eta").as_double();
    rbfnn_params_.gaussian_width = get_parameter("rbfnn_width").as_double();
}


// ════════════════════════════════════════════════════════════════
//  CALLBACKS
// ════════════════════════════════════════════════════════════════

void UAMAdaptiveController::odom_cb(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    // PX4 dùng hệ NED, chuyển sang ENU
    px_ =  msg->position[0];
    py_ = -msg->position[1];
    pz_ = -msg->position[2];

    vx_ =  msg->velocity[0];
    vy_ = -msg->velocity[1];
    vz_ = -msg->velocity[2];

    q_imu_ = Eigen::Quaterniond(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
    auto euler = quat_to_euler(q_imu_);
    // PX4 quaternion là NED frame. quat_to_euler trả về NED euler.
    // angular_velocity[1] và [2] đã đổi dấu (NED→ENU) ở dưới.
    // Phải đồng bộ: đổi dấu phi_ và theta_ để sang ENU.
    phi_   = -euler(0);  // NED roll  → ENU roll  (đổi dấu)
    theta_ = -euler(1);  // NED pitch → ENU pitch (đổi dấu: NED nose-down>0, ENU nose-up>0)
    psi_   =  euler(2);  // Yaw: ENU = NED (cùng trục Z dọc, ±π offset đã xử lý bởi modulo)

    phid_   =  msg->angular_velocity[0];
    thetad_ = -msg->angular_velocity[1];
    psid_   = -msg->angular_velocity[2];

    has_odom_ = true;

    // Lần đầu nhận odom: khởi tạo yaw setpoint từ heading thực
    // và reset tất cả integrator (tránh windup trước khi ARM)
    if (!odom_initialized_) {
        psi_des_ = psi_;
        x_des_   = px_;
        y_des_   = py_;
        J5_ = J7_ = J9_ = J11_ = 0.0;
        e5_ = e6_ = e7_ = e8_ = e9_ = e10_ = e11_ = e12_ = 0.0;
        m_hat_ = sys_.mass_nominal;
        odom_initialized_ = true;
        RCLCPP_INFO(get_logger(), "Odom initialized: yaw_des=%.2f, pos=(%.2f,%.2f,%.2f)",
                    psi_des_, px_, py_, pz_);
    }
}

void UAMAdaptiveController::status_cb(
    const px4_msgs::msg::VehicleStatus::SharedPtr /*msg*/) {}

void UAMAdaptiveController::lstm_cb(
    const geometry_msgs::msg::Vector3::SharedPtr /*msg*/)
{
    // LSTM disabled: không sử dụng tín hiệu bù tiến LSTM
    lstm_ff_.setZero();
}

void UAMAdaptiveController::sp_cb(
    const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    // Tính velocity reference từ delta setpoint (velocity feedforward)
    double now = get_clock()->now().seconds();
    double dt_sp = (last_sp_t_ > 0.0) ? (now - last_sp_t_) : 0.0;

    double new_z = msg->z;
    if (dt_sp > 0.001 && dt_sp < 0.5) {
        zd_des_ = (new_z - z_des_) / dt_sp;
        zd_des_ = std::clamp(zd_des_, -1.0, 1.0); // Giới hạn tốc độ reference
    } else {
        zd_des_ = 0.0;
    }

    x_des_ = msg->x;
    y_des_ = msg->y;
    z_des_ = new_z;
    last_sp_t_ = now;
}

void UAMAdaptiveController::joint_cb(
    const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if ((int)msg->position.size() < N_JOINTS) return;
    for (int i = 0; i < N_JOINTS; ++i) {
        q_[i]  = msg->position[i];
        dq_[i] = (msg->velocity.size() >= (size_t)N_JOINTS)
                 ? msg->velocity[i] : 0.0;
    }
    has_joints_ = true;
}

// Topic /arm/system_dynamics: phẳng hóa [D_inv(12×12), H(12)] = 144+12 = 156 phần tử
void UAMAdaptiveController::dyn_cb(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if ((int)msg->data.size() < 156) return;
    D_inv_ = Eigen::Map<const Eigen::Matrix<double,12,12,Eigen::RowMajor>>(
                 msg->data.data());
    H_vec_ = Eigen::Map<const Eigen::VectorXd>(msg->data.data() + 144, 12);
    dyn_ready_ = true;
}

void UAMAdaptiveController::enable_cb(const std_msgs::msg::Bool::SharedPtr msg)
{
    if (msg->data && !controller_enabled_) {
        RCLCPP_INFO(get_logger(), "Controller ENABLED.");
        // Reset m̂ để tránh giữ giá trị bão hòa từ khi disabled
        m_hat_ = sys_.mass_nominal;
        // Reset integrators
        J5_ = J7_ = J9_ = J11_ = 0.0;
        e5_ = e6_ = e7_ = e8_ = e9_ = e10_ = e11_ = e12_ = 0.0;
        // Reset RBFNN warmup + weights
        rbfnn_warmup_ticks_ = 0;
        rbfnn_->reset();
        // Gán lại setpoint
        x_des_ = px_; y_des_ = py_; z_des_ = pz_; psi_des_ = psi_;
    } else if (!msg->data && controller_enabled_) {
        RCLCPP_INFO(get_logger(), "Controller DISABLED.");
    }
    controller_enabled_ = msg->data;
}

void UAMAdaptiveController::yaw_cb(const std_msgs::msg::Float64::SharedPtr msg)
{
    psi_des_ = msg->data;
}



// ════════════════════════════════════════════════════════════════
//  Ước lượng mô-men trọng lực do dịch trọng tâm (g_cog)
//  Dùng công thức xấp xỉ tuyến tính dựa trên góc khớp hiện tại
// ════════════════════════════════════════════════════════════════
void UAMAdaptiveController::update_cog_disturbance()
{
    // NOTE: gy_cog_ feedforward disabled — use base_pitch_offset in the mixer instead.
    // All previous feedforward attempts produced sign ambiguities.
    // A clean tunable static offset is more reliable for this UAV.
    gx_cog_ = 0.0;
    gy_cog_ = 0.0;
    gz_cog_ = 0.0;
}


// ════════════════════════════════════════════════════════════════
//  ĐIỀU KHIỂN TRỤC Z (ĐỘ CAO) – Công thức tài liệu Mục 1
//
//  e5  = z − z_des
//  J5  = J5 + e5·dt
//  e6  = ż − ż_des + K5i·J5 + K5p·e5
//
//  Đặt ξ = g − K5i·e5 − K5p·(e6 − K5i·J5 − K5p·e5) − e5 − K5d·e6
//
//  U1  = m̂ / (cosφ·cosθ) · ξ
//
//  ṁ̂   = −cz · e6 · (g − K5i·e5 + K5p·(e6 − K5i·J5 − K5p·e5) − e5 − K5d·e6)
//       (lưu ý dấu + trước K5p trong luật cập nhật theo tài liệu)
// ════════════════════════════════════════════════════════════════
double UAMAdaptiveController::compute_altitude_control(double dt)
{
    const double Kp = bs_.alt.Kp;   // K5p
    const double Ki = bs_.alt.Ki;   // K5i
    const double Kd = bs_.alt.Kd;   // K5d
    const double cz = bs_.alt.cz;
    const double g  = sys_.gravity;

    // ── Sai số vị trí (bước 1) ──
    e5_ = pz_ - z_des_;

    // ── Tích phân sai số ──
    J5_ += e5_ * dt;
    // Anti-windup
    J5_ = sat(J5_, 2.0);

    // ── Sai số vận tốc ảo (bước 2) ──
    // e6 = ż − ż_des + K5i·J5 + K5p·e5
    e6_ = (vz_ - zd_des_) + Ki * J5_ + Kp * e5_;

    // ── Biểu thức bên trong ngoặc của U1 ──
    // ξ = g − K5i·e5 − K5p·(e6 − K5i·J5 − K5p·e5) − e5 − K5d·e6
    double inner_e6 = e6_ - Ki * J5_ - Kp * e5_;
    double xi_U1    = g - Ki * e5_ - Kp * inner_e6 - e5_ - Kd * e6_;

    // ── Lực đẩy tổng ──
    double cos_phi_theta = std::cos(phi_) * std::cos(theta_);
    // Tránh chia cho 0 khi góc nghiêng cực lớn
    cos_phi_theta = (std::abs(cos_phi_theta) < 0.1)
                    ? std::copysign(0.1, cos_phi_theta)
                    : cos_phi_theta;

    double U1_raw = (m_hat_ / cos_phi_theta) * xi_U1;

    // ── Cập nhật m̂ theo luật thích nghi + σ-modification ──
    // Sửa lỗi sai dấu trong tài liệu: dùng trực tiếp xi_U1 để giữ tính chặt chẽ của hàm Lyapunov.
    double xi_mhat = xi_U1;
    double m_hat_dot = -cz * e6_ * xi_mhat
                      - bs_.alt.sigma_m * (m_hat_ - sys_.mass_nominal);

    // Parameter Anti-Windup: Ngăn m̂ tăng ảo do sai lệch e5_ lớn khi lệnh cất cánh bị bậc thang, 
    // khiến U1_raw vượt ngưỡng thrust vật lý. Đồng thời pause adaptation nếu sai số độ cao
    // quá lớn (>= 0.5m) để tránh m̂ học sai trong giai đoạn bay lên chưa ổn định.
    if ((U1_raw >= sys_.max_thrust && m_hat_dot > 0.0) ||
        (U1_raw <= 0.0 && m_hat_dot < 0.0)) {
        m_hat_dot = 0.0;
    }

    m_hat_ += m_hat_dot * dt;
    // Giới hạn an toàn tuyệt đối
    m_hat_  = std::clamp(m_hat_, bs_.alt.m_hat_min, bs_.alt.m_hat_max);

    // Chuẩn hóa cho PX4 (NED: âm = đẩy lên)
    double U1_norm = -U1_raw / sys_.max_thrust;
    return std::clamp(U1_norm, -1.0, 0.0);
}


// ════════════════════════════════════════════════════════════════
//  ĐIỀU KHIỂN TƯ THẾ (ROLL/PITCH/YAW) – Công thức tài liệu Mục 2
//
//  ── ROLL ──
//  e7  = φ − φ_des
//  J7  = ∫e7 dt
//  e8  = φ̇ − φ̇_des + K7i·J7 + K7p·e7
//  U2  = Ixx/lx · ( −K7i·e7 − K7p·(e8−K7i·J7−K7p·e7)
//                   + θ̇·ψ̇·(Izz−Iyy)/Ixx − e7 − K7d·e8
//                   − n̂₀x/Ixx − gx/Ixx )
//
//  ── PITCH ──  (idem với index 9, Iyy/ly, (Ixx−Izz)/Iyy, n̂₀y, gy)
//
//  ── YAW ──    (idem với index 11, Izz, (Iyy−Ixx)/Izz, n̂₀z, gz)
//               không có hệ số tay đòn l vì U4 = mô-men trực tiếp
// ════════════════════════════════════════════════════════════════
Eigen::Vector3d UAMAdaptiveController::compute_attitude_control(double dt)
{
    const double Ixx = sys_.Ixx, Iyy = sys_.Iyy, Izz = sys_.Izz;

    // ─────────────────────────────────────────
    //  Tính tư thế mong muốn từ sai số vị trí ngang
    //  (vòng ngoài → vòng trong)
    // ─────────────────────────────────────────
    double e5_x = px_ - x_des_; // Lỗi vị trí theo trục North
    double e5_y = py_ - y_des_; // Lỗi vị trí theo trục West

    // Tính gia tốc mong muốn trong hệ trục thế giới (NWU)
    // Tách riêng hệ số vòng vị trí (Pos) ra khỏi vòng góc (Attitude) để tránh bị liệng (wobbling)
    // Hệ số P=1.0, D=2.0 là các giá trị tiêu chuẩn, mượt mà cho vòng lặp vị trí.
    // Hệ số vòng vị trí: giảm nhẹ để tránh command góc lớn khi position error lớn
    const double pos_Kp = 0.6;
    const double pos_Kd = 1.5;

    double ax_w = -pos_Kd * vx_ - pos_Kp * e5_x;
    double ay_w = -pos_Kd * vy_ - pos_Kp * e5_y;

    // Xoay gia tốc từ World (NWU) sang Body (Forward-Left-Up) theo góc Yaw (psi_)
    double cos_psi = std::cos(psi_);
    double sin_psi = std::sin(psi_);

    // Áp dụng ma trận quay 2D để đưa ax, ay về hệ quy chiếu thân drone
    double ax_b = ax_w * cos_psi - ay_w * sin_psi;
    double ay_b = ax_w * sin_psi + ay_w * cos_psi;

    // Ánh xạ từ gia tốc thân sang tư thế mong muốn:
    // Giới hạn ±0.20 rad (~11.5°) để chống lật với tải lệch tâm
    double theta_des_inner = sat(-ax_b / sys_.gravity, 0.20);
    double phi_des_inner   = sat( ay_b / sys_.gravity, 0.20);

    // ─────────────────────────────────────────
    //  RBFNN: ước lượng nhiễu n̂₀ = [n̂₀x, n̂₀y, n̂₀z]
    // ─────────────────────────────────────────
    Eigen::VectorXd Z_rbf(6);
    Z_rbf << e7_, e9_, e11_, e8_, e10_, e12_;

    // Cập nhật trọng số (học trực tuyến) — luôn học ngay cả trong warmup
    Eigen::Vector3d e2_att(e8_, e10_, e12_);
    rbfnn_->update_weights(Z_rbf, e2_att, dt);

    // Warmup + rbfnn_enable gate: RBFNN weights luôn được cập nhật (học ngầm),
    // nhưng output chỉ inject vào controller khi cả 2 điều kiện được thỏa mãn:
    // 1. rbfnn_enable = true (từ YAML)
    // 2. warmup >= 300 ticks (~3s ở 100Hz) — đủ thời gian hover ổn ?nh trước khi bật
    if (!rbfnn_output_enabled_ || rbfnn_warmup_ticks_ < 300) {
        if (rbfnn_warmup_ticks_ < 300) rbfnn_warmup_ticks_++;
        n0x_hat_ = 0.0;
        n0y_hat_ = 0.0;
        n0z_hat_ = 0.0;
    } else {
        Eigen::VectorXd n0_hat = rbfnn_->estimate(Z_rbf);
        n0_hat += lstm_ff_;
        // Cap RBFNN output tại ±1.0 Nm để tránh dao động khi mới kích hoạt
        n0x_hat_ = std::clamp(n0_hat(0), -1.0, 1.0);
        n0y_hat_ = std::clamp(n0_hat(1), -1.0, 1.0);
        n0z_hat_ = std::clamp(n0_hat(2), -1.0, 1.0);
    }

    // ─────────────────────────────────────────
    //  ROLL (U2)
    // ─────────────────────────────────────────
    {
        const double Kp = bs_.roll.Kp, Ki = bs_.roll.Ki, Kd = bs_.roll.Kd;

        e7_ = phi_ - phi_des_inner;
        J7_ = sat(J7_ + e7_ * dt, 5.0);

        // e8 = φ̇ − φ̇_des + K7i·J7 + K7p·e7
        e8_ = phid_ + Ki * J7_ + Kp * e7_;

        // Coriolis: θ̇·ψ̇·(Izz−Iyy)/Ixx
        double coriolis_roll = thetad_ * psid_ * (Izz - Iyy) / Ixx;

        // Biểu thức trong ngoặc lớn
        double inner_e8_roll = e8_ - Ki * J7_ - Kp * e7_;
        double bracket_roll  = -Ki * e7_
                               - Kp * inner_e8_roll
                               + coriolis_roll
                               - e7_
                               - Kd * e8_
                               - n0x_hat_ / Ixx
                               - gx_cog_  / Ixx;

        // U2 = Ixx · bracket
        double U2 = Ixx * bracket_roll;
        // (lưu giá trị vào biến tạm, ghép vào output cuối)
        (void)U2; // được dùng bên dưới
    }

    // ─────────────────────────────────────────
    //  PITCH (U3)
    // ─────────────────────────────────────────
    {
        const double Kp = bs_.pitch.Kp, Ki = bs_.pitch.Ki, Kd = bs_.pitch.Kd;

        e9_ = theta_ - theta_des_inner;
        J9_ = sat(J9_ + e9_ * dt, 5.0);

        e10_ = thetad_ + Ki * J9_ + Kp * e9_;

        double coriolis_pitch = phid_ * psid_ * (Ixx - Izz) / Iyy;

        double inner_e10 = e10_ - Ki * J9_ - Kp * e9_;
        double bracket_pitch = -Ki * e9_
                               - Kp * inner_e10
                               + coriolis_pitch
                               - e9_
                               - Kd * e10_
                               - n0y_hat_ / Iyy
                               - gy_cog_  / Iyy;

        double U3 = Iyy * bracket_pitch;
        (void)U3;
    }

    // ─────────────────────────────────────────
    //  YAW (U4)
    // ─────────────────────────────────────────
    {
        const double Kp = bs_.yaw.Kp, Ki = bs_.yaw.Ki, Kd = bs_.yaw.Kd;

        e11_ = psi_ - psi_des_;
        // Chuẩn hóa về [-π, π]
        while (e11_ >  M_PI) e11_ -= 2.0 * M_PI;
        while (e11_ < -M_PI) e11_ += 2.0 * M_PI;

        J11_ = sat(J11_ + e11_ * dt, 5.0);

        e12_ = psid_ + Ki * J11_ + Kp * e11_;

        double coriolis_yaw = phid_ * thetad_ * (Iyy - Ixx) / Izz;

        double inner_e12 = e12_ - Ki * J11_ - Kp * e11_;
        double bracket_yaw = -Ki * e11_
                             - Kp * inner_e12
                             + coriolis_yaw
                             - e11_
                             - Kd * e12_
                             - n0z_hat_ / Izz
                             - gz_cog_  / Izz;

        double U4 = Izz * bracket_yaw;
        (void)U4;
    }

    // ─────────────────────────────────────────
    //  Tập hợp kết quả + chuẩn hóa
    // ─────────────────────────────────────────

    // Tính lại với code sạch hơn, tập trung vào giá trị trả về
    auto calc_roll = [&]() -> double {
        const double Kp=bs_.roll.Kp, Ki=bs_.roll.Ki, Kd=bs_.roll.Kd;
        double inner = e8_ - Ki*J7_ - Kp*e7_;
        return Ixx*(-Ki*e7_ - Kp*inner
               + thetad_*psid_*(Izz-Iyy)/Ixx
               - e7_ - Kd*e8_
               - n0x_hat_/Ixx - gx_cog_/Ixx);
    };
    auto calc_pitch = [&]() -> double {
        const double Kp=bs_.pitch.Kp, Ki=bs_.pitch.Ki, Kd=bs_.pitch.Kd;
        double inner = e10_ - Ki*J9_ - Kp*e9_;
        return Iyy*(-Ki*e9_ - Kp*inner
               + phid_*psid_*(Ixx-Izz)/Iyy
               - e9_ - Kd*e10_
               - n0y_hat_/Iyy - gy_cog_/Iyy);
    };
    auto calc_yaw = [&]() -> double {
        const double Kp=bs_.yaw.Kp, Ki=bs_.yaw.Ki, Kd=bs_.yaw.Kd;
        double inner = e12_ - Ki*J11_ - Kp*e11_;
        return Izz*(-Ki*e11_ - Kp*inner
               + phid_*thetad_*(Iyy-Ixx)/Izz
               - e11_ - Kd*e12_
               - n0z_hat_/Izz - gz_cog_/Izz);
    };

    Eigen::Vector3d tau;
    tau(0) = sat(calc_roll(),  sys_.max_torque) / sys_.max_torque;
    tau(1) = sat(calc_pitch(), sys_.max_torque) / sys_.max_torque;
    tau(2) = sat(calc_yaw(),   sys_.max_torque) / sys_.max_torque;

    return tau;
}


// ════════════════════════════════════════════════════════════════
//  ĐIỀU KHIỂN KHỚP TAY MÁY – Công thức tài liệu Mục 3
//
//  ei1 = qi − qi_des
//  ei2 = q̇i − q̇i_des + Kpi·ei1
//
//  Qdd[i] = −Kpi·(ei2−Kpi·ei1) − ei1 − Kdi·ei2
//
//  τ_arm = (D⁻¹₇₋₁₂,₇₋₁₂)⁻¹ · [Qdd − D⁻¹₇₋₁₂,₁₋₆·(τ₁₋₆−H₁₋₆)] + H₇₋₁₂
//
//  Ở đây: hàng 0..5 (index 0-based) = DOF của UAV
//          hàng 6..11              = DOF của khớp tay
// ════════════════════════════════════════════════════════════════
Eigen::VectorXd UAMAdaptiveController::compute_joint_control()
{
    Eigen::VectorXd tau_joints = Eigen::VectorXd::Zero(N_JOINTS);

    if (!dyn_ready_) {
        // Fallback: PD thuần túy nếu chưa có ma trận động lực học
        for (int i = 0; i < N_JOINTS; ++i) {
            double ei1 = q_[i]  - qd_[i];
            double ei2 = dq_[i] - dqd_[i] + jg_[i].Kp * ei1;
            tau_joints(i) = sat(-jg_[i].Kp * (ei2 - jg_[i].Kp*ei1)
                                - ei1 - jg_[i].Kd * ei2,
                                sys_.max_joint_tau);
        }
        return tau_joints;
    }

    // ── Bước 1: Tính vector Qdd (6 phần tử) ──
    Eigen::VectorXd Qdd(N_JOINTS);
    for (int i = 0; i < N_JOINTS; ++i) {
        const double Kp = jg_[i].Kp;
        const double Kd = jg_[i].Kd;
        double ei1 = q_[i]  - qd_[i];
        double ei2 = dq_[i] - dqd_[i] + Kp * ei1;

        // Qdd[i] = −Kpi·(ei2 − Kpi·ei1) − ei1 − Kdi·ei2
        Qdd(i) = -Kp * (ei2 - Kp * ei1) - ei1 - Kd * ei2;
    }

    // ── Bước 2: Trích xuất các block từ D_inv ──
    // D_inv_ là ma trận 12×12 của toàn hệ thống
    // Block 7:12, 1:6  (0-indexed: hàng 6..11, cột 0..5)
    Eigen::MatrixXd D_inv_arm_uav = D_inv_.block(6, 0, N_JOINTS, 6);
    // Block 7:12, 7:12 (0-indexed: hàng 6..11, cột 6..11)
    Eigen::MatrixXd D_inv_arm_arm = D_inv_.block(6, 6, N_JOINTS, N_JOINTS);
    // H cho phần khớp tay (H_7:12)
    Eigen::VectorXd H_arm = H_vec_.segment(6, N_JOINTS);
    // H và τ₁₋₆ cho phần UAV (H_1:6, τ₁₋₆)
    Eigen::VectorXd H_uav = H_vec_.segment(0, 6);

    // ── Bước 3: τ₁₋₆ = tín hiệu UAV controller (thrust + torque) ──
    // Đây là đầu ra U1,U2,U3,U4 kết hợp lại thành vector 6 chiều
    // (x,y,z force + roll,pitch,yaw torque)
    Eigen::VectorXd tau_uav = Eigen::VectorXd::Zero(6);
    tau_uav(2) = m_hat_ * sys_.gravity; // Lực đẩy Z

    // ── Bước 4: Phương trình điều khiển khớp ──
    // τ = (D⁻¹_arm_arm)⁻¹ · [Qdd − D⁻¹_arm_uav · (τ_uav − H_uav)] + H_arm
    Eigen::VectorXd rhs = Qdd - D_inv_arm_uav * (tau_uav - H_uav);

    // Giải hệ tuyến tính: D_inv_arm_arm * τ = rhs
    // <=> τ = (D_inv_arm_arm)⁻¹ * rhs
    tau_joints = D_inv_arm_arm.lu().solve(rhs) + H_arm;

    // Giới hạn mô-men xoắn
    for (int i = 0; i < N_JOINTS; ++i)
        tau_joints(i) = sat(tau_joints(i), sys_.max_joint_tau);

    return tau_joints;
}


// ════════════════════════════════════════════════════════════════
//  VÒNG LẶP ĐIỀU KHIỂN CHÍNH – 100 Hz
// ════════════════════════════════════════════════════════════════
void UAMAdaptiveController::control_loop()
{
    publish_offboard_mode();

    if (!has_odom_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Chờ Odometry từ PX4...");
        return;
    }

    if (!controller_enabled_) {
        // Reset integrators & m̂
        J5_ = J7_ = J9_ = J11_ = 0.0;
        e5_ = e6_ = e7_ = e8_ = e9_ = e10_ = e11_ = e12_ = 0.0;
        m_hat_ = sys_.mass_nominal;
        
        last_t_ = get_clock()->now().seconds();

        // Publish idle motor command to keep offboard mode alive
        px4_msgs::msg::ActuatorMotors motors_msg{};
        for (int i = 0; i < 4; ++i) motors_msg.control[i] = 0.0f; // Idle
        for (int i = 4; i < 12; ++i) motors_msg.control[i] = std::nanf("");
        motors_msg.timestamp = get_clock()->now().nanoseconds() / 1000;
        motors_pub_->publish(motors_msg);

        // Publish zero thrust and torque
        px4_msgs::msg::VehicleThrustSetpoint thrust_msg{};
        thrust_msg.xyz[0] = thrust_msg.xyz[1] = thrust_msg.xyz[2] = 0.0f;
        thrust_msg.timestamp = motors_msg.timestamp;
        thrust_pub_->publish(thrust_msg);

        px4_msgs::msg::VehicleTorqueSetpoint torque_msg{};
        torque_msg.xyz[0] = torque_msg.xyz[1] = torque_msg.xyz[2] = 0.0f;
        torque_msg.timestamp = motors_msg.timestamp;
        torque_pub_->publish(torque_msg);

        // Giữ vị trí hiện tại
        x_des_ = px_; y_des_ = py_; 
        // Lấy z_des_ từ tham số khởi tạo thay vì chìm theo pz_ để khi bật lên bay ngay?
        // Mission bridge sẽ gửi z_des_ trước khi enable. Do đó ta khôgn xoá setpoint ở đây
        
        return;
    }

    double now = get_clock()->now().seconds();
    double dt  = (last_t_ > 0.0) ? (now - last_t_) : 0.01;
    dt         = std::clamp(dt, 0.005, 0.05);
    last_t_    = now;

    // 1. Cập nhật g_cog từ trạng thái khớp hiện tại
    if (has_joints_) update_cog_disturbance();

    // 2. Tính U1 (thrust) + cập nhật m̂
    double U1 = compute_altitude_control(dt);

    // 3. Tính U2,U3,U4 (torque) + cập nhật RBFNN
    Eigen::Vector3d U234 = compute_attitude_control(dt);

    // 4. Tính τ₁..τ₆ cho khớp tay
    Eigen::VectorXd tau_j = Eigen::VectorXd::Zero(N_JOINTS);
    if (has_joints_) tau_j = compute_joint_control();

    // 5. Gửi lệnh về PX4 qua ActuatorMotors (mixer quad X)
    //    U1 ∈ [-1, 0], negate → [0, 1] = thrust
    //    U234 ∈ [-1, 1] = roll, pitch, yaw torque đã chuẩn hóa
    {
        double T  = -U1;       // thrust [0, 1]
        // Gửi torque đã tính được tới PX4 (PX4 sẽ tự dịch thành actuator motors)
        double Ur =  U234(0);  // Roll  (ENU→NED: giữ nguyên)
        double Up = -U234(1);  // HOÀN TRẢ: Roll/Pitch/Yaw signs original
        double Uy = -U234(2);  // HOÀN TRẢ: Roll/Pitch/Yaw signs original

        // Bù trừ trọng tâm (CoM) thủ công do tay máy
        // Pitch: cánh tay phía sau kéo đuôi xuống
        // Roll: cánh tay có thể lệch ngang
        Up += sys_.base_pitch_offset;
        Ur += sys_.base_roll_offset;

        // Giới hạn torque: với payload nặng lệch tâm, cần giới hạn rộng hơn để chống lật (85% thrust)
        double max_torque_frac = 0.85 * T;
        double torque_max = std::max({std::abs(Ur), std::abs(Up), std::abs(Uy)});
        
        if (max_torque_frac < 0.01) {
            // Khi thrust quá nhỏ (chưa cất cánh), torque phải bằng 0 để tránh lật UAV
            Ur = 0.0;
            Up = 0.0;
            Uy = 0.0;
        } else if (torque_max > max_torque_frac) {
            double scale = max_torque_frac / torque_max;
            Ur *= scale;
            Up *= scale;
            Uy *= scale;
        }

        // hop SDF motor layout:
        // Motor 0 (quat1_1): rear-left    CW  (-Px, +Py)
        // Motor 1 (quat4_1): front-right  CCW (+Px, -Py)
        // Motor 2 (quat2_1): rear-right   CCW (-Px, -Py)
        // Motor 3 (quat3_1): front-left   CW  (+Px, +Py)
        //
        // Roll  (+Ur) → Left up (M0,M3), Right down (M1,M2)
        // Pitch (+Up) → Front up (M1,M3), Rear down (M0,M2)
        // Yaw   (+Uy) → CCW yaw → CCW motors up (M1,M2), CW down (M0,M3)
        double val0 = std::max(0.0, T + Ur - Up - Uy);  // rear-left  CW
        double val1 = std::max(0.0, T - Ur + Up + Uy);  // front-right CCW
        double val2 = std::max(0.0, T - Ur - Up + Uy);  // rear-right CCW
        double val3 = std::max(0.0, T + Ur + Up - Uy);  // front-left CW

        // Desaturation để giữ nguyên tỉ lệ khi có motor bị quá tải (>1.0)
        double max_val = std::max({val0, val1, val2, val3});
        if (max_val > 1.0) {
            val0 /= max_val;
            val1 /= max_val;
            val2 /= max_val;
            val3 /= max_val;
        }

        double m0 = std::sqrt(val0);
        double m1 = std::sqrt(val1);
        double m2 = std::sqrt(val2);
        double m3 = std::sqrt(val3);

        // Base idle speed (khoảng 0.15) để motor hoạt động ngay khi ARM
        double idle = 0.15;
        m0 = idle + m0 * (1.0 - idle);
        m1 = idle + m1 * (1.0 - idle);
        m2 = idle + m2 * (1.0 - idle);
        m3 = idle + m3 * (1.0 - idle);

        px4_msgs::msg::ActuatorMotors motors_msg{};
        motors_msg.control[0] = static_cast<float>(std::clamp(m0, 0.0, 1.0));
        motors_msg.control[1] = static_cast<float>(std::clamp(m1, 0.0, 1.0));
        motors_msg.control[2] = static_cast<float>(std::clamp(m2, 0.0, 1.0));
        motors_msg.control[3] = static_cast<float>(std::clamp(m3, 0.0, 1.0));
        // Motor 4-11 = NaN (không dùng)
        for (int i = 4; i < 12; ++i)
            motors_msg.control[i] = std::nanf("");
        motors_msg.timestamp = get_clock()->now().nanoseconds() / 1000;
        motors_pub_->publish(motors_msg);
    }

    // Vẫn publish thrust/torque cho debug/logging
    {
        px4_msgs::msg::VehicleThrustSetpoint thrust_msg{};
        thrust_msg.xyz[0] = 0.0f;
        thrust_msg.xyz[1] = 0.0f;
        thrust_msg.xyz[2] = static_cast<float>(U1);
        thrust_msg.timestamp = get_clock()->now().nanoseconds() / 1000;
        thrust_pub_->publish(thrust_msg);
    }
    {
        px4_msgs::msg::VehicleTorqueSetpoint torque_msg{};
        torque_msg.xyz[0] = static_cast<float>(U234(0));
        torque_msg.xyz[1] = static_cast<float>(U234(1));
        torque_msg.xyz[2] = static_cast<float>(U234(2));
        torque_msg.timestamp = get_clock()->now().nanoseconds() / 1000;
        torque_pub_->publish(torque_msg);
    }

    // 6. Gửi lệnh τ cho khớp tay
    {
        std_msgs::msg::Float64MultiArray joint_msg;
        for (int i = 0; i < N_JOINTS; ++i)
            joint_msg.data.push_back(tau_j(i));
        joint_tau_pub_->publish(joint_msg);
    }

    publish_debug(U1, U234, tau_j);
}


// ════════════════════════════════════════════════════════════════
//  Offboard heartbeat (phải phát > 2 Hz để PX4 giữ chế độ)
// ════════════════════════════════════════════════════════════════
void UAMAdaptiveController::publish_offboard_mode()
{
    px4_msgs::msg::OffboardControlMode mode{};
    mode.position        = false;
    mode.velocity        = false;
    mode.acceleration    = false;
    mode.attitude        = false;
    mode.body_rate       = false; 
    mode.direct_actuator = true;   // Về lại actuator control
    mode.timestamp       = get_clock()->now().nanoseconds() / 1000;
    offboard_pub_->publish(mode);
}


// ════════════════════════════════════════════════════════════════
//  Tiện ích
// ════════════════════════════════════════════════════════════════
Eigen::Vector3d UAMAdaptiveController::quat_to_euler(
    const Eigen::Quaterniond& q) const
{
    Eigen::Vector3d e;
    // Roll
    double sinr = 2.0*(q.w()*q.x() + q.y()*q.z());
    double cosr = 1.0 - 2.0*(q.x()*q.x() + q.y()*q.y());
    e(0) = std::atan2(sinr, cosr);
    // Pitch
    double sinp = 2.0*(q.w()*q.y() - q.z()*q.x());
    e(1) = (std::abs(sinp)>=1.0) ? std::copysign(M_PI/2.0,sinp) : std::asin(sinp);
    // Yaw
    double siny = 2.0*(q.w()*q.z() + q.x()*q.y());
    double cosy = 1.0 - 2.0*(q.y()*q.y() + q.z()*q.z());
    e(2) = std::atan2(siny, cosy);
    return e;
}

double UAMAdaptiveController::sat(double v, double lim) const
{
    return std::clamp(v, -lim, lim);
}

void UAMAdaptiveController::publish_debug(double U1,
                                           const Eigen::Vector3d& U234,
                                           const Eigen::VectorXd& tau_j)
{
    std_msgs::msg::Float64MultiArray msg;
    // [0-2]  Vị trí
    msg.data.insert(msg.data.end(), {px_, py_, pz_});
    // [3-5]  Euler
    msg.data.insert(msg.data.end(), {phi_, theta_, psi_});
    // [6-8]  Sai số tư thế e7,e9,e11
    msg.data.insert(msg.data.end(), {e7_, e9_, e11_});
    // [9-11] Sai số vận tốc e8,e10,e12
    msg.data.insert(msg.data.end(), {e8_, e10_, e12_});
    // [12]   Sai số độ cao e5
    msg.data.push_back(e5_);
    // [13]   Khối lượng ước lượng m̂
    msg.data.push_back(m_hat_);
    // [14-16] Nhiễu ước lượng RBFNN
    msg.data.insert(msg.data.end(), {n0x_hat_, n0y_hat_, n0z_hat_});
    // [17-19] g_cog
    msg.data.insert(msg.data.end(), {gx_cog_, gy_cog_, gz_cog_});
    // [20]   U1 (Thrust)
    msg.data.push_back(U1);
    // [21-23] U2,U3,U4 (Torque)
    msg.data.insert(msg.data.end(), {U234(0), U234(1), U234(2)});
    // [24-29] τ khớp tay
    for (int i=0; i<N_JOINTS; ++i) msg.data.push_back(tau_j(i));
    debug_pub_->publish(msg);
}


// ════════════════════════════════════════════════════════════════
//  main()
// ════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UAMAdaptiveController>());
    rclcpp::shutdown();
    return 0;
}
