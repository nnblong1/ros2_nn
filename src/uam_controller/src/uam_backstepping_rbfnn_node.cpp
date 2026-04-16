#include "uam_controller/uam_adaptive_controller.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std::chrono_literals;

// ════════════════════════════════════════════════════════════════
// RBFNN Implementation
// ════════════════════════════════════════════════════════════════

RBFNeuralNetwork::RBFNeuralNetwork(const RBFNNParams& p) : params_(p) {
    W_hat_ = Eigen::MatrixXd::Zero(p.num_neurons, p.output_dim);
    C_centers_ = Eigen::MatrixXd::Random(p.num_neurons, p.input_dim) * 0.5;
    B_widths_ = Eigen::VectorXd::Constant(p.num_neurons, p.gaussian_width);
    Gamma_ = Eigen::MatrixXd::Identity(p.num_neurons, p.num_neurons) * p.learning_rate;
}

Eigen::VectorXd RBFNeuralNetwork::compute_basis(const Eigen::VectorXd& Z) const {
    Eigen::VectorXd h(params_.num_neurons);
    for (int i = 0; i < params_.num_neurons; ++i) {
        double ns = (Z - C_centers_.row(i).transpose()).squaredNorm();
        h(i) = std::exp(-ns / (2.0 * B_widths_(i) * B_widths_(i)));
    }
    return h;
}

Eigen::VectorXd RBFNeuralNetwork::estimate(const Eigen::VectorXd& Z) const {
    return W_hat_.transpose() * compute_basis(Z);
}

void RBFNeuralNetwork::update_weights(const Eigen::VectorXd& Z, const Eigen::VectorXd& e2, double dt) {
    auto h = compute_basis(Z);
    auto dW = Gamma_ * (h * e2.transpose() - params_.e_modification * e2.norm() * W_hat_);
    W_hat_ += dW * dt;
    W_hat_ = W_hat_.cwiseMin(5.0).cwiseMax(-5.0);
}

void RBFNeuralNetwork::reset() {
    W_hat_.setZero();
}

// ════════════════════════════════════════════════════════════════
// Node ROS2
// ════════════════════════════════════════════════════════════════

UAMAdaptiveController::UAMAdaptiveController() : Node("uam_adaptive_controller") {
    declare_params();
    // RBFNN phải khởi tạo SAU declare_params() để nhận đúng learning_rate từ YAML
    rbfnn_ = std::make_unique<RBFNeuralNetwork>(rbfnn_params_);

    auto qos_r = rclcpp::QoS(10).reliable();
    auto qos_be = rclcpp::SensorDataQoS();

    torque_pub_ = create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>("/fmu/in/vehicle_torque_setpoint", qos_be);
    thrust_pub_ = create_publisher<px4_msgs::msg::VehicleThrustSetpoint>("/fmu/in/vehicle_thrust_setpoint", qos_be);
    debug_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/uam/debug_state", 10);
    joint_tau_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/uam/joint_torque_cmd", 10);

    odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", qos_be, std::bind(&UAMAdaptiveController::odom_cb, this, std::placeholders::_1));
        
    rates_sp_sub_ = create_subscription<px4_msgs::msg::VehicleRatesSetpoint>(
        "/fmu/out/vehicle_rates_setpoint", qos_be, std::bind(&UAMAdaptiveController::rates_sp_cb, this, std::placeholders::_1));

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&UAMAdaptiveController::joint_cb, this, std::placeholders::_1));
        
    dyn_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/arm/system_dynamics", 10, std::bind(&UAMAdaptiveController::dyn_cb, this, std::placeholders::_1));
        
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/uam/controller_enable", 10, std::bind(&UAMAdaptiveController::enable_cb, this, std::placeholders::_1));

    state_sub_ = create_subscription<std_msgs::msg::String>(
        "/uam/state", qos_r, std::bind(&UAMAdaptiveController::state_cb, this, std::placeholders::_1));

    timer_ = create_wall_timer(5ms, std::bind(&UAMAdaptiveController::control_loop, this));

    RCLCPP_INFO(get_logger(), "UAM Adaptive Rate Controller RBFNN Initialized | 200Hz");
}

void UAMAdaptiveController::declare_params() {
    declare_parameter("mass_nominal", sys_.mass_nominal);
    declare_parameter("Ixx", sys_.Ixx);
    declare_parameter("Iyy", sys_.Iyy);
    declare_parameter("Izz", sys_.Izz);
    declare_parameter("rbfnn_enable", false);

    declare_parameter("rate_Kp_roll", rate_gains_.K_roll);
    declare_parameter("rate_Kp_pitch", rate_gains_.K_pitch);
    declare_parameter("rate_Kp_yaw", rate_gains_.K_yaw);
    declare_parameter("rate_Ki_roll", rate_gains_.Ki_roll);
    declare_parameter("rate_Ki_pitch", rate_gains_.Ki_pitch);
    declare_parameter("rate_Ki_yaw", rate_gains_.Ki_yaw);
    declare_parameter("rate_Kd_roll", rate_gains_.Kd_roll);
    declare_parameter("rate_Kd_pitch", rate_gains_.Kd_pitch);
    declare_parameter("rate_Kd_yaw", rate_gains_.Kd_yaw);
    declare_parameter("base_pitch_offset", 0.0);
    declare_parameter("base_roll_offset", 0.0);
    declare_parameter("joint_kp", 50.0);
    declare_parameter("joint_kd", 5.0);

    sys_.mass_nominal = get_parameter("mass_nominal").as_double();
    sys_.Ixx          = get_parameter("Ixx").as_double();
    sys_.Iyy          = get_parameter("Iyy").as_double();
    sys_.Izz          = get_parameter("Izz").as_double();
    rbfnn_output_enabled_ = get_parameter("rbfnn_enable").as_bool();

    // ★ FIX #1: Đọc rbfnn_lr từ YAML vào rbfnn_params_ TRƯỚC KHI khởi tạo RBFNN
    declare_parameter("rbfnn_lr", rbfnn_params_.learning_rate);
    rbfnn_params_.learning_rate = get_parameter("rbfnn_lr").as_double();
    RCLCPP_INFO(get_logger(), "RBFNN learning_rate loaded from YAML: %.5f", rbfnn_params_.learning_rate);

    rate_gains_.K_roll  = get_parameter("rate_Kp_roll").as_double();
    rate_gains_.K_pitch = get_parameter("rate_Kp_pitch").as_double();
    rate_gains_.K_yaw   = get_parameter("rate_Kp_yaw").as_double();
    rate_gains_.Ki_roll  = get_parameter("rate_Ki_roll").as_double();
    rate_gains_.Ki_pitch = get_parameter("rate_Ki_pitch").as_double();
    rate_gains_.Ki_yaw   = get_parameter("rate_Ki_yaw").as_double();
    rate_gains_.Kd_roll  = get_parameter("rate_Kd_roll").as_double();
    rate_gains_.Kd_pitch = get_parameter("rate_Kd_pitch").as_double();
    rate_gains_.Kd_yaw   = get_parameter("rate_Kd_yaw").as_double();
    base_pitch_offset_  = get_parameter("base_pitch_offset").as_double();
    base_roll_offset_   = get_parameter("base_roll_offset").as_double();

    double jkp = get_parameter("joint_kp").as_double();
    double jkd = get_parameter("joint_kd").as_double();
    for (int i = 0; i < N_JOINTS; ++i) {
        jg_[i].Kp = jkp;
        jg_[i].Kd = jkd;
    }
}

void UAMAdaptiveController::odom_cb(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
    // PX4 Angular Velocity is in FRD (Forward-Right-Down)
    // Convert to NWU/ENU for consistency if you want, but for pure tracking, 
    // keeping FRD matching the Rates Setpoint mapping is better.
    // PX4 VehicleRatesSetpoint is FRD. VehicleOdometry angular_velocity is FRD.
    omega_(0) = msg->angular_velocity[0]; // Roll speed rad/s
    omega_(1) = msg->angular_velocity[1]; // Pitch speed 
    omega_(2) = msg->angular_velocity[2]; // Yaw speed
    has_odom_ = true;
}

void UAMAdaptiveController::rates_sp_cb(const px4_msgs::msg::VehicleRatesSetpoint::SharedPtr msg) {
    // Setpoints are in FRD
    omega_des_(0) = msg->roll;
    omega_des_(1) = msg->pitch;
    omega_des_(2) = msg->yaw;
    px4_timestamp_ = msg->timestamp;

    // Lấy thrust được tính từ PX4 Attitude/Position Controller
    thrust_des_(0) = msg->thrust_body[0];
    thrust_des_(1) = msg->thrust_body[1];
    thrust_des_(2) = msg->thrust_body[2];
    
    has_rates_sp_ = true;
}

void UAMAdaptiveController::joint_cb(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if ((int)msg->position.size() < N_JOINTS) return;
    for (int i = 0; i < N_JOINTS; ++i) {
        q_[i]  = msg->position[i];
        dq_[i] = (msg->velocity.size() >= (size_t)N_JOINTS) ? msg->velocity[i] : 0.0;
    }
    has_joints_ = true;
}

void UAMAdaptiveController::dyn_cb(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if ((int)msg->data.size() < 156) return;
    D_inv_ = Eigen::Map<const Eigen::Matrix<double,12,12,Eigen::RowMajor>>(msg->data.data());
    H_vec_ = Eigen::Map<const Eigen::VectorXd>(msg->data.data() + 144, 12);
    dyn_ready_ = true;
}


void UAMAdaptiveController::enable_cb(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data && !controller_enabled_) {
        RCLCPP_INFO(get_logger(), "Rate Controller ENABLED. RBFNN ramp-up reset.");
        rbfnn_->reset();
        controller_start_time_ = -1.0; // Reset ramp timer cho flight session mới
        e_omega_int_.setZero();        // Reset integral
    }
    controller_enabled_ = msg->data;
}

void UAMAdaptiveController::state_cb(const std_msgs::msg::String::SharedPtr msg) {
    // Simple JSON parsing to get mission_state
    std::string data = msg->data;
    std::string old_state = mission_state_;

    if (data.find("\"TAKEOFF\"") != std::string::npos) mission_state_ = "TAKEOFF";
    else if (data.find("\"HOLD\"") != std::string::npos)    mission_state_ = "HOLD";
    else if (data.find("\"GOTO\"") != std::string::npos)    mission_state_ = "GOTO";
    else if (data.find("\"IDLE\"") != std::string::npos)    mission_state_ = "IDLE";
    else if (data.find("\"ARMED\"") != std::string::npos)   mission_state_ = "ARMED";
    else mission_state_ = "OTHER";

    if (mission_state_ != old_state) {
        RCLCPP_INFO(get_logger(), "Mission State changed: %s -> %s", old_state.c_str(), mission_state_.c_str());
    }
}

double UAMAdaptiveController::sat(double v, double lim) const {
    return std::clamp(v, -lim, lim);
}

// ════════════════════════════════════════════════════════════════
// VÒNG LẶP ĐIỀU KHIỂN CHÍNH
// ════════════════════════════════════════════════════════════════
void UAMAdaptiveController::control_loop() {
    bool can_compute = (has_odom_ && has_rates_sp_ && controller_enabled_);

    double now = get_clock()->now().seconds();
    double dt  = (last_t_ > 0.0) ? (now - last_t_) : 0.005;
    dt = std::clamp(dt, 0.001, 0.02);
    last_t_ = now;

    Eigen::Vector3d tau_norm = Eigen::Vector3d::Zero();
    Eigen::Vector3d thrust_norm = Eigen::Vector3d::Zero();
    Eigen::Vector3d tau = Eigen::Vector3d::Zero();

    if (can_compute) {
        bool is_takeoff_or_idle = (mission_state_ == "TAKEOFF" || mission_state_ == "ARMED" || mission_state_ == "IDLE");

        // ★ Ghi nhận thời điểm controller bắt đầu hoạt động
        if (controller_start_time_ < 0.0) {
            controller_start_time_ = now;
            RCLCPP_INFO(get_logger(), "⏱️ Controller start time recorded. RBFNN Ramp-up begins.");
        }
        double elapsed = now - controller_start_time_;

        // 1. Tính toán Sai số (Error) LUÔN CHẠY
        e_omega_ = omega_ - omega_des_;

        // 2. ★ RBFNN RAMP-UP STRATEGY: Luôn học, output tăng dần
        //    Thay vì tắt cứng khi takeoff, RBFNN luôn cập nhật trọng số.
        //    Output bị clamp theo thời gian để đảm bảo an toàn.
        {
            Eigen::VectorXd Z(9);
            Z << e_omega_, omega_des_, omega_;

            // LUÔN cập nhật trọng số (học từ giây đầu tiên)
            if (rbfnn_output_enabled_) {
                rbfnn_->update_weights(Z, e_omega_, dt);

                // Xác định mức clamp theo thời gian (Ramp-up)
                double rbfnn_clamp;
                if (elapsed < RAMP_PHASE1_END) {
                    rbfnn_clamp = RAMP_P1_LIMIT;  // 0-3s: ±0.05 Nm
                } else if (elapsed < RAMP_PHASE2_END) {
                    rbfnn_clamp = RAMP_P2_LIMIT;  // 3-8s: ±0.15 Nm
                } else {
                    rbfnn_clamp = RAMP_FULL_LIMIT; // >8s:  ±0.50 Nm
                }

                Eigen::VectorXd n_est = rbfnn_->estimate(Z);
                n_hat_(0) = sat(n_est(0), rbfnn_clamp);
                n_hat_(1) = sat(n_est(1), rbfnn_clamp);
                n_hat_(2) = sat(n_est(2), rbfnn_clamp);
            } else {
                n_hat_.setZero();
            }
        }
        
        // 3. Luật Điều Khiển Rate - Full PID (LUÔN CHẠY)
        // PX4 internal rate gains = 0 → node này là nguồn torque DUY NHẤT
        Eigen::Matrix3d J_mat = Eigen::Vector3d(sys_.Ixx, sys_.Iyy, sys_.Izz).asDiagonal();
        Eigen::Vector3d coriolis = omega_.cross(J_mat * omega_);
        Eigen::Matrix3d Kp_mat = Eigen::Vector3d(rate_gains_.K_roll, rate_gains_.K_pitch, rate_gains_.K_yaw).asDiagonal();
        Eigen::Matrix3d Ki_mat = Eigen::Vector3d(rate_gains_.Ki_roll, rate_gains_.Ki_pitch, rate_gains_.Ki_yaw).asDiagonal();
        Eigen::Matrix3d Kd_mat = Eigen::Vector3d(rate_gains_.Kd_roll, rate_gains_.Kd_pitch, rate_gains_.Kd_yaw).asDiagonal();

        // Integral — LUÔN tích lũy, clamp chặt hơn khi takeoff
        double int_clamp = is_takeoff_or_idle ? 0.2 : 0.5;
        e_omega_int_ += e_omega_ * dt;
        e_omega_int_ = e_omega_int_.cwiseMax(-int_clamp).cwiseMin(int_clamp);

        // Derivative (trên sai số) + ★ FIX #4: Low-Pass Filter để lọc nhiễu cao tần
        Eigen::Vector3d e_omega_dot_raw = (e_omega_ - e_omega_prev_) / dt;
        Eigen::Vector3d e_omega_dot = lpf_alpha_ * e_omega_dot_raw + (1.0 - lpf_alpha_) * e_omega_dot_prev_;
        e_omega_dot_prev_ = e_omega_dot;
        e_omega_prev_ = e_omega_;
        
        tau = -Kp_mat * e_omega_ - Ki_mat * e_omega_int_ - Kd_mat * e_omega_dot + coriolis - n_hat_;

        // Normalized Torque for PX4
        tau_norm = tau / sys_.max_torque;
        
        // 4. Feedforward CG Offset Compensation (Cân bằng trọng lượng cánh tay tĩnh)
        tau_norm(0) += base_roll_offset_;
        tau_norm(1) += base_pitch_offset_;
        
        // ★ FIX #3: Thrust saturation để chống flyaway
        thrust_norm(0) = std::clamp(thrust_des_(0), -0.1, 0.1);
        thrust_norm(1) = std::clamp(thrust_des_(1), -0.1, 0.1);
        thrust_norm(2) = std::clamp(thrust_des_(2), -1.0, -0.05);
    } else {
        n_hat_.setZero();
        // Không nhận đủ điều kiện tính toán -> Lực = 0 để nuôi Control Allocator
    }
    
    // 4. LUÔN LUÔN Gửi ngược Lực đẩy (Thrust) và Mô-men (Torque) về cho PX4
    // Để giữ "mạng sống" cho Control Allocator
    if (px4_timestamp_ == 0) return; 

    // Approach B: Giữ position=True ở Mission Bridge để PX4 vẫn chạy Position/Attitude Controller
    // và sinh ra VehicleRatesSetpoint. PX4 Rate Controller gains = 0 nên không xung đột.
    // QUAN TRỌNG: Nếu dùng direct_actuator=true, PX4 sẽ không sinh RateSetpoint → hỏng!

    px4_msgs::msg::VehicleTorqueSetpoint torque_msg{};
    torque_msg.xyz[0] = static_cast<float>(sat(tau_norm(0), 1.0));
    torque_msg.xyz[1] = static_cast<float>(sat(tau_norm(1), 1.0));
    torque_msg.xyz[2] = static_cast<float>(sat(tau_norm(2), 1.0));
    torque_msg.timestamp = px4_timestamp_;
    torque_msg.timestamp_sample = px4_timestamp_;
    torque_pub_->publish(torque_msg);

    px4_msgs::msg::VehicleThrustSetpoint thrust_msg{};
    thrust_msg.xyz[0] = static_cast<float>(thrust_norm(0));
    thrust_msg.xyz[1] = static_cast<float>(thrust_norm(1));
    thrust_msg.xyz[2] = static_cast<float>(thrust_norm(2));
    thrust_msg.timestamp = px4_timestamp_;
    thrust_msg.timestamp_sample = px4_timestamp_;
    thrust_pub_->publish(thrust_msg);

    // 5. Cấp lệnh cho cánh tay máy (Đã tách biệt để hoạt động ngay khi controller_enabled)
    if (controller_enabled_ && has_joints_) {
        Eigen::VectorXd tau_j = compute_joint_control();
        std_msgs::msg::Float64MultiArray joint_msg;
        for (int i = 0; i < N_JOINTS; ++i) joint_msg.data.push_back(tau_j(i));
        joint_tau_pub_->publish(joint_msg);
    }
    
    // Pub Debug
    std_msgs::msg::Float64MultiArray dbg;
    dbg.data.insert(dbg.data.end(), {omega_(0), omega_(1), omega_(2)});
    dbg.data.insert(dbg.data.end(), {omega_des_(0), omega_des_(1), omega_des_(2)});
    dbg.data.insert(dbg.data.end(), {e_omega_(0), e_omega_(1), e_omega_(2)});
    dbg.data.insert(dbg.data.end(), {n_hat_(0), n_hat_(1), n_hat_(2)});
    dbg.data.insert(dbg.data.end(), {tau(0), tau(1), tau(2)});
    debug_pub_->publish(dbg);
}

// Hàm tính lực cho khớp tay máy
Eigen::VectorXd UAMAdaptiveController::compute_joint_control() {
    Eigen::VectorXd tau_joints = Eigen::VectorXd::Zero(N_JOINTS);
    
    // CỐ ĐỊNH CÁNH TAY TRONG KHI CẤT CÁNH (TAKEOFF) hoặc CHƯA SẴN SÀNG DYNAMICS
    // Force PD control only (ignore dynamics/coupling) during sensitive takeoff phase
    bool force_pd = (!dyn_ready_ || mission_state_ == "TAKEOFF" || mission_state_ == "ARMED" || mission_state_ == "IDLE");

    if (force_pd) {
        for (int i = 0; i < N_JOINTS; ++i) {
            double ei1 = q_[i] - qd_[i];
            double ei2 = dq_[i] - dqd_[i] + jg_[i].Kp * ei1;
            tau_joints(i) = sat(-jg_[i].Kp * (ei2 - jg_[i].Kp*ei1) - ei1 - jg_[i].Kd * ei2, sys_.max_joint_tau);
        }
        return tau_joints;
    }

    Eigen::VectorXd Qdd(N_JOINTS);
    for (int i = 0; i < N_JOINTS; ++i) {
        double ei1 = q_[i] - qd_[i];
        double ei2 = dq_[i] - dqd_[i] + jg_[i].Kp * ei1;
        Qdd(i) = -jg_[i].Kp * (ei2 - jg_[i].Kp * ei1) - ei1 - jg_[i].Kd * ei2;
    }

    Eigen::MatrixXd D_inv_arm_uav = D_inv_.block(6, 0, N_JOINTS, 6);
    Eigen::MatrixXd D_inv_arm_arm = D_inv_.block(6, 6, N_JOINTS, N_JOINTS);
    Eigen::VectorXd H_arm = H_vec_.segment(6, N_JOINTS);
    Eigen::VectorXd H_uav = H_vec_.segment(0, 6);

    Eigen::VectorXd tau_uav = Eigen::VectorXd::Zero(6);
    // Ở mode Rate Controller, Thrust xấp xỉ bằng Z force
    tau_uav(2) = sys_.mass_nominal * sys_.gravity; 

    Eigen::VectorXd rhs = Qdd - D_inv_arm_uav * (tau_uav - H_uav);
    tau_joints = D_inv_arm_arm.lu().solve(rhs) + H_arm;

    for (int i = 0; i < N_JOINTS; ++i) tau_joints(i) = sat(tau_joints(i), sys_.max_joint_tau);
    return tau_joints;
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<UAMAdaptiveController>());
    rclcpp::shutdown();
    return 0;
}
