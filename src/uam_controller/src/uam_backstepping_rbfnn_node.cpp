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
    C_centers_ = Eigen::MatrixXd::Zero(p.num_neurons, p.input_dim);

    // Deterministic low-discrepancy centers. This avoids a different controller
    // response at every process start while keeping broad RBF coverage.
    for (int i = 0; i < p.num_neurons; ++i) {
        for (int j = 0; j < p.input_dim; ++j) {
            const int hashed = (37 * (i + 1) + 17 * (j + 1)) % 101;
            C_centers_(i, j) = -1.0 + 2.0 * static_cast<double>(hashed) / 100.0;
        }
    }

    B_widths_ = Eigen::VectorXd::Constant(p.num_neurons, p.gaussian_width);
    Gamma_ = Eigen::MatrixXd::Identity(p.num_neurons, p.num_neurons) * p.learning_rate;
}

Eigen::VectorXd RBFNeuralNetwork::compute_basis(const Eigen::VectorXd& Z) const {
    Eigen::VectorXd h(params_.num_neurons);
    for (int i = 0; i < params_.num_neurons; ++i) {
        double ns = (Z - C_centers_.row(i).transpose()).squaredNorm();
        h(i) = std::exp(-ns / (B_widths_(i) * B_widths_(i)));
    }
    return h;
}

Eigen::VectorXd RBFNeuralNetwork::estimate(const Eigen::VectorXd& Z) const {
    return W_hat_.transpose() * compute_basis(Z);
}

void RBFNeuralNetwork::update_weights(const Eigen::VectorXd& Z, const Eigen::VectorXd& e2, double dt) {
    auto h = compute_basis(Z);
    Eigen::MatrixXd dW = params_.learning_rate * h * e2.transpose();

    if (params_.e_modification > 0.0) {
        dW -= params_.learning_rate * params_.e_modification * e2.norm() * W_hat_;
    }

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

    auto qos_be = rclcpp::SensorDataQoS();

    torque_pub_ = create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>("/fmu/in/vehicle_torque_setpoint", qos_be);
    thrust_pub_ = create_publisher<px4_msgs::msg::VehicleThrustSetpoint>("/fmu/in/vehicle_thrust_setpoint", qos_be);
    debug_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/uam/debug_state", 10);
    joint_tau_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/uam/joint_torque_cmd", 10);

    odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", qos_be, std::bind(&UAMAdaptiveController::odom_cb, this, std::placeholders::_1));
        
    rates_sp_sub_ = create_subscription<px4_msgs::msg::VehicleRatesSetpoint>(
        "/fmu/out/vehicle_rates_setpoint", qos_be, std::bind(&UAMAdaptiveController::rates_sp_cb, this, std::placeholders::_1));

    land_sub_ = create_subscription<px4_msgs::msg::VehicleLandDetected>(
        "/fmu/out/vehicle_land_detected", qos_be, std::bind(&UAMAdaptiveController::land_cb, this, std::placeholders::_1));

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&UAMAdaptiveController::joint_cb, this, std::placeholders::_1));
        
    dyn_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/arm/system_dynamics", 10, std::bind(&UAMAdaptiveController::dyn_cb, this, std::placeholders::_1));

    arm_wrench_sub_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
        "/arm/interaction_wrench", 10, std::bind(&UAMAdaptiveController::arm_wrench_cb, this, std::placeholders::_1));
        
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/uam/controller_enable", 10, std::bind(&UAMAdaptiveController::enable_cb, this, std::placeholders::_1));

    timer_ = create_wall_timer(5ms, std::bind(&UAMAdaptiveController::control_loop, this));

    RCLCPP_INFO(get_logger(), "UAM Adaptive Rate Controller RBFNN Initialized | 200Hz");
}

void UAMAdaptiveController::declare_params() {
    declare_parameter("mass_nominal", sys_.mass_nominal);
    declare_parameter("Ixx", sys_.Ixx);
    declare_parameter("Iyy", sys_.Iyy);
    declare_parameter("Izz", sys_.Izz);
    declare_parameter("Ixy", sys_.Ixy);
    declare_parameter("Ixz", sys_.Ixz);
    declare_parameter("Iyz", sys_.Iyz);
    declare_parameter("max_torque", sys_.max_torque);
    declare_parameter("max_joint_tau", sys_.max_joint_tau);
    declare_parameter("gravity", sys_.gravity);
    declare_parameter("rbfnn_enable", false);
    declare_parameter("rbfnn_num_neurons", rbfnn_params_.num_neurons);
    declare_parameter("rbfnn_gaussian_width", rbfnn_params_.gaussian_width);
    declare_parameter("rbfnn_e_modification", rbfnn_params_.e_modification);
    declare_parameter("rbfnn_output_gain", rbfnn_output_gain_);

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
    declare_parameter("tau_max_roll_nm", sys_.max_torque);
    declare_parameter("tau_max_pitch_nm", sys_.max_torque);
    declare_parameter("tau_max_yaw_nm", sys_.max_torque);
    declare_parameter("arm_ff_enable", false);
    declare_parameter("arm_ff_timeout_s", arm_ff_timeout_s_);
    declare_parameter("arm_ff_lpf_alpha", arm_ff_lpf_alpha_);
    declare_parameter("arm_ff_start_delay_s", arm_ff_start_delay_s_);
    declare_parameter("arm_ff_ramp_s", arm_ff_ramp_s_);
    declare_parameter("arm_ff_rate_limit_nm_s", arm_ff_rate_limit_nm_s_);
    declare_parameter("arm_ff_max_roll_nm", arm_ff_limit_(0));
    declare_parameter("arm_ff_max_pitch_nm", arm_ff_limit_(1));
    declare_parameter("arm_ff_max_yaw_nm", arm_ff_limit_(2));
    declare_parameter("arm_ff_scale_roll", arm_ff_scale_(0));
    declare_parameter("arm_ff_scale_pitch", arm_ff_scale_(1));
    declare_parameter("arm_ff_scale_yaw", arm_ff_scale_(2));
    declare_parameter("arm_ff_input_frame", arm_ff_input_frame_);
    declare_parameter("arm_ff_reaction_sign", arm_ff_reaction_sign_);
    declare_parameter("arm_virtual_disturbance_enable", arm_virtual_disturbance_enabled_);
    declare_parameter("arm_virtual_disturbance_max_roll_nm", 0.12);
    declare_parameter("arm_virtual_disturbance_max_pitch_nm", 0.12);
    declare_parameter("arm_virtual_disturbance_max_yaw_nm", 0.06);
    declare_parameter("arm_virtual_disturbance_scale_roll", arm_virtual_disturbance_scale_(0));
    declare_parameter("arm_virtual_disturbance_scale_pitch", arm_virtual_disturbance_scale_(1));
    declare_parameter("arm_virtual_disturbance_scale_yaw", arm_virtual_disturbance_scale_(2));
    declare_parameter("arm_virtual_disturbance_reaction_sign", arm_virtual_disturbance_reaction_sign_);
    declare_parameter("arm_cg_comp_enable", arm_cg_comp_enabled_);
    declare_parameter("arm_cg_roll_gain", arm_cg_roll_gain_);
    declare_parameter("arm_cg_pitch_gain", arm_cg_pitch_gain_);
    declare_parameter("arm_cg_max_norm", arm_cg_max_norm_);
    declare_parameter("arm_cg_lpf_alpha", arm_cg_lpf_alpha_);

    sys_.mass_nominal = get_parameter("mass_nominal").as_double();
    sys_.Ixx          = get_parameter("Ixx").as_double();
    sys_.Iyy          = get_parameter("Iyy").as_double();
    sys_.Izz          = get_parameter("Izz").as_double();
    sys_.Ixy          = get_parameter("Ixy").as_double();
    sys_.Ixz          = get_parameter("Ixz").as_double();
    sys_.Iyz          = get_parameter("Iyz").as_double();
    sys_.max_torque   = get_parameter("max_torque").as_double();
    sys_.max_joint_tau= get_parameter("max_joint_tau").as_double();
    sys_.gravity      = get_parameter("gravity").as_double();
    rbfnn_output_enabled_ = get_parameter("rbfnn_enable").as_bool();

    // ★ FIX #1: Đọc rbfnn_lr từ YAML vào rbfnn_params_ TRƯỚC KHI khởi tạo RBFNN
    declare_parameter("rbfnn_lr", rbfnn_params_.learning_rate);
    rbfnn_params_.num_neurons = std::max(12, static_cast<int>(get_parameter("rbfnn_num_neurons").as_int()));
    rbfnn_params_.input_dim = RBFNN_INPUT_DIM;
    rbfnn_params_.output_dim = 3;
    rbfnn_params_.learning_rate = std::clamp(get_parameter("rbfnn_lr").as_double(), 1.0e-5, 0.05);
    rbfnn_params_.gaussian_width = std::clamp(get_parameter("rbfnn_gaussian_width").as_double(), 2.0, 6.0);
    rbfnn_params_.e_modification = std::clamp(get_parameter("rbfnn_e_modification").as_double(), 0.0, 0.2);
    rbfnn_output_gain_ = std::clamp(get_parameter("rbfnn_output_gain").as_double(), 0.0, 1.0);
    RCLCPP_INFO(
        get_logger(),
        "RBFNN params: input_dim=%d, neurons=%d, lr=%.5f, width=%.3f, e_mod=%.4f, output_gain=%.3f",
        rbfnn_params_.input_dim,
        rbfnn_params_.num_neurons,
        rbfnn_params_.learning_rate,
        rbfnn_params_.gaussian_width,
        rbfnn_params_.e_modification,
        rbfnn_output_gain_);

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
    tau_axis_max_(0) = std::max(1e-6, get_parameter("tau_max_roll_nm").as_double());
    tau_axis_max_(1) = std::max(1e-6, get_parameter("tau_max_pitch_nm").as_double());
    tau_axis_max_(2) = std::max(1e-6, get_parameter("tau_max_yaw_nm").as_double());
    arm_ff_enabled_ = get_parameter("arm_ff_enable").as_bool();
    arm_ff_timeout_s_ = get_parameter("arm_ff_timeout_s").as_double();
    arm_ff_lpf_alpha_ = std::clamp(get_parameter("arm_ff_lpf_alpha").as_double(), 0.0, 1.0);
    arm_ff_start_delay_s_ = std::max(0.0, get_parameter("arm_ff_start_delay_s").as_double());
    arm_ff_ramp_s_ = std::max(0.1, get_parameter("arm_ff_ramp_s").as_double());
    arm_ff_rate_limit_nm_s_ = std::max(0.0, get_parameter("arm_ff_rate_limit_nm_s").as_double());
    arm_ff_limit_(0) = std::abs(get_parameter("arm_ff_max_roll_nm").as_double());
    arm_ff_limit_(1) = std::abs(get_parameter("arm_ff_max_pitch_nm").as_double());
    arm_ff_limit_(2) = std::abs(get_parameter("arm_ff_max_yaw_nm").as_double());
    arm_ff_scale_(0) = std::clamp(get_parameter("arm_ff_scale_roll").as_double(), -1.5, 1.5);
    arm_ff_scale_(1) = std::clamp(get_parameter("arm_ff_scale_pitch").as_double(), -1.5, 1.5);
    arm_ff_scale_(2) = std::clamp(get_parameter("arm_ff_scale_yaw").as_double(), -1.5, 1.5);
    arm_ff_input_frame_ = get_parameter("arm_ff_input_frame").as_string();
    arm_ff_reaction_sign_ = get_parameter("arm_ff_reaction_sign").as_double() < 0.0 ? -1.0 : 1.0;
    arm_virtual_disturbance_enabled_ = get_parameter("arm_virtual_disturbance_enable").as_bool();
    arm_virtual_disturbance_limit_(0) = std::abs(get_parameter("arm_virtual_disturbance_max_roll_nm").as_double());
    arm_virtual_disturbance_limit_(1) = std::abs(get_parameter("arm_virtual_disturbance_max_pitch_nm").as_double());
    arm_virtual_disturbance_limit_(2) = std::abs(get_parameter("arm_virtual_disturbance_max_yaw_nm").as_double());
    arm_virtual_disturbance_scale_(0) = std::clamp(get_parameter("arm_virtual_disturbance_scale_roll").as_double(), -3.0, 3.0);
    arm_virtual_disturbance_scale_(1) = std::clamp(get_parameter("arm_virtual_disturbance_scale_pitch").as_double(), -3.0, 3.0);
    arm_virtual_disturbance_scale_(2) = std::clamp(get_parameter("arm_virtual_disturbance_scale_yaw").as_double(), -3.0, 3.0);
    arm_virtual_disturbance_reaction_sign_ =
        get_parameter("arm_virtual_disturbance_reaction_sign").as_double() < 0.0 ? -1.0 : 1.0;
    arm_cg_comp_enabled_ = get_parameter("arm_cg_comp_enable").as_bool();
    arm_cg_roll_gain_ = get_parameter("arm_cg_roll_gain").as_double();
    arm_cg_pitch_gain_ = get_parameter("arm_cg_pitch_gain").as_double();
    arm_cg_max_norm_ = std::abs(get_parameter("arm_cg_max_norm").as_double());
    arm_cg_lpf_alpha_ = std::clamp(get_parameter("arm_cg_lpf_alpha").as_double(), 0.0, 1.0);

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
    altitude_m_ = -msg->position[2];
    vertical_speed_m_s_ = -msg->velocity[2];
    last_odom_rx_time_ = get_clock()->now().seconds();
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
    
    last_rates_sp_rx_time_ = get_clock()->now().seconds();
    has_rates_sp_ = true;
}

void UAMAdaptiveController::land_cb(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg) {
    landed_ = msg->landed;
    ground_contact_ = msg->ground_contact;
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

void UAMAdaptiveController::arm_wrench_cb(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
    const Eigen::Vector3d tau_msg(
        msg->wrench.torque.x,
        msg->wrench.torque.y,
        msg->wrench.torque.z);

    Eigen::Vector3d raw_tau = tau_msg;

    // arm_dynamics_node publishes in Gazebo/base_link convention by default:
    // X forward, Y left, Z up (FLU-like). PX4 VehicleTorqueSetpoint expects
    // body FRD: X forward, Y right, Z down. Convert before applying sign/scale.
    if (arm_ff_input_frame_ == "flu" || arm_ff_input_frame_ == "gazebo" || arm_ff_input_frame_ == "base_link") {
        raw_tau = Eigen::Vector3d(tau_msg(0), -tau_msg(1), -tau_msg(2));
    } else if (arm_ff_input_frame_ == "frd" || arm_ff_input_frame_ == "px4") {
        raw_tau = tau_msg;
    } else {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            5000,
            "Unknown arm_ff_input_frame='%s'. Assuming PX4 FRD.",
            arm_ff_input_frame_.c_str());
        raw_tau = tau_msg;
    }

    Eigen::Vector3d ff_tau = arm_ff_reaction_sign_ * raw_tau;
    ff_tau = arm_ff_scale_.cwiseProduct(ff_tau);
    ff_tau = sat_vec(ff_tau, arm_ff_limit_);
    tau_arm_ff_target_ = arm_ff_lpf_alpha_ * ff_tau + (1.0 - arm_ff_lpf_alpha_) * tau_arm_ff_target_;

    Eigen::Vector3d disturbance_tau = arm_virtual_disturbance_reaction_sign_ * raw_tau;
    disturbance_tau = arm_virtual_disturbance_scale_.cwiseProduct(disturbance_tau);
    disturbance_tau = sat_vec(disturbance_tau, arm_virtual_disturbance_limit_);
    tau_arm_virtual_disturbance_target_ =
        arm_ff_lpf_alpha_ * disturbance_tau
        + (1.0 - arm_ff_lpf_alpha_) * tau_arm_virtual_disturbance_target_;

    last_arm_wrench_rx_time_ = get_clock()->now().seconds();
    has_arm_wrench_ = true;
}


void UAMAdaptiveController::enable_cb(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data && !controller_enabled_) {
        RCLCPP_INFO(get_logger(), "Rate Controller ENABLED. RBFNN ramp-up reset.");
        rbfnn_->reset();
        controller_start_time_ = -1.0; // Reset ramp timer cho flight session mới
        e_omega_int_.setZero();        // Reset integral
        e_omega_prev_.setZero();
        e_omega_dot_prev_.setZero();
        n_hat_.setZero();
        tau_arm_ff_.setZero();
        tau_arm_ff_target_.setZero();
        tau_arm_virtual_disturbance_.setZero();
        tau_arm_virtual_disturbance_target_.setZero();
        arm_cg_bias_norm_.setZero();
    } else if (!msg->data && controller_enabled_) {
        RCLCPP_INFO(get_logger(), "Rate Controller DISABLED. PX4 internal rate controller fallback remains active.");
        rbfnn_->reset();
        controller_start_time_ = -1.0;
        e_omega_int_.setZero();
        e_omega_prev_.setZero();
        e_omega_dot_prev_.setZero();
        n_hat_.setZero();
        tau_arm_ff_.setZero();
        tau_arm_ff_target_.setZero();
        tau_arm_virtual_disturbance_.setZero();
        tau_arm_virtual_disturbance_target_.setZero();
        arm_cg_bias_norm_.setZero();
    }
    controller_enabled_ = msg->data;
}

bool UAMAdaptiveController::inputs_fresh(double now) const {
    const bool odom_fresh = has_odom_ && last_odom_rx_time_ > 0.0 && (now - last_odom_rx_time_) < 0.1;
    const bool rates_sp_fresh = has_rates_sp_ && last_rates_sp_rx_time_ > 0.0 && (now - last_rates_sp_rx_time_) < 0.1;
    return odom_fresh && rates_sp_fresh && arm_ff_fresh(now);
}

bool UAMAdaptiveController::arm_ff_fresh(double now) const {
    if (!arm_ff_enabled_ && !arm_virtual_disturbance_enabled_) return true;
    return has_arm_wrench_
        && last_arm_wrench_rx_time_ > 0.0
        && (now - last_arm_wrench_rx_time_) < arm_ff_timeout_s_;
}

bool UAMAdaptiveController::in_takeoff_sensitive_phase(double elapsed_since_enable) const {
    return landed_
        || ground_contact_
        || altitude_m_ < 1.8
        || std::abs(vertical_speed_m_s_) > 0.35
        || elapsed_since_enable < RAMP_PHASE2_END;
}

double UAMAdaptiveController::sat(double v, double lim) const {
    return std::clamp(v, -lim, lim);
}

Eigen::Vector3d UAMAdaptiveController::sat_vec(const Eigen::Vector3d& v, const Eigen::Vector3d& lim) const {
    Eigen::Vector3d out;
    for (int i = 0; i < 3; ++i) out(i) = sat(v(i), std::abs(lim(i)));
    return out;
}

Eigen::Vector3d UAMAdaptiveController::rate_limit_vec(
    const Eigen::Vector3d& current,
    const Eigen::Vector3d& target,
    double max_delta) const
{
    if (max_delta <= 0.0) return target;

    Eigen::Vector3d out = current;
    for (int i = 0; i < 3; ++i) {
        const double delta = std::clamp(target(i) - current(i), -max_delta, max_delta);
        out(i) += delta;
    }
    return out;
}

void UAMAdaptiveController::update_arm_feedforward(
    double elapsed_since_enable,
    double dt,
    bool takeoff_sensitive)
{
    Eigen::Vector3d target = Eigen::Vector3d::Zero();

    if (arm_ff_enabled_ && has_arm_wrench_ && !takeoff_sensitive) {
        const double ramp = std::clamp(
            (elapsed_since_enable - arm_ff_start_delay_s_) / arm_ff_ramp_s_,
            0.0,
            1.0);
        target = ramp * tau_arm_ff_target_;
    }

    const double max_delta = arm_ff_rate_limit_nm_s_ * dt;
    tau_arm_ff_ = rate_limit_vec(tau_arm_ff_, target, max_delta);
}

void UAMAdaptiveController::update_arm_virtual_disturbance(
    double elapsed_since_enable,
    double dt,
    bool takeoff_sensitive)
{
    Eigen::Vector3d target = Eigen::Vector3d::Zero();

    if (arm_virtual_disturbance_enabled_ && has_arm_wrench_ && !takeoff_sensitive) {
        const double ramp = std::clamp(
            (elapsed_since_enable - arm_ff_start_delay_s_) / arm_ff_ramp_s_,
            0.0,
            1.0);
        target = ramp * tau_arm_virtual_disturbance_target_;
    }

    const double max_delta = arm_ff_rate_limit_nm_s_ * dt;
    tau_arm_virtual_disturbance_ =
        rate_limit_vec(tau_arm_virtual_disturbance_, target, max_delta);
}

void UAMAdaptiveController::update_arm_cg_bias()
{
    Eigen::Vector2d target = Eigen::Vector2d::Zero();

    if (arm_cg_comp_enabled_ && has_joints_) {
        const double q1 = q_[0];
        const double q2 = q_[1];
        const double q3 = q_[2];
        const double q4 = q_[3];
        const double q5 = q_[4];
        const double q6 = q_[5];

        target(0) = arm_cg_roll_gain_ * (
            std::sin(q1) + 0.4 * std::sin(q4) + 0.2 * std::sin(q6));
        target(1) = arm_cg_pitch_gain_ * (
            std::sin(q2) + 0.7 * std::sin(q2 + q3) + 0.3 * std::sin(q5));

        target(0) = sat(target(0), arm_cg_max_norm_);
        target(1) = sat(target(1), arm_cg_max_norm_);
    }

    arm_cg_bias_norm_ =
        arm_cg_lpf_alpha_ * target + (1.0 - arm_cg_lpf_alpha_) * arm_cg_bias_norm_;
}

Eigen::VectorXd UAMAdaptiveController::build_rbfnn_input() const
{
    Eigen::VectorXd Z = Eigen::VectorXd::Zero(RBFNN_INPUT_DIM);
    int idx = 0;

    auto push_scaled = [&](double value, double scale) {
        if (idx >= RBFNN_INPUT_DIM) return;
        const double denom = std::max(std::abs(scale), 1.0e-6);
        Z(idx++) = sat(value / denom, 1.0);
    };

    for (int i = 0; i < 3; ++i) push_scaled(omega_(i), 1.0);       // rad/s
    for (int i = 0; i < 3; ++i) push_scaled(e_omega_(i), 1.0);     // rad/s
    for (int i = 0; i < N_JOINTS; ++i) push_scaled(q_[i], 1.0);    // rad
    for (int i = 0; i < N_JOINTS; ++i) push_scaled(dq_[i], 1.0);   // rad/s

    const Eigen::Vector3d residual_tau = tau_arm_virtual_disturbance_ - tau_arm_ff_;
    for (int i = 0; i < 3; ++i) push_scaled(residual_tau(i), std::max(arm_virtual_disturbance_limit_(i), 1.0e-3));

    return Z;
}

bool UAMAdaptiveController::rbfnn_ready(bool takeoff_sensitive, double now) const
{
    return rbfnn_output_enabled_
        && !takeoff_sensitive
        && has_joints_
        && has_arm_wrench_
        && arm_virtual_disturbance_enabled_
        && arm_ff_fresh(now)
        && tau_arm_virtual_disturbance_.norm() > 1.0e-4;
}

// ════════════════════════════════════════════════════════════════
// VÒNG LẶP ĐIỀU KHIỂN CHÍNH
// ════════════════════════════════════════════════════════════════
void UAMAdaptiveController::control_loop() {
    double now = get_clock()->now().seconds();
    double dt  = (last_t_ > 0.0) ? (now - last_t_) : 0.005;
    dt = std::clamp(dt, 0.001, 0.02);
    last_t_ = now;
    const bool can_compute = controller_enabled_ && inputs_fresh(now);

    Eigen::Vector3d tau_norm = Eigen::Vector3d::Zero();
    Eigen::Vector3d thrust_norm = Eigen::Vector3d::Zero();
    Eigen::Vector3d tau = Eigen::Vector3d::Zero();
    bool takeoff_sensitive = true;

    if (can_compute) {
        // ★ Ghi nhận thời điểm controller bắt đầu hoạt động
        if (controller_start_time_ < 0.0) {
            controller_start_time_ = now;
            RCLCPP_INFO(get_logger(), "⏱️ Controller start time recorded. RBFNN Ramp-up begins.");
        }
        double elapsed = now - controller_start_time_;
        takeoff_sensitive = in_takeoff_sensitive_phase(elapsed);

        // 1. Tính toán Sai số (Error) LUÔN CHẠY
        e_omega_ = omega_ - omega_des_;

        update_arm_feedforward(elapsed, dt, takeoff_sensitive);
        update_arm_virtual_disturbance(elapsed, dt, takeoff_sensitive);
        update_arm_cg_bias();

        // 2. RBFNN residual compensation. Chỉ học/xuất khi nhiễu cánh tay ảo
        //    đã active để tránh học bias hover và gây drift XY.
        {
            Eigen::VectorXd Z = build_rbfnn_input();

            if (rbfnn_ready(takeoff_sensitive, now)) {
                rbfnn_->update_weights(Z, e_omega_, dt);

                // Xác định mức clamp theo thời gian (Ramp-up), đơn vị rad/s^2
                double rbfnn_clamp;
                if (elapsed < RAMP_PHASE1_END) {
                    rbfnn_clamp = RAMP_P1_LIMIT;  // 0-3s: ±0.05 rad/s^2
                } else if (elapsed < RAMP_PHASE2_END) {
                    rbfnn_clamp = RAMP_P2_LIMIT;  // 3-8s: ±0.15 rad/s^2
                } else {
                    rbfnn_clamp = RAMP_FULL_LIMIT; // >8s:  ±0.50 rad/s^2
                }

                Eigen::VectorXd n_est = rbfnn_->estimate(Z);
                n_hat_(0) = rbfnn_output_gain_ * sat(n_est(0), rbfnn_clamp);
                n_hat_(1) = rbfnn_output_gain_ * sat(n_est(1), rbfnn_clamp);
                n_hat_(2) = rbfnn_output_gain_ * sat(n_est(2), rbfnn_clamp);
            } else {
                n_hat_.setZero();
            }
        }
        
        // 3. Luật Điều Khiển Rate - Backstepping + RBFNN + RNE feedforward
        // PX4 internal rate gains = 0 → node này là nguồn torque DUY NHẤT
        Eigen::Matrix3d J_mat;
        J_mat << sys_.Ixx, sys_.Ixy, sys_.Ixz,
                 sys_.Ixy, sys_.Iyy, sys_.Iyz,
                 sys_.Ixz, sys_.Iyz, sys_.Izz;
        Eigen::Vector3d coriolis = omega_.cross(J_mat * omega_);
        Eigen::Matrix3d Kp_mat = Eigen::Vector3d(rate_gains_.K_roll, rate_gains_.K_pitch, rate_gains_.K_yaw).asDiagonal();
        Eigen::Matrix3d Ki_mat = Eigen::Vector3d(rate_gains_.Ki_roll, rate_gains_.Ki_pitch, rate_gains_.Ki_yaw).asDiagonal();
        Eigen::Matrix3d Kd_mat = Eigen::Vector3d(rate_gains_.Kd_roll, rate_gains_.Kd_pitch, rate_gains_.Kd_yaw).asDiagonal();

        Eigen::Vector3d e_omega_dot = Eigen::Vector3d::Zero();
        if (elapsed > 2.0 * dt) {
            Eigen::Vector3d raw_dot = (e_omega_ - e_omega_prev_) / dt;
            raw_dot = raw_dot.cwiseMax(-20.0).cwiseMin(20.0);
            e_omega_dot_prev_ = lpf_alpha_ * raw_dot + (1.0 - lpf_alpha_) * e_omega_dot_prev_;
            e_omega_dot = e_omega_dot_prev_;
        } else {
            e_omega_dot_prev_.setZero();
        }

        // Integral — LUÔN tích lũy, clamp chặt hơn khi takeoff
        double int_clamp = takeoff_sensitive ? 0.2 : 0.5;
        e_omega_int_ += e_omega_ * dt;
        e_omega_int_ = e_omega_int_.cwiseMax(-int_clamp).cwiseMin(int_clamp);

        const Eigen::Vector3d tau_arm_comp = arm_ff_enabled_ ? tau_arm_ff_ : Eigen::Vector3d::Zero();
        const Eigen::Vector3d tau_arm_disturbance =
            arm_virtual_disturbance_enabled_ ? tau_arm_virtual_disturbance_ : Eigen::Vector3d::Zero();
        tau = J_mat * (-Kp_mat * e_omega_ - Ki_mat * e_omega_int_ - Kd_mat * e_omega_dot - n_hat_)
              + coriolis
              + tau_arm_disturbance
              - tau_arm_comp;
        e_omega_prev_ = e_omega_;

        // Normalized Torque for PX4
        tau_norm(0) = tau(0) / tau_axis_max_(0);
        tau_norm(1) = tau(1) / tau_axis_max_(1);
        tau_norm(2) = tau(2) / tau_axis_max_(2);
        
        // 4. Feedforward CG Offset Compensation (Cân bằng trọng lượng cánh tay tĩnh)
        tau_norm(0) += base_roll_offset_ + arm_cg_bias_norm_(0);
        tau_norm(1) += base_pitch_offset_ + arm_cg_bias_norm_(1);
        
        // ★ FIX #3: Thrust saturation để chống flyaway
        thrust_norm(0) = std::clamp(thrust_des_(0), -0.1, 0.1);
        thrust_norm(1) = std::clamp(thrust_des_(1), -0.1, 0.1);
        thrust_norm(2) = std::clamp(thrust_des_(2), -1.0, -0.05);
    } else {
        if (controller_enabled_) {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "External rate controller enabled but PX4 inputs are stale. "
                "Holding ROS torque/thrust publish; PX4 internal fallback should stay active. "
                "Check odometry/rate-setpoint/RNE freshness.");
        }
        n_hat_.setZero();
        // Không nhận đủ điều kiện tính toán -> Lực = 0 để nuôi Control Allocator
    }
    
    // 4. Chỉ gửi lực đẩy và mô-men về cho PX4 khi controller_enabled_ = true
    // Khi controller_enabled_ = false, KHÔNG GỬI để PX4 tự động fallback về internal rate controller
    if (px4_timestamp_ == 0) return; 

    if (controller_enabled_ && can_compute) {
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
    }

    // 5. Cấp lệnh cho cánh tay máy (Đã tách biệt để hoạt động ngay khi controller_enabled)
    if (controller_enabled_ && has_joints_) {
        Eigen::VectorXd tau_j = compute_joint_control(takeoff_sensitive);
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
    dbg.data.insert(dbg.data.end(), {tau_arm_ff_(0), tau_arm_ff_(1), tau_arm_ff_(2)});
    dbg.data.insert(dbg.data.end(), {tau_norm(0), tau_norm(1), tau_norm(2)});
    dbg.data.push_back(arm_ff_enabled_ ? 1.0 : 0.0);
    dbg.data.push_back(arm_ff_fresh(now) ? 1.0 : 0.0);
    dbg.data.insert(dbg.data.end(), {
        tau_arm_virtual_disturbance_(0),
        tau_arm_virtual_disturbance_(1),
        tau_arm_virtual_disturbance_(2)
    });
    dbg.data.push_back(arm_virtual_disturbance_enabled_ ? 1.0 : 0.0);
    debug_pub_->publish(dbg);
}

// Hàm tính lực cho khớp tay máy
Eigen::VectorXd UAMAdaptiveController::compute_joint_control(bool takeoff_sensitive) {
    Eigen::VectorXd tau_joints = Eigen::VectorXd::Zero(N_JOINTS);
    
    // CỐ ĐỊNH CÁNH TAY TRONG KHI TAKEOFF
    // Force PD control only (ignore dynamics/coupling) during sensitive takeoff phase
    bool force_pd = (!dyn_ready_ || takeoff_sensitive);

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
