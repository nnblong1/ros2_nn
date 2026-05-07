#pragma once

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_torque_setpoint.hpp>
#include <px4_msgs/msg/vehicle_thrust_setpoint.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/actuator_motors.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>

using namespace std::chrono_literals;

struct UAMSystemParams {
    double mass_nominal = 2.5; 
    double gravity      = 9.81;
    double Ixx          = 0.029; 
    double Iyy          = 0.029;
    double Izz          = 0.055;
    double max_torque   = 5.0;   
    double max_joint_tau= 20.0;  
};

struct RateGains {
    double K_roll  = 0.6;
    double K_pitch = 0.6;
    double K_yaw   = 0.4;
    double Ki_roll  = 0.05;
    double Ki_pitch = 0.05;
    double Ki_yaw   = 0.02;
    double Kd_roll  = 0.003;
    double Kd_pitch = 0.003;
    double Kd_yaw   = 0.001;
};

struct JointGains { double Kp=50.0, Kd=5.0; };

struct RBFNNParams {
    int    num_neurons    = 25;
    int    input_dim      = 9;    // Z = [e_p, e_q, e_r, p_d, q_d, r_d, p, q, r]
    int    output_dim     = 3;    // [n_x, n_y, n_z]
    double learning_rate  = 0.1;
    double e_modification = 0.01;
    double gaussian_width = 1.0;
};

class RBFNeuralNetwork {
public:
    explicit RBFNeuralNetwork(const RBFNNParams& p);

    Eigen::VectorXd compute_basis(const Eigen::VectorXd& Z) const;
    Eigen::VectorXd estimate(const Eigen::VectorXd& Z) const;
    void update_weights(const Eigen::VectorXd& Z,
                        const Eigen::VectorXd& e2,
                        double dt);
    void reset();

private:
    RBFNNParams     params_;
    Eigen::MatrixXd W_hat_;
    Eigen::MatrixXd C_centers_;
    Eigen::VectorXd B_widths_;
    Eigen::MatrixXd Gamma_;
};

class UAMAdaptiveController : public rclcpp::Node {
public:
    explicit UAMAdaptiveController();
    ~UAMAdaptiveController() = default;

private:
    UAMSystemParams sys_;
    RateGains       rate_gains_;
    RBFNNParams     rbfnn_params_;
    std::array<JointGains, 6> jg_;

    std::unique_ptr<RBFNeuralNetwork> rbfnn_;

    // States
    Eigen::Vector3d omega_      = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_des_  = Eigen::Vector3d::Zero();
    Eigen::Vector3d thrust_des_ = Eigen::Vector3d::Zero();

    Eigen::Vector3d e_omega_      = Eigen::Vector3d::Zero();
    Eigen::Vector3d e_omega_int_  = Eigen::Vector3d::Zero();  // Integral accumulator
    Eigen::Vector3d e_omega_prev_ = Eigen::Vector3d::Zero();  // Previous error for D-term
    Eigen::Vector3d e_omega_dot_prev_ = Eigen::Vector3d::Zero(); // LPF on D-term
    Eigen::Vector3d n_hat_        = Eigen::Vector3d::Zero();

    double last_t_ = -1.0;
    static constexpr double lpf_alpha_ = 0.2; // LPF cutoff ~20Hz @ 200Hz sample rate
    double last_odom_rx_time_ = -1.0;
    double last_rates_sp_rx_time_ = -1.0;
    double altitude_m_ = 0.0;
    double vertical_speed_m_s_ = 0.0;
    bool landed_ = true;
    bool ground_contact_ = true;

    // ★ RBFNN Ramp-up Strategy: Cho phép RBFNN học từ đầu, output tăng dần
    double controller_start_time_ = -1.0;      // Thời điểm controller được enable lần đầu
    static constexpr double RAMP_PHASE1_END = 3.0;   // 0-3s: output max ±0.05
    static constexpr double RAMP_PHASE2_END = 8.0;   // 3-8s: output max ±0.15
    static constexpr double RAMP_FULL_LIMIT = 0.50;  // >8s:  output max ±0.50
    static constexpr double RAMP_P1_LIMIT   = 0.05;
    static constexpr double RAMP_P2_LIMIT   = 0.15;


    // Khớp tay máy
    static constexpr int N_JOINTS = 6;
    std::array<double, N_JOINTS> q_{},  dq_{};
    std::array<double, N_JOINTS> qd_{}, dqd_{};
    Eigen::MatrixXd D_inv_  = Eigen::MatrixXd::Identity(12, 12);
    Eigen::VectorXd H_vec_  = Eigen::VectorXd::Zero(12);
    bool dyn_ready_ = false;

    // Cờ
    bool has_odom_ = false;
    bool has_rates_sp_ = false;
    bool has_joints_ = false;
    bool controller_enabled_ = false;
    bool rbfnn_output_enabled_ = false;
    uint64_t px4_timestamp_ = 0;
    
    double base_pitch_offset_ = 0.0;
    double base_roll_offset_ = 0.0;

    // I/O (Chuyển tiếp nguyên xi thrust_body từ Attitude Controller)
    rclcpp::Publisher<px4_msgs::msg::VehicleTorqueSetpoint>::SharedPtr  torque_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleThrustSetpoint>::SharedPtr  thrust_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr      debug_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr      joint_tau_pub_;

    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr     odom_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rates_sp_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr land_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr       joint_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr   dyn_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr                enable_sub_;

    rclcpp::TimerBase::SharedPtr timer_;

    // Callbacks
    void odom_cb(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
    void rates_sp_cb(const px4_msgs::msg::VehicleRatesSetpoint::SharedPtr msg);
    void land_cb(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);
    void joint_cb(const sensor_msgs::msg::JointState::SharedPtr msg);
    void dyn_cb(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void enable_cb(const std_msgs::msg::Bool::SharedPtr msg);


    // Xử lý vòng lặp
    void control_loop();
    Eigen::VectorXd compute_joint_control(bool takeoff_sensitive);
    void declare_params();
    double sat(double v, double lim) const;
    bool inputs_fresh(double now) const;
    bool in_takeoff_sensitive_phase(double elapsed_since_enable) const;
};
