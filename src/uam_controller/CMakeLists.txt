cmake_minimum_required(VERSION 3.8)
project(uam_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Phụ thuộc
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# ── C++ node ──
add_executable(uam_backstepping_rbfnn_node
  src/uam_backstepping_rbfnn_node.cpp
)
target_include_directories(uam_backstepping_rbfnn_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(uam_backstepping_rbfnn_node
  rclcpp px4_msgs geometry_msgs sensor_msgs std_msgs std_srvs trajectory_msgs
)

install(TARGETS uam_backstepping_rbfnn_node
  DESTINATION lib/${PROJECT_NAME}
)

# ── Header files ──
install(DIRECTORY include/
  DESTINATION include
)

# ── Launch, Config, Models ──
install(DIRECTORY
  launch config models
  DESTINATION share/${PROJECT_NAME}
)

# ── Python module ──
ament_python_install_package(${PROJECT_NAME})

# ── Python scripts ──
install(PROGRAMS
  scripts/arm_dynamics_node.py
  scripts/arm_gazebo_command_node.py
  scripts/uam_mission_bridge.py
  scripts/uam_telemetry_monitor.py
  scripts/rbfnn_data_logger.py
  scripts/rbfnn_training_supervisor.py
  scripts/arm_trajectory_generator.py
  scripts/arm_initial_pose.py
  scripts/qgc_rbfnn_trigger.py
  
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
