#!/usr/bin/env bash
set -euo pipefail

# Sync ROS 2 px4_msgs from the PX4 firmware tree used on the flight controller.
# Usage:
#   tools/sync_px4_msgs_from_px4.sh /path/to/PX4-Autopilot
# If no argument is given, PX4_AUTOPILOT_DIR or ~/PX4-Autopilot is used.

PX4_DIR="${1:-${PX4_AUTOPILOT_DIR:-$HOME/PX4-Autopilot}}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_MSG_DIR="$PX4_DIR/msg"
PKG_DIR="$WORKSPACE_DIR/src/px4_msgs"

if [[ ! -d "$PX4_MSG_DIR" ]]; then
  echo "PX4 msg directory not found: $PX4_MSG_DIR" >&2
  exit 1
fi

mkdir -p "$PKG_DIR/msg" "$PKG_DIR/srv"
rm -f "$PKG_DIR/msg/"*.msg

cp "$PX4_MSG_DIR/"*.msg "$PKG_DIR/msg/"

if compgen -G "$PX4_MSG_DIR/versioned/*.msg" > /dev/null; then
  cp "$PX4_MSG_DIR/versioned/"*.msg "$PKG_DIR/msg/"
fi

cat > "$PKG_DIR/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.5)

project(px4_msgs)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(MSGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/msg")
file(GLOB PX4_MSGS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${MSGS_DIR}/*.msg")

set(SRVS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/srv")
file(GLOB PX4_SRVS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${SRVS_DIR}/*.srv")

rosidl_generate_interfaces(${PROJECT_NAME}
	${PX4_MSGS}
	${PX4_SRVS}
	DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
CMAKE

cat > "$PKG_DIR/package.xml" <<'XML'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>px4_msgs</name>
  <version>2.0.1</version>
  <description>Package with the ROS-equivalent of PX4 uORB msgs</description>
  <maintainer email="info@px4.io">PX4</maintainer>
  <license>BSD 3-Clause</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>builtin_interfaces</depend>
  <depend>ros_environment</depend>

  <exec_depend>rosidl_default_runtime</exec_depend>

  <test_depend>ament_lint_common</test_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
XML

if [[ ! -f "$PKG_DIR/srv/VehicleCommand.srv" ]]; then
  cat > "$PKG_DIR/srv/VehicleCommand.srv" <<'SRV'
VehicleCommand request
---
VehicleCommandAck reply
SRV
fi

echo "Synced $(find "$PKG_DIR/msg" -maxdepth 1 -type f -name '*.msg' | wc -l) PX4 messages into $PKG_DIR"
