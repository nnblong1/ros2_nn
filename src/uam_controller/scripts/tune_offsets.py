#!/usr/bin/env python3
import subprocess
import time
import yaml
import sys
import os

PARAMS_FILE = "/home/wicom/ros2_ws/src/uam_controller/config/uam_controller_params.yaml"

def update_params(pitch, roll):
    with open(PARAMS_FILE, "r") as f:
        data = yaml.safe_load(f)
    
    data["uam_adaptive_controller"]["ros__parameters"]["base_pitch_offset"] = float(pitch)
    data["uam_adaptive_controller"]["ros__parameters"]["base_roll_offset"] = float(roll)
    data["uam_adaptive_controller"]["ros__parameters"]["alt_Kp"] = 8.0
    data["uam_adaptive_controller"]["ros__parameters"]["alt_Ki"] = 0.1
    data["uam_adaptive_controller"]["ros__parameters"]["alt_Kd"] = 3.5
    data["uam_adaptive_controller"]["ros__parameters"]["alt_cz"] = 0.05
    data["uam_adaptive_controller"]["ros__parameters"]["rbfnn_enable"] = False

    with open(PARAMS_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Updated YAML with Pitch: {pitch}, Roll: {roll}")

def kill_all():
    print("Killing existing processes...")
    subprocess.run("killall -9 px4 ruby MicroXRCEAgent gzserver gzclient 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'ros2 launch' 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'uam_backstepping_rbfnn_node' 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'uam_mission_bridge' 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'uam_telemetry_monitor' 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'arm_dynamics_node' 2>/dev/null", shell=True)
    time.sleep(2)

def main():
    pitch_values = [-0.04, -0.06, -0.08, -0.10, -0.12, -0.14, -0.16]
    roll_values = [0.0, -0.05, -0.10, -0.15, -0.20]
    
    results = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checker_script = os.path.join(script_dir, "hover_checker.py")

    with open(checker_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry
from std_srvs.srv import Trigger
import math
import sys
import time

class HoverChecker(Node):
    def __init__(self):
        super().__init__('hover_checker')
        self.sub = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_callback, rclpy.qos.qos_profile_sensor_data)
        self.cli = self.create_client(Trigger, '/uam/arm_takeoff')
        self.start_time = None
        self.takeoff_triggered = False
        self.takeoff_time = 0
        self.get_logger().info("Hover checker started. Waiting for odometry...")

    def odom_callback(self, msg):
        now = time.time()

        if self.start_time is None:
            self.start_time = now
            self.get_logger().info("Receiving Odom! Waiting 5s before takeoff...")
        
        # Trigger takeoff after 5s of getting data
        if not self.takeoff_triggered and (now - self.start_time > 5.0):
            if not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("Service not ready yet...")
                return
            req = Trigger.Request()
            self.cli.call_async(req)
            self.takeoff_triggered = True
            self.takeoff_time = now
            self.get_logger().info("Takeoff triggered.")
            return

        if not self.takeoff_triggered:
            return

        # Check for crash or success
        elapsed = now - self.takeoff_time
        
        q0, q1, q2, q3 = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1: pitch = math.copysign(math.pi / 2.0, sinp)
        else: pitch = math.asin(sinp)

        enu_z = -msg.position[2]

        if elapsed > 1.0:
            if abs(roll) > math.radians(45) or abs(pitch) > math.radians(45):
                self.get_logger().error(f"FLIPPED! Roll: {math.degrees(roll):.1f}, Pitch: {math.degrees(pitch):.1f}")
                sys.exit(1)
            if enu_z < 0.2 and elapsed > 8.0:
                self.get_logger().error("FAILED TO CLIMB")
                sys.exit(1)
        
        if elapsed > 15.0:
            if enu_z > 1.0:
                self.get_logger().info("SUCCESS: Hovered.")
                sys.exit(0)
            else:
                self.get_logger().error("FAILED TO CLIMB")
                sys.exit(1)

def main(args=None):
    rclpy.init(args=args)
    checker = HoverChecker()
    try:
        rclpy.spin(checker)
    except SystemExit as e:
        sys.exit(e.code)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
''')
    os.chmod(checker_script, 0o755)

    print("===========================================")
    print("   AUTOMATED STATIC HOVER TUNING SCRIPT    ")
    print("===========================================")

    for p in pitch_values:
        for r in roll_values:
            print(f"\n--- Testing configuration: Pitch={p}, Roll={r} ---")
            
            kill_all()
            update_params(p, r)

            print("Starting PX4 and Gazebo...")
            px4_process = subprocess.Popen("cd ~/PX4-Autopilot && make px4_sitl gz_x500_hop", shell=True)
            time.sleep(10)

            print("Starting ROS 2 system...")
            ros_process = subprocess.Popen("cd ~/ros2_ws && source install/setup.bash && ros2 launch uam_controller uam_system.launch.py sim:=true", shell=True, executable='/bin/bash')
            time.sleep(6)

            print("Starting hover checker...")
            
            try:
                checker_process = subprocess.run(
                    f"cd ~/ros2_ws && source install/setup.bash && {checker_script}",
                    shell=True, executable='/bin/bash', timeout=30, check=False)
                
                exit_code = checker_process.returncode
            except subprocess.TimeoutExpired:
                print("Checker timed out. Assuming failure.")
                exit_code = 1
            
            results.append({
                "pitch": p,
                "roll": r,
                "success": (exit_code == 0)
            })

            print("Cleaning up this run...")
            ros_process.terminate()
            px4_process.terminate()
            kill_all()

    print("\n\n===========================================")
    print("           TUNING RESULTS SUMMARY          ")
    print("===========================================")
    for res in results:
        status = "SUCCESS" if res["success"] else "CRASH/FLIP"
        print(f"Pitch: {res['pitch']:>6}, Roll: {res['roll']:>6} -> {status}")

if __name__ == "__main__":
    main()
