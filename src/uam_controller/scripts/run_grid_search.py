#!/usr/bin/env python3
"""
run_grid_search.py  v5.0
-------------------------
Script tự động tìm kiếm 3 tham số tối ưu cho UAM RBFNN Adaptive Controller:
  1. base_pitch_offset  — bù trọng tâm lệch trước/sau
  2. base_roll_offset   — bù trọng tâm lệch trái/phải
  3. rbfnn_lr           — tốc độ học RBFNN

v5.0 - Chỉ tìm 3 tham số (PID giữ cố định, RBFNN lo phần còn lại):
  - RBFNN được bật ngay từ takeoff (Ramp-up Strategy)
  - Không cần tune PID vì RBFNN bù adaptive
  - Mutation rate 40% quanh best, 60% random

Cách chạy: python3 src/uam_controller/scripts/run_grid_search.py
"""

import os
import sys
import time
import subprocess
import yaml
import csv
import math
import numpy as np
import signal
import datetime
import glob
import shutil

WORKSPACE_DIR = "/home/wicom/ros2_ws"
PX4_DIR = "/home/wicom/PX4-Autopilot"
YAML_PATH = f"{WORKSPACE_DIR}/src/uam_controller/config/uam_controller_params.yaml"
RESULTS_FILE = f"{WORKSPACE_DIR}/ket_qua_tim_kiem_v5.csv"
LOG_FILE = "rbfnn_flight_data.csv"

# ═══════════════════════════════════════════════════════════════
# CẤU HÌNH VÙNG TÌM KIẾM (CHỈ 3 THAM SỐ)
# ═══════════════════════════════════════════════════════════════
SEARCH_RANGES = {
    'base_pitch_offset':  (-0.15, 0.10),
    'base_roll_offset':   (-0.08, 0.08),
    'rbfnn_lr':           (0.003, 0.025),
}

TARGET_ALT = 2.0
EXPERIMENT_TIME = 45   
TAKEOFF_PHASE = 15.0
HOVER_PHASE_START = 25.0

PX4_START_TIMEOUT = 25
ROS2_START_TIMEOUT = 12
TAKEOFF_CALL_TIMEOUT = 10
MAX_ROUND_TIME = 130


def nuke_all_processes():
    """Tắt sạch hoàn toàn các tiến trình."""
    print("🧹 Cleaning environment...", flush=True)
    os.system("killall -9 px4 ruby gz gzserver gzclient MicroXRCEAgent micro_xrce_dds_agent 2>/dev/null")
    os.system("pkill -9 -f 'uam_adaptive_controller|uam_mission_bridge|arm_dynamics_node|rbfnn_data_logger' 2>/dev/null")
    os.system("pkill -9 -f 'ros2|gz-sim' 2>/dev/null")
    # Clean SHM
    for f in glob.glob("/dev/shm/fastrtps_*") + glob.glob("/dev/shm/FastRTPS_*"):
        try: os.remove(f)
        except: pass
    # Clean Tmp
    for f in glob.glob("/tmp/px4*"):
        try: 
            if os.path.isdir(f): shutil.rmtree(f, ignore_errors=True)
            else: os.remove(f)
        except: pass
    time.sleep(3.0)


def modify_yaml(params_dict):
    """Cập nhật 3 tham số vào YAML (giữ nguyên PID)."""
    with open(YAML_PATH, 'r') as file:
        data = yaml.safe_load(file)

    ctrl = data['uam_adaptive_controller']['ros__parameters']
    for key, val in params_dict.items():
        ctrl[key] = float(val)

    # Đảm bảo RBFNN luôn bật
    ctrl['rbfnn_enable'] = True

    with open(YAML_PATH, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def compute_cost_from_csv():
    """Chấm điểm."""
    if not os.path.exists(LOG_FILE): return float('inf'), "NO_DATA"
    try:
        data = np.genfromtxt(LOG_FILE, delimiter=',', names=True)
        if data.size < 40: return float('inf'), "NOT_ENOUGH_DATA"
    except: return float('inf'), "READ_ERROR"

    times, alts, rolls, pitchs = data['time'], data['alt_z'], data['roll'], data['pitch']

    # 1. Vibration Penalty
    tk_mask = (times > 2.0) & (times <= TAKEOFF_PHASE)
    vibration_cost = np.sqrt(np.mean(rolls[tk_mask]**2)) + np.sqrt(np.mean(pitchs[tk_mask]**2)) if any(tk_mask) else 100.0

    # 2. Altitude Penalty
    final_alt = np.mean(alts[times > (times[-1]-5.0)]) if len(times)>10 else 0
    alt_error = abs(final_alt - TARGET_ALT)
    altitude_cost = alt_error * 15.0 + (100.0 if alt_error > 1.2 else 0.0)

    # 3. Hover Stability
    hv_mask = times >= HOVER_PHASE_START
    stability_cost = (np.std(rolls[hv_mask]) + np.std(pitchs[hv_mask]) + np.std(alts[hv_mask])*5.0) if any(hv_mask) else 50.0

    # 4. Status
    max_alt = np.max(alts) if len(alts)>0 else 0
    max_ang = max(np.max(np.abs(rolls)), np.max(np.abs(pitchs)))
    verdict = "FLEW"
    penalty = 0.0
    if max_alt < 0.3: penalty += 500.0; verdict = "STUCK"
    elif max_alt < 1.0: penalty += 200.0; verdict = "LOW_ALT"
    if max_ang > 60: penalty += 400.0; verdict = "NEAR_FLIP"
    if max_ang > 85: penalty += 1000.0; verdict = "FLIPPED"

    total = 0.35*vibration_cost + 0.40*altitude_cost + 0.25*stability_cost + penalty
    print(f"  📊 Cost: {total:.2f} | Alt: {max_alt:.2f}m | Ang: {max_ang:.1f}° | {verdict}")
    return total, verdict


def run_experiment(params_dict):
    nuke_all_processes()
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    modify_yaml(params_dict)

    print("🚀 SITL...", end=" ", flush=True)
    px4 = subprocess.Popen(["make", "px4_sitl", "gz_x500_hop"], cwd=PX4_DIR, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(PX4_START_TIMEOUT)

    print("📡 ROS2...", end=" ", flush=True)
    ros = subprocess.Popen(["ros2", "launch", "uam_controller", "uam_system.launch.py", "sim:=true"], 
                           stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(ROS2_START_TIMEOUT)

    print("✈️ Takeoff...", end=" ", flush=True)
    success = False
    for attempt in range(12):
        try:
            res = subprocess.run(["ros2", "service", "call", "/uam/arm_takeoff", "std_srvs/srv/Trigger"], 
                                 capture_output=True, text=True, timeout=10)
            if "success=true" in res.stdout.lower(): success = True; break
        except subprocess.TimeoutExpired:
            print(f"⏳ Retry {attempt+1}/12...", end=" ", flush=True)
        time.sleep(2.0)

    if not success:
        print("❌ FAIL"); os.killpg(os.getpgid(px4.pid), 9); os.killpg(os.getpgid(ros.pid), 9); return 9999, "BRIDGE_FAIL"

    logger = subprocess.Popen(["ros2", "run", "uam_controller", "rbfnn_data_logger.py"], stdout=subprocess.DEVNULL)
    print(f"📈 Logging ({EXPERIMENT_TIME}s, early-stop on flip)...")
    
    elapsed = 0
    early_stop = False
    while elapsed < EXPERIMENT_TIME:
        time.sleep(2.0)
        elapsed += 2
        # Kiểm tra lật sớm
        if os.path.exists(LOG_FILE):
            try:
                data = np.genfromtxt(LOG_FILE, delimiter=',', names=True)
                if data.size > 10:
                    max_ang = max(np.max(np.abs(data['roll'])), np.max(np.abs(data['pitch'])))
                    if max_ang > 60:
                        print(f"⚠️ FLIP DETECTED ({max_ang:.1f}°) → Abort sớm!")
                        early_stop = True
                        break
            except:
                pass

    os.killpg(os.getpgid(px4.pid), 9); os.killpg(os.getpgid(ros.pid), 9); logger.terminate()
    return compute_cost_from_csv()


def main():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            f.write("timestamp,pitch_offset,roll_offset,rbfnn_lr,cost,verdict\n")

    best_cost = 9999
    best_params = None
    iter_count = 1

    print("\n🚀 STARTING OPTIMIZATION v5.0 (3-Param RBFNN Search) 🚀")
    print(f"   Tham số: base_pitch_offset, base_roll_offset, rbfnn_lr")
    print(f"   PID giữ cố định, RBFNN bật từ takeoff (Ramp-up)")
    print(f"   File kết quả: {RESULTS_FILE}\n")

    try:
        while True:
            # Thuật toán: 60% Random, 40% Mutation quanh Best
            is_mutation = best_params is not None and (np.random.random() < 0.4)
            
            current_params = {}
            for key, (vmin, vmax) in SEARCH_RANGES.items():
                if is_mutation:
                    sigma = (vmax - vmin) * 0.1
                    val = np.clip(np.random.normal(best_params[key], sigma), vmin, vmax)
                else:
                    val = np.random.uniform(vmin, vmax)
                current_params[key] = val

            print(f"\n{'='*60}")
            print(f"[Iteration {iter_count}] {'🔹 Mutation' if is_mutation else '🎲 Random'}")
            print(f"  pitch_off={current_params['base_pitch_offset']:.4f}  "
                  f"roll_off={current_params['base_roll_offset']:.4f}  "
                  f"lr={current_params['rbfnn_lr']:.5f}")
            
            cost, verdict = run_experiment(current_params)

            # Log results
            with open(RESULTS_FILE, 'a') as f:
                p = current_params
                f.write(f"{datetime.datetime.now()},"
                        f"{p['base_pitch_offset']:.5f},"
                        f"{p['base_roll_offset']:.5f},"
                        f"{p['rbfnn_lr']:.5f},"
                        f"{cost:.2f},{verdict}\n")

            if cost < best_cost:
                best_cost = cost
                best_params = current_params.copy()
                print(f"🏆 NEW BEST: Cost {best_cost:.2f} | "
                      f"pitch={best_params['base_pitch_offset']:.4f} "
                      f"roll={best_params['base_roll_offset']:.4f} "
                      f"lr={best_params['rbfnn_lr']:.5f}")

            iter_count += 1

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
        if best_params:
            modify_yaml(best_params)
            print(f"✅ Best params saved to YAML:")
            print(f"   pitch_offset = {best_params['base_pitch_offset']:.5f}")
            print(f"   roll_offset  = {best_params['base_roll_offset']:.5f}")
            print(f"   rbfnn_lr     = {best_params['rbfnn_lr']:.5f}")
            print(f"   Best Cost    = {best_cost:.2f}")

if __name__ == "__main__":
    main()
