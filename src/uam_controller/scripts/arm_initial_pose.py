#!/usr/bin/env python3
"""
arm_initial_pose.py
-------------------
Gửi tư thế "co tay" ban đầu tới cánh tay robot khi khởi động.

Cách tiếp cận: Gọi trực tiếp `gz topic -p` cho TỪNG khớp một (sequential),
BYPASS hoàn toàn bridge ROS2 để tránh ZeroMQ buffer drop.

Chạy 1 lần rồi tự thoát.
"""

import subprocess
import time
import sys

# ═══════════════════════════════════════════════════════
#  GÓC KHỞI ĐẦU (rad) – Tư thế "co tay" tránh va chạm
#
#  J1 (Rev20): Yaw vai       limit ±π      = 0.0
#  J2 (Rev22): Pitch shoulder <lower>-1.6</lower><upper>1.6</upper> (nâng vai LÊN TRÊN)
#  J3 (Rev23): Pitch elbow   <lower>-2.617993</lower> <upper>1.570796</upper> (gập khuỷu)
#  J4 (Rev26): Yaw cổ tay    limit ±π      = 0.0
#  J5 (Rev28): Pitch cổ tay  limit ±1.57   = -0.9  (co cổ tay)
#  J6 (Rev30): Roll kẹp      limit ±1.57   = 0.0
# ═══════════════════════════════════════════════════════
INITIAL_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Delay giữa các khớp (giây) – đủ thời gian cho khớp di chuyển đến đích
JOINT_DELAY = 2.0

# Delay ban đầu để Gazebo sẵn sàng sau khi spawn
STARTUP_DELAY = 3.0

def send_joint_cmd(joint_num: int, position: float, model: str = 'x500_hop_0') -> bool:
    """Gửi lệnh vị trí trực tiếp qua gz topic. Trả về True nếu thành công."""
    topic = f'/model/{model}/arm/joint{joint_num}/cmd_pos'
    cmd = [
        'gz', 'topic',
        '-t', topic,
        '-m', 'gz.msgs.Double',
        '-p', f'data: {position}'
    ]
    try:
        result = subprocess.run(cmd, timeout=5.0,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f'[WARN] joint{joint_num}: gz topic returncode={result.returncode} '
                  f'stderr={result.stderr.strip()}')
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f'[ERROR] joint{joint_num}: timeout gửi lệnh!')
        return False
    except FileNotFoundError:
        print('[ERROR] Không tìm thấy lệnh `gz`. Kiểm tra PATH.')
        sys.exit(1)


def main():
    print(f'[arm_initial_pose] Chờ {STARTUP_DELAY}s cho Gazebo sẵn sàng...')
    time.sleep(STARTUP_DELAY)

    print('[arm_initial_pose] Bắt đầu gửi lệnh co tay (sequential bypass mode)...')
    for i, pos in enumerate(INITIAL_POSITIONS, start=1):
        print(f'  → Khớp {i}: {pos:.3f} rad')
        ok = send_joint_cmd(i, pos)
        if not ok:
            print(f'  [WARN] Khớp {i} có thể chưa nhận được lệnh, tiếp tục...')
        time.sleep(JOINT_DELAY)

    print('[arm_initial_pose] ✅ Hoàn tất gập cánh tay!')


if __name__ == '__main__':
    main()
