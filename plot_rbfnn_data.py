#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

file_path = '/home/wicom/ros2_ws/rbfnn_flight_data.csv'
out_dir = '/home/wicom/ros2_ws/src/plot/'
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(file_path):
    print("CSV file not found!")
    exit(1)

df = pd.read_csv(file_path)
time = df['time']

# 1. Altitude Plot
plt.figure(figsize=(10, 5))
plt.plot(time, df['alt_z'], label='Actual Altitude (m)', color='blue')
plt.axhline(2.0, color='red', linestyle='--', label='Setpoint (2m)')
plt.title('UAV Altitude over Time under Arm Disturbance')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, 'alt_plot.png'))
plt.close()

# 2. Attitude Plot
plt.figure(figsize=(10, 5))
plt.plot(time, df['roll'], label='Roll (deg)', color='orange')
plt.plot(time, df['pitch'], label='Pitch (deg)', color='green')
plt.title('UAV Attitude (Roll & Pitch)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, 'attitude_plot.png'))
plt.close()

# 3. m_hat Plot
plt.figure(figsize=(10, 5))
plt.plot(time, df['m_hat'], label='Estimated Mass (m_hat)', color='purple')
plt.axhline(1.95, color='gray', linestyle='--', label='Nominal Mass')
plt.title('RBFNN Mass/Disturbance Estimation (m_hat)')
plt.xlabel('Time (s)')
plt.ylabel('m_hat (kg)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, 'm_hat_plot.png'))
plt.close()

print(f"Plots generated successfully in {out_dir}")
