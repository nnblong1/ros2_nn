from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'uav_control'

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name,
            ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wicom',
    maintainer_email='longnguyen2124@gmail.com',
    description="UAV control package for PX4 + RPi4 + ROS2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "offboard_control  = uav_control.offboard_control:main",
            "mission_manager   = uav_control.mission_manager:main",
            "telemetry_monitor = uav_control.telemetry_monitor:main",
        ],
    },
)
