import airsim
import numpy as np
import cv2
import os
import csv
import time


# =====================================================
# CONFIG
# =====================================================

SAVE_DIR = "dataset"

CAMERA_HZ = 20
IMU_HZ = 200

DURATION = 30  # seconds

VELOCITY = 2.0


# =====================================================
# CREATE FOLDERS
# =====================================================

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)


# =====================================================
# CONNECT
# =====================================================

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")

client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()


# =====================================================
# CSV LOGGERS
# =====================================================

imu_file = open(f"{SAVE_DIR}/imu.csv", "w", newline="")
imu_writer = csv.writer(imu_file)

imu_writer.writerow([
    "timestamp",
    "ax", "ay", "az",
    "gx", "gy", "gz"
])


pose_file = open(f"{SAVE_DIR}/poses.csv", "w", newline="")
pose_writer = csv.writer(pose_file)

pose_writer.writerow([
    "timestamp",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "vx", "vy", "vz",
    "wx", "wy", "wz"
])


# =====================================================
# TRAJECTORY
# =====================================================

print("Starting motion...")

client.moveByVelocityAsync(
    vx=VELOCITY,
    vy=0,
    vz=0,
    duration=DURATION
)


# =====================================================
# MAIN LOOP
# =====================================================

t0 = time.time()

next_cam = t0
next_imu = t0

frame_idx = 0

while True:

    now = time.time()
    elapsed = now - t0

    if elapsed > DURATION:
        break

    # =====================================
    # IMU
    # =====================================

    if now >= next_imu:

        imu = client.getImuData()

        acc = imu.linear_acceleration
        gyro = imu.angular_velocity

        imu_writer.writerow([
            elapsed,
            acc.x_val,
            acc.y_val,
            acc.z_val,
            gyro.x_val,
            gyro.y_val,
            gyro.z_val
        ])

        next_imu += 1 / IMU_HZ

    # =====================================
    # CAMERA
    # =====================================

    if now >= next_cam:

        response = client.simGetImages([
            airsim.ImageRequest(
                "0",
                airsim.ImageType.Scene,
                False,
                False
            )
        ])[0]

        img = np.frombuffer(
            response.image_data_uint8,
            dtype=np.uint8
        )

        img = img.reshape(
            response.height,
            response.width,
            3
        )

        cv2.imwrite(
            f"{SAVE_DIR}/images/frame_{frame_idx:06d}.png",
            img
        )

        # =================================
        # Ground truth
        # =================================

        state = client.getMultirotorState()

        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation

        vel = state.kinematics_estimated.linear_velocity
        ang = state.kinematics_estimated.angular_velocity

        pose_writer.writerow([
            elapsed,

            pos.x_val,
            pos.y_val,
            pos.z_val,

            ori.x_val,
            ori.y_val,
            ori.z_val,
            ori.w_val,

            vel.x_val,
            vel.y_val,
            vel.z_val,

            ang.x_val,
            ang.y_val,
            ang.z_val
        ])

        frame_idx += 1
        next_cam += 1 / CAMERA_HZ



print("sijfij")
