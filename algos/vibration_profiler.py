from video_stabilizer import VideoStabilizer, GyroBuffer

# Shared between IMU thread and stabilizer
gyro_buf = GyroBuffer()
stab = VideoStabilizer(
    K                 = your_camera_K,
    dist_coeffs       = your_dist_coeffs,   # or None if already undistorted
    gyro_buffer       = gyro_buf,
    img_shape         = (480, 640),
    cutoff_hz         = 1.5,                # tune with VibrationProfiler first
    rolling_shutter_s = 0.016,              # 0.0 for global shutter
)

# IMU thread — unchanged from your existing code, add one line:
gyro_buf.push(timestamp, omega_raw)

# EKF thread — when bias estimate updates:
stab.update_bias(ekf_state.b_gyro)

# Camera thread:
stable_frame = stab.stabilize(raw_frame, timestamp)