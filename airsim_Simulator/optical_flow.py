import cv2
import numpy as np
import glob


MIN_POSE_FLOW_PX = 0.35


def _validate_frame_pair(frame1, frame2):
    if frame1 is None or frame2 is None:
        raise ValueError("frame1 and frame2 must be valid images, got None")
    if frame1.ndim != 3 or frame2.ndim != 3:
        raise ValueError("frame1 and frame2 must be color images with 3 channels")
    if frame1.shape != frame2.shape:
        raise ValueError(f"frame shapes must match, got {frame1.shape} and {frame2.shape}")


def _tracked_points(frame1, frame2):
    _validate_frame_pair(frame1, frame2)

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    features = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7
    )
    if features is None or len(features) < 8:
        raise ValueError("not enough trackable features in frame1")

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, features, None, **lk_params
    )
    if new_points is None or status is None:
        raise ValueError("optical flow failed to track points")

    good_old = features[status == 1]
    good_new = new_points[status == 1]
    if len(good_old) < 8:
        raise ValueError("not enough valid tracked points")

    return gray1, good_old, good_new


def _median_flow_px(good_old, good_new):
    old = good_old.reshape(-1, 2)
    new = good_new.reshape(-1, 2)
    return float(np.median(np.linalg.norm(new - old, axis=1)))


def Lucas_Kanade(frame1, frame2):
    gray1, good_old, good_new = _tracked_points(frame1, frame2)


    # --- RANSAC filtering using Fundamental Matrix ---
    good_old_pts = good_old.reshape(-1,1,2)
    good_new_pts = good_new.reshape(-1,1,2)

    F, mask = cv2.findFundamentalMat(
        good_old_pts,
        good_new_pts,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99
    )
    if mask is None:
        raise ValueError("fundamental matrix estimation failed")

    # Keep only inliers
    inliers_old = good_old[mask.ravel() == 1]
    inliers_new = good_new[mask.ravel() == 1]

    return inliers_old, inliers_new



def estimate_Rt(frame1, frame2):
    gray1, good_old, good_new = _tracked_points(frame1, frame2)
    if _median_flow_px(good_old, good_new) < MIN_POSE_FLOW_PX:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    h, w = gray1.shape


    # --- Fake camera intrinsics ---
    f = w / (2*np.tan(np.radians(90/2)))
    K = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ])


    # --- Essential matrix with RANSAC ---
    E, mask = cv2.findEssentialMat(
        good_old, good_new,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None or mask is None:
        raise ValueError("essential matrix estimation failed")

    # mask = inliers (1) / outliers (0)
    inliers1 = good_old[mask.ravel() == 1]
    inliers2 = good_new[mask.ravel() == 1]
    if len(inliers1) < 5:
        raise ValueError("not enough inliers to recover pose")

    # --- Recover pose (R, t direction) ---
    _, R_camera, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K)

    #align camera frame to drone frame convention 
    Permute = np.array([
    [0,0,1],
    [1,0,0],
    [0,1,0]
])
    R_drone = Permute @ R_camera @ Permute.T



    euler_angles = rotation_to_euler(R_drone)
    return t[0][0], t[1][0], t[2][0], euler_angles[0], euler_angles[1], euler_angles[2]


def rotation_to_euler(R):
    # Assuming R = Rz(yaw) * Ry(pitch) * Rx(roll)

    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)

    singular = sy < 1e-6

    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0

    return -np.degrees([roll, pitch, yaw])
