import cv2
import numpy as np
import glob


def Lucas_Kanade(frame1, frame2):

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect good features to track (Shi-Tomasi corners)
    features = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7
    )


    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,  # pyramid levels
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )



    # Compute optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, features, None, **lk_params
    )



    # Select valid points
    good_old = features[status == 1]
    good_new = new_points[status == 1]


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

    # Keep only inliers
    inliers_old = good_old[mask.ravel() == 1]
    inliers_new = good_new[mask.ravel() == 1]

    return inliers_old, inliers_new



def estimate_Rt(frame1, frame2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect good features to track (Shi-Tomasi corners)
    features = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7
    )


    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,  # pyramid levels
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )


    # Compute optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, features, None, **lk_params
    )

    # Select valid points
    good_old = features[status == 1]
    good_new = new_points[status == 1]


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

    # mask = inliers (1) / outliers (0)
    inliers1 = good_old[mask.ravel() == 1]
    inliers2 = good_new[mask.ravel() == 1]

    # --- Recover pose (R, t direction) ---
    _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K)

    euler_angles = rotation_to_euler(R)
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

    return np.degrees([roll, pitch, yaw])