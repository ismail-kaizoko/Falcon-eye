import cv2
import numpy as np

def recover_motion_from_F(F, pts1, pts2, shape):
    h, w = shape

    # --- Fake camera intrinsics ---
    f = w
    K = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]
    ])

    # --- Convert F → E ---
    E = K.T @ F @ K

    # --- Recover pose ---
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return R, t


def estimate_motion(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- features ---
    pts1 = cv2.goodFeaturesToTrack(gray1, 200, 0.01, 7)

    # --- optical flow ---
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, pts1, None,
        winSize=(15,15), maxLevel=2
    )

    # --- filter valid ---
    pts1 = pts1[status == 1].reshape(-1,1,2)
    pts2 = pts2[status == 1].reshape(-1,1,2)

    if len(pts1) < 8:
        return None, None

    # --- Fundamental matrix with RANSAC ---
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        1.0, 0.99
    )

    if F is None:
        return None, None

    pts1_in = pts1[mask.ravel()==1]
    pts2_in = pts2[mask.ravel()==1]

    if len(pts1_in) < 8:
        return None, None

    # --- recover motion ---
    R, t = recover_motion_from_F(F, pts1_in, pts2_in, gray1.shape)

    return R, t


import glob

Path = "docs/imgs/corridor/"
images = sorted(glob.glob(Path + "*.png"))

frames = [cv2.imread(img) for img in images]


def draw_motion(frame, R, t):
    h, w, _ = frame.shape
    cx, cy = int(w/2), int(h/2)

    output = frame.copy()

    if t is not None:
        # --- translation arrow ---
        tx, ty = t[0][0], t[1][0]
        scale = 100

        end = (int(cx + scale*tx), int(cy + scale*ty))
        cv2.arrowedLine(output, (cx, cy), end, (0,255,0), 3)

    if R is not None:
        # --- rotation visualization (yaw approx) ---
        yaw = np.arctan2(R[1,0], R[0,0])

        text = f"Yaw: {np.degrees(yaw):.2f} deg"
        cv2.putText(output, text, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return output

import numpy as np

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

    return roll, pitch, yaw


output_frames = []
import matplotlib.pyplot as plt


plt.ion()  # interactive mode
fig, ax = plt.subplots()

for i in range(len(frames)-1):
    f1 = frames[i]
    f2 = frames[i+1]

    R, t = estimate_motion(f1, f2)

    img = f2.copy()

    h, w, _ = img.shape
    cx, cy = w//2, h//2

    if t is not None:
        # --- translation arrow ---
        tx, ty = t[0][0], t[1][0]
        scale = 100

        end_x = int(cx + scale * tx)
        end_y = int(cy + scale * ty)

        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0,255,0), 3)

    if R is not None:
        roll, pitch, yaw = rotation_to_euler(R)

        text = f"Roll: {np.degrees(roll):.1f} | Pitch: {np.degrees(pitch):.1f} | Yaw: {np.degrees(yaw):.1f}"

        cv2.putText(img, text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # --- show with matplotlib ---
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame {i}")
    ax.axis('off')

    plt.pause(0.5)  # <-- slow down here (adjust!)