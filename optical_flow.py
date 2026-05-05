import cv2
import numpy as np
import glob


# Load two consecutive frames
Path = "docs/imgs/corridor/"


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
    f = w
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

    return roll, pitch, yaw


import matplotlib.pyplot as plt
import numpy as np


# # Create figure with n rows, 2 columns
# fig, axes = plt.subplots(8, 2, figsize=(10, 2*8))

# for i in range(8):
#     # Display two consecutive frames
#     frame1 = cv2.imread(Path+f"bt.00{i}.png")
#     frame2 = cv2.imread(Path+f"bt.00{i+1}.png")
#     axes[i, 0].imshow(frame1, cmap='gray')
#     axes[i, 1].imshow(frame2, cmap='gray')
#     axes[i, 0].axis('off')
#     axes[i, 1].axis('off')
#     axes[i, 0].set_title(f'Frame {i}')
#     axes[i, 1].set_title(f'Frame {i+1}')
    
#     # Get transformation
#     tx, ty, tz, rx, ry, rz = estimate_Rt(frame1, frame2)
    
#     # Print 6 variables


#     print(f"Pair {i}: trans=[{tx:.3f}, {ty:.3f}, {tz:.3f}], rot=[{rx:.3f}, {ry:.3f}, {rz:.3f}]")
#     # Add text below images
#     axes[i, 0].text(0.5, -0.2, f'T:[{tx:.2f},{ty:.2f},{tz:.2f}] R:[{rx:.2f},{ry:.2f},{rz:.2f}]',
#                     transform=axes[i, 0].transAxes, ha='center', fontsize=8)

# plt.tight_layout()
# plt.show()

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

    tx, ty, tz, roll, pitch, yaw = estimate_Rt(f1, f2)

    img = f2.copy()

    h, w, _ = img.shape
    cx, cy = w//2, h//2


    scale = 1000

    end_x = int(cx + scale * tx)
    end_y = int(cy + scale * ty)

    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (0,255,0), 3)

    text = f"Roll: {np.degrees(roll):.1f} | Pitch: {np.degrees(pitch):.1f} | Yaw: {np.degrees(yaw):.1f}"

    cv2.putText(img, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # --- show with matplotlib ---
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame {i}")
    ax.axis('off')

    plt.pause(2)  # <-- slow down here (adjust!)