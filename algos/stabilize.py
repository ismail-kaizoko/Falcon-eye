import cv2
import numpy as np


INPUT_VIDEO = "./docs/vids/road_noisy.mp4"
OUTPUT_VIDEO = "./docs/vids/road_denoised.mp4"

SMOOTHING_RADIUS = 30


cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (w, h)
)

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

transforms = []


# ------------------------------------------------
# Estimate frame-to-frame motion
# ------------------------------------------------

for i in range(n_frames - 1):

    ret, curr = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=30
    )

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None
    )

    idx = status.flatten() == 1

    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Estimate affine motion
    M, _ = cv2.estimateAffinePartial2D(
        prev_pts,
        curr_pts
    )

    if M is None:
        M = np.eye(2, 3)

    dx = M[0, 2]
    dy = M[1, 2]

    da = np.arctan2(M[1, 0], M[0, 0])

    transforms.append([dx, dy, da])

    prev_gray = curr_gray

transforms = np.array(transforms)


# ------------------------------------------------
# Smooth trajectory
# ------------------------------------------------

trajectory = np.cumsum(transforms, axis=0)


def moving_average(curve, radius):
    window = 2 * radius + 1

    filt = np.ones(window) / window

    curve_pad = np.pad(
        curve,
        (radius, radius),
        mode='edge'
    )

    smoothed = np.convolve(
        curve_pad,
        filt,
        mode='same'
    )

    return smoothed[radius:-radius]


smoothed_trajectory = np.copy(trajectory)

for i in range(3):
    smoothed_trajectory[:, i] = moving_average(
        trajectory[:, i],
        SMOOTHING_RADIUS
    )

difference = smoothed_trajectory - trajectory

smooth_transforms = transforms + difference


# ------------------------------------------------
# Apply correction
# ------------------------------------------------

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret, frame = cap.read()

for i in range(len(smooth_transforms)):

    ret, frame = cap.read()
    if not ret:
        break

    dx, dy, da = smooth_transforms[i]

    M = np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da),  np.cos(da), dy]
    ])

    stabilized = cv2.warpAffine(
        frame,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    writer.write(stabilized)

writer.release()
cap.release()

print("Done → filtered.mp4")