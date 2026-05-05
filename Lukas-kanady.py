import cv2
import numpy as np

# Load two consecutive frames
Path = "docs/imgs/corridor/"
frame1 = cv2.imread(Path+"bt.009.png")
frame2 = cv2.imread(Path+"bt.010.png")

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



# Draw motion vectors
output = frame2.copy()
for (old, new) in zip(good_old, good_new):
    x1, y1 = old.ravel()
    x2, y2 = new.ravel()
    cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.circle(output, (int(x2), int(y2)), 3, (0,0,255), -1)

cv2.imshow("Optical Flow", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


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

# Draw filtered (inlier) motion vectors
output = frame2.copy()

for (old, new) in zip(inliers_old, inliers_new):
    x1, y1 = old.ravel()
    x2, y2 = new.ravel()

    cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.circle(output, (int(x2), int(y2)), 3, (0,0,255), -1)

cv2.imshow("Filtered Optical Flow (RANSAC)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()




pts1_in = good_old[mask.ravel() == 1].reshape(-1,2)
pts2_in = good_new[mask.ravel() == 1].reshape(-1,2)

# --- Compute flow ---
flow = pts2_in - pts1_in
print(flow)


# --- Mean direction (image space) ---
mean_flow = np.mean(flow, axis=0)

print("Mean flow direction:", mean_flow)

# normalize
dir2D = mean_flow / np.linalg.norm(mean_flow)
print("Normalized 2D direction:", dir2D)


def estimate_foe(points, flows):
    A = []
    b = []

    for (x, y), (u, v) in zip(points, flows):
        if np.linalg.norm([u, v]) < 1e-4:
            continue

        A.append([v, -u])
        b.append(v*x - u*y)

    A = np.array(A)
    b = np.array(b)

    foe = np.linalg.lstsq(A, b, rcond=None)[0]
    return foe


foe = estimate_foe(pts1_in, flow)

print("FOE:", foe)

cv2.circle(output, (int(foe[0]), int(foe[1])), 8, (255,0,0), -1)

h, w = gray1.shape
cx, cy = w/2, h/2

x = (foe[0] - cx)
y = (foe[1] - cy)

t_dir = np.array([x, y, 1.0])
t_dir = t_dir / np.linalg.norm(t_dir)

print("3D direction (approx):", t_dir)