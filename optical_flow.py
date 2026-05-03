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

