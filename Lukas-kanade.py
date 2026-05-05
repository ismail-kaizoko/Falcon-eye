import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load two consecutive frames
Path = "docs/imgs/corridor/"

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



images = sorted(glob.glob(Path + "*.png"))
frames = [cv2.imread(img) for img in images]


plt.ion()  # interactive mode

for i in range(len(frames)):
    if i==len(frames)-1 :
        last_frame = frames[-1]
        plt.imshow(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        plt.pause(2)

    f1 = frames[i]
    f2 = frames[i+1]

    inliers_old, inliers_new = Lucas_Kanade(f1,f2)

    output = f1.copy()
    for (old, new) in zip(inliers_old, inliers_new):
        x1, y1 = old.ravel()
        x2, y2 = new.ravel()

        cv2.arrowedLine(output,(int(x2), int(y2)), (int(x1), int(y1)), (0,255,255), 2, tipLength=0.2)
        # cv2.arrowedLine(output, (int(x1), int()), 3, (0,0,255), -1)
    

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.pause(1)



