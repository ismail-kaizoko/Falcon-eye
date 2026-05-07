import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load two consecutive frames
Path = "docs/imgs/corridor/"


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



