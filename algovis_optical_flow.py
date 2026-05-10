import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from algos.optical_flow import *



# Load two consecutive frames
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



output_frames = []
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

    text = f"Roll: {roll:.1f} | Pitch: {pitch:.1f} | Yaw: {yaw:.1f}"

    cv2.putText(img, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # --- show with matplotlib ---
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Frame {i}")
    ax.axis('off')

    plt.pause(2)  # <-- slow down here (adjust!)
