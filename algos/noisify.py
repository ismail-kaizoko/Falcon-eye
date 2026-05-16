import cv2
import numpy as np


INPUT_VIDEO = "./docs/vids/road.mp4"
OUTPUT_VIDEO = "./docs/vids/road_noisy.mp4"

MAX_TRANSLATION = 8     # pixels
MAX_ROTATION = 2.0      # degrees

SMOOTHNESS = 0.9


cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (w, h)
)

dx, dy, angle = 0, 0, 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Smooth random motion
    dx = SMOOTHNESS * dx + np.random.randn() * 2
    dy = SMOOTHNESS * dy + np.random.randn() * 2
    angle = SMOOTHNESS * angle + np.random.randn() * 0.5

    dx = np.clip(dx, -MAX_TRANSLATION, MAX_TRANSLATION)
    dy = np.clip(dy, -MAX_TRANSLATION, MAX_TRANSLATION)
    angle = np.clip(angle, -MAX_ROTATION, MAX_ROTATION)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(
        (w//2, h//2),
        angle,
        1.0
    )

    # Add translation
    M[0, 2] += dx
    M[1, 2] += dy

    noisy = cv2.warpAffine(
        frame,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    writer.write(noisy)

cap.release()
writer.release()

print("Done → noisy.mp4")