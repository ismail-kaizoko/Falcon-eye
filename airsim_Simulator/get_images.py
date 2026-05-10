#Get images (this is what you need for VO)
import airsim
import numpy as np
import cv2
from algos import optical_flow


client = airsim.MultirotorClient()
client.confirmConnection()

response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]

img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img = img1d.reshape(response.height, response.width, 3)

cv2.imshow("AirSim Image", img)
cv2.waitKey(0)




#Get IMU data (VERY IMPORTANT for you)
imu = client.getImuData()

print("Angular velocity:", imu.angular_velocity)
print("Linear acceleration:", imu.linear_acceleration)




#Move the drone (generate your dataset)
client.moveByVelocityAsync(1, 0, 0, 2).join()  # forward


frames = []

for i in range(50):
    client.moveByVelocityAsync(1, 0, 0, 0.1).join()

    response = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])[0]

    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = img.reshape(response.height, response.width, 3)

    frames.append(img)



#Plug into your VO pipeline
R, t = estimate_motion(frames[i], frames[i+1])



#Ground truth (THIS IS GOLD)
state = client.getMultirotorState()

pos = state.kinematics_estimated.position
ori = state.kinematics_estimated.orientation

#Ground truth (THIS IS GOLD)
info = client.simGetCameraInfo("0")

K = np.array(info.proj_mat.matrix).reshape(4,4)