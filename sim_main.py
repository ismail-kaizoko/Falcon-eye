#Get images (this is what you need for VO)
import airsim
import numpy as np
import cv2
from matplotlib import pyplot as plt
from algos.optical_flow import *
from skimage.io import imsave


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()


print("Connected and flying!")



# add command to return drone to square-0
client.rotateToYawAsync(0).join()
client.moveToPositionAsync(0,0,0,5).join()

print("setup finished")





#record full rotation : 
step = 10

for i in range(9) :
    deg = i*step
    client.rotateToYawAsync(deg).join()
    print(f"moved by {step}")
    response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 3)
    imsave(f'data/rot{deg}.png', img)





# stream_frames(rot_frames)
print("rotation finished")


#record x-translation :
n = 5
step = 1
for i in range(n*step):
    client.moveByVelocityAsync(0, 2, 0, 1).join()  # forward
    response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(response.height, response.width, 3)
    imsave(f'data/x-transl{i*step}.png', img)



# stream_frames(forward_frames)

print("translation finished")