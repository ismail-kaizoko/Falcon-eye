# Commands to interact with AirSim :

## Connection & control

```python
import airsim

client=airsim.MultirotorClient()
client.confirmConnection()
```

---

## 🎮 Motion control

### Velocity control (most useful)

```python
client.moveByVelocityAsync(vx,vy,vz,duration).join()
```

- vx, vy, vz → m/s (body frame)
- duration → seconds

### Position control

```python
client.moveToPositionAsync(x,y,z,velocity).join()
```

### Orientation (yaw)

```python
client.rotateToYawAsync(yaw_deg).join()
```

---

### Combined motion

```python
client.moveByVelocityZAsync(vx,vy,z,duration).join()
```

---

## 📷 Cameras

### Get RGB image

```python
response=client.simGetImages([
airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)])[0]
```

### Convert to numpy

```python
import numpy as np

img=np.frombuffer(response.image_data_uint8,dtype=np.uint8)
img=img.reshape(response.height,response.width,3)
```

### Other image types

```python
{
  "SimMode":"Multirotor",
  "Vehicles": {
    "drone1": {
      "VehicleType":"SimpleFlight",
      "AutoCreate":true
    }
  }
}
```

```python
airsim.ImageType.DepthPerspective
airsim.ImageType.Segmentation
airsim.ImageType.SurfaceNormals
```

## Camera info (intrinsics)

```python
info=client.simGetCameraInfo("0")

info.fov
info.pose
```

---

## 🎯 State Ground-Truth

### IMU (for VIO)

```python
imu=client.getImuData()

imu.angular_velocity
imu.linear_acceleration
imu.orientation
```

### State

```python
state=client.getMultirotorState()

pos=state.kinematics_estimated.position
vel=state.kinematics_estimated.linear_velocity
ori=state.kinematics_estimated.orientation
```

### Pose (direct)

```python
pose=client.simGetVehiclePose()

pose.position
pose.orientation
```

## ⏱ Time control

```
client.simPause(True)
client.simContinueForTime(1.0)
```

👉 useful for reproducible experiments
