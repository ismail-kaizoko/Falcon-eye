"""Tunable variables for the AirSim keyboard driving demo."""

# AirSim RPC target. Leave host empty for the default localhost endpoint.
AIRSIM_HOST = ""
VEHICLE_NAME = ""

# Camera stream.
CAMERA_NAME = "0"
CAMERA_IMAGE_TYPE = "Scene"
STREAM_FPS = 30.0
WINDOW_NAME = "Falcon-EYE Drone Game"
DISPLAY_STATUS_OVERLAY = True

# Control loop.
COMMAND_HZ = 20.0
COMMAND_DURATION_SECONDS = 0.12

# Body-frame velocities in meters/second.
FORWARD_SPEED_MPS = 4.0
STRAFE_SPEED_MPS = 3.0
VERTICAL_SPEED_MPS = 2.0
FAST_MULTIPLIER = 1.75

# Yaw rate in degrees/second.
YAW_RATE_DEG_PER_SEC = 55.0

# Startup / shutdown behavior.
AUTO_TAKEOFF = True
TAKEOFF_TIMEOUT_SECONDS = 12.0
LAND_ON_EXIT = False
HOVER_ON_EXIT = True

# Safety clamps. AirSim uses NED coordinates: negative Z is up.
MAX_ALTITUDE_UP_METERS = 30.0
MIN_ALTITUDE_UP_METERS = 0.5
