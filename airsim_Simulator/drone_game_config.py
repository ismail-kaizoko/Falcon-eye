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
TELEMETRY_HZ = 10.0
SAVE_ANGLE_PLOTS = True
ANGLE_PLOT_DIR = "data/drone_game_plots"
ANGLE_ESTIMATE_EVERY_N_FRAMES = 1

# Control loop.
COMMAND_HZ = 4.0
COMMAND_DURATION_SECONDS = 0.35

# AZERTY keyboard map.
KEY_FORWARD = "z"
KEY_BACKWARD = "s"
KEY_STRAFE_LEFT = "q"
KEY_STRAFE_RIGHT = "d"
KEY_YAW_LEFT = "a"
KEY_YAW_RIGHT = "e"
KEY_UP = "space"
KEY_DOWN = "ctrl"
KEY_FAST = "shift"
KEY_HOVER = "h"
KEY_TAKEOFF = "t"
KEY_LAND = "l"

# Body-frame velocities in meters/second.
FORWARD_SPEED_MPS = 10.0
STRAFE_SPEED_MPS = 10.0
VERTICAL_SPEED_MPS = 10.0
FAST_MULTIPLIER = 1.75

# Yaw rate in degrees/second.
YAW_RATE_DEG_PER_SEC = 55.0

# Startup / shutdown behavior.
AUTO_TAKEOFF = True
TAKEOFF_TIMEOUT_SECONDS = 12.0
LAND_ON_EXIT = False
HOVER_ON_EXIT = True
