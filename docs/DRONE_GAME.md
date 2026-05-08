# AirSim Drone Game Mode

This branch adds a small "game mode" runner for the AirSim multirotor:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python airsim\drone_game.py
```

AirSim must already be running in a multirotor environment, for example Blocks, with API control allowed.

## Controls

| Key | Action |
| --- | --- |
| `Z` / `S` | Move forward / backward in the drone body frame |
| `Q` / `D` | Strafe left / right in the drone body frame |
| `Space` / `Ctrl` | Move up / down |
| `A` / `E` | Yaw left / right |
| `Shift` | Fast movement modifier |
| `H` | Hover |
| `T` | Take off |
| `L` | Land |
| `Esc` | Quit |

## Tunable Variables

All gameplay and streaming variables are in `airsim/drone_game_config.py`:

| Variable | Meaning |
| --- | --- |
| `STREAM_FPS` | Target camera refresh rate for the OpenCV window |
| `COMMAND_HZ` | Rate of velocity command updates sent to AirSim |
| `COMMAND_DURATION_SECONDS` | Duration attached to each short AirSim velocity command |
| `TELEMETRY_HZ` | Rate of IMU reads used by the overlay |
| `DISPLAY_IMU_OVERLAY` | Show/hide the IMU overlay lines |
| `KEY_*` | AZERTY key bindings for movement and actions |
| `FORWARD_SPEED_MPS` | Forward/back velocity in meters per second |
| `STRAFE_SPEED_MPS` | Lateral velocity in meters per second |
| `VERTICAL_SPEED_MPS` | Up/down velocity in meters per second |
| `FAST_MULTIPLIER` | Speed multiplier when `Shift` is held |
| `YAW_RATE_DEG_PER_SEC` | Rotation speed for `Q`/`E` |
| `MAX_ALTITUDE_UP_METERS` | Upward altitude clamp, converted from AirSim NED `z` |
| `MIN_ALTITUDE_UP_METERS` | Minimum altitude before down commands are blocked |

## Mechanism

The runner uses the official AirSim Python RPC client. AirSim exposes APIs for vehicle control, state, and images; multirotors can be moved with `move*` commands, and camera frames are fetched with `simGetImages`. AirSim coordinates use NED: positive `X` is north/forward, positive `Y` is east/right, and positive `Z` is down.

The script creates three `MultirotorClient` instances:

- The control client owns arming, takeoff, hover/land, state reads, and motion commands.
- The camera client owns `simGetImages` calls and live display.
- The telemetry client owns low-rate `getImuData` calls for the overlay.

The split matters because `simGetImages` is a blocking RPC call. Instead of letting camera acquisition interrupt the flight loop, `drone_game.py` runs flight control in a background thread at `COMMAND_HZ` while the main thread fetches camera frames at `STREAM_FPS` and renders them with OpenCV. IMU reads are also isolated in their own throttled loop so telemetry does not pause video.

Keyboard state is captured by `pynput` in event callbacks. The control loop reads a snapshot of currently pressed keys, converts it to body-frame velocity components, and sends:

```python
client.moveByVelocityBodyFrameAsync(
    vx,
    vy,
    vz,
    duration,
    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
)
```

`moveByVelocityBodyFrameAsync` gives the GTA-like feel: `Z` always moves toward the drone camera's current forward direction after yaw, instead of moving along a fixed global axis. The command duration is deliberately short and refreshed many times per second. Releasing keys naturally causes the next command to become zero velocity plus zero yaw rate.

To avoid frozen video while keys are held, movement commands are joined before the next movement command is issued, idle zero-velocity commands are not spammed, and altitude checks are throttled. This keeps AirSim's RPC queue from being dominated by movement/state calls while the camera stream is trying to fetch frames.

The camera stream requests uncompressed RGB scene frames:

```python
airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)
```

Each `ImageResponse.image_data_uint8` buffer is reshaped into a NumPy image, converted from RGB to OpenCV's BGR format, annotated with a status overlay, and shown in a `cv2.imshow` window.

The IMU overlay shows:

- Linear acceleration from `imu.linear_acceleration`, in `m/s2`.
- Angular acceleration estimated from the derivative of `imu.angular_velocity`, in `rad/s2`.
- Raw angular velocity, labelled `gyro`, in `rad/s`.

## AirSim References

- Official AirSim API overview: https://microsoft.github.io/AirSim/apis/
- Official AirSim image API docs: https://microsoft.github.io/AirSim/image_apis/
