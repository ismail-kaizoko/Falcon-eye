"""Keyboard-controlled AirSim drone with a live camera window.

Run from the repository root:
    python airsim/drone_game.py
"""

from __future__ import annotations

# import sys
# sys.path.append("../")


import os
import threading
import time
from dataclasses import dataclass, field
from typing import Set, Tuple

import airsim
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from airsim import to_eularian_angles

try:
    from .optical_flow import estimate_Rt
except ImportError:  # Support `python airsim_Simulator/drone_game.py`.
    from optical_flow import estimate_Rt

try:
    from pynput import keyboard
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency: pynput. Install project dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

try:
    from .drone_game_config import (
        AIRSIM_HOST,
        AUTO_TAKEOFF,
        ANGLE_ESTIMATE_EVERY_N_FRAMES,
        ANGLE_PLOT_DIR,
        CAMERA_IMAGE_TYPE,
        CAMERA_NAME,
        COMMAND_DURATION_SECONDS,
        COMMAND_HZ,
        DISPLAY_STATUS_OVERLAY,
        FAST_MULTIPLIER,
        FORWARD_SPEED_MPS,
        HOVER_ON_EXIT,
        KEY_BACKWARD,
        KEY_DOWN,
        KEY_FAST,
        KEY_FORWARD,
        KEY_HOVER,
        KEY_LAND,
        KEY_STRAFE_LEFT,
        KEY_STRAFE_RIGHT,
        KEY_TAKEOFF,
        KEY_UP,
        KEY_YAW_LEFT,
        KEY_YAW_RIGHT,
        LAND_ON_EXIT,
        SAVE_ANGLE_PLOTS,
        STRAFE_SPEED_MPS,
        STREAM_FPS,
        TAKEOFF_TIMEOUT_SECONDS,
        TELEMETRY_HZ,
        VEHICLE_NAME,
        VERTICAL_SPEED_MPS,
        WINDOW_NAME,
        YAW_RATE_DEG_PER_SEC,
    )
except ImportError:  # Support `python airsim_Simulator/drone_game.py`.
    from drone_game_config import (
        AIRSIM_HOST,
        AUTO_TAKEOFF,
        ANGLE_ESTIMATE_EVERY_N_FRAMES,
        ANGLE_PLOT_DIR,
        CAMERA_IMAGE_TYPE,
        CAMERA_NAME,
        COMMAND_DURATION_SECONDS,
        COMMAND_HZ,
        DISPLAY_STATUS_OVERLAY,
        FAST_MULTIPLIER,
        FORWARD_SPEED_MPS,
        HOVER_ON_EXIT,
        KEY_BACKWARD,
        KEY_DOWN,
        KEY_FAST,
        KEY_FORWARD,
        KEY_HOVER,
        KEY_LAND,
        KEY_STRAFE_LEFT,
        KEY_STRAFE_RIGHT,
        KEY_TAKEOFF,
        KEY_UP,
        KEY_YAW_LEFT,
        KEY_YAW_RIGHT,
        LAND_ON_EXIT,
        SAVE_ANGLE_PLOTS,
        STRAFE_SPEED_MPS,
        STREAM_FPS,
        TAKEOFF_TIMEOUT_SECONDS,
        TELEMETRY_HZ,
        VEHICLE_NAME,
        VERTICAL_SPEED_MPS,
        WINDOW_NAME,
        YAW_RATE_DEG_PER_SEC,
    )


KEY_HELP = (
    "Z/S forward/back | Q/D strafe | Space/Ctrl up/down | A/E yaw | "
    "Shift fast | H hover | L land | T takeoff | Esc quit"
)


@dataclass
class KeyboardState:
    pressed: Set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_requested: bool = False

    def press(self, key: str) -> None:
        with self.lock:
            if key == "esc":
                self.stop_requested = True
            if key in self.pressed:
                return
            self.pressed.add(key)

    def release(self, key: str) -> None:
        with self.lock:
            self.pressed.discard(key)

    def snapshot(self) -> tuple[Set[str], bool]:
        with self.lock:
            return set(self.pressed), self.stop_requested

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True


AngleTriplet = Tuple[float, float, float]


@dataclass
class OrientationState:
    true_angles_deg: AngleTriplet = (0.0, 0.0, 0.0)
    estimated_angles_deg: AngleTriplet = (0.0, 0.0, 0.0)
    delta_angles_deg: AngleTriplet = (0.0, 0.0, 0.0)
    last_estimation_ok: bool = False
    updated_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_true(self, true_angles_deg: AngleTriplet) -> None:
        with self.lock:
            self.true_angles_deg = true_angles_deg
            self.updated_at = time.perf_counter()

    def update_estimate(
        self,
        estimated_angles_deg: AngleTriplet,
        delta_angles_deg: AngleTriplet,
        ok: bool,
    ) -> None:
        with self.lock:
            self.estimated_angles_deg = estimated_angles_deg
            self.delta_angles_deg = delta_angles_deg
            self.last_estimation_ok = ok

    def snapshot(self) -> tuple[AngleTriplet, AngleTriplet, AngleTriplet, bool, float]:
        with self.lock:
            return (
                self.true_angles_deg,
                self.estimated_angles_deg,
                self.delta_angles_deg,
                self.last_estimation_ok,
                self.updated_at,
            )


@dataclass
class AngleHistory:
    times: list[float] = field(default_factory=list)
    true_angles_deg: list[AngleTriplet] = field(default_factory=list)
    estimated_angles_deg: list[AngleTriplet] = field(default_factory=list)
    delta_angles_deg: list[AngleTriplet] = field(default_factory=list)
    start_time: float = field(default_factory=time.perf_counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(
        self,
        true_angles_deg: AngleTriplet,
        estimated_angles_deg: AngleTriplet,
        delta_angles_deg: AngleTriplet,
    ) -> None:
        with self.lock:
            self.times.append(time.perf_counter() - self.start_time)
            self.true_angles_deg.append(true_angles_deg)
            self.estimated_angles_deg.append(estimated_angles_deg)
            self.delta_angles_deg.append(delta_angles_deg)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            return (
                np.asarray(self.times, dtype=float),
                np.asarray(self.true_angles_deg, dtype=float),
                np.asarray(self.estimated_angles_deg, dtype=float),
                np.asarray(self.delta_angles_deg, dtype=float),
            )


def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if isinstance(key, keyboard.KeyCode) and key.char:
        return key.char.lower()

    special_keys = {
        keyboard.Key.esc: "esc",
        keyboard.Key.space: "space",
        keyboard.Key.shift: "shift",
        keyboard.Key.shift_l: "shift",
        keyboard.Key.shift_r: "shift",
        keyboard.Key.ctrl: "ctrl",
        keyboard.Key.ctrl_l: "ctrl",
        keyboard.Key.ctrl_r: "ctrl",
    }
    return special_keys.get(key)


def make_client() -> airsim.MultirotorClient:
    if AIRSIM_HOST:
        client = airsim.MultirotorClient(ip=AIRSIM_HOST)
    else:
        client = airsim.MultirotorClient()
    client.confirmConnection()
    return client


def airsim_image_type() -> int:
    try:
        return getattr(airsim.ImageType, CAMERA_IMAGE_TYPE)
    except AttributeError as exc:
        raise ValueError(f"Unsupported AirSim image type: {CAMERA_IMAGE_TYPE}") from exc


def setup_drone(client: airsim.MultirotorClient) -> None:
    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)
    if AUTO_TAKEOFF:
        client.takeoffAsync(timeout_sec=TAKEOFF_TIMEOUT_SECONDS, vehicle_name=VEHICLE_NAME).join()
        client.hoverAsync(vehicle_name=VEHICLE_NAME).join()


def axis(pressed: Set[str], positive: str, negative: str) -> float:
    return float(positive in pressed) - float(negative in pressed)


@dataclass
class CommandScheduler:
    last_motion_refresh: float = 0.0
    last_discrete_action: float = 0.0
    last_command: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    zero_sent: bool = False


def command_from_keys(pressed: Set[str]) -> tuple[float, float, float, float]:
    speed_multiplier = FAST_MULTIPLIER if KEY_FAST in pressed else 1.0
    vx = axis(pressed, positive=KEY_FORWARD, negative=KEY_BACKWARD) * FORWARD_SPEED_MPS * speed_multiplier
    vy = axis(pressed, positive=KEY_STRAFE_RIGHT, negative=KEY_STRAFE_LEFT) * STRAFE_SPEED_MPS * speed_multiplier
    vz = axis(pressed, positive=KEY_DOWN, negative=KEY_UP) * VERTICAL_SPEED_MPS
    yaw_rate = axis(pressed, positive=KEY_YAW_RIGHT, negative=KEY_YAW_LEFT) * YAW_RATE_DEG_PER_SEC * speed_multiplier
    return vx, vy, vz, yaw_rate


def send_motion_command(
    client: airsim.MultirotorClient,
    command: tuple[float, float, float, float],
) -> None:
    vx, vy, vz, yaw_rate = command
    client.moveByVelocityBodyFrameAsync(
        vx,
        vy,
        vz,
        COMMAND_DURATION_SECONDS,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
        vehicle_name=VEHICLE_NAME,
    )


def service_controls(
    client: airsim.MultirotorClient,
    pressed: Set[str],
    scheduler: CommandScheduler,
) -> None:
    now = time.perf_counter()

    if KEY_TAKEOFF in pressed and now - scheduler.last_discrete_action > 0.8:
        client.takeoffAsync(timeout_sec=TAKEOFF_TIMEOUT_SECONDS, vehicle_name=VEHICLE_NAME)
        scheduler.last_discrete_action = now
        return
    if KEY_LAND in pressed and now - scheduler.last_discrete_action > 0.8:
        client.landAsync(vehicle_name=VEHICLE_NAME)
        scheduler.last_discrete_action = now
        return
    if KEY_HOVER in pressed and now - scheduler.last_discrete_action > 0.4:
        client.hoverAsync(vehicle_name=VEHICLE_NAME)
        scheduler.last_discrete_action = now
        scheduler.zero_sent = True
        scheduler.last_command = (0.0, 0.0, 0.0, 0.0)
        return

    command = command_from_keys(pressed)
    command_is_zero = command == (0.0, 0.0, 0.0, 0.0)

    if command_is_zero:
        if not scheduler.zero_sent:
            send_motion_command(client, command)
            scheduler.zero_sent = True
            scheduler.last_command = command
        return

    command_changed = command != scheduler.last_command
    refresh_due = now - scheduler.last_motion_refresh >= 1.0 / COMMAND_HZ
    if command_changed or refresh_due:
        send_motion_command(client, command)
        scheduler.last_motion_refresh = now
        scheduler.last_command = command
        scheduler.zero_sent = False


def add_angles(a: AngleTriplet, b: AngleTriplet) -> AngleTriplet:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def radians_to_degrees(angles_rad: AngleTriplet) -> AngleTriplet:
    return tuple(float(np.degrees(value)) for value in angles_rad)  # type: ignore[return-value]


def true_angles_from_state(client: airsim.MultirotorClient) -> AngleTriplet:
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    pitch, roll, yaw = to_eularian_angles(state.kinematics_estimated.orientation)
    return radians_to_degrees((roll, pitch, yaw))


def telemetry_loop(
    client: airsim.MultirotorClient,
    orientation: OrientationState,
    stop_event: threading.Event,
) -> None:
    period = 1.0 / TELEMETRY_HZ

    while not stop_event.is_set():
        start = time.perf_counter()
        orientation.update_true(true_angles_from_state(client))

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, period - elapsed))


def frame_from_response(response: airsim.ImageResponse) -> np.ndarray | None:
    if response.height == 0 or response.width == 0 or not response.image_data_uint8:
        return None

    frame = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    frame = frame.reshape(response.height, response.width, 3)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def estimate_delta_angles(previous_frame: np.ndarray, frame: np.ndarray) -> tuple[AngleTriplet, bool]:
    try:
        _, _, _, roll, pitch, yaw = estimate_Rt(previous_frame, frame)
    except Exception:
        return (0.0, 0.0, 0.0), False

    delta = (roll, pitch, yaw)
    if not np.all(np.isfinite(delta)):
        return (0.0, 0.0, 0.0), False
    return delta, True


def save_angle_plots(history: AngleHistory) -> list[str]:
    if not SAVE_ANGLE_PLOTS:
        return []

    times, true_angles, estimated_angles, _ = history.snapshot()
    if len(times) < 2:
        return []

    os.makedirs(ANGLE_PLOT_DIR, exist_ok=True)
    saved_paths: list[str] = []
    labels = ("roll", "pitch", "yaw")

    for index, label in enumerate(labels):
        plt.figure(figsize=(10, 5))
        plt.plot(times, true_angles[:, index], label=f"true {label}")
        plt.plot(times, estimated_angles[:, index], label=f"estimated {label}")
        plt.xlabel("time (s)")
        plt.ylabel("angle (deg)")
        plt.title(f"{label.title()} true vs estimated")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()

        path = os.path.join(ANGLE_PLOT_DIR, f"{label}_true_vs_estimated.png")
        plt.savefig(path, dpi=140)
        plt.close()
        saved_paths.append(path)

    return saved_paths


def format_angles(angles: AngleTriplet) -> str:
    roll, pitch, yaw = angles
    return f"yaw {yaw:7.2f}  pitch {pitch:7.2f}  roll {roll:7.2f}"


def draw_overlay(frame: np.ndarray, fps: float, keys: Set[str], orientation: OrientationState) -> None:
    if not DISPLAY_STATUS_OVERLAY:
        return

    true_angles, estimated_angles, delta_angles, estimate_ok, updated_at = orientation.snapshot()
    age = time.perf_counter() - updated_at if updated_at else 0.0
    status = "ok" if estimate_ok else "waiting"

    cv2.putText(
        frame,
        KEY_HELP,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"stream {fps:4.1f} fps | keys: {' '.join(sorted(keys))}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (80, 220, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "true  " + format_angles(true_angles),
        (12, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 225, 110),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "est   " + format_angles(estimated_angles) + f"  {status}",
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 225, 110),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "delta " + format_angles(delta_angles) + f"  state {age:0.1f}s",
        (12, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 225, 110),
        1,
        cv2.LINE_AA,
    )


def stream_loop(
    control_client: airsim.MultirotorClient,
    client: airsim.MultirotorClient,
    keys: KeyboardState,
    orientation: OrientationState,
    history: AngleHistory,
    stop_event: threading.Event,
) -> None:
    image_type = airsim_image_type()
    request = airsim.ImageRequest(CAMERA_NAME, image_type, False, False)
    period = 1.0 / STREAM_FPS
    last_frame_time = time.perf_counter()
    measured_fps = 0.0
    scheduler = CommandScheduler()
    previous_frame: np.ndarray | None = None
    frame_index = 0
    estimated_angles: AngleTriplet | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        start = time.perf_counter()
        responses = client.simGetImages([request], vehicle_name=VEHICLE_NAME)
        frame = frame_from_response(responses[0]) if responses else None
        if frame is None:
            time.sleep(period)
            continue
        raw_frame = frame.copy()

        now = time.perf_counter()
        measured_fps = 0.85 * measured_fps + 0.15 * (1.0 / max(1e-6, now - last_frame_time))
        last_frame_time = now

        pressed, stop_requested = keys.snapshot()
        if stop_requested:
            stop_event.set()
            break

        true_angles, _, _, _, _ = orientation.snapshot()
        if estimated_angles is None:
            estimated_angles = true_angles
            orientation.update_estimate(estimated_angles, (0.0, 0.0, 0.0), False)

        if previous_frame is not None and frame_index % ANGLE_ESTIMATE_EVERY_N_FRAMES == 0:
            delta_angles, estimate_ok = estimate_delta_angles(previous_frame, raw_frame)
            if estimate_ok:
                estimated_angles = add_angles(estimated_angles, delta_angles)
            orientation.update_estimate(estimated_angles, delta_angles, estimate_ok)
        else:
            _, current_estimated, current_delta, current_ok, _ = orientation.snapshot()
            orientation.update_estimate(current_estimated, current_delta, current_ok)

        true_angles, current_estimated, current_delta, _, _ = orientation.snapshot()
        history.append(true_angles, current_estimated, current_delta)

        draw_overlay(frame, measured_fps, pressed, orientation)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            keys.request_stop()
            stop_event.set()
            break

        service_controls(control_client, pressed, scheduler)
        previous_frame = raw_frame
        frame_index += 1

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, period - elapsed))


def main() -> None:
    keys = KeyboardState()
    orientation = OrientationState()
    history = AngleHistory()
    stop_event = threading.Event()

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = normalize_key(key)
        if normalized:
            keys.press(normalized)

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        normalized = normalize_key(key)
        if normalized:
            keys.release(normalized)

    control_client = make_client()
    camera_client = make_client()
    telemetry_client = make_client()
    setup_drone(control_client)
    orientation.update_true(true_angles_from_state(telemetry_client))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    telemetry_worker = threading.Thread(
        target=telemetry_loop,
        args=(telemetry_client, orientation, stop_event),
        daemon=True,
    )
    telemetry_worker.start()

    saved_plots: list[str] = []
    try:
        stream_loop(control_client, camera_client, keys, orientation, history, stop_event)
    finally:
        stop_event.set()
        telemetry_worker.join(timeout=2.0)
        listener.stop()
        if LAND_ON_EXIT:
            control_client.landAsync(vehicle_name=VEHICLE_NAME).join()
        elif HOVER_ON_EXIT:
            control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
        cv2.destroyAllWindows()
        saved_plots = save_angle_plots(history)
        for path in saved_plots:
            print(f"Saved angle plot: {path}")


if __name__ == "__main__":
    main()
