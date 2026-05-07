"""Keyboard-controlled AirSim drone with a live camera window.

Run from the repository root:
    python airsim/drone_game.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Set

import airsim
import cv2
import numpy as np

try:
    from pynput import keyboard
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency: pynput. Install project dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from drone_game_config import (
    AIRSIM_HOST,
    AUTO_TAKEOFF,
    CAMERA_IMAGE_TYPE,
    CAMERA_NAME,
    COMMAND_DURATION_SECONDS,
    COMMAND_HZ,
    DISPLAY_STATUS_OVERLAY,
    FAST_MULTIPLIER,
    FORWARD_SPEED_MPS,
    HOVER_ON_EXIT,
    LAND_ON_EXIT,
    MAX_ALTITUDE_UP_METERS,
    MIN_ALTITUDE_UP_METERS,
    STRAFE_SPEED_MPS,
    STREAM_FPS,
    TAKEOFF_TIMEOUT_SECONDS,
    VEHICLE_NAME,
    VERTICAL_SPEED_MPS,
    WINDOW_NAME,
    YAW_RATE_DEG_PER_SEC,
)


KEY_HELP = (
    "W/S forward/back | A/D strafe | Space/Ctrl up/down | Q/E yaw | "
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


def current_altitude_up(client: airsim.MultirotorClient) -> float:
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    return -float(state.kinematics_estimated.position.z_val)


def axis(pressed: Set[str], positive: str, negative: str) -> float:
    return float(positive in pressed) - float(negative in pressed)


def control_loop(
    client: airsim.MultirotorClient,
    keys: KeyboardState,
    stop_event: threading.Event,
) -> None:
    period = 1.0 / COMMAND_HZ
    last_discrete_action = 0.0

    while not stop_event.is_set():
        start = time.perf_counter()
        pressed, stop_requested = keys.snapshot()
        if stop_requested:
            stop_event.set()
            break

        if "t" in pressed and start - last_discrete_action > 0.8:
            client.takeoffAsync(timeout_sec=TAKEOFF_TIMEOUT_SECONDS, vehicle_name=VEHICLE_NAME).join()
            last_discrete_action = start
            continue
        elif "l" in pressed and start - last_discrete_action > 0.8:
            client.landAsync(vehicle_name=VEHICLE_NAME).join()
            last_discrete_action = start
            continue
        elif "h" in pressed and start - last_discrete_action > 0.4:
            client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
            last_discrete_action = start
            continue

        speed_multiplier = FAST_MULTIPLIER if "shift" in pressed else 1.0
        vx = axis(pressed, positive="w", negative="s") * FORWARD_SPEED_MPS * speed_multiplier
        vy = axis(pressed, positive="d", negative="a") * STRAFE_SPEED_MPS * speed_multiplier
        vz = axis(pressed, positive="ctrl", negative="space") * VERTICAL_SPEED_MPS
        yaw_rate = axis(pressed, positive="e", negative="q") * YAW_RATE_DEG_PER_SEC * speed_multiplier

        altitude = current_altitude_up(client)
        if altitude >= MAX_ALTITUDE_UP_METERS and vz < 0:
            vz = 0.0
        if altitude <= MIN_ALTITUDE_UP_METERS and vz > 0:
            vz = 0.0

        client.moveByVelocityBodyFrameAsync(
            vx,
            vy,
            vz,
            COMMAND_DURATION_SECONDS,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=VEHICLE_NAME,
        )

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, period - elapsed))


def frame_from_response(response: airsim.ImageResponse) -> np.ndarray | None:
    if response.height == 0 or response.width == 0 or not response.image_data_uint8:
        return None

    frame = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    frame = frame.reshape(response.height, response.width, 3)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def draw_overlay(frame: np.ndarray, fps: float, keys: Set[str]) -> None:
    if not DISPLAY_STATUS_OVERLAY:
        return

    cv2.putText(
        frame,
        KEY_HELP,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"stream {fps:4.1f} fps | keys: {' '.join(sorted(keys))}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 220, 120),
        1,
        cv2.LINE_AA,
    )


def stream_loop(
    client: airsim.MultirotorClient,
    keys: KeyboardState,
    stop_event: threading.Event,
) -> None:
    image_type = airsim_image_type()
    request = airsim.ImageRequest(CAMERA_NAME, image_type, False, False)
    period = 1.0 / STREAM_FPS
    last_frame_time = time.perf_counter()
    measured_fps = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        start = time.perf_counter()
        responses = client.simGetImages([request], vehicle_name=VEHICLE_NAME)
        frame = frame_from_response(responses[0]) if responses else None
        if frame is None:
            time.sleep(period)
            continue

        now = time.perf_counter()
        measured_fps = 0.85 * measured_fps + 0.15 * (1.0 / max(1e-6, now - last_frame_time))
        last_frame_time = now

        pressed, stop_requested = keys.snapshot()
        if stop_requested:
            stop_event.set()
            break

        draw_overlay(frame, measured_fps, pressed)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            keys.request_stop()
            stop_event.set()
            break

        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, period - elapsed))


def main() -> None:
    keys = KeyboardState()
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
    setup_drone(control_client)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    worker = threading.Thread(target=control_loop, args=(control_client, keys, stop_event), daemon=True)
    worker.start()

    try:
        stream_loop(camera_client, keys, stop_event)
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        listener.stop()
        if LAND_ON_EXIT:
            control_client.landAsync(vehicle_name=VEHICLE_NAME).join()
        elif HOVER_ON_EXIT:
            control_client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
