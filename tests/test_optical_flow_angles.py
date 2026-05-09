import importlib

import numpy as np
import pytest


MODULE_NAMES = (
    "algos.optical_flow",
    "airsim_Simulator.optical_flow",
)


def rotation_matrix_from_degrees(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rz @ ry @ rx


@pytest.mark.parametrize("module_name", MODULE_NAMES)
@pytest.mark.parametrize(
    "expected_degrees",
    [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.0, -15.0, 0.0),
        (0.0, 0.0, 30.0),
        (12.0, -7.0, 25.0),
    ],
)
def test_rotation_to_euler_returns_roll_pitch_yaw_in_degrees(module_name, expected_degrees):
    module = importlib.import_module(module_name)
    rotation = rotation_matrix_from_degrees(*expected_degrees)

    recovered = module.rotation_to_euler(rotation)

    assert np.allclose(recovered, expected_degrees, atol=1e-6)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_rotation_to_euler_handles_gimbal_lock_in_degrees(module_name):
    module = importlib.import_module(module_name)
    rotation = rotation_matrix_from_degrees(20.0, 90.0, 0.0)

    recovered = module.rotation_to_euler(rotation)

    assert np.allclose(recovered, (20.0, 90.0, 0.0), atol=1e-6)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_estimate_rt_rejects_featureless_frames(module_name):
    module = importlib.import_module(module_name)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="not enough trackable features"):
        module.estimate_Rt(frame, frame.copy())


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_lucas_kanade_rejects_none_frames(module_name):
    module = importlib.import_module(module_name)

    with pytest.raises(ValueError, match="valid images"):
        module.Lucas_Kanade(None, None)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_estimate_rt_returns_zero_for_identical_feature_rich_frames(module_name):
    module = importlib.import_module(module_name)
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    for y in range(20, 150, 30):
        for x in range(20, 210, 30):
            frame[y - 2 : y + 3, x - 2 : x + 3] = 255

    recovered = module.estimate_Rt(frame, frame.copy())

    assert np.allclose(recovered, np.zeros(6), atol=1e-9)
