from airsim_Simulator.drone_game import AngleIntegrator, print_initial_angles


def test_angle_integrator_rejects_huge_single_frame_delta():
    integrator = AngleIntegrator()
    integrator.reset((1.0, 2.0, 3.0))

    estimate, delta, ok = integrator.update((0.0, -400.0, 0.0), True)

    assert estimate == (1.0, 2.0, 3.0)
    assert delta == (0.0, 0.0, 0.0)
    assert ok is False


def test_angle_integrator_ignores_deadband_delta():
    integrator = AngleIntegrator()
    integrator.reset((1.0, 2.0, 3.0))

    estimate, delta, ok = integrator.update((0.05, -0.05, 0.05), True)

    assert estimate == (1.0, 2.0, 3.0)
    assert delta == (0.0, 0.0, 0.0)
    assert ok is False


def test_angle_integrator_smooths_and_accumulates_valid_delta():
    integrator = AngleIntegrator()
    integrator.reset((0.0, 0.0, 0.0))

    estimate, delta, ok = integrator.update((2.0, 0.0, 0.0), True)

    assert delta == (0.7, 0.0, 0.0)
    assert estimate == (0.7, 0.0, 0.0)
    assert ok is True


def test_print_initial_angles_labels_yaw_pitch_roll(capsys):
    print_initial_angles((1.0, 2.0, 3.0))

    output = capsys.readouterr().out

    assert "yaw=3.000, pitch=2.000, roll=1.000" in output
    assert "Internal angle order (roll, pitch, yaw) deg: (1.000, 2.000, 3.000)" in output
