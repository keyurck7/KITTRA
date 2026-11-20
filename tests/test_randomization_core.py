"""Randomizers and actuator/sensor models, tested on a fake model (plain NumPy arrays)."""
from types import SimpleNamespace

import numpy as np
import pytest

from kittra_sim.config import load_config
from kittra_sim.randomization.core import (
    DelayLine,
    PhysicsRandomizer,
    VisualRandomizer,
    apply_brightness_contrast,
    apply_motion_blur,
    backlash_step,
    to_uint8,
)


def fake_model():
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        geom_friction=np.tile([1.0, 0.02, 0.001], (10, 1)),
        body_mass=rng.uniform(0.01, 0.5, 8),
        body_inertia=rng.uniform(1e-6, 1e-4, (8, 3)),
        dof_damping=np.full(12, 2.0),
        actuator_gainprm=np.tile(np.r_[100.0, np.zeros(9)], (6, 1)),
        actuator_biasprm=np.tile(np.r_[0.0, -100.0, 0.0, np.zeros(7)], (6, 1)),
        light_pos=np.array([[0.15, 0.0, 1.0]]),
        light_diffuse=np.array([[0.7, 0.7, 0.7]]),
        geom_rgba=np.tile([0.5, 0.5, 0.5, 1.0], (10, 1)),
        cam_pos=np.array([[0.15, 0.0, 0.55], [0.0, 0.0, 0.045]]),
        cam_quat=np.array([[0.5, 0.5, -0.5, 0.5], [1.0, 0, 0, 0]]),
    )


def make_physics(model, cfg):
    return PhysicsRandomizer.from_model(model, cfg, friction_geoms=[6, 7, 8], mass_bodies=[5, 6],
                                        damping_dofs=[0, 1, 2, 3, 4], kp_actuators=[0, 1, 2, 3, 4])


def test_physics_ranges_and_selection():
    model = fake_model()
    nominal = {k: np.array(getattr(model, k)) for k in vars(model)}
    cfg = load_config("randomized")["physics"]
    r = make_physics(model, cfg)
    rng = np.random.default_rng(1)
    for _ in range(50):
        r.apply(model, rng)
        f = model.geom_friction[[6, 7, 8], 0] / nominal["geom_friction"][[6, 7, 8], 0]
        assert np.all((f >= cfg["friction"][0]) & (f <= cfg["friction"][1]))
        # untouched elements stay nominal
        np.testing.assert_array_equal(model.geom_friction[:6], nominal["geom_friction"][:6])
        np.testing.assert_array_equal(model.geom_friction[:, 1:], nominal["geom_friction"][:, 1:])
        np.testing.assert_array_equal(model.body_mass[:5], nominal["body_mass"][:5])
        np.testing.assert_array_equal(model.dof_damping[5:], nominal["dof_damping"][5:])
        # inertia scales with mass
        np.testing.assert_allclose(model.body_inertia[5] / nominal["body_inertia"][5],
                                   model.body_mass[5] / nominal["body_mass"][5])
        # position actuator stays consistent: bias = -gain
        np.testing.assert_allclose(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
        np.testing.assert_array_equal(model.actuator_gainprm[5], nominal["actuator_gainprm"][5])


def test_physics_does_not_accumulate_and_restores():
    model = fake_model()
    nominal_mass = model.body_mass.copy()
    cfg = load_config("randomized")["physics"]
    r = make_physics(model, cfg)
    rng = np.random.default_rng(2)
    for _ in range(200):          # would drift far outside the range if it accumulated
        r.apply(model, rng)
    ratio = model.body_mass[[5, 6]] / nominal_mass[[5, 6]]
    assert np.all((ratio >= 0.7) & (ratio <= 1.3))
    r.cfg = load_config("nominal")["physics"]
    r.apply(model, rng)
    np.testing.assert_array_equal(model.body_mass, nominal_mass)


def test_visual_randomizer_bounds_and_restore():
    model = fake_model()
    nominal = {k: np.array(getattr(model, k)) for k in ("light_pos", "geom_rgba", "cam_pos", "cam_quat")}
    cfg = load_config("randomized")["visual"]
    r = VisualRandomizer.from_model(model, cfg, light_ids=[0], color_geoms=[0, 1, 2],
                                    target_geoms=[3], camera_ids=[0])
    rng = np.random.default_rng(3)
    for _ in range(50):
        r.apply(model, rng)
        assert np.all(np.abs(model.light_pos - nominal["light_pos"]) <= cfg["light_pos_offset"] + 1e-12)
        assert np.all(np.abs(model.cam_pos[0] - nominal["cam_pos"][0]) <= cfg["cam_pos_offset"] + 1e-12)
        np.testing.assert_array_equal(model.cam_pos[1], nominal["cam_pos"][1])     # wrist camera untouched
        assert np.linalg.norm(model.cam_quat[0]) == pytest.approx(1.0)
        dot = abs(float(model.cam_quat[0] @ nominal["cam_quat"][0]))
        assert 2 * np.degrees(np.arccos(min(dot, 1.0))) <= cfg["cam_rot_deg"] + 1e-6
        assert np.all(np.abs(model.geom_rgba[:3, :3] - 0.5) <= cfg["color_jitter"] + 1e-12)
        np.testing.assert_array_equal(model.geom_rgba[4:], nominal["geom_rgba"][4:])
    r.cfg = load_config("nominal")["visual"]
    r.apply(model, rng)
    for k, v in nominal.items():
        np.testing.assert_array_equal(getattr(model, k), v)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_delay_line(delay):
    line = DelayLine(delay, fill=np.zeros(2))
    outs = [line.push(np.array([i, -i], float)) for i in range(1, 8)]
    for i, out in enumerate(outs, start=1):
        expected = i - delay if i - delay >= 1 else 0
        np.testing.assert_array_equal(out, [expected, -expected])


def test_backlash():
    play = np.array([0.02])
    out = np.array([0.0])
    out = backlash_step(out, np.array([0.005]), play)   # inside the play: no motion
    assert out[0] == pytest.approx(0.0)
    out = backlash_step(out, np.array([0.03]), play)    # beyond: follows with lag play/2
    assert out[0] == pytest.approx(0.02)
    out = backlash_step(out, np.array([0.025]), play)   # reversal inside the play: no motion
    assert out[0] == pytest.approx(0.02)
    np.testing.assert_allclose(backlash_step(np.array([0.3]), np.array([-0.2]), np.zeros(1)), [-0.2])


def test_image_ops():
    img = np.full((16, 16, 3), 100.0)
    np.testing.assert_allclose(apply_brightness_contrast(img, 1.2, 1.0), 120.0)
    stripes = np.zeros((16, 16, 3), np.float32)
    stripes[:, ::2] = 255
    blurred = apply_motion_blur(stripes, 4, horizontal=True)
    assert blurred[:, 2:-2].std() < stripes.std()
    assert to_uint8(np.array([-5.0, 300.0, 12.4])).tolist() == [0, 255, 12]
