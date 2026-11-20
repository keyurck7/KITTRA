import numpy as np
import pytest

from kittra_sim.transforms import (
    closest_symmetric_angle,
    gripper_yaw_from_mat,
    mat_to_quat,
    quat_conj,
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
    quat_to_yaw,
    random_small_rotation,
    topdown_gripper_mat,
    wrap_angle,
    yaw_quat,
)


def test_quat_mat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        np.testing.assert_allclose(mat_to_quat(quat_to_mat(q)), q, atol=1e-9)


def test_quat_mul_matches_matrix_product():
    rng = np.random.default_rng(1)
    a = quat_from_axis_angle(rng.normal(size=3), 0.7)
    b = quat_from_axis_angle(rng.normal(size=3), -1.3)
    np.testing.assert_allclose(quat_to_mat(quat_mul(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-12)
    np.testing.assert_allclose(quat_mul(a, quat_conj(a)), [1, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("yaw", [-3.0, -1.0, 0.0, 0.4, 2.5])
def test_yaw_helpers(yaw):
    assert quat_to_yaw(yaw_quat(yaw)) == pytest.approx(yaw)
    m = topdown_gripper_mat(yaw)
    np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-12)
    assert np.linalg.det(m) == pytest.approx(1.0)
    np.testing.assert_allclose(m[:, 0], [0, 0, -1])          # approach axis points down
    assert gripper_yaw_from_mat(m) == pytest.approx(yaw)


def test_closest_symmetric_angle():
    assert closest_symmetric_angle(0.1, 1.6, np.pi / 2) == pytest.approx(0.1 + np.pi / 2)
    assert closest_symmetric_angle(3.0, -3.0, 2 * np.pi) == pytest.approx(3.0 - 2 * np.pi)
    assert wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)


def test_random_small_rotation_bounded():
    rng = np.random.default_rng(2)
    for _ in range(100):
        q = random_small_rotation(rng, np.deg2rad(3))
        angle = 2 * np.arccos(np.clip(abs(q[0]), -1, 1))
        assert angle <= np.deg2rad(3) + 1e-9
