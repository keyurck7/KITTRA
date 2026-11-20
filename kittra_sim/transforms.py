"""Small rotation helpers. Quaternions use MuJoCo's (w, x, y, z) convention. Pure NumPy."""
from __future__ import annotations

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def yaw_quat(yaw: float) -> np.ndarray:
    """Rotation about world +z."""
    return quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w >= 0), numerically robust (Shepperd's method)."""
    m = np.asarray(m, dtype=float).reshape(3, 3)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


def quat_to_yaw(q: np.ndarray) -> float:
    """Yaw (rotation about world z) of a body, from its x-axis projected on the xy-plane."""
    m = quat_to_mat(q)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def topdown_gripper_mat(yaw: float) -> np.ndarray:
    """Orientation of the gripper frame for a top-down grasp.

    Gripper frame convention (see assets/so101/so101_simplified.xml):
      x = approach direction (points into the fingers), y = finger opening direction.
    Top-down: x = world -z; the finger opening direction lies in the table plane at angle ``yaw``.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.array([c, s, 0.0])
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def gripper_yaw_from_mat(m: np.ndarray) -> float:
    """Inverse of :func:`topdown_gripper_mat` for the yaw: angle of the gripper y-axis in the xy-plane."""
    m = np.asarray(m).reshape(3, 3)
    return float(np.arctan2(m[1, 1], m[0, 1]))


def closest_symmetric_angle(angle: float, reference: float, symmetry: float) -> float:
    """Among ``angle + k * symmetry`` return the value closest to ``reference``."""
    k = np.round((reference - angle) / symmetry)
    return float(angle + k * symmetry)


def random_small_rotation(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    """Quaternion of a rotation about a random axis by an angle uniform in [-max_angle, max_angle]."""
    axis = rng.normal(size=3)
    angle = rng.uniform(-max_angle, max_angle)
    return quat_from_axis_angle(axis, angle)
