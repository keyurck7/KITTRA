"""Conversions between NumPy and ROS 2 messages (no cv_bridge, so no OpenCV/NumPy ABI issues)."""
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image


def numpy_to_image(img: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    return msg


def image_to_numpy(msg: Image) -> np.ndarray:
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
    return img[..., ::-1].copy() if msg.encoding == "bgr8" else img.copy()


def make_transform(stamp, parent: str, child: str, pos, quat_wxyz) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in pos)
    w, x, y, z = (float(v) for v in quat_wxyz)
    t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = w, x, y, z
    return t
