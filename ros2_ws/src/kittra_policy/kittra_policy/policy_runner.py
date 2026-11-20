"""Run a trained image-based policy against the robot interface.

The node only talks to topics, never to the simulator, so the identical node drives the MuJoCo
bridge in Phase 2 and the real SO-101 driver in Phase 3.

    subscribes  /joint_states, /camera/overhead/image_raw, /camera/wrist/image_raw
                /kittra/episode_info   (resets the policy, e.g. clears ACT's action queue, on "reset")
    publishes   /joint_commands        sensor_msgs/JointState
    services    /kittra_policy/reset   std_srvs/Trigger
                /kittra_policy/enable  std_srvs/SetBool  (starts enabled unless start_enabled:=false)

Parameters
    policy_type   act | diffusion | smolvla
    policy_path   LeRobot pretrained_model directory
    device        cpu | cuda (default cpu)
    rate_hz       control rate (default 30.0)
    start_enabled bool (default true)
"""
from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from kittra_bridge.ros_conversions import image_to_numpy
from kittra_sim import constants as C


class PolicyRunner(Node):
    def __init__(self):
        super().__init__("policy_runner")
        self.declare_parameter("policy_type", "act")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("rate_hz", float(C.CONTROL_HZ))
        self.declare_parameter("start_enabled", True)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        ptype, path = str(p("policy_type")), str(p("policy_path"))
        if ptype not in ("act", "diffusion", "smolvla"):
            raise ValueError("policy_runner supports image-based LeRobot policies: act, diffusion, smolvla")
        if not path:
            raise ValueError("set the policy_path parameter to a LeRobot pretrained_model directory")
        from kittra_learning.policies import load_policy

        self.policy = load_policy(ptype, path, device=str(p("device")))
        self.policy.reset()
        self.enabled = bool(p("start_enabled"))

        self.joint_pos: np.ndarray | None = None
        self.images: dict[str, np.ndarray] = {}
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Image, "/camera/overhead/image_raw",
                                 lambda m: self._on_img(C.OVERHEAD_CAMERA, m), qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/wrist/image_raw",
                                 lambda m: self._on_img(C.WRIST_CAMERA, m), qos_profile_sensor_data)
        self.create_subscription(String, "/kittra/episode_info", self._on_episode, 10)
        self.cmd_pub = self.create_publisher(JointState, "/joint_commands", 10)
        self.create_service(Trigger, "/kittra_policy/reset", self._srv_reset)
        self.create_service(SetBool, "/kittra_policy/enable", self._srv_enable)
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)
        self._inf_ms: list[float] = []
        self.get_logger().info(f"loaded {ptype} policy from {path}; enabled={self.enabled}")

    def _on_js(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in C.JOINT_NAMES):
            self.joint_pos = np.array([msg.position[idx[n]] for n in C.JOINT_NAMES], dtype=np.float32)

    def _on_img(self, cam: str, msg: Image) -> None:
        self.images[cam] = image_to_numpy(msg)

    def _on_episode(self, msg: String) -> None:
        try:
            event = json.loads(msg.data).get("event")
        except json.JSONDecodeError:
            return
        if event == "reset":
            self.policy.reset()

    def _srv_reset(self, request, response):
        self.policy.reset()
        response.success, response.message = True, "policy reset"
        return response

    def _srv_enable(self, request, response):
        self.enabled = bool(request.data)
        if self.enabled:
            self.policy.reset()
        response.success, response.message = True, f"enabled={self.enabled}"
        return response

    def _tick(self) -> None:
        if not self.enabled or self.joint_pos is None or len(self.images) < 2:
            return
        obs = {"joint_pos": self.joint_pos,
               "overhead_rgb": self.images[C.OVERHEAD_CAMERA],
               "wrist_rgb": self.images[C.WRIST_CAMERA]}
        t0 = time.perf_counter()
        action = self.policy.act(obs, {})
        self._inf_ms.append(1000 * (time.perf_counter() - t0))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(C.JOINT_NAMES)
        msg.position = [float(v) for v in action]
        self.cmd_pub.publish(msg)
        if len(self._inf_ms) == 300:
            self.get_logger().info(f"mean inference {np.mean(self._inf_ms):.1f} ms/step")
            self._inf_ms.clear()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
