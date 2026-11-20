"""MuJoCo <-> ROS 2 bridge.

Runs the Kittra simulation at 30 Hz and exposes it through the same interface the Phase 3
hardware driver will provide:

    publishes   /joint_states                  sensor_msgs/JointState   (measured, incl. sensor noise if configured)
                /camera/overhead/image_raw     sensor_msgs/Image        rgb8
                /camera/wrist/image_raw        sensor_msgs/Image        rgb8
                /tf, /tf_static                world -> base_link, world -> part frames
                /kittra/scene_markers          visualization_msgs/MarkerArray
                /kittra/episode_info           std_msgs/String (JSON: reset / success / step count)
    subscribes  /joint_commands                sensor_msgs/JointState   (absolute targets, by joint name)
    services    /kittra/reset                  std_srvs/Trigger

Parameters
    env_config      config name or YAML path (default "nominal")
    image_size      camera resolution (default 128)
    publish_images  bool (default true)
    rate_hz         control rate (default 30.0)
    autopilot       "none" or "expert": run the scripted expert inside the bridge (Phase 1 demo)
    auto_reset      start a new episode when the current one ends (default true)
    seed            first episode seed (default 0)
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from kittra_bridge.ros_conversions import make_transform, numpy_to_image
from kittra_sim import constants as C
from kittra_sim.factory import make_pick_place_env


class MujocoBridge(Node):
    def __init__(self):
        super().__init__("mujoco_bridge")
        self.declare_parameter("env_config", "nominal")
        self.declare_parameter("image_size", 128)
        self.declare_parameter("publish_images", True)
        self.declare_parameter("rate_hz", float(C.CONTROL_HZ))
        self.declare_parameter("autopilot", "none")
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("seed", 0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.publish_images = bool(p("publish_images"))
        self.autopilot = str(p("autopilot"))
        self.auto_reset = bool(p("auto_reset"))
        self.seed = int(p("seed"))
        obs_mode = "rgb" if self.publish_images else "state"
        self.env = make_pick_place_env(p("env_config"), obs_mode=obs_mode, image_size=int(p("image_size")))
        self.base = self.env.unwrapped

        self.expert = None
        if self.autopilot == "expert":
            from kittra_sim.scripted_expert import ScriptedPickPlaceExpert

            self.expert = ScriptedPickPlaceExpert(self.env)
        elif self.autopilot != "none":
            raise ValueError("autopilot must be 'none' or 'expert'")

        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.img_pubs = {
            C.OVERHEAD_CAMERA: self.create_publisher(Image, "/camera/overhead/image_raw", qos_profile_sensor_data),
            C.WRIST_CAMERA: self.create_publisher(Image, "/camera/wrist/image_raw", qos_profile_sensor_data),
        }
        self.marker_pub = self.create_publisher(MarkerArray, "/kittra/scene_markers", 10)
        self.info_pub = self.create_publisher(String, "/kittra/episode_info", 10)
        self.tf_pub = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.create_subscription(JointState, "/joint_commands", self._on_cmd, 10)
        self.create_service(Trigger, "/kittra/reset", self._on_reset)

        now = self.get_clock().now().to_msg()
        self.static_tf.sendTransform(make_transform(now, "world", "base_link", (0, 0, 0), (1, 0, 0, 0)))

        self.episode = 0
        self._reset_episode()
        self.create_timer(1.0 / float(p("rate_hz")), self._tick)
        self.get_logger().info(
            f"bridge running: config={p('env_config')} images={self.publish_images} autopilot={self.autopilot}")

    # ------------------------------------------------------------------ #
    def _reset_episode(self) -> None:
        self.obs, self.info = self.env.reset(seed=self.seed + self.episode)
        self.cmd = self.base.current_command()
        if self.expert is not None:
            self.expert.reset(self.obs, self.info)
        self.steps = 0
        self._publish_info("reset")

    def _on_cmd(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(C.JOINT_NAMES)}
        cmd = self.cmd.copy()
        if msg.name:
            for n, v in zip(msg.name, msg.position):
                if n in name_to_idx:
                    cmd[name_to_idx[n]] = v
        elif len(msg.position) == len(C.JOINT_NAMES):
            cmd[:] = msg.position
        else:
            self.get_logger().warn("ignoring /joint_commands without names and with wrong length")
            return
        self.cmd = cmd

    def _on_reset(self, request, response):
        self.episode += 1
        self._reset_episode()
        response.success, response.message = True, f"episode {self.episode} started"
        return response

    def _tick(self) -> None:
        if self.expert is not None:
            self.cmd = np.asarray(self.expert.act(self.obs, self.info), dtype=float)
        self.obs, _, terminated, truncated, self.info = self.env.step(self.cmd)
        self.steps += 1
        self._publish_state()
        if terminated or truncated:
            self._publish_info("success" if self.info["is_success"] else "failure")
            if self.auto_reset:
                self.episode += 1
                self._reset_episode()

    # ------------------------------------------------------------------ #
    def _publish_state(self) -> None:
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = list(C.ROS_JOINT_NAMES)
        q = [float(v) for v in self.obs["joint_pos"]]
        js.position = q + [q[5]]  # gripper_mirror follows gripper
        js.velocity = [float(v) for v in self.obs["joint_vel"]] + [float(self.obs["joint_vel"][5])]
        self.js_pub.publish(js)

        if self.publish_images:
            for cam, pub in self.img_pubs.items():
                pub.publish(numpy_to_image(self.obs[f"{cam}_rgb"], stamp, f"{cam}_camera"))

        tfs = []
        markers = MarkerArray()
        for i, name in enumerate(C.ALL_PARTS):
            pos, quat = self.base.h.part_pose(self.base.data, name)
            tfs.append(make_transform(stamp, "world", name, pos, quat))
            markers.markers.append(self._part_marker(i, name, pos, quat, stamp))
        for j, (site, rgba) in enumerate(((C.TRAY_SPAWN_SITE, (0.3, 0.35, 0.4, 0.6)),
                                          (C.KIT_AREA_SITE, (0.85, 0.6, 0.2, 0.6)),
                                          (C.REJECT_AREA_SITE, (0.7, 0.15, 0.15, 0.6)))):
            area = self.base.h.area(self.base.data, site)
            markers.markers.append(self._box_marker(100 + j, site, area, rgba, stamp))
        self.tf_pub.sendTransform(tfs)
        self.marker_pub.publish(markers)

    def _part_marker(self, mid, name, pos, quat, stamp) -> Marker:
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, "world"
        m.ns, m.id = "parts", mid
        m.type = Marker.CUBE if name == C.TARGET_PART else Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in quat)
        m.scale.x = m.scale.y = 0.03 if name == C.TARGET_PART else 0.028
        m.scale.z = 0.03
        gid = self.base.h.part_geom[name]
        r, g, b, a = (float(v) for v in self.base.model.geom_rgba[gid])
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        return m

    def _box_marker(self, mid, name, area, rgba, stamp) -> Marker:
        m = Marker()
        m.header.stamp, m.header.frame_id = stamp, "world"
        m.ns, m.id = "containers", mid
        m.type, m.action = Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y = float(area.center[0]), float(area.center[1])
        m.pose.position.z = float(area.z) - 0.002
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 2 * float(area.half[0]), 2 * float(area.half[1]), 0.004
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _publish_info(self, event: str) -> None:
        msg = String()
        msg.data = json.dumps({"event": event, "episode": self.episode, "steps": self.steps,
                               "success": bool(self.info.get("is_success", False))})
        self.info_pub.publish(msg)

    def destroy_node(self):
        self.env.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoBridge()
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
