# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import numpy as np
import rclpy
from bdai_ros2_wrappers.callback_groups import NonReentrantCallbackGroup
from bdai_ros2_wrappers.node import Node
from geometry_msgs.msg import Point
from rclpy.executors import MultiThreadedExecutor
from spatialmath import SE3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from utils.conversions import to_ros_pose


class RootNodeVis(Node):
    def __init__(self) -> None:
        super().__init__("RootNodeVis")
        self.nonreentrant_callback_group = NonReentrantCallbackGroup()
        self.create_subscription(
            MarkerArray,
            "/mobile_exp/node_3d",
            callback=self.add_root,
            callback_group=self.nonreentrant_callback_group,
            qos_profile=10,
        )
        self.root_pub = self.create_publisher(
            msg_type=MarkerArray,
            topic="/mobile_exp/root_node",
            qos_profile=20,
        )

    def add_root(self, markers: MarkerArray) -> None:
        node_markers_pos_ls = []
        for marker in markers._markers:  # noqa
            if marker.ns == "node":
                node_markers_pos_ls.append(
                    np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
                )
                frame_id = marker.header.frame_id
        if not node_markers_pos_ls:
            return
        node_markers_pos = np.stack(node_markers_pos_ls)
        max_z = node_markers_pos.max(axis=0)[2]
        node_markers_pos = node_markers_pos[node_markers_pos[:, 2] > max_z - 0.05]
        avg_xy = node_markers_pos.mean(axis=0)[:2]
        root_pos = [avg_xy[0], avg_xy[1], max_z + 0.2]
        root_pose = np.eye(4)
        root_pose[:3, 3] = root_pos

        pub_markers = []

        marker = Marker()
        marker.type = Marker.SPHERE
        marker.id = 0
        marker.header.frame_id = frame_id
        marker.ns = "node"
        marker.text = "root"
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = ColorRGBA(r=140.0 / 255.0, g=140.0 / 255.0, b=140.0 / 255.0, a=1.0)
        marker.pose = to_ros_pose(SE3(root_pose))
        pub_markers.append(marker)

        for n_i, node_pos in enumerate(node_markers_pos):
            line_strip = Marker()
            line_strip.header.frame_id = frame_id
            line_strip.ns = f"edge_{n_i}"
            line_strip.type = Marker.LINE_STRIP
            line_strip.id = n_i + 1
            line_strip.action = Marker.ADD

            node_mean_i = root_pos
            node_mean_j = node_pos

            from_pt = Point()
            from_pt.x = node_mean_i[0]
            from_pt.y = node_mean_i[1]
            from_pt.z = node_mean_i[2]
            line_strip.points.append(from_pt)

            to_pt = Point()
            to_pt.x = node_mean_j[0]
            to_pt.y = node_mean_j[1]
            to_pt.z = node_mean_j[2]
            line_strip.points.append(to_pt)

            line_strip.color.r = 0.5
            line_strip.color.g = 0.5
            line_strip.color.b = 0.5
            line_strip.color.a = 1.0

            line_strip.lifetime.sec = 0  # zero means forever
            line_strip.lifetime.nanosec = 0
            line_strip.frame_locked = True

            line_strip.scale.x = 0.02
            pub_markers.append(line_strip)
        self.root_pub.publish(MarkerArray(markers=pub_markers))


def pipeline() -> None:
    rclpy.init()

    node = RootNodeVis()
    # executor = SingleThreadedExecutor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    pipeline()
