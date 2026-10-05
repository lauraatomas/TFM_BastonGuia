#!/usr/bin/env python3
"""RViz and image debugging for the mapless cane simulation (v10).

This node is deliberately isolated from the controller. It only subscribes to
existing diagnostic topics and publishes visual products, so disabling it does
not alter steering, perception or odometry behaviour.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Float64, Float64MultiArray, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .image_utils import image_to_bgr8, image_to_depth_metres


@dataclass
class Candidate:
    steer: float
    valid: bool
    clearance: float
    observed: float
    collision_distance: float
    score: float
    far_clearance: float
    tail_clearance: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)
    return q


def ros_image_from_bgr(image: np.ndarray, source: Image) -> Image:
    out = Image()
    out.header = source.header
    out.height = int(image.shape[0])
    out.width = int(image.shape[1])
    out.encoding = 'bgr8'
    out.is_bigendian = False
    out.step = int(image.shape[1] * 3)
    out.data = image.astype(np.uint8, copy=False).tobytes()
    return out


class MaplessDebugVisualizerNode(Node):
    CANDIDATE_WIDTH = 8

    def __init__(self) -> None:
        super().__init__('mapless_debug_visualizer_node')

        self.declare_parameter('fixed_frame', 'local_odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('local_odom_topic', '/local_odom')
        self.declare_parameter('ground_truth_topic', '/unused_ground_truth')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('front_distance_topic', '/front_collision_distance')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('path_max_points', 2500)
        self.declare_parameter('path_min_spacing_m', 0.025)
        self.declare_parameter('wheelbase_m', 0.275)
        self.declare_parameter('arc_draw_horizon_m', 1.65)
        self.declare_parameter('arc_sample_step_m', 0.05)
        self.declare_parameter('reference_line_back_m', 2.5)
        self.declare_parameter('reference_line_forward_m', 8.0)
        self.declare_parameter('depth_display_min_m', 0.20)
        self.declare_parameter('depth_display_max_m', 4.0)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_camera_overlay', True)
        self.declare_parameter('publish_depth_colormap', True)

        gp = lambda name: self.get_parameter(name).value
        self.fixed_frame = str(gp('fixed_frame'))
        self.base_frame = str(gp('base_frame'))
        self.local_odom_topic = str(gp('local_odom_topic'))
        self.gt_topic = str(gp('ground_truth_topic'))
        self.camera_topic = str(gp('camera_topic'))
        self.depth_topic = str(gp('depth_topic'))
        self.front_distance_topic = str(gp('front_distance_topic'))
        self.path_max_points = max(100, int(gp('path_max_points')))
        self.path_min_spacing_m = max(0.001, float(gp('path_min_spacing_m')))
        self.wheelbase_m = max(0.05, float(gp('wheelbase_m')))
        self.arc_draw_horizon_m = max(0.25, float(gp('arc_draw_horizon_m')))
        self.arc_sample_step_m = max(0.01, float(gp('arc_sample_step_m')))
        self.reference_line_back_m = max(0.0, float(gp('reference_line_back_m')))
        self.reference_line_forward_m = max(0.5, float(gp('reference_line_forward_m')))
        self.depth_display_min_m = max(0.01, float(gp('depth_display_min_m')))
        self.depth_display_max_m = max(
            self.depth_display_min_m + 0.05, float(gp('depth_display_max_m'))
        )
        self.publish_tf = bool(gp('publish_tf'))
        self.publish_camera_overlay = bool(gp('publish_camera_overlay'))
        self.publish_depth_colormap = bool(gp('publish_depth_colormap'))

        self.local_odom: Optional[Odometry] = None
        self.gt_odom: Optional[Odometry] = None
        self.gt_origin: Optional[tuple[float, float, float]] = None
        self.pending_rebase = True
        self.local_path_points: deque[PoseStamped] = deque(maxlen=self.path_max_points)
        self.gt_path_points: deque[PoseStamped] = deque(maxlen=self.path_max_points)

        self.candidates: list[Candidate] = []
        self.best_steer = 0.0
        self.assist_steer = 0.0
        self.state = 'WAITING'
        self.odom_confidence = 0.0
        self.odom_valid = False
        self.front_distance = math.inf
        self.lateral_error = 0.0
        self.heading_error = 0.0
        self.warning = False
        self.warning_reason = ''
        self.vo_translation_scale = 0.0
        self.vo_motion_curvature = 0.0
        self.vo_arc_lateral_residual = 0.0
        self.vo_distance_since_reset = 0.0
        self.vo_estimated_drift = 0.0

        self.route_line = [0.0] * 8
        self.return_target = [0.0] * 5
        self.tracked_obstacle = [0.0] * 5
        self.locked_obstacle = [0.0] * 4
        self.autonomous_turn = [0.0] * 4

        self.latest_camera: Optional[Image] = None
        self.latest_depth: Optional[Image] = None

        self.create_subscription(Odometry, self.local_odom_topic, self.local_odom_cb, 20)
        self.create_subscription(Odometry, self.gt_topic, self.gt_odom_cb, 20)
        self.create_subscription(Bool, '/local_odom_reset_event', self.reset_cb, 10)
        self.create_subscription(Float32MultiArray, '/local_arc_candidates', self.candidates_cb, 10)
        self.create_subscription(Float64, '/arc_planner_best_steer', self.best_steer_cb, 10)
        self.create_subscription(Float64, '/assist_steering_cmd', self.assist_steer_cb, 10)
        self.create_subscription(Float64MultiArray, '/debug/route_line', self.route_line_cb, 10)
        self.create_subscription(Float64MultiArray, '/debug/return_target', self.return_target_cb, 10)
        self.create_subscription(Float32MultiArray, '/tracked_front_obstacle', self.tracked_obstacle_cb, 10)
        self.create_subscription(Float64MultiArray, '/debug/locked_obstacle', self.locked_obstacle_cb, 10)
        self.create_subscription(Float64MultiArray, '/debug/autonomous_turn', self.autonomous_turn_cb, 10)
        self.create_subscription(String, '/mapless_bypass_state', self.state_cb, 10)
        self.create_subscription(Float64, '/local_odom_confidence', self.confidence_cb, 10)
        self.create_subscription(Bool, '/local_odom_valid', self.valid_cb, 10)
        self.create_subscription(Float64, self.front_distance_topic, self.front_cb, 10)
        self.create_subscription(Float64, '/lateral_error', self.lateral_cb, 10)
        self.create_subscription(Float64, '/heading_error', self.heading_cb, 10)
        self.create_subscription(Bool, '/safety_warning', self.warning_cb, 10)
        self.create_subscription(String, '/safety_warning_reason', self.warning_reason_cb, 10)
        self.create_subscription(Float64, '/debug/vo_translation_scale', self.vo_scale_cb, 10)
        self.create_subscription(Float64, '/debug/vo_motion_curvature', self.vo_curvature_cb, 10)
        self.create_subscription(Float64, '/debug/vo_arc_lateral_residual', self.vo_arc_residual_cb, 10)
        self.create_subscription(Float64, '/debug/vo_distance_since_reset', self.vo_distance_cb, 10)
        self.create_subscription(Float64, '/debug/vo_estimated_drift', self.vo_drift_cb, 10)
        self.create_subscription(Image, self.camera_topic, self.camera_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data)

        self.markers_pub = self.create_publisher(MarkerArray, '/debug/markers', 10)
        self.local_path_pub = self.create_publisher(Path, '/debug/local_path', 10)
        self.gt_path_pub = self.create_publisher(Path, '/debug/ground_truth_path', 10)
        self.camera_overlay_pub = self.create_publisher(Image, '/debug/camera_overlay', 2)
        self.depth_colormap_pub = self.create_publisher(Image, '/debug/depth_colormap', 2)

        self.tf_broadcaster = TransformBroadcaster(self)
        rate = max(1.0, float(gp('publish_rate_hz')))
        self.timer = self.create_timer(1.0 / rate, self.publish_debug)
        self.get_logger().info(
            'Mapless debug visualizer v10 started | '
            f'fixed_frame={self.fixed_frame} base_frame={self.base_frame} '
            'topics=/debug/markers,/debug/local_path,/debug/ground_truth_path,'
            '/debug/camera_overlay,/debug/depth_colormap'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def local_odom_cb(self, msg: Odometry) -> None:
        self.local_odom = msg
        self.append_path_pose(self.local_path_points, msg.pose.pose.position.x,
                              msg.pose.pose.position.y,
                              yaw_from_quaternion(msg.pose.pose.orientation), msg)

    def gt_odom_cb(self, msg: Odometry) -> None:
        self.gt_odom = msg
        gx = float(msg.pose.pose.position.x)
        gy = float(msg.pose.pose.position.y)
        gyaw = yaw_from_quaternion(msg.pose.pose.orientation)
        if self.pending_rebase or self.gt_origin is None:
            self.gt_origin = (gx, gy, gyaw)
            self.gt_path_points.clear()
            self.pending_rebase = False
        rx, ry, ryaw = self.rebase_ground_truth(gx, gy, gyaw)
        self.append_path_pose(self.gt_path_points, rx, ry, ryaw, msg)

    def reset_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.pending_rebase = True
        self.gt_origin = None
        self.local_path_points.clear()
        self.gt_path_points.clear()
        self.get_logger().info('Debug paths and ground-truth origin rebased.')

    def candidates_cb(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) % self.CANDIDATE_WIDTH != 0:
            return
        parsed: list[Candidate] = []
        for i in range(0, len(data), self.CANDIDATE_WIDTH):
            parsed.append(
                Candidate(
                    steer=float(data[i]),
                    valid=bool(data[i + 1] > 0.5),
                    clearance=float(data[i + 2]),
                    observed=float(data[i + 3]),
                    collision_distance=float(data[i + 4]),
                    score=float(data[i + 5]),
                    far_clearance=float(data[i + 6]),
                    tail_clearance=float(data[i + 7]),
                )
            )
        self.candidates = parsed

    def best_steer_cb(self, msg: Float64) -> None:
        self.best_steer = float(msg.data)

    def assist_steer_cb(self, msg: Float64) -> None:
        self.assist_steer = float(msg.data)

    def route_line_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 8:
            self.route_line = [float(v) for v in msg.data[:8]]

    def return_target_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 5:
            self.return_target = [float(v) for v in msg.data[:5]]

    def tracked_obstacle_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 5:
            self.tracked_obstacle = [float(v) for v in msg.data[:5]]

    def locked_obstacle_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self.locked_obstacle = [float(v) for v in msg.data[:4]]

    def autonomous_turn_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self.autonomous_turn = [float(v) for v in msg.data[:4]]

    def state_cb(self, msg: String) -> None:
        self.state = str(msg.data)

    def confidence_cb(self, msg: Float64) -> None:
        self.odom_confidence = float(msg.data)

    def valid_cb(self, msg: Bool) -> None:
        self.odom_valid = bool(msg.data)

    def front_cb(self, msg: Float64) -> None:
        value = float(msg.data)
        self.front_distance = value if value > 0.0 else math.inf

    def lateral_cb(self, msg: Float64) -> None:
        self.lateral_error = float(msg.data)

    def heading_cb(self, msg: Float64) -> None:
        self.heading_error = float(msg.data)

    def warning_cb(self, msg: Bool) -> None:
        self.warning = bool(msg.data)

    def warning_reason_cb(self, msg: String) -> None:
        self.warning_reason = str(msg.data)

    def vo_scale_cb(self, msg: Float64) -> None:
        self.vo_translation_scale = float(msg.data)

    def vo_curvature_cb(self, msg: Float64) -> None:
        self.vo_motion_curvature = float(msg.data)

    def vo_arc_residual_cb(self, msg: Float64) -> None:
        self.vo_arc_lateral_residual = float(msg.data)

    def vo_distance_cb(self, msg: Float64) -> None:
        self.vo_distance_since_reset = float(msg.data)

    def vo_drift_cb(self, msg: Float64) -> None:
        self.vo_estimated_drift = float(msg.data)

    def camera_cb(self, msg: Image) -> None:
        self.latest_camera = msg

    def depth_cb(self, msg: Image) -> None:
        self.latest_depth = msg

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def rebase_ground_truth(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        if self.gt_origin is None:
            return 0.0, 0.0, 0.0
        ox, oy, oyaw = self.gt_origin
        dx = x - ox
        dy = y - oy
        ct = math.cos(oyaw)
        st = math.sin(oyaw)
        return ct * dx + st * dy, -st * dx + ct * dy, wrap_angle(yaw - oyaw)

    def append_path_pose(
        self,
        path: deque[PoseStamped],
        x: float,
        y: float,
        yaw: float,
        source: Odometry,
    ) -> None:
        if path:
            last = path[-1].pose.position
            if math.hypot(x - last.x, y - last.y) < self.path_min_spacing_m:
                return
        pose = PoseStamped()
        pose.header.stamp = source.header.stamp
        pose.header.frame_id = self.fixed_frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.03
        pose.pose.orientation = quaternion_from_yaw(yaw)
        path.append(pose)

    def local_pose(self) -> Optional[tuple[float, float, float]]:
        if self.local_odom is None:
            return None
        pose = self.local_odom.pose.pose
        return float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation)

    def current_gt_pose(self) -> Optional[tuple[float, float, float]]:
        if self.gt_odom is None or self.gt_origin is None:
            return None
        pose = self.gt_odom.pose.pose
        return self.rebase_ground_truth(
            float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation)
        )

    def arc_points(self, steer: float, distance: float, pose: tuple[float, float, float]) -> list[Point]:
        x0, y0, yaw = pose
        curvature = math.tan(steer) / self.wheelbase_m
        points: list[Point] = []
        samples = np.arange(0.0, max(distance, self.arc_sample_step_m) + 0.5 * self.arc_sample_step_m,
                            self.arc_sample_step_m)
        for s in samples:
            if abs(curvature) < 1e-6:
                xb = float(s)
                yb = 0.0
            else:
                xb = math.sin(curvature * float(s)) / curvature
                yb = (1.0 - math.cos(curvature * float(s))) / curvature
            point = Point()
            point.x = x0 + math.cos(yaw) * xb - math.sin(yaw) * yb
            point.y = y0 + math.sin(yaw) * xb + math.cos(yaw) * yb
            point.z = 0.055
            points.append(point)
        return points

    # ------------------------------------------------------------------
    # Marker helpers
    # ------------------------------------------------------------------
    def marker(self, marker_id: int, marker_type: int, namespace: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.fixed_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 350_000_000
        return marker

    @staticmethod
    def set_color(marker: Marker, r: float, g: float, b: float, a: float = 1.0) -> None:
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

    def line_marker(self, marker_id: int, namespace: str, points: list[Point],
                    width: float, color: tuple[float, float, float, float]) -> Marker:
        marker = self.marker(marker_id, Marker.LINE_STRIP, namespace)
        marker.scale.x = width
        marker.points = points
        self.set_color(marker, *color)
        return marker

    def make_markers(self) -> MarkerArray:
        result = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        result.markers.append(delete_all)
        pose = self.local_pose()
        if pose is None:
            return result
        x, y, yaw = pose
        next_id = 1

        # Estimated robot pose.
        robot = self.marker(next_id, Marker.ARROW, 'robot_estimate')
        next_id += 1
        robot.pose.position.x = x
        robot.pose.position.y = y
        robot.pose.position.z = 0.08
        robot.pose.orientation = quaternion_from_yaw(yaw)
        robot.scale.x = 0.45
        robot.scale.y = 0.10
        robot.scale.z = 0.10
        self.set_color(robot, 0.10, 0.55, 1.0, 1.0)
        result.markers.append(robot)

        # Rebased Gazebo ground truth and error vector.
        gt = self.current_gt_pose()
        if gt is not None:
            gx, gy, gyaw = gt
            gt_marker = self.marker(next_id, Marker.ARROW, 'ground_truth')
            next_id += 1
            gt_marker.pose.position.x = gx
            gt_marker.pose.position.y = gy
            gt_marker.pose.position.z = 0.10
            gt_marker.pose.orientation = quaternion_from_yaw(gyaw)
            gt_marker.scale.x = 0.45
            gt_marker.scale.y = 0.10
            gt_marker.scale.z = 0.10
            self.set_color(gt_marker, 1.0, 0.25, 0.15, 0.95)
            result.markers.append(gt_marker)
            p1 = Point(x=x, y=y, z=0.08)
            p2 = Point(x=gx, y=gy, z=0.08)
            result.markers.append(
                self.line_marker(next_id, 'localization_error', [p1, p2], 0.025,
                                 (1.0, 0.15, 0.15, 0.90))
            )
            next_id += 1

        # Frozen route line, orthogonal projection and return target.
        if self.route_line[0] > 0.5:
            line_x0, line_y0, line_yaw = self.route_line[1:4]
            progress = self.route_line[5]
            ct = math.cos(line_yaw)
            st = math.sin(line_yaw)
            s0 = progress - self.reference_line_back_m
            s1 = progress + self.reference_line_forward_m
            p0 = Point(x=line_x0 + s0 * ct, y=line_y0 + s0 * st, z=0.045)
            p1 = Point(x=line_x0 + s1 * ct, y=line_y0 + s1 * st, z=0.045)
            result.markers.append(
                self.line_marker(next_id, 'reference_line', [p0, p1], 0.045,
                                 (0.05, 1.0, 0.25, 0.95))
            )
            next_id += 1
            projection = Point(
                x=line_x0 + progress * ct,
                y=line_y0 + progress * st,
                z=0.06,
            )
            robot_point = Point(x=x, y=y, z=0.06)
            result.markers.append(
                self.line_marker(next_id, 'cross_track_error', [robot_point, projection],
                                 0.035, (1.0, 0.80, 0.10, 0.95))
            )
            next_id += 1

        if self.return_target[0] > 0.5:
            tx, ty, target_yaw = self.return_target[1:4]
            target = self.marker(next_id, Marker.SPHERE, 'return_target')
            next_id += 1
            target.pose.position.x = tx
            target.pose.position.y = ty
            target.pose.position.z = 0.11
            target.scale.x = 0.20
            target.scale.y = 0.20
            target.scale.z = 0.20
            self.set_color(target, 0.10, 1.0, 1.0, 0.95)
            result.markers.append(target)
            target_arrow = self.marker(next_id, Marker.ARROW, 'return_target_heading')
            next_id += 1
            target_arrow.pose.position.x = tx
            target_arrow.pose.position.y = ty
            target_arrow.pose.position.z = 0.12
            target_arrow.pose.orientation = quaternion_from_yaw(target_yaw)
            target_arrow.scale.x = 0.45
            target_arrow.scale.y = 0.07
            target_arrow.scale.z = 0.07
            self.set_color(target_arrow, 0.10, 1.0, 1.0, 0.95)
            result.markers.append(target_arrow)

        # Candidate arcs. Valid = green, invalid = red. Planner best = yellow.
        for index, candidate in enumerate(self.candidates):
            draw_distance = clamp(
                candidate.collision_distance,
                0.10,
                self.arc_draw_horizon_m,
            )
            points = self.arc_points(candidate.steer, draw_distance, pose)
            is_best = abs(candidate.steer - self.best_steer) < 0.015
            if is_best:
                color = (1.0, 0.85, 0.05, 0.95)
                width = 0.035
            elif candidate.valid:
                color = (0.15, 0.85, 0.25, 0.55)
                width = 0.018
            else:
                color = (1.0, 0.15, 0.10, 0.42)
                width = 0.014
            result.markers.append(
                self.line_marker(next_id + index, 'candidate_arcs', points, width, color)
            )
        next_id += len(self.candidates)

        # Controller's actual assist command, shown independently from planner best.
        assist_points = self.arc_points(self.assist_steer, self.arc_draw_horizon_m, pose)
        result.markers.append(
            self.line_marker(next_id, 'selected_assist_arc', assist_points, 0.050,
                             (0.0, 0.75, 1.0, 1.0))
        )
        next_id += 1

        # Current and locked obstacle estimates.
        for namespace, data, color in (
            ('tracked_obstacle', self.tracked_obstacle, (0.72, 0.15, 1.0, 0.70)),
            ('locked_obstacle', self.locked_obstacle, (1.0, 0.45, 0.05, 0.90)),
        ):
            if data[0] <= 0.5:
                continue
            obstacle = self.marker(next_id, Marker.CYLINDER, namespace)
            next_id += 1
            obstacle.pose.position.x = data[1]
            obstacle.pose.position.y = data[2]
            obstacle.pose.position.z = 0.16
            radius = max(0.08, data[3])
            obstacle.scale.x = 2.0 * radius
            obstacle.scale.y = 2.0 * radius
            obstacle.scale.z = 0.32
            self.set_color(obstacle, *color)
            result.markers.append(obstacle)

        # Autonomous turn anchor / guard.
        if self.autonomous_turn[0] > 0.5:
            anchor_yaw = self.autonomous_turn[1]
            guard_active = self.autonomous_turn[3] > 0.5
            anchor = self.marker(next_id, Marker.ARROW, 'autonomous_heading_anchor')
            next_id += 1
            anchor.pose.position.x = x
            anchor.pose.position.y = y
            anchor.pose.position.z = 0.20
            anchor.pose.orientation = quaternion_from_yaw(anchor_yaw)
            anchor.scale.x = 0.62
            anchor.scale.y = 0.055
            anchor.scale.z = 0.055
            self.set_color(anchor, 1.0, 0.05 if guard_active else 0.70, 0.05, 0.95)
            result.markers.append(anchor)

        # Compact status text above the robot.
        text = self.marker(next_id, Marker.TEXT_VIEW_FACING, 'status')
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = 1.15
        text.scale.z = 0.20
        front_text = 'inf' if not math.isfinite(self.front_distance) else f'{self.front_distance:.2f}m'
        guard_text = ' GUARD' if self.autonomous_turn[3] > 0.5 else ''
        warning_text = f'\nWARN: {self.warning_reason}' if self.warning else ''
        text.text = (
            f'{self.state}{guard_text}\n'
            f'VO={self.odom_confidence:.2f} valid={self.odom_valid} '
            f'front={front_text}\n'
            f'e_y={self.lateral_error:+.2f}m e_th={self.heading_error:+.2f}rad '
            f'steer={self.assist_steer:+.2f}\n'
            f'scale={self.vo_translation_scale:.2f} curv={self.vo_motion_curvature:.2f} '
            f'arc_lat={self.vo_arc_lateral_residual:+.3f}\n'
            f'dist={self.vo_distance_since_reset:.1f}m '
            f'drift_est={self.vo_estimated_drift:.2f}m{warning_text}'
        )
        self.set_color(text, 1.0, 0.25, 0.15, 1.0) if self.warning else self.set_color(
            text, 1.0, 1.0, 1.0, 1.0
        )
        result.markers.append(text)
        return result

    # ------------------------------------------------------------------
    # Image products
    # ------------------------------------------------------------------
    def publish_camera_image(self) -> None:
        if not self.publish_camera_overlay or self.latest_camera is None:
            return
        bgr = image_to_bgr8(self.latest_camera)
        if bgr is None:
            return
        canvas = bgr.copy()
        h, w = canvas.shape[:2]
        cv2.line(canvas, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.rectangle(canvas, (4, 4), (min(w - 4, 720), 151), (0, 0, 0), -1)
        front_text = 'inf' if not math.isfinite(self.front_distance) else f'{self.front_distance:.2f} m'
        lines = [
            f'STATE: {self.state}',
            f'front={front_text} planner={self.best_steer:+.2f} assist={self.assist_steer:+.2f}',
            f'VO valid={self.odom_valid} conf={self.odom_confidence:.2f}  e_y={self.lateral_error:+.2f}  e_th={self.heading_error:+.2f}',
            f'scale={self.vo_translation_scale:.2f} curv={self.vo_motion_curvature:.2f} arc_lat={self.vo_arc_lateral_residual:+.3f}',
            f'dist={self.vo_distance_since_reset:.1f}m drift_est={self.vo_estimated_drift:.2f}m',
        ]
        if self.warning:
            lines.append(f'WARNING: {self.warning_reason}')
        for index, line in enumerate(lines):
            color = (0, 80, 255) if self.warning and index == len(lines) - 1 else (255, 255, 255)
            cv2.putText(canvas, line, (12, 27 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color, 2, cv2.LINE_AA)
        self.camera_overlay_pub.publish(ros_image_from_bgr(canvas, self.latest_camera))

    def publish_depth_image(self) -> None:
        if not self.publish_depth_colormap or self.latest_depth is None:
            return
        depth = image_to_depth_metres(self.latest_depth)
        if depth is None:
            return
        valid = np.isfinite(depth) & (depth >= self.depth_display_min_m)
        clipped = np.clip(depth, self.depth_display_min_m, self.depth_display_max_m)
        normalized = 255.0 * (1.0 - (
            clipped - self.depth_display_min_m
        ) / (self.depth_display_max_m - self.depth_display_min_m))
        gray = normalized.astype(np.uint8)
        gray[~valid] = 0
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color[~valid] = (0, 0, 0)
        h, w = color.shape[:2]
        cv2.line(color, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.putText(
            color,
            f'{self.depth_display_min_m:.1f}-{self.depth_display_max_m:.1f} m | front={self.front_distance:.2f} m',
            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
        )
        self.depth_colormap_pub.publish(ros_image_from_bgr(color, self.latest_depth))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish_paths(self) -> None:
        stamp = self.get_clock().now().to_msg()
        local = Path()
        local.header.frame_id = self.fixed_frame
        local.header.stamp = stamp
        local.poses = list(self.local_path_points)
        self.local_path_pub.publish(local)

        gt = Path()
        gt.header.frame_id = self.fixed_frame
        gt.header.stamp = stamp
        gt.poses = list(self.gt_path_points)
        self.gt_path_pub.publish(gt)

    def publish_transform(self) -> None:
        if not self.publish_tf or self.local_odom is None:
            return
        pose = self.local_odom.pose.pose
        transform = TransformStamped()
        transform.header.stamp = self.local_odom.header.stamp
        transform.header.frame_id = self.fixed_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = float(pose.position.x)
        transform.transform.translation.y = float(pose.position.y)
        transform.transform.translation.z = 0.0
        transform.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def publish_debug(self) -> None:
        self.publish_transform()
        self.publish_paths()
        self.markers_pub.publish(self.make_markers())
        self.publish_camera_image()
        self.publish_depth_image()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaplessDebugVisualizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
