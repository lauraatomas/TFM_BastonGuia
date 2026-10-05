#!/usr/bin/env python3
"""Generate collision-checked steering arcs from RGB-D plus short-lived memory.

Real-hardware revision: floor-band rejection prevents depth noise around z=0
from being classified simultaneously as ground and obstacle.

This is not a global map.  Obstacle cells are retained only for a few seconds in
/local_odom so that a cone or bollard does not disappear the instant it leaves
the frontal camera field of view.  The memory is pruned continuously and is
cleared when local odometry is unreliable.

Candidate array format, repeated every eight values:
    [steer_rad, valid, minimum_clearance_m, observed_ratio,
     collision_distance_m, score, far_clearance_m, tail_clearance_m]

Tracked obstacle format on /tracked_front_obstacle:
    [valid, x_in_local_odom, y_in_local_odom, radius_m, quality]
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32MultiArray, Float64, String

from .image_utils import image_to_depth_metres


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TraversabilityArcPlannerNode(Node):
    CANDIDATE_WIDTH = 8

    def __init__(self) -> None:
        super().__init__('traversability_arc_planner_node')

        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('odom_topic', '/local_odom')
        self.declare_parameter('process_rate_hz', 8.0)
        self.declare_parameter('depth_timeout_s', 1.0)

        # Camera mount in base_footprint. Positive pitch looks downward.
        self.declare_parameter('camera_height_m', 0.58)
        self.declare_parameter('camera_x_m', 0.18)
        self.declare_parameter('camera_y_m', 0.0)
        self.declare_parameter('camera_pitch_down_rad', 0.18)
        self.declare_parameter('camera_yaw_offset_rad', 0.0)

        # Depth and local grid.
        self.declare_parameter('depth_stride', 4)
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('grid_resolution_m', 0.05)
        self.declare_parameter('grid_x_max_m', 3.8)
        self.declare_parameter('grid_y_half_m', 2.2)
        self.declare_parameter('min_obstacle_height_m', 0.08)
        self.declare_parameter('max_obstacle_height_m', 1.70)
        self.declare_parameter('ground_height_tolerance_m', 0.05)
        self.declare_parameter('observation_height_min_m', -0.18)
        self.declare_parameter('observation_height_max_m', 2.0)
        self.declare_parameter('observation_dilation_cells', 3)
        self.declare_parameter('obstacle_dilation_cells', 1)

        # Robot model and arc sampling.
        self.declare_parameter('wheelbase_m', 0.55)
        self.declare_parameter('robot_width_m', 0.48)
        self.declare_parameter('safety_margin_m', 0.16)
        self.declare_parameter('speed_margin_gain_s', 0.08)
        self.declare_parameter('max_steer_rad', 0.48)
        self.declare_parameter('num_steering_candidates', 21)
        self.declare_parameter('arc_sample_step_m', 0.06)
        self.declare_parameter('min_horizon_m', 1.10)
        self.declare_parameter('max_horizon_m', 3.0)
        self.declare_parameter('lookahead_time_s', 1.8)
        self.declare_parameter('base_trigger_distance_m', 0.82)
        self.declare_parameter('reaction_time_s', 1.25)
        self.declare_parameter('emergency_distance_m', 0.42)
        self.declare_parameter('speed_filter_window', 7)
        self.declare_parameter('max_planning_speed_m_s', 0.80)

        # Unknown-space and ground support.
        self.declare_parameter('min_observed_ratio', 0.36)
        self.declare_parameter('near_observed_distance_m', 1.0)
        self.declare_parameter('enable_ground_support_check', False)
        self.declare_parameter('min_ground_support_ratio', 0.30)

        # Short-lived obstacle memory. This is a few-second local cache, not SLAM.
        self.declare_parameter('enable_obstacle_memory', True)
        self.declare_parameter('obstacle_memory_time_s', 4.0)
        self.declare_parameter('obstacle_memory_resolution_m', 0.08)
        self.declare_parameter('obstacle_memory_max_cells', 5000)
        self.declare_parameter('memory_min_odom_confidence', 0.25)

        # Nearest blocking obstacle track.
        self.declare_parameter('track_corridor_extra_m', 0.22)
        self.declare_parameter('track_search_extra_m', 0.45)
        self.declare_parameter('track_min_component_cells', 3)
        self.declare_parameter('track_min_radius_m', 0.08)
        self.declare_parameter('track_max_radius_m', 0.55)

        # Scores.
        self.declare_parameter('clearance_score_weight', 1.0)
        self.declare_parameter('observed_score_weight', 0.55)
        self.declare_parameter('steer_penalty_weight', 0.18)
        # Side selection must prefer the genuinely wider corridor, not merely
        # the single arc whose nearest point happens to be clear.
        self.declare_parameter('side_choice_far_start_m', 0.48)
        self.declare_parameter('side_choice_tail_fraction', 0.30)
        self.declare_parameter('side_summary_top_k', 3)
        self.declare_parameter('publish_debug_grid', True)
        self.declare_parameter('debug_grid_scale', 5)

        gp = lambda name: self.get_parameter(name).value
        self.depth_topic = str(gp('depth_topic'))
        self.camera_info_topic = str(gp('camera_info_topic'))
        self.odom_topic = str(gp('odom_topic'))
        self.process_rate_hz = float(gp('process_rate_hz'))
        self.depth_timeout_s = float(gp('depth_timeout_s'))

        self.camera_height_m = float(gp('camera_height_m'))
        self.camera_x_m = float(gp('camera_x_m'))
        self.camera_y_m = float(gp('camera_y_m'))
        self.camera_pitch_down_rad = float(gp('camera_pitch_down_rad'))
        self.camera_yaw_offset_rad = float(gp('camera_yaw_offset_rad'))

        self.depth_stride = int(gp('depth_stride'))
        self.min_depth_m = float(gp('min_depth_m'))
        self.max_depth_m = float(gp('max_depth_m'))
        self.grid_resolution_m = float(gp('grid_resolution_m'))
        self.grid_x_max_m = float(gp('grid_x_max_m'))
        self.grid_y_half_m = float(gp('grid_y_half_m'))
        self.min_obstacle_height_m = float(gp('min_obstacle_height_m'))
        self.max_obstacle_height_m = float(gp('max_obstacle_height_m'))
        self.ground_height_tolerance_m = float(gp('ground_height_tolerance_m'))
        self.observation_height_min_m = float(gp('observation_height_min_m'))
        self.observation_height_max_m = float(gp('observation_height_max_m'))
        self.observation_dilation_cells = int(gp('observation_dilation_cells'))
        self.obstacle_dilation_cells = int(gp('obstacle_dilation_cells'))

        self.wheelbase_m = float(gp('wheelbase_m'))
        self.robot_width_m = float(gp('robot_width_m'))
        self.safety_margin_m = float(gp('safety_margin_m'))
        self.speed_margin_gain_s = float(gp('speed_margin_gain_s'))
        self.max_steer_rad = float(gp('max_steer_rad'))
        self.num_steering_candidates = int(gp('num_steering_candidates'))
        if self.num_steering_candidates % 2 == 0:
            self.num_steering_candidates += 1
        self.arc_sample_step_m = float(gp('arc_sample_step_m'))
        self.min_horizon_m = float(gp('min_horizon_m'))
        self.max_horizon_m = float(gp('max_horizon_m'))
        self.lookahead_time_s = float(gp('lookahead_time_s'))
        self.base_trigger_distance_m = float(gp('base_trigger_distance_m'))
        self.reaction_time_s = float(gp('reaction_time_s'))
        self.emergency_distance_m = float(gp('emergency_distance_m'))
        self.speed_filter_window = max(1, int(gp('speed_filter_window')))
        self.max_planning_speed_m_s = float(gp('max_planning_speed_m_s'))

        self.min_observed_ratio = float(gp('min_observed_ratio'))
        self.near_observed_distance_m = float(gp('near_observed_distance_m'))
        self.enable_ground_support_check = bool(gp('enable_ground_support_check'))
        self.min_ground_support_ratio = float(gp('min_ground_support_ratio'))

        self.enable_obstacle_memory = bool(gp('enable_obstacle_memory'))
        self.obstacle_memory_time_s = float(gp('obstacle_memory_time_s'))
        self.obstacle_memory_resolution_m = float(gp('obstacle_memory_resolution_m'))
        self.obstacle_memory_max_cells = int(gp('obstacle_memory_max_cells'))
        self.memory_min_odom_confidence = float(gp('memory_min_odom_confidence'))

        self.track_corridor_extra_m = float(gp('track_corridor_extra_m'))
        self.track_search_extra_m = float(gp('track_search_extra_m'))
        self.track_min_component_cells = int(gp('track_min_component_cells'))
        self.track_min_radius_m = float(gp('track_min_radius_m'))
        self.track_max_radius_m = float(gp('track_max_radius_m'))

        self.clearance_score_weight = float(gp('clearance_score_weight'))
        self.observed_score_weight = float(gp('observed_score_weight'))
        self.steer_penalty_weight = float(gp('steer_penalty_weight'))
        self.side_choice_far_start_m = float(gp('side_choice_far_start_m'))
        self.side_choice_tail_fraction = float(gp('side_choice_tail_fraction'))
        self.side_summary_top_k = max(1, int(gp('side_summary_top_k')))
        self.publish_debug_grid_enabled = bool(gp('publish_debug_grid'))
        self.debug_grid_scale = max(2, int(gp('debug_grid_scale')))

        self.camera_info: Optional[CameraInfo] = None
        self.latest_depth: Optional[Image] = None
        self.latest_depth_stamp = 0.0
        self.last_processed_stamp = -1.0

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_valid = False
        self.odom_confidence = 0.0
        self.have_odom = False
        self.speed_samples: deque[float] = deque(maxlen=self.speed_filter_window)
        self.speed_mps = 0.0

        # Local-odom voxel cell -> last seen time.
        self.obstacle_memory: dict[tuple[int, int], float] = {}
        self.last_debug_time = self.get_clock().now()

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(Bool, '/local_odom_valid', self.odom_valid_callback, 10)
        self.create_subscription(Float64, '/local_odom_confidence', self.odom_confidence_callback, 10)
        self.create_subscription(Bool, '/planner/clear_obstacle_memory', self.clear_memory_callback, 10)

        self.candidates_pub = self.create_publisher(Float32MultiArray, '/local_arc_candidates', 10)
        self.front_collision_pub = self.create_publisher(Float64, '/front_collision_distance', 10)
        self.trigger_pub = self.create_publisher(Float64, '/avoidance_trigger_distance', 10)
        self.best_steer_pub = self.create_publisher(Float64, '/arc_planner_best_steer', 10)
        self.best_valid_pub = self.create_publisher(Bool, '/arc_planner_best_valid', 10)
        self.emergency_pub = self.create_publisher(Bool, '/arc_planner_emergency', 10)
        self.track_pub = self.create_publisher(Float32MultiArray, '/tracked_front_obstacle', 10)
        self.memory_count_pub = self.create_publisher(Float64, '/obstacle_memory_cell_count', 10)
        self.side_summary_pub = self.create_publisher(Float32MultiArray, '/arc_side_summary', 10)
        self.debug_pub = self.create_publisher(String, '/arc_planner_debug', 10)
        self.debug_grid_pub = self.create_publisher(Image, '/debug/traversability_image', 2)

        self.timer = self.create_timer(1.0 / max(self.process_rate_hz, 1.0), self.process_latest_depth)
        self.get_logger().info(
            f'Traversability planner v9 started | depth={self.depth_topic} '
            f'candidates={self.num_steering_candidates} memory={self.enable_obstacle_memory}'
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def odom_callback(self, msg: Odometry) -> None:
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.have_odom = True
        speed = max(0.0, abs(float(msg.twist.twist.linear.x)))
        self.speed_samples.append(min(speed, self.max_planning_speed_m_s))
        self.speed_mps = float(np.median(np.asarray(self.speed_samples, dtype=np.float32)))

    def odom_valid_callback(self, msg: Bool) -> None:
        self.odom_valid = bool(msg.data)
        if not self.odom_valid:
            self.obstacle_memory.clear()

    def odom_confidence_callback(self, msg: Float64) -> None:
        self.odom_confidence = float(msg.data)
        if self.odom_confidence < self.memory_min_odom_confidence:
            self.obstacle_memory.clear()

    def clear_memory_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        count = len(self.obstacle_memory)
        self.obstacle_memory.clear()
        self.get_logger().info(
            f'Obstacle memory cleared on controller request ({count} cells).'
        )

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth = msg
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest_depth_stamp = stamp if stamp > 0.0 else self.now_seconds()

    def process_latest_depth(self) -> None:
        if self.camera_info is None or self.latest_depth is None:
            return
        now = self.now_seconds()
        if (now - self.latest_depth_stamp) > self.depth_timeout_s:
            self.publish_empty('DEPTH_TIMEOUT')
            return
        if self.latest_depth_stamp <= self.last_processed_stamp:
            return
        self.last_processed_stamp = self.latest_depth_stamp

        depth_m = image_to_depth_metres(self.latest_depth)
        if depth_m is None:
            self.publish_empty(f'UNSUPPORTED_DEPTH_ENCODING {self.latest_depth.encoding}')
            return
        points_base = self.depth_to_base_points(depth_m)
        if points_base is None or points_base.shape[0] < 50:
            self.publish_empty('TOO_FEW_VALID_DEPTH_POINTS')
            return

        current_obstacle, observed, ground = self.build_current_masks(points_base)

        horizon = clamp(
            self.min_horizon_m + self.speed_mps * self.lookahead_time_s,
            self.min_horizon_m,
            self.max_horizon_m,
        )
        trigger_distance = clamp(
            self.base_trigger_distance_m + self.speed_mps * self.reaction_time_s,
            self.base_trigger_distance_m,
            self.max_horizon_m,
        )

        # Track the nearest current component before memory is injected. This
        # prevents an old wall cell from becoming the locked obstacle identity.
        track = self.find_front_obstacle_track(current_obstacle, trigger_distance)
        self.publish_track(track)

        if self.enable_obstacle_memory:
            self.update_obstacle_memory(points_base, now)
            self.prune_obstacle_memory(now)
            obstacle = current_obstacle.copy()
            self.inject_obstacle_memory(obstacle)
        else:
            obstacle = current_obstacle

        clearance_grid = self.compute_clearance_grid(obstacle)
        steer_values = np.linspace(
            -self.max_steer_rad,
            self.max_steer_rad,
            self.num_steering_candidates,
            dtype=np.float64,
        )
        candidates = [
            self.evaluate_arc(float(steer), horizon, clearance_grid, observed, ground)
            for steer in steer_values
        ]
        side_summary = self.compute_side_summary(candidates)

        straight_index = int(np.argmin(np.abs(steer_values)))
        straight = candidates[straight_index]
        front_collision_distance = float(straight[4])
        valid_candidates = [candidate for candidate in candidates if candidate[1] > 0.5]
        if valid_candidates:
            best = max(valid_candidates, key=lambda candidate: candidate[5])
            best_valid = True
        else:
            # Even when none is fully valid, report the arc with the greatest
            # collision distance instead of blindly returning the straight arc.
            best = max(
                candidates,
                key=lambda candidate: (
                    candidate[4] + 0.5 * candidate[2] + 0.2 * candidate[3]
                ),
            )
            best_valid = False

        emergency = math.isfinite(front_collision_distance) and front_collision_distance < self.emergency_distance_m
        flat: list[float] = []
        for candidate in candidates:
            flat.extend(float(value) for value in candidate)

        self.candidates_pub.publish(Float32MultiArray(data=flat))
        self.front_collision_pub.publish(Float64(data=front_collision_distance))
        self.trigger_pub.publish(Float64(data=trigger_distance))
        self.best_steer_pub.publish(Float64(data=float(best[0])))
        self.best_valid_pub.publish(Bool(data=best_valid))
        self.emergency_pub.publish(Bool(data=bool(emergency)))
        self.memory_count_pub.publish(Float64(data=float(len(self.obstacle_memory))))
        self.side_summary_pub.publish(Float32MultiArray(data=[float(v) for v in side_summary]))
        if self.publish_debug_grid_enabled:
            self.publish_debug_grid(obstacle, observed, candidates, float(best[0]))

        track_text = 'none' if track is None else f'({track[0]:.2f},{track[1]:.2f},r={track[2]:.2f},q={track[3]:.2f})'
        debug = (
            f'v={self.speed_mps:.2f} horizon={horizon:.2f} trigger={trigger_distance:.2f} '
            f'front_collision={front_collision_distance:.2f} valid={len(valid_candidates)}/{len(candidates)} '
            f'best={best[0]:.2f} clear={best[2]:.2f} observed={best[3]:.2f} '
            f'memory_cells={len(self.obstacle_memory)} track={track_text} emergency={emergency} '
            f'side_scores(L={side_summary[0]:.2f},R={side_summary[1]:.2f}) '
            f'side_tail(L={side_summary[4]:.2f},R={side_summary[5]:.2f})'
        )
        self.debug_pub.publish(String(data=debug))
        now_clock = self.get_clock().now()
        if (now_clock - self.last_debug_time).nanoseconds * 1e-9 > 0.75:
            self.get_logger().info(debug)
            self.last_debug_time = now_clock

    def publish_debug_grid(
        self,
        obstacle: np.ndarray,
        observed: np.ndarray,
        candidates: list[Tuple[float, float, float, float, float, float, float, float]],
        best_steer: float,
    ) -> None:
        """Publish a top-down image of the exact local grid used by the planner."""
        nx, ny = obstacle.shape
        canvas = np.zeros((nx, ny, 3), dtype=np.uint8)
        canvas[:, :] = (35, 35, 35)              # unknown
        canvas[observed > 0] = (65, 95, 65)      # observed free / support
        canvas[obstacle > 0] = (25, 25, 230)     # obstacle (BGR)

        required_clearance = (
            0.5 * self.robot_width_m
            + self.safety_margin_m
            + self.speed_margin_gain_s * self.speed_mps
        )
        half_cells = max(1, int(round(required_clearance / self.grid_resolution_m)))
        centre_y = int(round(self.grid_y_half_m / self.grid_resolution_m))
        cv2.line(canvas, (max(0, centre_y - half_cells), 0),
                 (max(0, centre_y - half_cells), nx - 1), (90, 90, 90), 1)
        cv2.line(canvas, (min(ny - 1, centre_y + half_cells), 0),
                 (min(ny - 1, centre_y + half_cells), nx - 1), (90, 90, 90), 1)

        for candidate in candidates:
            steer, valid, _clearance, _observed, collision_distance = candidate[:5]
            distance = clamp(float(collision_distance), 0.10, self.max_horizon_m)
            curvature = math.tan(float(steer)) / max(self.wheelbase_m, 1e-3)
            samples = np.arange(0.0, distance + 0.5 * self.arc_sample_step_m,
                                self.arc_sample_step_m)
            points: list[tuple[int, int]] = []
            for sample in samples:
                if abs(curvature) < 1e-6:
                    x = float(sample)
                    y = 0.0
                else:
                    x = math.sin(curvature * float(sample)) / curvature
                    y = (1.0 - math.cos(curvature * float(sample))) / curvature
                ix = int(round(x / self.grid_resolution_m))
                iy = int(round((y + self.grid_y_half_m) / self.grid_resolution_m))
                if 0 <= ix < nx and 0 <= iy < ny:
                    points.append((iy, ix))
            if len(points) < 2:
                continue
            is_best = abs(float(steer) - best_steer) < 0.015
            color = (0, 230, 255) if is_best else ((40, 210, 40) if valid > 0.5 else (30, 30, 180))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color,
                          2 if is_best else 1, cv2.LINE_AA)

        # The array's x axis is rows; flip it so forward appears upward.
        view = cv2.flip(canvas, 0)
        view = cv2.resize(
            view,
            (ny * self.debug_grid_scale, nx * self.debug_grid_scale),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(
            view,
            f'forward up | v={self.speed_mps:.2f} m/s | best={best_steer:+.2f} rad',
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        msg = Image()
        if self.latest_depth is not None:
            msg.header = self.latest_depth.header
        msg.height = int(view.shape[0])
        msg.width = int(view.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = int(view.shape[1] * 3)
        msg.data = view.tobytes()
        self.debug_grid_pub.publish(msg)

    # ------------------------------------------------------------------
    # Point cloud and masks
    # ------------------------------------------------------------------
    def depth_to_base_points(self, depth_m: np.ndarray) -> Optional[np.ndarray]:
        info = self.camera_info
        if info is None:
            return None
        h, w = depth_m.shape
        stride = max(self.depth_stride, 1)
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        z_cam = depth_m[0:h:stride, 0:w:stride]
        valid = np.isfinite(z_cam) & (z_cam >= self.min_depth_m) & (z_cam <= self.max_depth_m)
        if np.count_nonzero(valid) < 50:
            return None
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        z = z_cam[valid].astype(np.float64)
        x_right = ((u[valid] - cx) * z / fx).astype(np.float64)
        y_down = ((v[valid] - cy) * z / fy).astype(np.float64)
        camera_points = np.vstack((x_right, y_down, z))
        base_points = self.camera_to_base_rotation() @ camera_points
        base_points[0, :] += self.camera_x_m
        base_points[1, :] += self.camera_y_m
        base_points[2, :] += self.camera_height_m
        return base_points.T

    def camera_to_base_rotation(self) -> np.ndarray:
        optical_to_level_base = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        p = self.camera_pitch_down_rad
        cp, sp = math.cos(p), math.sin(p)
        pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        y = self.camera_yaw_offset_rad
        cy, sy = math.cos(y), math.sin(y)
        yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return yaw @ pitch @ optical_to_level_base

    def grid_shape(self) -> tuple[int, int]:
        nx = int(math.ceil(self.grid_x_max_m / self.grid_resolution_m)) + 1
        ny = int(math.ceil((2.0 * self.grid_y_half_m) / self.grid_resolution_m)) + 1
        return nx, ny

    def build_current_masks(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny = self.grid_shape()
        obstacle = np.zeros((nx, ny), dtype=np.uint8)
        observed = np.zeros((nx, ny), dtype=np.uint8)
        ground = np.zeros((nx, ny), dtype=np.uint8)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        in_grid = (
            (x >= 0.0) & (x <= self.grid_x_max_m)
            & (y >= -self.grid_y_half_m) & (y <= self.grid_y_half_m)
        )
        x, y, z = x[in_grid], y[in_grid], z[in_grid]
        ix = np.clip(np.floor(x / self.grid_resolution_m).astype(np.int32), 0, nx - 1)
        iy = np.clip(
            np.floor((y + self.grid_y_half_m) / self.grid_resolution_m).astype(np.int32),
            0,
            ny - 1,
        )

        observation_valid = (z >= self.observation_height_min_m) & (z <= self.observation_height_max_m)
        # El suelo y su ruido de profundidad no deben convertirse a la vez en
        # obstáculo. La banda |z| <= ground_height_tolerance_m queda reservada
        # como suelo; un obstáculo debe sobresalir por encima de esa banda.
        ground_valid = np.abs(z) <= self.ground_height_tolerance_m
        obstacle_floor = max(
            self.min_obstacle_height_m,
            self.ground_height_tolerance_m + 0.01,
        )
        obstacle_valid = (
            (z >= obstacle_floor)
            & (z <= self.max_obstacle_height_m)
            & (~ground_valid)
        )
        observed[ix[observation_valid], iy[observation_valid]] = 1
        obstacle[ix[obstacle_valid], iy[obstacle_valid]] = 1
        ground[ix[ground_valid], iy[ground_valid]] = 1

        obs_kernel_size = max(1, 2 * self.observation_dilation_cells + 1)
        obs_kernel = np.ones((obs_kernel_size, obs_kernel_size), dtype=np.uint8)
        observed = cv2.dilate(observed, obs_kernel, iterations=1)
        ground = cv2.dilate(ground, obs_kernel, iterations=1)
        if self.obstacle_dilation_cells > 0:
            k = 2 * self.obstacle_dilation_cells + 1
            obstacle = cv2.dilate(obstacle, np.ones((k, k), dtype=np.uint8), iterations=1)
        return obstacle, observed, ground

    def compute_clearance_grid(self, obstacle: np.ndarray) -> np.ndarray:
        free_image = ((1 - obstacle) * 255).astype(np.uint8)
        distance_cells = cv2.distanceTransform(free_image, cv2.DIST_L2, 5)
        return distance_cells.astype(np.float32) * self.grid_resolution_m

    # ------------------------------------------------------------------
    # Short-lived memory and track
    # ------------------------------------------------------------------
    def memory_usable(self) -> bool:
        return (
            self.have_odom
            and self.odom_valid
            and self.odom_confidence >= self.memory_min_odom_confidence
        )

    def update_obstacle_memory(self, points_base: np.ndarray, now: float) -> None:
        if not self.memory_usable():
            self.obstacle_memory.clear()
            return
        obstacle_points = points_base[
            (points_base[:, 2] >= self.min_obstacle_height_m)
            & (points_base[:, 2] <= self.max_obstacle_height_m)
            & (points_base[:, 0] >= 0.0)
            & (points_base[:, 0] <= self.grid_x_max_m)
            & (np.abs(points_base[:, 1]) <= self.grid_y_half_m)
        ]
        if obstacle_points.size == 0:
            return
        # Subsample to keep CPU and memory bounded.
        obstacle_points = obstacle_points[::2]
        ct, st = math.cos(self.odom_yaw), math.sin(self.odom_yaw)
        x_local = self.odom_x + ct * obstacle_points[:, 0] - st * obstacle_points[:, 1]
        y_local = self.odom_y + st * obstacle_points[:, 0] + ct * obstacle_points[:, 1]
        res = self.obstacle_memory_resolution_m
        keys_x = np.floor(x_local / res).astype(np.int32)
        keys_y = np.floor(y_local / res).astype(np.int32)
        for key in zip(keys_x.tolist(), keys_y.tolist()):
            self.obstacle_memory[key] = now

        if len(self.obstacle_memory) > self.obstacle_memory_max_cells:
            oldest = sorted(self.obstacle_memory.items(), key=lambda item: item[1])
            for key, _stamp in oldest[: len(self.obstacle_memory) - self.obstacle_memory_max_cells]:
                self.obstacle_memory.pop(key, None)

    def prune_obstacle_memory(self, now: float) -> None:
        cutoff = now - self.obstacle_memory_time_s
        stale = [key for key, stamp in self.obstacle_memory.items() if stamp < cutoff]
        for key in stale:
            self.obstacle_memory.pop(key, None)

    def inject_obstacle_memory(self, obstacle: np.ndarray) -> None:
        if not self.memory_usable() or not self.obstacle_memory:
            return
        res_mem = self.obstacle_memory_resolution_m
        keys = np.asarray(list(self.obstacle_memory.keys()), dtype=np.float64)
        x_local = (keys[:, 0] + 0.5) * res_mem
        y_local = (keys[:, 1] + 0.5) * res_mem
        dx = x_local - self.odom_x
        dy = y_local - self.odom_y
        ct, st = math.cos(self.odom_yaw), math.sin(self.odom_yaw)
        x_base = ct * dx + st * dy
        y_base = -st * dx + ct * dy
        in_grid = (
            (x_base >= 0.0) & (x_base <= self.grid_x_max_m)
            & (y_base >= -self.grid_y_half_m) & (y_base <= self.grid_y_half_m)
        )
        if not np.any(in_grid):
            return
        ix = np.floor(x_base[in_grid] / self.grid_resolution_m).astype(np.int32)
        iy = np.floor((y_base[in_grid] + self.grid_y_half_m) / self.grid_resolution_m).astype(np.int32)
        ix = np.clip(ix, 0, obstacle.shape[0] - 1)
        iy = np.clip(iy, 0, obstacle.shape[1] - 1)
        obstacle[ix, iy] = 1

    def find_front_obstacle_track(
        self, obstacle: np.ndarray, trigger_distance: float
    ) -> Optional[tuple[float, float, float, float]]:
        if not self.memory_usable():
            return None
        # Merge sparse depth cells only for component extraction.
        merged = cv2.dilate(obstacle, np.ones((3, 3), dtype=np.uint8), iterations=1)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(merged, connectivity=8)
        corridor_half = 0.5 * self.robot_width_m + self.safety_margin_m + self.track_corridor_extra_m
        best = None
        best_min_x = math.inf
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.track_min_component_cells:
                continue
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            min_x = left * self.grid_resolution_m
            max_x = (left + width) * self.grid_resolution_m
            min_y = top * self.grid_resolution_m - self.grid_y_half_m
            max_y = (top + height) * self.grid_resolution_m - self.grid_y_half_m
            intersects_corridor = min_y <= corridor_half and max_y >= -corridor_half
            if not intersects_corridor:
                continue
            if min_x > trigger_distance + self.track_search_extra_m or max_x < 0.05:
                continue
            if min_x < best_min_x:
                cx_cell, cy_cell = centroids[label]
                x_base = float(cx_cell * self.grid_resolution_m)
                y_base = float(cy_cell * self.grid_resolution_m - self.grid_y_half_m)
                radius = 0.5 * max(width, height) * self.grid_resolution_m
                radius = clamp(radius, self.track_min_radius_m, self.track_max_radius_m)
                quality = clamp(area / 30.0, 0.0, 1.0)
                best = (x_base, y_base, radius, quality)
                best_min_x = min_x
        return best

    def publish_track(self, track: Optional[tuple[float, float, float, float]]) -> None:
        if track is None or not self.memory_usable():
            self.track_pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0]))
            return
        x_base, y_base, radius, quality = track
        ct, st = math.cos(self.odom_yaw), math.sin(self.odom_yaw)
        x_local = self.odom_x + ct * x_base - st * y_base
        y_local = self.odom_y + st * x_base + ct * y_base
        self.track_pub.publish(
            Float32MultiArray(data=[1.0, float(x_local), float(y_local), float(radius), float(quality)])
        )

    # ------------------------------------------------------------------
    # Arc evaluation
    # ------------------------------------------------------------------
    def evaluate_arc(
        self,
        steer: float,
        horizon: float,
        clearance_grid: np.ndarray,
        observed_mask: np.ndarray,
        ground_mask: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        sample_s = np.arange(
            self.arc_sample_step_m,
            horizon + 0.5 * self.arc_sample_step_m,
            self.arc_sample_step_m,
            dtype=np.float64,
        )
        curvature = math.tan(steer) / max(self.wheelbase_m, 1e-3)
        if abs(curvature) < 1e-5:
            x = sample_s
            y = np.zeros_like(sample_s)
        else:
            x = np.sin(curvature * sample_s) / curvature
            y = (1.0 - np.cos(curvature * sample_s)) / curvature

        ix = np.floor(x / self.grid_resolution_m).astype(np.int32)
        iy = np.floor((y + self.grid_y_half_m) / self.grid_resolution_m).astype(np.int32)
        in_bounds = (
            (ix >= 0) & (ix < clearance_grid.shape[0])
            & (iy >= 0) & (iy < clearance_grid.shape[1])
        )
        if not np.all(in_bounds):
            first_bad = int(np.argmax(~in_bounds))
            return (
                float(steer), 0.0, 0.0, 0.0,
                float(sample_s[first_bad]), -1e6, 0.0, 0.0,
            )

        clearance = clearance_grid[ix, iy]
        observed = observed_mask[ix, iy].astype(np.float32)
        ground = ground_mask[ix, iy].astype(np.float32)
        required_clearance = (
            0.5 * self.robot_width_m
            + self.safety_margin_m
            + self.speed_margin_gain_s * self.speed_mps
        )
        collision_indices = np.flatnonzero(clearance < required_clearance)
        collision_distance = float(sample_s[int(collision_indices[0])]) if collision_indices.size else float(horizon)
        min_clearance = float(np.min(clearance)) if clearance.size else 0.0
        observed_ratio = float(np.mean(observed)) if observed.size else 0.0
        ground_ratio = float(np.mean(ground)) if ground.size else 0.0
        near = sample_s <= min(self.near_observed_distance_m, horizon)
        near_observed_ratio = float(np.mean(observed[near])) if np.any(near) else observed_ratio

        far_mask = sample_s >= min(self.side_choice_far_start_m, 0.75 * horizon)
        far_clearance = (
            float(np.percentile(clearance[far_mask], 20.0))
            if np.any(far_mask) else min_clearance
        )
        tail_count = max(1, int(math.ceil(clearance.size * clamp(
            self.side_choice_tail_fraction, 0.10, 0.60
        ))))
        tail_clearance = (
            float(np.percentile(clearance[-tail_count:], 25.0))
            if clearance.size else 0.0
        )

        valid = collision_indices.size == 0 and near_observed_ratio >= self.min_observed_ratio
        if self.enable_ground_support_check:
            valid = valid and ground_ratio >= self.min_ground_support_ratio
        score = (
            0.58 * self.clearance_score_weight * min(min_clearance, 1.5)
            + 0.27 * self.clearance_score_weight * min(far_clearance, 1.5)
            + 0.15 * self.clearance_score_weight * min(tail_clearance, 1.5)
            + self.observed_score_weight * observed_ratio
            - self.steer_penalty_weight * abs(steer) / max(self.max_steer_rad, 1e-3)
        )
        if not valid:
            score -= 1000.0
        return (
            float(steer),
            1.0 if valid else 0.0,
            min_clearance,
            observed_ratio,
            collision_distance,
            float(score),
            far_clearance,
            tail_clearance,
        )

    def compute_side_summary(
        self,
        candidates: list[Tuple[float, float, float, float, float, float, float, float]],
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """Aggregate both sides so the controller can compare gap width.

        Returns:
          left_score, right_score, left_far, right_far, left_tail, right_tail,
          left_best_steer, right_best_steer, left_viable_fraction,
          right_viable_fraction.
        """
        def one_side(sign: int) -> tuple[float, float, float, float, float]:
            side = [
                c for c in candidates
                if (c[0] > 0.035 if sign > 0 else c[0] < -0.035)
            ]
            if not side:
                return -1e6, 0.0, 0.0, 0.0, 0.0
            viable = [c for c in side if c[1] > 0.5]
            pool = viable or [c for c in side if c[4] > 0.08 and c[3] >= 0.08]
            if not pool:
                return -1e6, 0.0, 0.0, float(side[0][0]), 0.0

            def quality(c):
                return (
                    1.45 * min(c[6], 1.5)
                    + 1.15 * min(c[7], 1.5)
                    + 0.85 * min(c[4], 2.0)
                    + 0.40 * c[3]
                    + 0.30 * min(c[2], 1.5)
                    - 0.12 * abs(c[0]) / max(self.max_steer_rad, 1e-3)
                    - (1.5 if c[1] <= 0.5 else 0.0)
                )

            ranked = sorted(pool, key=quality, reverse=True)
            top = ranked[: self.side_summary_top_k]
            score = float(np.mean([quality(c) for c in top]))
            far = float(np.mean([c[6] for c in top]))
            tail = float(np.mean([c[7] for c in top]))
            best_steer = float(ranked[0][0])
            viable_fraction = len(viable) / max(len(side), 1)
            score += 0.80 * viable_fraction
            return score, far, tail, best_steer, viable_fraction

        left = one_side(+1)
        right = one_side(-1)
        return (
            left[0], right[0], left[1], right[1], left[2], right[2],
            left[3], right[3], left[4], right[4],
        )

    def publish_empty(self, reason: str) -> None:
        self.candidates_pub.publish(Float32MultiArray(data=[]))
        self.front_collision_pub.publish(Float64(data=0.0))
        self.trigger_pub.publish(Float64(data=self.base_trigger_distance_m))
        self.best_steer_pub.publish(Float64(data=0.0))
        self.best_valid_pub.publish(Bool(data=False))
        self.emergency_pub.publish(Bool(data=True))
        self.track_pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0, 0.0, 0.0]))
        self.side_summary_pub.publish(Float32MultiArray(data=[-1e6, -1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.debug_pub.publish(String(data=reason))

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TraversabilityArcPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        # rclpy/Jazzy may raise this exact pybind error while a subscription is
        # being torn down after SIGINT.  Other RuntimeError instances propagate.
        if 'Unable to convert call argument' not in str(exc):
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
