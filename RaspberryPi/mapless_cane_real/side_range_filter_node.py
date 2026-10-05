#!/usr/bin/env python3
"""Robust filter for left/right HC-SR04-style ranges.

Safety policy:
* unknown / no echo is valid=False, never "clear";
* a sudden closer measurement is accepted immediately;
* a sudden farther measurement must persist before it is accepted, preventing a
  single long echo from falsely announcing that the obstacle has ended;
* stale history is cleared after timeout so an old baseline cannot poison the
  next object.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float64, String


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass
class SideState:
    values: Deque[float]
    stamp: Optional[float] = None
    clear: bool = False
    pending_rise_value: Optional[float] = None
    pending_rise_count: int = 0


class SideRangeFilterNode(Node):
    def __init__(self) -> None:
        super().__init__('side_range_filter_node')

        self.declare_parameter('left_topic', '/ultrasound_left/range')
        self.declare_parameter('right_topic', '/ultrasound_right/range')
        self.declare_parameter('window_size', 5)
        self.declare_parameter('timeout_s', 0.35)
        self.declare_parameter('min_range_m', 0.05)
        self.declare_parameter('max_range_m', 2.80)
        self.declare_parameter('occupied_threshold_m', 0.62)
        self.declare_parameter('clear_threshold_m', 0.90)
        self.declare_parameter('max_rise_jump_m', 0.45)
        self.declare_parameter('rise_confirm_count', 2)
        self.declare_parameter('publish_rate_hz', 20.0)

        gp = lambda name: self.get_parameter(name).value
        self.left_topic = str(gp('left_topic'))
        self.right_topic = str(gp('right_topic'))
        self.window_size = max(1, int(gp('window_size')))
        self.timeout_s = float(gp('timeout_s'))
        self.min_range_m = float(gp('min_range_m'))
        self.max_range_m = float(gp('max_range_m'))
        self.occupied_threshold_m = float(gp('occupied_threshold_m'))
        self.clear_threshold_m = float(gp('clear_threshold_m'))
        self.max_rise_jump_m = float(gp('max_rise_jump_m'))
        self.rise_confirm_count = max(1, int(gp('rise_confirm_count')))

        self.left = SideState(values=deque(maxlen=self.window_size))
        self.right = SideState(values=deque(maxlen=self.window_size))
        self.last_debug_time = self.get_clock().now()

        self.create_subscription(Range, self.left_topic, self.left_callback, qos_profile_sensor_data)
        self.create_subscription(Range, self.right_topic, self.right_callback, qos_profile_sensor_data)

        self.left_dist_pub = self.create_publisher(Float64, '/side_left_dist', 10)
        self.right_dist_pub = self.create_publisher(Float64, '/side_right_dist', 10)
        self.left_valid_pub = self.create_publisher(Bool, '/side_left_valid', 10)
        self.right_valid_pub = self.create_publisher(Bool, '/side_right_valid', 10)
        self.left_clear_pub = self.create_publisher(Bool, '/side_left_clear', 10)
        self.right_clear_pub = self.create_publisher(Bool, '/side_right_clear', 10)
        self.debug_pub = self.create_publisher(String, '/side_range_debug', 10)

        rate = float(gp('publish_rate_hz'))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.timer_callback)
        self.get_logger().info(
            f'Side range filter v4 started | left={self.left_topic} right={self.right_topic}'
        )

    def left_callback(self, msg: Range) -> None:
        self.accept_measurement(msg, self.left)

    def right_callback(self, msg: Range) -> None:
        self.accept_measurement(msg, self.right)

    def accept_measurement(self, msg: Range, state: SideState) -> None:
        value = float(msg.range)
        lower = max(self.min_range_m, float(msg.min_range) if msg.min_range > 0.0 else 0.0)
        upper = self.max_range_m
        if msg.max_range > 0.0 and math.isfinite(msg.max_range):
            upper = min(upper, float(msg.max_range))
        if not math.isfinite(value) or value < lower or value > upper:
            return

        stamp = stamp_to_seconds(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.now_seconds()

        if not state.values:
            state.values.append(value)
            state.stamp = stamp
            return

        median = float(np.median(np.asarray(state.values, dtype=np.float32)))

        # Closer readings are safety-critical and are never rejected as a jump.
        if value <= median + self.max_rise_jump_m:
            state.values.append(value)
            state.stamp = stamp
            state.pending_rise_value = None
            state.pending_rise_count = 0
            return

        # A much larger distance can mean that the object ended, but it can also
        # be one specular/long echo. Require persistence before replacing the
        # window with the new level.
        if state.pending_rise_value is None or abs(value - state.pending_rise_value) > 0.20:
            state.pending_rise_value = value
            state.pending_rise_count = 1
            return

        state.pending_rise_count += 1
        state.pending_rise_value = 0.5 * (state.pending_rise_value + value)
        if state.pending_rise_count >= self.rise_confirm_count:
            state.values.clear()
            state.values.append(float(state.pending_rise_value))
            state.stamp = stamp
            state.pending_rise_value = None
            state.pending_rise_count = 0

    def timer_callback(self) -> None:
        now_sec = self.now_seconds()
        left_dist, left_valid = self.current_value(self.left, now_sec)
        right_dist, right_valid = self.current_value(self.right, now_sec)

        self.left.clear = self.update_clear(left_dist, left_valid, self.left.clear)
        self.right.clear = self.update_clear(right_dist, right_valid, self.right.clear)

        self.left_dist_pub.publish(Float64(data=left_dist if left_valid else 999.0))
        self.right_dist_pub.publish(Float64(data=right_dist if right_valid else 999.0))
        self.left_valid_pub.publish(Bool(data=left_valid))
        self.right_valid_pub.publish(Bool(data=right_valid))
        self.left_clear_pub.publish(Bool(data=self.left.clear))
        self.right_clear_pub.publish(Bool(data=self.right.clear))

        debug = (
            f'left={left_dist:.2f} valid={left_valid} clear={self.left.clear} | '
            f'right={right_dist:.2f} valid={right_valid} clear={self.right.clear}'
        )
        self.debug_pub.publish(String(data=debug))
        now = self.get_clock().now()
        if (now - self.last_debug_time).nanoseconds * 1e-9 > 0.75:
            self.get_logger().info(debug)
            self.last_debug_time = now

    def current_value(self, state: SideState, now_sec: float) -> tuple[float, bool]:
        if state.stamp is None or (now_sec - state.stamp) > self.timeout_s or not state.values:
            state.values.clear()
            state.pending_rise_value = None
            state.pending_rise_count = 0
            state.clear = False
            return math.inf, False
        return float(np.median(np.asarray(state.values, dtype=np.float32))), True

    def update_clear(self, distance: float, valid: bool, previous: bool) -> bool:
        if not valid:
            return False
        if distance <= self.occupied_threshold_m:
            return False
        if distance >= self.clear_threshold_m:
            return True
        return previous

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SideRangeFilterNode()
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
