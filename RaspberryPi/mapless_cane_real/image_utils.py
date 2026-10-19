#!/usr/bin/env python3
"""Safe conversion of sensor_msgs/Image without cv_bridge.

ROS Jazzy's cv_bridge binary may be built against NumPy 1.x while a user-level
NumPy 2.x shadows the system package.  Converting directly from the message
buffer avoids that binary ABI dependency and is also cheaper for depth images.
"""

from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np
from sensor_msgs.msg import Image


def _native_array(msg: Image, dtype: np.dtype, channels: int) -> Optional[np.ndarray]:
    try:
        itemsize = np.dtype(dtype).itemsize
        if msg.step <= 0 or msg.height <= 0 or msg.width <= 0:
            return None
        row_items = int(msg.step // itemsize)
        expected_items = int(msg.height * row_items)
        data = np.frombuffer(msg.data, dtype=dtype, count=expected_items)
        if data.size != expected_items:
            return None
        if bool(msg.is_bigendian) != (sys.byteorder == 'big') and itemsize > 1:
            data = data.byteswap().view(data.dtype.newbyteorder('='))
        rows = data.reshape((msg.height, row_items))
        useful = msg.width * channels
        if useful > row_items:
            return None
        rows = rows[:, :useful]
        if channels == 1:
            return rows.reshape((msg.height, msg.width))
        return rows.reshape((msg.height, msg.width, channels))
    except (ValueError, TypeError, BufferError):
        return None


def image_to_depth_metres(msg: Image) -> Optional[np.ndarray]:
    encoding = str(msg.encoding).lower()
    if encoding in ('16uc1', 'mono16'):
        arr = _native_array(msg, np.uint16, 1)
        return None if arr is None else arr.astype(np.float32) * 0.001
    if encoding in ('32fc1', '32fc'):
        arr = _native_array(msg, np.float32, 1)
        return None if arr is None else arr.astype(np.float32, copy=False)

    bytes_per_pixel = int(msg.step // max(msg.width, 1))
    if bytes_per_pixel == 2:
        arr = _native_array(msg, np.uint16, 1)
        return None if arr is None else arr.astype(np.float32) * 0.001
    if bytes_per_pixel == 4:
        arr = _native_array(msg, np.float32, 1)
        return None if arr is None else arr.astype(np.float32, copy=False)
    return None


def image_to_bgr8(msg: Image) -> Optional[np.ndarray]:
    encoding = str(msg.encoding).lower()
    if encoding == 'bgr8':
        return _native_array(msg, np.uint8, 3)
    if encoding == 'rgb8':
        arr = _native_array(msg, np.uint8, 3)
        return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if encoding == 'bgra8':
        arr = _native_array(msg, np.uint8, 4)
        return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if encoding == 'rgba8':
        arr = _native_array(msg, np.uint8, 4)
        return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if encoding in ('mono8', '8uc1'):
        arr = _native_array(msg, np.uint8, 1)
        return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return None
