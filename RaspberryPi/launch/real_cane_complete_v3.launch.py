#!/usr/bin/env python3
"""
ARRANQUE COMPLETO V3 DEL BASTÓN REAL.

Un único comando arranca:
  - el real_cane_full.launch.py ya existente (RealSense, odometría, planner,
    controlador de evasión, steering mux, bridge ESP32, etc.)
  - el nuevo coordinador háptico V3

Uso:
    ros2 launch mapless_cane_real real_cane_complete_v3.launch.py \
      serial_device:=/dev/ttyACM0

No sustituye la lógica de navegación que ya funciona: la incluye y añade
únicamente la coordinación háptica nueva.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('mapless_cane_real')

    existing_main_launch = os.path.join(
        package_share,
        'launch',
        'real_cane_full.launch.py',
    )

    serial_device = LaunchConfiguration('serial_device')

    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_device',
            default_value='/dev/ttyACM0',
            description='Puerto serie del ESP32-S3.',
        ),

        LogInfo(
            msg='[MAPLESS V3] Arrancando sistema completo + háptica V3...'
        ),

        # Mantiene intacto el launch principal que ya contiene la cadena
        # RealSense -> planner -> controller -> mux -> ESP32.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                existing_main_launch
            ),
            launch_arguments={
                'serial_device': serial_device,
            }.items(),
        ),

        # No depende de una entrada console_scripts adicional:
        # el módulo Python forma parte del paquete mapless_cane_real.
        ExecuteProcess(
            cmd=[
                'python3',
                '-m',
                'mapless_cane_real.haptic_event_coordinator_node',
                '--ros-args',
                '-p',
                'output_topic:=/haptic_pattern_final',
            ],
            output='screen',
        ),
    ])
