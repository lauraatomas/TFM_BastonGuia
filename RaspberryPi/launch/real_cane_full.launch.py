#!/usr/bin/env python3

"""
Launch FULL del bastón real.

Este archivo actúa como wrapper del launch base `real_cane.launch.py`.

La configuración completa del hardware, RealSense, odometría, planner,
controlador, ESP32, sensores y steering se encuentra en:

    real_cane.launch.py

De esta forma evitamos duplicar parámetros y tener calibraciones diferentes
entre distintos launch.

Cadena de arranque:

    real_cane_complete_v3.launch.py
        └── real_cane_full.launch.py
                └── real_cane.launch.py

El coordinador háptico V3 se arranca desde real_cane_complete_v3.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:

    # ------------------------------------------------------------------
    # Paquete
    # ------------------------------------------------------------------

    package_share = get_package_share_directory('mapless_cane_real')

    # Launch base que contiene TODO el sistema físico.
    base_launch = os.path.join(
        package_share,
        'launch',
        'real_cane.launch.py',
    )

    # ------------------------------------------------------------------
    # Argumentos recibidos
    # ------------------------------------------------------------------

    serial_device = LaunchConfiguration('serial_device')
    start_realsense = LaunchConfiguration('start_realsense')
    start_debug = LaunchConfiguration('start_debug')
    enable_pointcloud = LaunchConfiguration('enable_pointcloud')
    rviz = LaunchConfiguration('rviz')

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    return LaunchDescription([

        # ==============================================================
        # ARGUMENTOS GENERALES
        # ==============================================================

        DeclareLaunchArgument(
            'serial_device',
            default_value='/dev/ttyACM0',
            description='Puerto serie del ESP32-S3.',
        ),

        DeclareLaunchArgument(
            'start_realsense',
            default_value='true',
            description='Arrancar Intel RealSense D435.',
        ),

        DeclareLaunchArgument(
            'start_debug',
            default_value='false',
            description='Arrancar visualizador/debug ROS.',
        ),

        DeclareLaunchArgument(
            'enable_pointcloud',
            default_value='false',
            description='Publicar pointcloud de la RealSense.',
        ),

        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Arrancar RViz.',
        ),

        # ==============================================================
        # INFORMACIÓN DE ARRANQUE
        # ==============================================================

        LogInfo(
            msg='[MAPLESS FULL] Arrancando sistema completo del bastón real...'
        ),

        LogInfo(
            msg='[MAPLESS FULL] Configuración principal: real_cane.launch.py'
        ),

        # ==============================================================
        # LAUNCH BASE
        #
        # real_cane.launch.py contiene:
        #
        #   - Intel RealSense D435
        #   - calibración y geometría de cámara
        #   - TF / robot_state_publisher
        #   - bridge ESP32-S3
        #   - MPU-6050
        #   - HC-SR04 laterales
        #   - botones
        #   - filtrado lateral
        #   - odometría RGB-D + IMU
        #   - planner de transitabilidad
        #   - controlador de evasión
        #   - steering command mux
        #   - hardware readiness
        #   - debug opcional
        #   - RViz opcional
        #
        # IMPORTANTE:
        # Los parámetros de calibración de cámara NO se vuelven a definir
        # aquí. Se utilizan los valores por defecto de real_cane.launch.py,
        # que constituye la única fuente de verdad.
        # ==============================================================

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                base_launch
            ),

            launch_arguments={

                # ESP32
                'serial_device': serial_device,

                # RealSense
                'start_realsense': start_realsense,
                'enable_pointcloud': enable_pointcloud,

                # Debug
                'run_debug_visualizer': start_debug,
                'rviz': rviz,

            }.items(),
        ),

    ])