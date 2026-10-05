#!/usr/bin/env python3
"""Supervisa si las entradas mínimas necesarias del hardware real están
disponibles, actualizadas y en condiciones de ser utilizadas por el sistema."""

# ============================================================================
# Importaciones
# ============================================================================
# El nodo utiliza mensajes de cámara, IMU, ultrasonidos y estado lógico para
# comprobar la disponibilidad de todos los elementos mínimos del sistema.
# ============================================================================

from __future__ import annotations

from typing import Dict, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, Range
from std_msgs.msg import Bool, Float64, String


# ============================================================================
# Monitor de disponibilidad del hardware
# ============================================================================
# Este nodo no controla actuadores ni procesa percepción. Su única función es
# verificar periódicamente que las fuentes esenciales siguen publicando datos
# recientes y que la odometría local es suficientemente fiable.
# ============================================================================

class HardwareReadinessNode(Node):
    def __init__(self) -> None:
        super().__init__('hardware_readiness_node')
        # ------------------------------------------------------------------
        # Parámetros de entrada
        # ------------------------------------------------------------------
        # Topics requeridos para cámara RGB-D, IMU, ultrasonidos y bridge hardware.
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('imu_topic', '/imu/data_raw')
        self.declare_parameter('left_range_topic', '/ultrasound_left/range')
        self.declare_parameter('right_range_topic', '/ultrasound_right/range')
        self.declare_parameter('require_imu', True)
        self.declare_parameter('require_hardware_bridge', True)
        self.declare_parameter('sensor_timeout_s', 1.0)
        self.declare_parameter('hardware_timeout_s', 0.6)
        self.declare_parameter('minimum_odom_confidence', 0.25)
        self.declare_parameter('publish_rate_hz', 2.0)

        # Lectura de parámetros y configuración de criterios de frescura.
        gp = lambda n: self.get_parameter(n).value
        self.require_imu = bool(gp('require_imu'))
        self.require_hardware_bridge = bool(gp('require_hardware_bridge'))
        self.sensor_timeout_s = float(gp('sensor_timeout_s'))
        self.hardware_timeout_s = float(gp('hardware_timeout_s'))
        self.minimum_odom_confidence = float(gp('minimum_odom_confidence'))
        # Cada entrada guarda únicamente la marca temporal de su último mensaje.
        # No se necesita conservar el contenido, solo saber si sigue activa.
        self.stamps: Dict[str, Optional[float]] = {
            key: None for key in ['color', 'depth', 'camera_info', 'imu', 'left', 'right', 'hardware']
        }
        self.hardware_alive = False
        self.odom_valid = False
        self.odom_confidence = 0.0
        self.last_reason = ''

        # Suscripciones a las fuentes mínimas necesarias para considerar operativo
        # el sistema real.
        self.create_subscription(Image, str(gp('color_topic')), lambda _: self.mark('color'), qos_profile_sensor_data)
        self.create_subscription(Image, str(gp('depth_topic')), lambda _: self.mark('depth'), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(gp('camera_info_topic')), lambda _: self.mark('camera_info'), qos_profile_sensor_data)
        self.create_subscription(Imu, str(gp('imu_topic')), lambda _: self.mark('imu'), qos_profile_sensor_data)
        self.create_subscription(Range, str(gp('left_range_topic')), lambda _: self.mark('left'), qos_profile_sensor_data)
        self.create_subscription(Range, str(gp('right_range_topic')), lambda _: self.mark('right'), qos_profile_sensor_data)
        self.create_subscription(Bool, '/hardware_alive', self.hardware_cb, 10)
        self.create_subscription(Bool, '/local_odom_valid', self.odom_valid_cb, 10)
        self.create_subscription(Float64, '/local_odom_confidence', self.odom_conf_cb, 10)

        # /system_ready publica el resultado global y /system_ready_reason explica
        # qué elemento falta cuando el sistema aún no está preparado.
        self.ready_pub = self.create_publisher(Bool, '/system_ready', 10)
        self.reason_pub = self.create_publisher(String, '/system_ready_reason', 10)
        self.timer = self.create_timer(1.0 / max(float(gp('publish_rate_hz')), 0.5), self.timer_cb)
        self.get_logger().info('Hardware readiness monitor started.')

    # Actualiza la marca temporal de una fuente cada vez que llega un mensaje.
    def mark(self, key: str) -> None:
        self.stamps[key] = self.now_seconds()

    # Registra el heartbeat procedente del bridge ESP32.
    def hardware_cb(self, msg: Bool) -> None:
        self.hardware_alive = bool(msg.data)
        self.mark('hardware')

    # Actualiza el bit de validez de la odometría local.
    def odom_valid_cb(self, msg: Bool) -> None:
        self.odom_valid = bool(msg.data)

    # Actualiza la confianza numérica de la odometría.
    def odom_conf_cb(self, msg: Float64) -> None:
        self.odom_confidence = float(msg.data)

    # =========================================================================
    # Evaluación periódica de disponibilidad
    # =========================================================================
    # Se comprueba:
    #   - que RGB, depth y CameraInfo sean recientes;
    #   - que ambos ultrasonidos estén activos;
    #   - que la IMU esté disponible si es obligatoria;
    #   - que el bridge ESP32 siga vivo;
    #   - que la odometría sea válida y supere la confianza mínima.
    def timer_cb(self) -> None:
        now = self.now_seconds()
        missing = []
        for key in ['color', 'depth', 'camera_info', 'left', 'right']:
            stamp = self.stamps[key]
            if stamp is None or now - stamp > self.sensor_timeout_s:
                missing.append(key)
        if self.require_imu:
            stamp = self.stamps['imu']
            if stamp is None or now - stamp > self.sensor_timeout_s:
                missing.append('imu')
        if self.require_hardware_bridge:
            stamp = self.stamps['hardware']
            if stamp is None or now - stamp > self.hardware_timeout_s or not self.hardware_alive:
                missing.append('esp32')
        if not self.odom_valid or self.odom_confidence < self.minimum_odom_confidence:
            missing.append(f'local_odom(conf={self.odom_confidence:.2f})')

        # El sistema solo se considera listo cuando no existe ninguna fuente
        # obligatoria ausente, obsoleta o inválida.
        ready = not missing
        reason = 'READY' if ready else 'WAITING_FOR_' + ','.join(missing)
        self.ready_pub.publish(Bool(data=ready))
        self.reason_pub.publish(String(data=reason))
        if reason != self.last_reason:
            if ready:
                self.get_logger().info('System ready: all required sensors are fresh.')
            else:
                self.get_logger().warn(reason)
            self.last_reason = reason

    # Conversión del reloj ROS a segundos en coma flotante.
    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


# ============================================================================
# Punto de entrada del nodo
# ============================================================================
# Inicializa ROS 2, crea el monitor y mantiene activa la supervisión periódica.
# ============================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareReadinessNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
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
