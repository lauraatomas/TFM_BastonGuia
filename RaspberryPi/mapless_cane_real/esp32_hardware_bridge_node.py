#!/usr/bin/env python3
"""
Bridge ROS 2 <-> ESP32-S3 para el bastón Mapless Cane.

VERSIÓN V3 - compatible con:
    esp32_mapless_cane_full_v3_haptics.ino

ESP32 -> Raspberry:
    TEL2,seq,uptime_ms,
         left_mm,right_mm,
         left_valid,right_valid,
         button_left,button_right,
         imu_valid,
         ax,ay,az,gx,gy,gz,
         servo_us,haptic

Raspberry -> ESP32:
    CMD,seq,steer_rad,haptic

Convención de dirección:
    steer_rad > 0  -> izquierda
    steer_rad < 0  -> derecha

Haptic recibido por el ESP32:
    0 -> ninguno / rearme
    1 -> inicio giro izquierda
    2 -> inicio giro derecha
    3 -> obstáculo al lado izquierdo
    4 -> obstáculo al lado derecho
    5 -> maniobra terminada
    6 -> localización perdida
    7 -> peligro general

El sensor ultrasónico SUPERIOR no pasa por ROS: se procesa localmente
en el ESP32 y tiene prioridad háptica allí.
"""

# ============================================================================
# Importaciones
# ============================================================================
# El bridge combina tres grupos de dependencias:
#   - librerías estándar de Python para operaciones matemáticas, gestión de
#     hilos y definición de estructuras de datos;
#   - rclpy y mensajes ROS 2 para la comunicación con el resto del sistema;
#   - pyserial para el enlace físico USB/serie con el ESP32-S3.
# ============================================================================

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu, Range
from std_msgs.msg import Bool, Float64, Int8, String, UInt8

# La importación de pyserial se protege para permitir que el nodo pueda cargar
# incluso si la dependencia no está disponible. En ese caso, el propio nodo
# informa del error y evita intentar abrir el puerto serie.

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None
    SerialException = Exception


# Función auxiliar utilizada para limitar valores a un intervalo seguro.
# Se emplea, entre otros casos, para acotar órdenes de dirección y códigos
# hápticos antes de transmitirlos al microcontrolador.

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ============================================================================
# Estructura de telemetría
# ============================================================================
# Cada trama TEL2 recibida desde el ESP32 se convierte en una instancia de
# Telemetry. De este modo, los diferentes campos del protocolo se manipulan
# mediante nombres explícitos en lugar de índices numéricos.
# ============================================================================

@dataclass
class Telemetry:
    seq: int
    uptime_ms: int

    left_mm: int
    right_mm: int
    left_valid: bool
    right_valid: bool

    button_left: bool
    button_right: bool

    imu_valid: bool
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    servo_us: int
    haptic: int


# ============================================================================
# Nodo principal de comunicación hardware
# ============================================================================
# Este nodo implementa el enlace bidireccional entre ROS 2 y el ESP32-S3.
#
# Flujo ESP32 -> Raspberry:
#   - distancias de los sensores ultrasónicos laterales;
#   - estado de los pulsadores;
#   - aceleración y velocidad angular de la IMU;
#   - posición reportada del servo;
#   - estado del patrón háptico.
#
# Flujo Raspberry -> ESP32:
#   - orden final de dirección en radianes;
#   - patrón háptico final.
#
# Los datos recibidos se publican posteriormente mediante mensajes ROS 2
# estándar, de forma que puedan ser utilizados por los demás nodos del sistema.
# ============================================================================

class Esp32HardwareBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('esp32_hardware_bridge_node')

        # --------------------------------------------------------------
        # Parámetros
        # --------------------------------------------------------------
        # Parámetros asociados al enlace serie con el ESP32-S3.
        # Permiten configurar el dispositivo, velocidad de comunicación,
        # timeouts, frecuencia de reconexión, frecuencia de envío de comandos
        # y periodicidad de los mensajes de diagnóstico.
        self.declare_parameter('serial_device', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('reconnect_period_s', 1.0)
        self.declare_parameter('serial_timeout_s', 0.05)
        self.declare_parameter('hardware_timeout_s', 0.40)
        self.declare_parameter('command_rate_hz', 25.0)
        self.declare_parameter('telemetry_log_period_s', 1.0)

        # Topic que contiene la orden final de dirección generada por ROS 2.
        # El valor se expresa en radianes y se transmite posteriormente al ESP32.
        self.declare_parameter('steering_topic', '/steering_pos_cmd')

        # IMPORTANTE:
        # Se usa un topic NUEVO para no mezclarlo con el antiguo lenguaje
        # háptico que todavía pueda publicar el controlador.
        self.declare_parameter('haptic_topic', '/haptic_pattern_final')

        # Topics asociados a la interacción manual mediante los pulsadores.
        # Permiten representar una orden de giro manual, detectar si el usuario
        # está solicitando un giro y conocer el sentido de dicho giro.
        self.declare_parameter('human_steer_topic', '/hardware/human_steer_cmd')
        self.declare_parameter(
            'manual_turn_active_topic',
            '/hardware/manual_turn_active',
        )
        self.declare_parameter(
            'manual_turn_direction_topic',
            '/hardware/manual_turn_direction',
        )

        # Topics utilizados para publicar las medidas de los sensores
        # ultrasónicos laterales recibidas desde el ESP32.
        self.declare_parameter(
            'left_range_topic',
            '/ultrasound_left/range',
        )
        self.declare_parameter(
            'right_range_topic',
            '/ultrasound_right/range',
        )
        # Topic de salida de la MPU6050 con aceleración lineal y velocidad angular.
        self.declare_parameter('imu_topic', '/imu/data_raw')

        # Frames asociados a cada sensor. Deben ser coherentes con la estructura
        # TF definida en el modelo URDF/Xacro del bastón.
        self.declare_parameter(
            'left_frame_id',
            'ultrasound_left_link',
        )
        self.declare_parameter(
            'right_frame_id',
            'ultrasound_right_link',
        )
        self.declare_parameter('imu_frame_id', 'imu_link')

        # Parámetros descriptivos de los sensores ultrasónicos empleados al
        # construir mensajes sensor_msgs/Range.
        self.declare_parameter(
            'ultrasound_field_of_view_rad',
            0.35,
        )
        self.declare_parameter(
            'ultrasound_min_range_m',
            0.05,
        )
        self.declare_parameter(
            'ultrasound_max_range_m',
            2.80,
        )
        # Correcciones opcionales aplicadas a las distancias medidas por cada
        # ultrasonido. Con valor 0.0 no se introduce ninguna compensación.
        self.declare_parameter(
            'left_range_offset_m',
            0.0,
        )
        self.declare_parameter(
            'right_range_offset_m',
            0.0,
        )

        # Magnitud de la orden manual asociada a los pulsadores.
        # La inversión permite intercambiar izquierda y derecha sin modificar
        # el cableado físico.
        self.declare_parameter(
            'max_human_steer_rad',
            0.08,
        )
        self.declare_parameter(
            'invert_turn_buttons',
            False,
        )

        # Servo: Raspberry manda RADIANES, NO microsegundos.
        self.declare_parameter('max_steer_rad', 0.08)
        self.declare_parameter('invert_servo', False)

        # ------------------------------------------------------------------
        # Lectura y almacenamiento de parámetros
        # ------------------------------------------------------------------
        # A partir de este punto se copian los valores efectivos de los
        # parámetros ROS a atributos internos del nodo.
        gp = lambda name: self.get_parameter(name).value

        self.serial_device = str(gp('serial_device'))
        self.baud_rate = int(gp('baud_rate'))
        self.reconnect_period_s = float(gp('reconnect_period_s'))
        self.serial_timeout_s = float(gp('serial_timeout_s'))
        self.hardware_timeout_s = float(gp('hardware_timeout_s'))
        self.telemetry_log_period_s = float(
            gp('telemetry_log_period_s')
        )

        self.steering_topic = str(gp('steering_topic'))
        self.haptic_topic = str(gp('haptic_topic'))
        self.human_steer_topic = str(gp('human_steer_topic'))
        self.manual_turn_active_topic = str(
            gp('manual_turn_active_topic')
        )
        self.manual_turn_direction_topic = str(
            gp('manual_turn_direction_topic')
        )
        self.left_range_topic = str(gp('left_range_topic'))
        self.right_range_topic = str(gp('right_range_topic'))
        self.imu_topic = str(gp('imu_topic'))

        self.left_frame_id = str(gp('left_frame_id'))
        self.right_frame_id = str(gp('right_frame_id'))
        self.imu_frame_id = str(gp('imu_frame_id'))

        self.range_fov = float(
            gp('ultrasound_field_of_view_rad')
        )
        self.range_min = float(
            gp('ultrasound_min_range_m')
        )
        self.range_max = float(
            gp('ultrasound_max_range_m')
        )
        self.left_range_offset_m = float(
            gp('left_range_offset_m')
        )
        self.right_range_offset_m = float(
            gp('right_range_offset_m')
        )

        self.max_human_steer_rad = float(
            gp('max_human_steer_rad')
        )
        self.invert_turn_buttons = bool(
            gp('invert_turn_buttons')
        )

        self.max_steer_rad = float(gp('max_steer_rad'))
        self.invert_servo = bool(gp('invert_servo'))

        # --------------------------------------------------------------
        # Estado
        # --------------------------------------------------------------
        # ------------------------------------------------------------------
        # Estado de la comunicación serie
        # ------------------------------------------------------------------
        # El acceso al puerto se protege mediante un Lock, ya que la lectura
        # de telemetría se ejecuta en un hilo independiente del temporizador
        # encargado del envío de comandos.
        self.serial_port = None
        self.serial_lock = threading.Lock()
        self.read_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Marcas temporales utilizadas para gestionar la reconexión,
        # la supervisión de actividad del hardware y la generación de logs.
        self.last_connect_attempt = 0.0
        self.last_telemetry_stamp: Optional[float] = None
        self.last_log_stamp = 0.0

        # Últimas órdenes de dirección y háptica recibidas desde ROS 2.
        # Estos valores se retransmiten de forma periódica al ESP32.
        self.latest_steering_rad = 0.0
        self.latest_haptic = 0

        # Último estado conocido de los dos pulsadores físicos del mango.
        self.button_left = False
        self.button_right = False

        # Contadores utilizados para monitorización y diagnóstico del enlace.
        self.command_seq = 0
        self.rx_count = 0
        self.parse_errors = 0
        self.ignored_lines = 0

        # Últimos valores reportados por el ESP32 para fines de diagnóstico.
        # La Raspberry transmite la dirección en radianes; el valor servo_us
        # únicamente refleja la señal aplicada por el microcontrolador.
        self.last_servo_us = 2000
        self.last_haptic_reported = 0
        self.last_left_mm = 0
        self.last_right_mm = 0
        self.last_left_valid = False
        self.last_right_valid = False
        self.last_imu_valid = False

        # --------------------------------------------------------------
        # Suscripciones
        # --------------------------------------------------------------
        # Suscripción a la orden final de dirección del sistema.
        self.create_subscription(
            Float64,
            self.steering_topic,
            self.steering_cb,
            10,
        )
        # Suscripción al patrón háptico final generado por el coordinador.
        self.create_subscription(
            UInt8,
            self.haptic_topic,
            self.haptic_cb,
            10,
        )

        # --------------------------------------------------------------
        # Publicadores
        # --------------------------------------------------------------
        # Publicadores asociados a los datos procedentes del ESP32:
        # sensores ultrasónicos, IMU, interacción manual, estado del hardware
        # y mensajes de diagnóstico del enlace serie.
        self.left_range_pub = self.create_publisher(
            Range,
            self.left_range_topic,
            qos_profile_sensor_data,
        )
        self.right_range_pub = self.create_publisher(
            Range,
            self.right_range_topic,
            qos_profile_sensor_data,
        )
        self.imu_pub = self.create_publisher(
            Imu,
            self.imu_topic,
            qos_profile_sensor_data,
        )

        self.human_steer_pub = self.create_publisher(
            Float64,
            self.human_steer_topic,
            10,
        )
        self.turn_active_pub = self.create_publisher(
            Bool,
            self.manual_turn_active_topic,
            10,
        )
        self.turn_direction_pub = self.create_publisher(
            Int8,
            self.manual_turn_direction_topic,
            10,
        )

        self.hardware_alive_pub = self.create_publisher(
            Bool,
            '/hardware_alive',
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            '/hardware/serial_debug',
            10,
        )

        # Temporizador periódico del bridge. En cada ciclo se supervisa la
        # conexión, se publica el estado del hardware, se procesan los pulsadores
        # y se envía al ESP32 la última orden disponible.
        command_rate = float(gp('command_rate_hz'))
        self.timer = self.create_timer(
            1.0 / max(command_rate, 1.0),
            self.timer_cb,
        )

        self.get_logger().info(
            'ESP32 bridge V3 iniciado | '
            f'{self.serial_device}@{self.baud_rate} | '
            f'steering={self.steering_topic} | '
            f'haptic={self.haptic_topic}'
        )

        if serial is None:
            self.get_logger().error(
                'Falta pyserial. Instala: sudo apt install python3-serial'
            )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    # Callback asociado al topic de dirección. El valor recibido se valida y
    # se limita al rango permitido antes de almacenarlo.
    def steering_cb(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            return
        self.latest_steering_rad = clamp(
            value,
            -self.max_steer_rad,
            self.max_steer_rad,
        )

    # Callback del patrón háptico. El código se limita al rango definido por
    # el protocolo de comunicación utilizado por el firmware.
    def haptic_cb(self, msg: UInt8) -> None:
        # Solo existen códigos 0..7 en este protocolo.
        self.latest_haptic = int(
            clamp(int(msg.data), 0, 7)
        )

    # ------------------------------------------------------------------
    # Serie
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Gestión de la conexión serie
    # ------------------------------------------------------------------
    # Intenta establecer la comunicación con el ESP32 únicamente cuando no
    # existe una conexión activa y ha transcurrido el periodo de reconexión.
    def connect_if_needed(self, now: float) -> None:
        if serial is None:
            return

        if self.serial_port is not None:
            return

        if now - self.last_connect_attempt < self.reconnect_period_s:
            return

        self.last_connect_attempt = now

        try:
            port = serial.Serial(
                self.serial_device,
                self.baud_rate,
                timeout=self.serial_timeout_s,
                write_timeout=self.serial_timeout_s,
            )

            # El ESP32-S3 puede reiniciarse al abrir el puerto.
            # No bloqueamos el nodo; simplemente vaciamos lo que haya.
            try:
                port.reset_input_buffer()
                port.reset_output_buffer()
            except Exception:
                pass

            with self.serial_lock:
                self.serial_port = port

            self.stop_event.clear()
            self.read_thread = threading.Thread(
                target=self.read_loop,
                daemon=True,
            )
            self.read_thread.start()

            self.get_logger().info(
                f'ESP32 conectado en {self.serial_device}.'
            )

        except (SerialException, OSError) as exc:
            self.get_logger().warn(
                f'No se puede abrir {self.serial_device}: {exc}',
                throttle_duration_sec=3.0,
            )

    # Cierre controlado del puerto serie. Una vez liberado el recurso,
    # el temporizador podrá realizar nuevos intentos de conexión.
    def disconnect(self, reason: str) -> None:
        with self.serial_lock:
            port = self.serial_port
            self.serial_port = None

        if port is not None:
            try:
                port.close()
            except Exception:
                pass

        self.get_logger().warn(
            f'ESP32 desconectado: {reason}'
        )

    # Hilo dedicado a la recepción de telemetría.
    # Solo las líneas que comienzan por TEL2 se consideran tramas de datos;
    # los mensajes informativos del firmware se ignoran sin tratarlos como
    # errores del protocolo.
    def read_loop(self) -> None:
        while not self.stop_event.is_set():

            with self.serial_lock:
                port = self.serial_port

            if port is None:
                return

            try:
                raw = port.readline()
            except (SerialException, OSError) as exc:
                self.disconnect(str(exc))
                return

            if not raw:
                continue

            try:
                line = raw.decode(
                    'ascii',
                    errors='strict',
                ).strip()
            except UnicodeDecodeError:
                self.parse_errors += 1
                continue

            if not line:
                continue

            # Solo TEL2 es telemetría estructurada.
            # Líneas OK/INFO/BOOT del firmware se ignoran sin contarlas
            # como fallos de protocolo.
            if not line.startswith('TEL2,'):
                self.ignored_lines += 1
                continue

            try:
                telemetry = self.parse_telemetry(line)
            except ValueError as exc:
                self.parse_errors += 1
                if (
                    self.parse_errors <= 5
                    or self.parse_errors % 50 == 0
                ):
                    self.get_logger().warn(
                        f'TEL2 inválida: {exc}'
                    )
                continue

            self.handle_telemetry(telemetry)

    # Conversión de una trama TEL2 en una estructura Telemetry.
    # Se comprueba la cabecera, el número de campos y el tipo de cada dato
    # antes de aceptar la trama como válida.
    @staticmethod
    def parse_telemetry(line: str) -> Telemetry:
        fields = line.split(',')

        # TEL2 + 17 datos = 18 campos.
        if len(fields) != 18:
            raise ValueError(
                f'número de campos={len(fields)}; esperado=18 | {line!r}'
            )

        if fields[0] != 'TEL2':
            raise ValueError(
                f'cabecera inesperada {fields[0]!r}'
            )

        try:
            return Telemetry(
                seq=int(fields[1]),
                uptime_ms=int(fields[2]),

                left_mm=int(fields[3]),
                right_mm=int(fields[4]),
                left_valid=bool(int(fields[5])),
                right_valid=bool(int(fields[6])),

                button_left=bool(int(fields[7])),
                button_right=bool(int(fields[8])),

                imu_valid=bool(int(fields[9])),
                ax=float(fields[10]),
                ay=float(fields[11]),
                az=float(fields[12]),
                gx=float(fields[13]),
                gy=float(fields[14]),
                gz=float(fields[15]),

                servo_us=int(fields[16]),
                haptic=int(fields[17]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'dato no convertible: {exc} | {line!r}'
            ) from exc

    # ------------------------------------------------------------------
    # Publicación de telemetría
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Procesamiento de telemetría ESP32 -> ROS 2
    # ------------------------------------------------------------------
    # Una trama válida actualiza el watchdog del hardware, almacena el estado
    # recibido y publica las medidas de los ultrasonidos y de la IMU.
    def handle_telemetry(
        self,
        telemetry: Telemetry,
    ) -> None:

        now = self.now_seconds()
        self.last_telemetry_stamp = now
        self.rx_count += 1

        self.button_left = telemetry.button_left
        self.button_right = telemetry.button_right

        self.last_servo_us = telemetry.servo_us
        self.last_haptic_reported = telemetry.haptic

        self.last_left_mm = telemetry.left_mm
        self.last_right_mm = telemetry.right_mm
        self.last_left_valid = telemetry.left_valid
        self.last_right_valid = telemetry.right_valid
        self.last_imu_valid = telemetry.imu_valid

        stamp = self.get_clock().now().to_msg()

        self.publish_range(
            self.left_range_pub,
            self.left_frame_id,
            telemetry.left_mm,
            telemetry.left_valid,
            self.left_range_offset_m,
            stamp,
        )

        self.publish_range(
            self.right_range_pub,
            self.right_frame_id,
            telemetry.right_mm,
            telemetry.right_valid,
            self.right_range_offset_m,
            stamp,
        )

        if telemetry.imu_valid:
            self.publish_imu(
                telemetry,
                stamp,
            )

    # Construcción de un mensaje sensor_msgs/Range.
    # La distancia se convierte de milímetros a metros, se aplica el offset
    # configurado y se representa como infinito cuando la medida no es válida.
    def publish_range(
        self,
        publisher,
        frame_id: str,
        millimetres: int,
        valid: bool,
        offset_m: float,
        stamp,
    ) -> None:

        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = self.range_fov
        msg.min_range = self.range_min
        msg.max_range = self.range_max

        if valid and millimetres > 0:
            value = (
                float(millimetres) * 0.001
                + offset_m
            )
            msg.range = clamp(
                value,
                self.range_min,
                self.range_max,
            )
        else:
            msg.range = math.inf

        publisher.publish(msg)

    # Construcción del mensaje sensor_msgs/Imu.
    # El firmware proporciona aceleración lineal y velocidad angular, pero no
    # una orientación absoluta; por ello se marca la orientación como no válida
    # mediante orientation_covariance[0] = -1.
    def publish_imu(
        self,
        telemetry: Telemetry,
        stamp,
    ) -> None:

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame_id

        # El MPU6050 en este firmware no entrega orientación absoluta.
        msg.orientation_covariance[0] = -1.0

        msg.linear_acceleration.x = telemetry.ax
        msg.linear_acceleration.y = telemetry.ay
        msg.linear_acceleration.z = telemetry.az

        msg.angular_velocity.x = telemetry.gx
        msg.angular_velocity.y = telemetry.gy
        msg.angular_velocity.z = telemetry.gz

        # Covarianzas conservadoras; pueden calibrarse después.
        accel_var = 0.20 ** 2
        gyro_var = 0.03 ** 2

        msg.linear_acceleration_covariance = [
            accel_var, 0.0, 0.0,
            0.0, accel_var, 0.0,
            0.0, 0.0, accel_var,
        ]

        msg.angular_velocity_covariance = [
            gyro_var, 0.0, 0.0,
            0.0, gyro_var, 0.0,
            0.0, 0.0, gyro_var,
        ]

        self.imu_pub.publish(msg)

    # ------------------------------------------------------------------
    # Bucle periódico
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Bucle periódico principal
    # ------------------------------------------------------------------
    # Este callback mantiene activo el enlace con el ESP32, genera la señal
    # /hardware_alive, traduce los pulsadores físicos a topics ROS y transmite
    # las órdenes de dirección y háptica.
    def timer_cb(self) -> None:

        now = self.now_seconds()
        self.connect_if_needed(now)

        # El hardware se considera operativo únicamente cuando el puerto está
        # abierto y se ha recibido telemetría dentro del tiempo máximo permitido.
        alive = (
            self.serial_port is not None
            and self.last_telemetry_stamp is not None
            and (
                now - self.last_telemetry_stamp
                <= self.hardware_timeout_s
            )
        )

        self.hardware_alive_pub.publish(
            Bool(data=bool(alive))
        )

        # Conversión del estado de los pulsadores a una dirección discreta:
        #   izquierda -> +1
        #   derecha   -> -1
        #   ninguno o ambos -> 0
        # Botones físicos del mango.
        direction = 0

        if self.button_left and not self.button_right:
            direction = 1
        elif self.button_right and not self.button_left:
            direction = -1

        if self.invert_turn_buttons:
            direction *= -1

        human_steer = (
            float(direction)
            * self.max_human_steer_rad
        )

        self.human_steer_pub.publish(
            Float64(data=human_steer)
        )
        self.turn_active_pub.publish(
            Bool(data=bool(direction != 0))
        )
        self.turn_direction_pub.publish(
            Int8(data=int(direction))
        )

        # Transmisión periódica de la última orden disponible al ESP32.
        self.send_command()

        if (
            now - self.last_log_stamp
            >= self.telemetry_log_period_s
        ):
            left_text = (
                f'{self.last_left_mm}mm'
                if self.last_left_valid
                else 'INVALID'
            )
            right_text = (
                f'{self.last_right_mm}mm'
                if self.last_right_valid
                else 'INVALID'
            )

            debug = (
                f'alive={alive} '
                f'rx={self.rx_count} '
                f'parse_errors={self.parse_errors} '
                f'ignored={self.ignored_lines} | '
                f'L={left_text} R={right_text} '
                f'imu={self.last_imu_valid} | '
                f'buttons=({int(self.button_left)},'
                f'{int(self.button_right)}) '
                f'turn={direction:+d} | '
                f'steer_tx={self.current_tx_steer():+.3f}rad '
                f'servo_report={self.last_servo_us}us | '
                f'haptic_tx={self.latest_haptic} '
                f'haptic_report={self.last_haptic_reported}'
            )

            self.debug_pub.publish(
                String(data=debug)
            )
            self.get_logger().info(debug)
            self.last_log_stamp = now

    # Obtiene la orden de dirección que se enviará realmente, aplicando la
    # inversión configurada y el límite máximo de giro.
    def current_tx_steer(self) -> float:
        steering = self.latest_steering_rad

        if self.invert_servo:
            steering = -steering

        return clamp(
            steering,
            -self.max_steer_rad,
            self.max_steer_rad,
        )

    # Construcción y envío de la trama de control:
    #   CMD,<secuencia>,<steer_rad>,<haptic>
    #
    # La conversión de radianes a la señal PWM del servo se realiza en el ESP32.
    def send_command(self) -> None:

        with self.serial_lock:
            port = self.serial_port

        if port is None:
            return

        steering = self.current_tx_steer()

        self.command_seq = (
            self.command_seq + 1
        ) & 0x7FFFFFFF

        # IMPORTANTE:
        # Se mandan RADIANES directamente.
        line = (
            f'CMD,{self.command_seq},'
            f'{steering:.4f},'
            f'{self.latest_haptic}\n'
        ).encode('ascii')

        try:
            with self.serial_lock:
                if self.serial_port is not None:
                    self.serial_port.write(line)

        except (SerialException, OSError) as exc:
            self.disconnect(str(exc))

    # Conversión del reloj ROS a segundos en formato de coma flotante.
    def now_seconds(self) -> float:
        return (
            self.get_clock().now().nanoseconds
            * 1e-9
        )

    # ------------------------------------------------------------------
    # Apagado seguro del nodo
    # ------------------------------------------------------------------
    # Antes de cerrar la comunicación se intenta enviar una última orden con
    # dirección centrada y háptica desactivada, y posteriormente se libera
    # el puerto serie.
    def destroy_node(self):
        self.stop_event.set()

        # Intento final de centro + haptic=0.
        try:
            with self.serial_lock:
                if self.serial_port is not None:
                    self.command_seq = (
                        self.command_seq + 1
                    ) & 0x7FFFFFFF

                    line = (
                        f'CMD,{self.command_seq},'
                        '0.0000,0\n'
                    ).encode('ascii')

                    self.serial_port.write(line)
        except Exception:
            pass

        self.disconnect('node shutdown')
        return super().destroy_node()


# ============================================================================
# Punto de entrada del nodo
# ============================================================================
# Inicializa ROS 2, crea la instancia del bridge y mantiene el nodo activo.
# El bloque finally asegura un cierre ordenado ante Ctrl+C o shutdown de ROS.
# ============================================================================

def main(args=None) -> None:
    rclpy.init(args=args)

    node = Esp32HardwareBridgeNode()

    try:
        rclpy.spin(node)

    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
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
