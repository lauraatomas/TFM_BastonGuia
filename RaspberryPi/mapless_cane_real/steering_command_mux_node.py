#!/usr/bin/env python3
"""
Arbitraje de DIRECCIÓN para el bastón real.

El bastón es pasivo longitudinalmente:
- la persona empuja para avanzar;
- la persona deja de empujar/retiene para detenerse;
- este nodo NO genera velocidad, fuerza, frenado ni una orden de avance.

Prioridades de dirección:
1. MANUAL_TURN:
   la intención humana izquierda/derecha manda directamente sobre el servo.
2. MANUAL_SETTLE:
   al soltar el botón/flecha, la rueda vuelve progresivamente hacia el centro.
3. AUTONOMOUS_ASSIST:
   durante evasión/retorno pasa la dirección calculada por el controlador.
4. HEADING_HOLD:
   fuera de evasión se conserva el nuevo rumbo elegido por la persona.
5. USER_GUIDED / ODOM_DEGRADED:
   sin una referencia fiable se centra la rueda.

La detección de si el bastón se está desplazando se obtiene exclusivamente de
/local_odom. Ya no existe /hardware/user_push_hint en este nodo.
"""

# ============================================================================
# Importaciones
# ============================================================================
# Este nodo combina lógica de arbitraje de dirección, estimación de movimiento
# a partir de odometría y publicación de la orden final de servo.
# ============================================================================

from __future__ import annotations

import math
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String


# Limita un valor al intervalo indicado.

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Normaliza un ángulo al intervalo [-pi, pi].

def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# Extrae el ángulo de guiñada (yaw) de un cuaternión de orientación ROS.

def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# Desplaza progresivamente un valor hacia un objetivo sin superar el incremento
# máximo permitido. Se utiliza para suavizar la orden del servo.

def move_towards(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


# ============================================================================
# Multiplexor de dirección
# ============================================================================
# Este nodo decide qué fuente tiene autoridad sobre la rueda delantera en cada
# instante. No genera propulsión ni frenado: únicamente determina el ángulo final
# de dirección que se enviará al ESP32.
# ============================================================================

class SteeringCommandMuxNode(Node):
    USER_GUIDED = 'USER_GUIDED'
    MANUAL_TURN = 'MANUAL_TURN'
    MANUAL_SETTLE = 'MANUAL_SETTLE'
    HEADING_HOLD = 'HEADING_HOLD'
    AUTONOMOUS_ASSIST = 'AUTONOMOUS_ASSIST'
    ODOM_DEGRADED = 'ODOM_DEGRADED'

    def __init__(self) -> None:
        super().__init__('steering_command_mux_node')

        # ------------------------------------------------------------------
        # Interfaces ROS
        # ------------------------------------------------------------------
        # Todos los topics de este nodo están relacionados con dirección,
        # asistencia o localización. El movimiento longitudinal sigue dependiendo
        # exclusivamente del usuario.
        # Topics: todos son de DIRECCIÓN/localización.
        self.declare_parameter(
            'human_steer_topic',
            '/hardware/human_steer_cmd',
        )
        self.declare_parameter(
            'manual_turn_active_topic',
            '/hardware/manual_turn_active',
        )
        self.declare_parameter(
            'assist_steer_topic',
            '/assist_steering_cmd',
        )
        self.declare_parameter(
            'assist_active_topic',
            '/assist_active',
        )
        self.declare_parameter(
            'stop_requested_topic',
            '/stop_requested',
        )
        self.declare_parameter('odom_topic', '/local_odom')
        self.declare_parameter(
            'odom_valid_topic',
            '/local_odom_valid',
        )
        self.declare_parameter(
            'odom_confidence_topic',
            '/local_odom_confidence',
        )
        self.declare_parameter(
            'output_topic',
            '/steering_pos_cmd',
        )

        # ------------------------------------------------------------------
        # Temporización y límites de dirección
        # ------------------------------------------------------------------
        # Define frecuencia de control, timeouts de entrada, límite angular,
        # inversión de signo y offset mecánico del servo.
        # Temporización y límites.
        self.declare_parameter('control_rate_hz', 40.0)
        self.declare_parameter('input_timeout_s', 0.60)
        self.declare_parameter('odom_timeout_s', 0.80)
        self.declare_parameter('max_steer_rad', 0.08)
        self.declare_parameter('steer_sign', 1.0)
        self.declare_parameter('steer_offset_rad', 0.0)

        # ------------------------------------------------------------------
        # Entrada manual del usuario
        # ------------------------------------------------------------------
        # Determina cuándo una orden humana se considera suficientemente grande
        # como para tomar prioridad sobre cualquier asistencia automática.
        # Entrada humana.
        self.declare_parameter(
            'manual_enter_deadband_rad',
            0.08,
        )
        self.declare_parameter(
            'manual_slew_rate_rad_s',
            1.80,
        )

        # ------------------------------------------------------------------
        # Recentrado tras un giro manual
        # ------------------------------------------------------------------
        # Al soltar el botón o flecha se reduce progresivamente la dirección y se
        # usa la velocidad angular de la odometría para amortiguar el giro residual.
        # Centrado posterior al giro humano.
        self.declare_parameter(
            'settle_slew_rate_rad_s',
            0.90,
        )
        self.declare_parameter(
            'settle_yaw_rate_gain',
            0.28,
        )
        self.declare_parameter(
            'settle_max_countersteer_rad',
            0.10,
        )
        self.declare_parameter(
            'settle_center_tolerance_rad',
            0.035,
        )
        self.declare_parameter(
            'settle_yaw_rate_tolerance_rad_s',
            0.10,
        )
        self.declare_parameter(
            'settle_stable_time_s',
            0.45,
        )
        self.declare_parameter(
            'settle_min_time_s',
            0.25,
        )
        self.declare_parameter(
            'settle_timeout_s',
            2.50,
        )

        # ------------------------------------------------------------------
        # Mantenimiento de rumbo
        # ------------------------------------------------------------------
        # Una vez finalizado el giro manual, el nuevo yaw se memoriza como referencia
        # y se corrigen pequeñas desviaciones mientras el bastón sigue avanzando.
        # Mantenimiento del rumbo elegido por la persona.
        self.declare_parameter('heading_kp', 0.55)
        self.declare_parameter('heading_kd', 0.18)
        self.declare_parameter(
            'heading_max_steer_rad',
            0.12,
        )
        self.declare_parameter(
            'heading_slew_rate_rad_s',
            0.65,
        )
        self.declare_parameter(
            'heading_error_deadband_rad',
            0.018,
        )
        self.declare_parameter(
            'heading_yaw_rate_deadband_rad_s',
            0.025,
        )
        self.declare_parameter(
            'heading_breakout_error_rad',
            0.50,
        )
        self.declare_parameter(
            'heading_capture_yaw_rate_rad_s',
            0.08,
        )
        self.declare_parameter(
            'heading_capture_stable_time_s',
            0.50,
        )

        # ------------------------------------------------------------------
        # Detección de movimiento
        # ------------------------------------------------------------------
        # El estado MOVING/STOPPED se deduce exclusivamente de /local_odom.
        # Se utiliza histéresis para evitar cambios de estado por ruido cerca de cero.
        # Movimiento físico inferido de /local_odom.
        # Histeresis para que no cambie MOVING/STOPPED por ruido cerca de 0.
        self.declare_parameter(
            'moving_enter_speed_m_s',
            0.050,
        )
        self.declare_parameter(
            'moving_exit_speed_m_s',
            0.025,
        )

        # ------------------------------------------------------------------
        # Criterios de validez de la orientación
        # ------------------------------------------------------------------
        # Permiten exigir una odometría reciente, válida y con confianza mínima
        # antes de utilizar el yaw para mantener rumbo.
        # Calidad de orientación.
        self.declare_parameter(
            'require_odom_valid_for_heading',
            False,
        )
        self.declare_parameter(
            'min_odom_confidence_for_heading',
            0.0,
        )
        self.declare_parameter(
            'heading_reference_max_stale_s',
            3.0,
        )

        # Lectura y almacenamiento de parámetros ROS.
        gp = lambda name: self.get_parameter(name).value

        self.human_steer_topic = str(gp('human_steer_topic'))
        self.manual_turn_active_topic = str(
            gp('manual_turn_active_topic')
        )
        self.assist_steer_topic = str(gp('assist_steer_topic'))
        self.assist_active_topic = str(gp('assist_active_topic'))
        self.stop_requested_topic = str(
            gp('stop_requested_topic')
        )
        self.odom_topic = str(gp('odom_topic'))
        self.odom_valid_topic = str(gp('odom_valid_topic'))
        self.odom_confidence_topic = str(
            gp('odom_confidence_topic')
        )
        self.output_topic = str(gp('output_topic'))

        self.control_rate_hz = float(gp('control_rate_hz'))
        self.input_timeout_s = float(gp('input_timeout_s'))
        self.odom_timeout_s = float(gp('odom_timeout_s'))
        self.max_steer_rad = abs(float(gp('max_steer_rad')))
        self.steer_sign = float(gp('steer_sign'))
        self.steer_offset_rad = float(gp('steer_offset_rad'))

        self.manual_enter_deadband_rad = float(
            gp('manual_enter_deadband_rad')
        )
        self.manual_slew_rate_rad_s = float(
            gp('manual_slew_rate_rad_s')
        )

        self.settle_slew_rate_rad_s = float(
            gp('settle_slew_rate_rad_s')
        )
        self.settle_yaw_rate_gain = float(
            gp('settle_yaw_rate_gain')
        )
        self.settle_max_countersteer_rad = float(
            gp('settle_max_countersteer_rad')
        )
        self.settle_center_tolerance_rad = float(
            gp('settle_center_tolerance_rad')
        )
        self.settle_yaw_rate_tolerance_rad_s = float(
            gp('settle_yaw_rate_tolerance_rad_s')
        )
        self.settle_stable_time_s = float(
            gp('settle_stable_time_s')
        )
        self.settle_min_time_s = float(
            gp('settle_min_time_s')
        )
        self.settle_timeout_s = float(
            gp('settle_timeout_s')
        )

        self.heading_kp = float(gp('heading_kp'))
        self.heading_kd = float(gp('heading_kd'))
        self.heading_max_steer_rad = float(
            gp('heading_max_steer_rad')
        )
        self.heading_slew_rate_rad_s = float(
            gp('heading_slew_rate_rad_s')
        )
        self.heading_error_deadband_rad = float(
            gp('heading_error_deadband_rad')
        )
        self.heading_yaw_rate_deadband_rad_s = float(
            gp('heading_yaw_rate_deadband_rad_s')
        )
        self.heading_breakout_error_rad = float(
            gp('heading_breakout_error_rad')
        )
        self.heading_capture_yaw_rate_rad_s = float(
            gp('heading_capture_yaw_rate_rad_s')
        )
        self.heading_capture_stable_time_s = float(
            gp('heading_capture_stable_time_s')
        )

        self.moving_enter_speed_m_s = max(
            0.0,
            float(gp('moving_enter_speed_m_s')),
        )
        self.moving_exit_speed_m_s = max(
            0.0,
            float(gp('moving_exit_speed_m_s')),
        )
        if self.moving_exit_speed_m_s > self.moving_enter_speed_m_s:
            self.moving_exit_speed_m_s = self.moving_enter_speed_m_s

        self.require_odom_valid_for_heading = bool(
            gp('require_odom_valid_for_heading')
        )
        self.min_odom_confidence_for_heading = float(
            gp('min_odom_confidence_for_heading')
        )
        self.heading_reference_max_stale_s = float(
            gp('heading_reference_max_stale_s')
        )

        # Estado de las entradas manuales y automáticas de dirección.
        # Entradas de dirección.
        self.human_steer = 0.0
        self.manual_turn_active = False
        self.assist_steer = 0.0
        self.assist_active = False

        # /stop_requested se conserva únicamente como información diagnóstica.
        # Este nodo nunca convierte ese aviso en una orden de frenado.
        # /stop_requested se conserva SOLO como aviso/diagnóstico.
        self.stop_requested = False

        self.human_stamp: Optional[float] = None
        self.assist_stamp: Optional[float] = None
        self.manual_stamp: Optional[float] = None

        # Estado de odometría utilizado para yaw, velocidad y detección de movimiento.
        # Odometría.
        self.have_odom = False
        self.odom_valid = False
        self.odom_confidence = 0.0
        self.odom_stamp: Optional[float] = None
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.speed = 0.0
        self.moving = False

        # Estado interno de la máquina de arbitraje y referencia de rumbo.
        # Estado interno.
        self.state = self.USER_GUIDED
        self.state_started_at = self.now_seconds()
        self.output_cmd = 0.0

        self.heading_reference_valid = False
        self.heading_reference = 0.0
        self.heading_reference_stamp: Optional[float] = None
        self.initial_capture_since: Optional[float] = None
        self.settle_stable_since: Optional[float] = None

        self.last_control_time = self.get_clock().now()
        self.last_debug_time = self.get_clock().now()

        # ------------------------------------------------------------------
        # Suscripciones ROS 2
        # ------------------------------------------------------------------
        # Reciben intención humana, asistencia automática, odometría y señales
        # auxiliares de estado.
        # Suscripciones.
        self.create_subscription(
            Float64,
            self.human_steer_topic,
            self.human_steer_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.manual_turn_active_topic,
            self.manual_turn_active_callback,
            10,
        )
        self.create_subscription(
            Float64,
            self.assist_steer_topic,
            self.assist_steer_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.assist_active_topic,
            self.assist_active_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.stop_requested_topic,
            self.stop_requested_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )
        self.create_subscription(
            Bool,
            self.odom_valid_topic,
            self.odom_valid_callback,
            10,
        )
        self.create_subscription(
            Float64,
            self.odom_confidence_topic,
            self.odom_confidence_callback,
            10,
        )

        # ------------------------------------------------------------------
        # Publicadores ROS 2
        # ------------------------------------------------------------------
        # El resultado principal es /steering_pos_cmd. También se publican estado,
        # prioridad humana, heading hold y variables de diagnóstico.
        # Publicaciones.
        self.output_pub = self.create_publisher(
            Float64,
            self.output_topic,
            10,
        )
        self.state_pub = self.create_publisher(
            String,
            '/steering_mux_state',
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            '/steering_mux_debug',
            10,
        )
        self.human_priority_pub = self.create_publisher(
            Bool,
            '/human_priority_active',
            10,
        )
        self.heading_hold_pub = self.create_publisher(
            Bool,
            '/heading_hold_active',
            10,
        )
        self.heading_reference_pub = self.create_publisher(
            Float64,
            '/heading_reference',
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(self.control_rate_hz, 1.0),
            self.control_callback,
        )

        self.get_logger().info(
            'SteeringCommandMuxNode DIRECCION-ONLY iniciado | '
            f'human={self.human_steer_topic} '
            f'assist={self.assist_steer_topic} '
            f'odom={self.odom_topic} out={self.output_topic} | '
            'avance/parada = físicos, inferidos solo desde odometría'
        )

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # =========================================================================
    # Callbacks de entrada
    # =========================================================================
    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    # Actualiza la petición manual de dirección y limita el valor al rango máximo.
    def human_steer_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            return

        self.human_steer = clamp(
            value,
            -self.max_steer_rad,
            self.max_steer_rad,
        )
        self.human_stamp = self.now_seconds()

    # Indica si existe una maniobra manual activa según la interfaz física.
    def manual_turn_active_callback(self, msg: Bool) -> None:
        self.manual_turn_active = bool(msg.data)
        self.manual_stamp = self.now_seconds()

    # Recibe la dirección propuesta por el controlador de evasión.
    def assist_steer_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            return

        self.assist_steer = clamp(
            value,
            -self.max_steer_rad,
            self.max_steer_rad,
        )
        self.assist_stamp = self.now_seconds()

    def assist_active_callback(self, msg: Bool) -> None:
        self.assist_active = bool(msg.data)

    # Conserva el aviso de parada únicamente para diagnóstico. No modifica
    # directamente el movimiento del bastón.
    def stop_requested_callback(self, msg: Bool) -> None:
        # NO-BRAKE: nunca se convierte en una orden de movimiento.
        self.stop_requested = bool(msg.data)

    # Extrae yaw, velocidad lineal y velocidad angular de /local_odom.
    # También actualiza el estado moving mediante histéresis.
    def odom_callback(self, msg: Odometry) -> None:
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.speed = math.hypot(vx, vy)

        self.yaw_rate = float(msg.twist.twist.angular.z)

        self.have_odom = all(
            math.isfinite(v)
            for v in (
                self.yaw,
                self.speed,
                self.yaw_rate,
            )
        )
        self.odom_stamp = self.now_seconds()

        # Histeresis de movimiento.
        if self.moving:
            if self.speed <= self.moving_exit_speed_m_s:
                self.moving = False
        else:
            if self.speed >= self.moving_enter_speed_m_s:
                self.moving = True

    def odom_valid_callback(self, msg: Bool) -> None:
        self.odom_valid = bool(msg.data)

    def odom_confidence_callback(self, msg: Float64) -> None:
        self.odom_confidence = float(msg.data)

    # =========================================================================
    # Funciones auxiliares
    # =========================================================================
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    # Comprueba si una entrada ha sido recibida dentro del timeout permitido.
    def is_fresh(
        self,
        stamp: Optional[float],
        timeout: float,
        now_sec: float,
    ) -> bool:
        return (
            stamp is not None
            and (now_sec - stamp) <= timeout
        )

    # Determina si la odometría puede utilizarse para control de rumbo.
    def orientation_usable(self, now_sec: float) -> bool:
        if not self.have_odom:
            return False

        if not self.is_fresh(
            self.odom_stamp,
            self.odom_timeout_s,
            now_sec,
        ):
            return False

        if (
            self.require_odom_valid_for_heading
            and not self.odom_valid
        ):
            return False

        if (
            self.odom_confidence
            < self.min_odom_confidence_for_heading
        ):
            return False

        return True

    # Centraliza los cambios de estado del multiplexor.
    def change_state(
        self,
        new_state: str,
        now_sec: float,
        reason: str,
    ) -> None:
        if new_state == self.state:
            return

        self.get_logger().info(
            f'State: {self.state} -> {new_state} ({reason})'
        )

        self.state = new_state
        self.state_started_at = now_sec
        self.settle_stable_since = None

    # Memoriza el yaw actual como nuevo rumbo elegido por el usuario.
    def capture_heading_reference(
        self,
        now_sec: float,
        reason: str,
    ) -> None:
        self.heading_reference = self.yaw
        self.heading_reference_valid = True
        self.heading_reference_stamp = now_sec
        self.initial_capture_since = None

        self.get_logger().info(
            'Nuevo rumbo humano fijado: '
            f'yaw={self.heading_reference:.3f} rad '
            f'({math.degrees(self.heading_reference):.1f} deg) | '
            f'{reason}'
        )

    # Invalida la referencia de rumbo cuando deja de ser fiable o el usuario
    # inicia una nueva maniobra.
    def invalidate_heading_reference(
        self,
        reason: str,
    ) -> None:
        if self.heading_reference_valid:
            self.get_logger().warn(
                f'Referencia de rumbo invalidada: {reason}'
            )

        self.heading_reference_valid = False
        self.heading_reference_stamp = None
        self.initial_capture_since = None

    # Determina si existe una petición manual válida y reciente.
    def manual_requested(self, now_sec: float) -> bool:
        active_topic_fresh = self.is_fresh(
            self.manual_stamp,
            self.input_timeout_s,
            now_sec,
        )
        human_topic_fresh = self.is_fresh(
            self.human_stamp,
            self.input_timeout_s,
            now_sec,
        )

        active_by_button = (
            active_topic_fresh
            and self.manual_turn_active
        )

        active_by_command = (
            human_topic_fresh
            and abs(self.human_steer)
            >= self.manual_enter_deadband_rad
        )

        return active_by_button or active_by_command

    # Determina si la asistencia automática está activa y su comando es reciente.
    def assist_requested(self, now_sec: float) -> bool:
        return (
            self.assist_active
            and self.is_fresh(
                self.assist_stamp,
                self.input_timeout_s,
                now_sec,
            )
        )

    # Aplica límite angular y velocidad máxima de cambio a la orden del servo.
    def apply_output_dynamics(
        self,
        target: float,
        rate: float,
        dt: float,
    ) -> float:
        target = clamp(
            target,
            -self.max_steer_rad,
            self.max_steer_rad,
        )

        self.output_cmd = move_towards(
            self.output_cmd,
            target,
            max(0.0, rate) * dt,
        )

        return clamp(
            self.output_cmd,
            -self.max_steer_rad,
            self.max_steer_rad,
        )

    # Control PD de mantenimiento de rumbo:
    #   - término proporcional sobre error angular;
    #   - término derivativo sobre velocidad de yaw.
    def compute_heading_hold(
        self,
    ) -> tuple[float, float]:
        error = wrap_angle(
            self.heading_reference - self.yaw
        )

        effective_error = (
            0.0
            if abs(error)
            < self.heading_error_deadband_rad
            else error
        )

        effective_rate = (
            0.0
            if abs(self.yaw_rate)
            < self.heading_yaw_rate_deadband_rad_s
            else self.yaw_rate
        )

        target = (
            self.heading_kp * effective_error
            - self.heading_kd * effective_rate
        )

        target = clamp(
            target,
            -self.heading_max_steer_rad,
            self.heading_max_steer_rad,
        )

        return target, error

    # =========================================================================
    # Bucle principal de arbitraje
    # =========================================================================
    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    # Ejecuta la jerarquía completa de prioridades:
    #   1. MANUAL_TURN
    #   2. MANUAL_SETTLE
    #   3. AUTONOMOUS_ASSIST
    #   4. HEADING_HOLD
    #   5. USER_GUIDED / ODOM_DEGRADED
    def control_callback(self) -> None:
        now = self.get_clock().now()
        now_sec = self.now_seconds()

        dt = (
            now - self.last_control_time
        ).nanoseconds * 1e-9
        self.last_control_time = now
        dt = clamp(dt, 1e-3, 0.10)

        # Evaluación de las fuentes disponibles y de la calidad de odometría.
        manual = self.manual_requested(now_sec)
        assist = self.assist_requested(now_sec)
        odom_ok = self.orientation_usable(now_sec)

        source = 'CENTER'
        heading_error = 0.0
        human_priority = False
        heading_hold_active = False

        # --------------------------------------------------------------
        # 1) Prioridad humana
        # --------------------------------------------------------------
        # Si existe una intención manual, la dirección humana pasa directamente
        # al servo y anula temporalmente cualquier asistencia.
        # --------------------------------------------------------------
        # 1) Intención humana: máxima prioridad sobre la DIRECCIÓN.
        # --------------------------------------------------------------
        if manual:
            if self.state != self.MANUAL_TURN:
                self.change_state(
                    self.MANUAL_TURN,
                    now_sec,
                    'HUMAN_DIRECTION_REQUEST',
                )
                self.invalidate_heading_reference(
                    'NEW_MANUAL_TURN'
                )

            output = self.apply_output_dynamics(
                self.human_steer,
                self.manual_slew_rate_rad_s,
                dt,
            )

            source = 'HUMAN_DIRECTION'
            human_priority = True

        # --------------------------------------------------------------
        # 2) Inicio del recentrado
        # --------------------------------------------------------------
        # Al desaparecer la orden manual comienza una fase de transición suave.
        # --------------------------------------------------------------
        # 2) Al soltar la flecha/botón, empezar centrado.
        # --------------------------------------------------------------
        elif self.state == self.MANUAL_TURN:
            self.change_state(
                self.MANUAL_SETTLE,
                now_sec,
                'HUMAN_DIRECTION_RELEASED',
            )

            output = self.apply_output_dynamics(
                0.0,
                self.settle_slew_rate_rad_s,
                dt,
            )

            source = 'SETTLE_START'
            human_priority = True

        # --------------------------------------------------------------
        # 3) Recentrado y captura del nuevo rumbo
        # --------------------------------------------------------------
        # Se espera a que la rueda esté prácticamente centrada y el yaw sea estable
        # antes de fijar la nueva referencia.
        # --------------------------------------------------------------
        # 3) Centrado progresivo y captura del nuevo rumbo.
        # --------------------------------------------------------------
        elif self.state == self.MANUAL_SETTLE:
            elapsed = now_sec - self.state_started_at

            if odom_ok:
                countersteer = clamp(
                    -self.settle_yaw_rate_gain
                    * self.yaw_rate,
                    -self.settle_max_countersteer_rad,
                    self.settle_max_countersteer_rad,
                )
            else:
                countersteer = 0.0

            output = self.apply_output_dynamics(
                countersteer,
                self.settle_slew_rate_rad_s,
                dt,
            )

            source = 'SETTLE'
            human_priority = True

            centered = (
                abs(output)
                <= self.settle_center_tolerance_rad
            )

            yaw_stable = (
                odom_ok
                and abs(self.yaw_rate)
                <= self.settle_yaw_rate_tolerance_rad_s
            )

            if (
                elapsed >= self.settle_min_time_s
                and centered
                and yaw_stable
            ):
                if self.settle_stable_since is None:
                    self.settle_stable_since = now_sec

                elif (
                    now_sec - self.settle_stable_since
                    >= self.settle_stable_time_s
                ):
                    self.capture_heading_reference(
                        now_sec,
                        'MANUAL_TURN_COMPLETED',
                    )
                    self.change_state(
                        self.HEADING_HOLD,
                        now_sec,
                        'NEW_HEADING_CAPTURED',
                    )
            else:
                self.settle_stable_since = None

            if elapsed >= self.settle_timeout_s:
                if odom_ok:
                    self.capture_heading_reference(
                        now_sec,
                        'SETTLE_TIMEOUT_CAPTURE_CURRENT_YAW',
                    )
                    self.change_state(
                        self.HEADING_HOLD,
                        now_sec,
                        'SETTLE_TIMEOUT',
                    )
                else:
                    self.invalidate_heading_reference(
                        'SETTLE_TIMEOUT_NO_ODOM'
                    )
                    self.change_state(
                        self.ODOM_DEGRADED,
                        now_sec,
                        'SETTLE_TIMEOUT_NO_ODOM',
                    )

        # --------------------------------------------------------------
        # 4) Asistencia automática
        # --------------------------------------------------------------
        # Durante evasión o retorno se utiliza la orden procedente del controlador.
        # --------------------------------------------------------------
        # 4) Evasión/retorno automático: solo dirección.
        # --------------------------------------------------------------
        elif assist:
            if self.state != self.AUTONOMOUS_ASSIST:
                self.change_state(
                    self.AUTONOMOUS_ASSIST,
                    now_sec,
                    'ASSIST_ACTIVE',
                )

            output = self.apply_output_dynamics(
                self.assist_steer,
                self.manual_slew_rate_rad_s,
                dt,
            )

            source = 'ASSIST_DIRECTION'

        # --------------------------------------------------------------
        # 5) Fin de asistencia
        # --------------------------------------------------------------
        # Si existe una referencia válida, se vuelve a mantener el rumbo humano.
        # --------------------------------------------------------------
        # 5) Fin de la evasión: recuperar rumbo humano previo.
        # --------------------------------------------------------------
        elif self.state == self.AUTONOMOUS_ASSIST:
            if self.heading_reference_valid and odom_ok:
                self.change_state(
                    self.HEADING_HOLD,
                    now_sec,
                    'ASSIST_FINISHED',
                )

                target, heading_error = (
                    self.compute_heading_hold()
                )

                if self.moving:
                    output = self.apply_output_dynamics(
                        target,
                        self.heading_slew_rate_rad_s,
                        dt,
                    )
                    source = 'HEADING_HOLD_AFTER_ASSIST'
                else:
                    output = self.apply_output_dynamics(
                        0.0,
                        self.settle_slew_rate_rad_s,
                        dt,
                    )
                    source = 'HEADING_HOLD_STOPPED'

                heading_hold_active = True

            else:
                self.change_state(
                    self.USER_GUIDED,
                    now_sec,
                    'ASSIST_FINISHED_NO_REFERENCE',
                )

                output = self.apply_output_dynamics(
                    0.0,
                    self.settle_slew_rate_rad_s,
                    dt,
                )

                source = 'CENTER_AFTER_ASSIST'

        # --------------------------------------------------------------
        # 6) Mantenimiento de rumbo
        # --------------------------------------------------------------
        # Solo aplica correcciones mientras la odometría indica movimiento real.
        # --------------------------------------------------------------
        # 6) Mantenimiento de rumbo.
        #    Solo aplica correcciones cuando la odometría indica movimiento.
        # --------------------------------------------------------------
        elif self.heading_reference_valid and odom_ok:
            if self.state != self.HEADING_HOLD:
                self.change_state(
                    self.HEADING_HOLD,
                    now_sec,
                    'REFERENCE_AVAILABLE',
                )

            target, heading_error = (
                self.compute_heading_hold()
            )

            if (
                abs(heading_error)
                > self.heading_breakout_error_rad
            ):
                self.invalidate_heading_reference(
                    'HUMAN_BREAKOUT_OR_STALE_REFERENCE'
                )
                self.change_state(
                    self.USER_GUIDED,
                    now_sec,
                    'HEADING_BREAKOUT',
                )

                output = self.apply_output_dynamics(
                    0.0,
                    self.settle_slew_rate_rad_s,
                    dt,
                )

                source = 'BREAKOUT_CENTER'

            elif not self.moving:
                # La persona se ha detenido físicamente.
                # No se ordena ninguna parada: simplemente centramos la rueda.
                output = self.apply_output_dynamics(
                    0.0,
                    self.settle_slew_rate_rad_s,
                    dt,
                )

                source = 'HEADING_HOLD_STOPPED'
                heading_hold_active = True

            else:
                output = self.apply_output_dynamics(
                    target,
                    self.heading_slew_rate_rad_s,
                    dt,
                )

                source = 'HEADING_HOLD_MOVING'
                heading_hold_active = True

        # --------------------------------------------------------------
        # 7) Sin referencia válida
        # --------------------------------------------------------------
        # La rueda permanece centrada y se espera a detectar un tramo recto y
        # estable para capturar una nueva referencia.
        # --------------------------------------------------------------
        # 7) Sin referencia: rueda centrada.
        #    Se captura rumbo inicial cuando REALMENTE se detecta movimiento
        #    mediante /local_odom y el giro es estable.
        # --------------------------------------------------------------
        else:
            if self.state not in (
                self.USER_GUIDED,
                self.ODOM_DEGRADED,
            ):
                self.change_state(
                    self.USER_GUIDED,
                    now_sec,
                    'NO_ACTIVE_SOURCE',
                )

            output = self.apply_output_dynamics(
                0.0,
                self.settle_slew_rate_rad_s,
                dt,
            )

            source = 'CENTER_WAIT_REFERENCE'

            if (
                odom_ok
                and self.moving
                and abs(self.yaw_rate)
                <= self.heading_capture_yaw_rate_rad_s
            ):
                if self.initial_capture_since is None:
                    self.initial_capture_since = now_sec

                elif (
                    now_sec - self.initial_capture_since
                    >= self.heading_capture_stable_time_s
                ):
                    self.capture_heading_reference(
                        now_sec,
                        'INITIAL_STRAIGHT_HEADING',
                    )
                    self.change_state(
                        self.HEADING_HOLD,
                        now_sec,
                        'INITIAL_HEADING_CAPTURED',
                    )
            else:
                self.initial_capture_since = None

            if not odom_ok:
                self.change_state(
                    self.ODOM_DEGRADED,
                    now_sec,
                    'ODOM_NOT_USABLE',
                )

        # Si la referencia no puede comprobarse durante demasiado tiempo,
        # descartarla en vez de seguir corrigiendo con información vieja.
        if (
            self.heading_reference_valid
            and self.heading_reference_stamp is not None
            and not odom_ok
            and (
                now_sec - self.heading_reference_stamp
                > self.heading_reference_max_stale_s
            )
        ):
            self.invalidate_heading_reference(
                'ODOM_STALE_TOO_LONG'
            )

        # Aplicación final de inversión de signo y offset mecánico antes de
        # publicar la orden de servo.
        servo_output = clamp(
            self.steer_sign * output
            + self.steer_offset_rad,
            -self.max_steer_rad,
            self.max_steer_rad,
        )

        self.output_pub.publish(
            Float64(data=servo_output)
        )
        self.state_pub.publish(
            String(data=self.state)
        )
        self.human_priority_pub.publish(
            Bool(data=human_priority)
        )
        self.heading_hold_pub.publish(
            Bool(data=heading_hold_active)
        )
        self.heading_reference_pub.publish(
            Float64(
                data=(
                    self.heading_reference
                    if self.heading_reference_valid
                    else math.nan
                )
            )
        )

        debug = (
            f'state={self.state} source={source} '
            f'human={self.human_steer:+.3f} manual={manual} '
            f'assist={self.assist_steer:+.3f} active={assist} '
            f'v={self.speed:.3f} moving={self.moving} '
            f'yaw={self.yaw:+.3f} wz={self.yaw_rate:+.3f} '
            f'odom_ok={odom_ok} '
            f'h_ref={self.heading_reference:+.3f} '
            f'ref_valid={self.heading_reference_valid} '
            f'e_h={heading_error:+.3f} '
            f'warning_only_stop_topic={self.stop_requested} '
            f'out={servo_output:+.3f}'
        )

        self.debug_pub.publish(
            String(data=debug)
        )

        if (
            (now - self.last_debug_time).nanoseconds * 1e-9
            >= 0.75
        ):
            self.get_logger().info(debug)
            self.last_debug_time = now


# ============================================================================
# Punto de entrada del nodo
# ============================================================================
# Inicializa ROS 2, crea el multiplexor y mantiene activo el bucle de control.
# Antes de cerrar se intenta publicar una última orden de dirección centrada.
# ============================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SteeringCommandMuxNode()

    try:
        rclpy.spin(node)

    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
        pass

    finally:
        try:
            node.output_pub.publish(
                Float64(data=0.0)
            )
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
