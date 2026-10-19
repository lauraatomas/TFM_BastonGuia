#!/usr/bin/env python3
"""
Coordinador háptico para Mapless Cane.

Convierte el estado de navegación + dirección final + ultrasonidos laterales
en eventos hápticos espaciales sencillos para el firmware V3 del ESP32.

Lenguaje:
    0 -> ninguno / rearme
    1 -> inicio de giro a la IZQUIERDA
    2 -> inicio de giro a la DERECHA
    3 -> obstáculo confirmado al LADO IZQUIERDO
    4 -> obstáculo confirmado al LADO DERECHO
    5 -> maniobra de evasión TERMINADA
    6 -> localización perdida             (opcional)
    7 -> peligro general                  (opcional)

Secuencia típica:

    USER_GUIDED
       |
       | obstáculo frontal / planner decide evasión
       v
    AVOID_OPEN
       |
       +--> giro izquierda -> evento 1
       |         o
       +--> giro derecha   -> evento 2
       |
       | ultrasonido del lado del obstáculo lo confirma
       +--> obstáculo izquierda -> evento 3
       |         o
       +--> obstáculo derecha   -> evento 4
       v
    PASS_OBSTACLE
       |
       v
    RETURN_LINE
       |
       | recuperación terminada
       v
    USER_GUIDED
       |
       +--> evento 5 (vibrador inferior)

El sensor superior NO participa aquí. Su aviso es local en el ESP32.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Range
from std_msgs.msg import Float64, String, UInt8


class HapticEventCoordinatorNode(Node):

    USER_GUIDED = 'USER_GUIDED'
    MANUAL_TURN = 'MANUAL_TURN'
    AVOID_OPEN = 'AVOID_OPEN'
    PASS_OBSTACLE = 'PASS_OBSTACLE'
    RETURN_LINE = 'RETURN_LINE'

    HAPTIC_NONE = 0
    HAPTIC_TURN_LEFT = 1
    HAPTIC_TURN_RIGHT = 2
    HAPTIC_OBJECT_LEFT = 3
    HAPTIC_OBJECT_RIGHT = 4
    HAPTIC_RETURN_COMPLETE = 5
    HAPTIC_LOCALIZATION_LOST = 6
    HAPTIC_DANGER = 7

    def __init__(self) -> None:
        super().__init__('haptic_event_coordinator_node')

        # --------------------------------------------------------------
        # Topics
        # --------------------------------------------------------------
        self.declare_parameter(
            'state_topic',
            '/mapless_bypass_state',
        )
        self.declare_parameter(
            'steering_topic',
            '/steering_pos_cmd',
        )
        self.declare_parameter(
            'left_range_topic',
            '/ultrasound_left/range',
        )
        self.declare_parameter(
            'right_range_topic',
            '/ultrasound_right/range',
        )
        self.declare_parameter(
            'warning_reason_topic',
            '/safety_warning_reason',
        )
        self.declare_parameter(
            'output_topic',
            '/haptic_pattern_final',
        )

        # --------------------------------------------------------------
        # Temporización de eventos hacia ESP32
        # --------------------------------------------------------------
        self.declare_parameter(
            'update_rate_hz',
            25.0,
        )

        # No es la duración física de vibración: el ESP32 genera el patrón.
        # Es el tiempo que mantenemos el código en serie antes de mandar 0.
        self.declare_parameter(
            'event_code_hold_s',
            0.18,
        )
        self.declare_parameter(
            'rearm_zero_s',
            0.14,
        )

        # --------------------------------------------------------------
        # Detección del inicio de giro
        # --------------------------------------------------------------
        self.declare_parameter(
            'turn_min_steer_rad',
            0.06,
        )
        self.declare_parameter(
            'turn_confirm_s',
            0.12,
        )
        self.declare_parameter(
            'steering_timeout_s',
            0.50,
        )

        # --------------------------------------------------------------
        # Confirmación del obstáculo lateral
        # --------------------------------------------------------------
        self.declare_parameter(
            'side_detect_distance_m',
            0.75,
        )
        self.declare_parameter(
            'side_detect_confirm_s',
            0.12,
        )
        self.declare_parameter(
            'range_timeout_s',
            0.50,
        )

        # Si el controlador cambia de lado durante AVOID_OPEN,
        # permitimos actualizar el lado esperado mientras aún no se haya
        # emitido el aviso lateral.
        self.declare_parameter(
            'allow_turn_side_update',
            False,
        )

        # Por defecto dejamos desactivados los avisos 6/7 para no sobrecargar
        # a la persona durante las primeras pruebas de usabilidad.
        self.declare_parameter(
            'enable_safety_haptics',
            False,
        )
        self.declare_parameter(
            'safety_event_cooldown_s',
            3.0,
        )

        gp = lambda name: self.get_parameter(name).value

        self.state_topic = str(gp('state_topic'))
        self.steering_topic = str(gp('steering_topic'))
        self.left_range_topic = str(gp('left_range_topic'))
        self.right_range_topic = str(gp('right_range_topic'))
        self.warning_reason_topic = str(
            gp('warning_reason_topic')
        )
        self.output_topic = str(gp('output_topic'))

        self.event_code_hold_s = float(
            gp('event_code_hold_s')
        )
        self.rearm_zero_s = float(
            gp('rearm_zero_s')
        )

        self.turn_min_steer_rad = float(
            gp('turn_min_steer_rad')
        )
        self.turn_confirm_s = float(
            gp('turn_confirm_s')
        )
        self.steering_timeout_s = float(
            gp('steering_timeout_s')
        )

        self.side_detect_distance_m = float(
            gp('side_detect_distance_m')
        )
        self.side_detect_confirm_s = float(
            gp('side_detect_confirm_s')
        )
        self.range_timeout_s = float(
            gp('range_timeout_s')
        )

        self.allow_turn_side_update = bool(
            gp('allow_turn_side_update')
        )

        self.enable_safety_haptics = bool(
            gp('enable_safety_haptics')
        )
        self.safety_event_cooldown_s = float(
            gp('safety_event_cooldown_s')
        )

        # --------------------------------------------------------------
        # Estado de navegación
        # --------------------------------------------------------------
        self.state = ''
        self.previous_state = ''

        self.latest_steer = 0.0
        self.latest_steer_stamp: Optional[float] = None

        self.left_range = math.inf
        self.right_range = math.inf
        self.left_range_valid = False
        self.right_range_valid = False
        self.left_range_stamp: Optional[float] = None
        self.right_range_stamp: Optional[float] = None

        # --------------------------------------------------------------
        # Memoria por maniobra
        # --------------------------------------------------------------
        self.manoeuvre_active = False
        self.turn_cue_sent = False
        self.side_cue_sent = False

        # +1 = izquierda, -1 = derecha
        self.turn_sign = 0
        self.turn_candidate_sign = 0
        self.turn_candidate_since: Optional[float] = None

        # 'LEFT' / 'RIGHT'
        self.expected_object_side = 'UNKNOWN'
        self.side_detect_since: Optional[float] = None

        # Para detectar "RETURN_LINE -> USER_GUIDED".
        self.saw_return_line = False

        # --------------------------------------------------------------
        # Cola de eventos
        # --------------------------------------------------------------
        self.event_queue: Deque[Tuple[int, str]] = deque()

        self.active_code = self.HAPTIC_NONE
        self.active_label = ''
        self.active_until = 0.0
        self.rearm_until = 0.0

        self.last_published = -1

        # Safety opcional.
        self.last_warning_reason = ''
        self.last_safety_event_time = -1e9

        # --------------------------------------------------------------
        # ROS
        # --------------------------------------------------------------
        self.create_subscription(
            String,
            self.state_topic,
            self.state_cb,
            10,
        )

        self.create_subscription(
            Float64,
            self.steering_topic,
            self.steering_cb,
            10,
        )

        self.create_subscription(
            Range,
            self.left_range_topic,
            self.left_range_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Range,
            self.right_range_topic,
            self.right_range_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            String,
            self.warning_reason_topic,
            self.warning_reason_cb,
            10,
        )

        self.pub = self.create_publisher(
            UInt8,
            self.output_topic,
            10,
        )

        rate = float(gp('update_rate_hz'))
        self.timer = self.create_timer(
            1.0 / max(rate, 1.0),
            self.timer_cb,
        )

        self.get_logger().info(
            'Coordinador háptico V3 iniciado | '
            f'out={self.output_topic} | '
            f'side_detect<={self.side_detect_distance_m:.2f}m'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def state_cb(self, msg: String) -> None:
        new_state = msg.data.strip()

        if not new_state:
            return

        if new_state == self.state:
            return

        old_state = self.state
        self.previous_state = old_state
        self.state = new_state

        self.get_logger().info(
            f'Estado navegación: {old_state or "NONE"} -> {new_state}'
        )

        # Inicio de una maniobra de evasión.
        if new_state == self.AVOID_OPEN:
            if not self.manoeuvre_active:
                self.start_manoeuvre()
            else:
                # Una reentrada desde RETURN_LINE puede significar nueva
                # evasión. Reiniciamos las pistas hápticas de esa evasión.
                if old_state == self.RETURN_LINE:
                    self.start_manoeuvre()

        if new_state == self.RETURN_LINE:
            if self.manoeuvre_active:
                self.saw_return_line = True

        # La maniobra se considera terminada cuando realmente se ha pasado
        # por RETURN_LINE y el controlador vuelve a USER_GUIDED.
        if (
            new_state == self.USER_GUIDED
            and self.manoeuvre_active
            and self.saw_return_line
        ):
            self.enqueue_event(
                self.HAPTIC_RETURN_COMPLETE,
                'MANIOBRA_TERMINADA',
            )
            self.finish_manoeuvre()

        # Giro manual: no queremos interpretar el ángulo del servo como
        # aviso automático de evasión.
        if new_state == self.MANUAL_TURN:
            self.reset_turn_candidate()

    def steering_cb(self, msg: Float64) -> None:
        value = float(msg.data)

        if not math.isfinite(value):
            return

        self.latest_steer = value
        self.latest_steer_stamp = self.now_seconds()

    def left_range_cb(self, msg: Range) -> None:
        now = self.now_seconds()

        valid = (
            math.isfinite(msg.range)
            and msg.range >= msg.min_range
            and msg.range <= msg.max_range
        )

        self.left_range_valid = valid
        self.left_range = (
            float(msg.range)
            if valid
            else math.inf
        )
        self.left_range_stamp = now

    def right_range_cb(self, msg: Range) -> None:
        now = self.now_seconds()

        valid = (
            math.isfinite(msg.range)
            and msg.range >= msg.min_range
            and msg.range <= msg.max_range
        )

        self.right_range_valid = valid
        self.right_range = (
            float(msg.range)
            if valid
            else math.inf
        )
        self.right_range_stamp = now

    def warning_reason_cb(self, msg: String) -> None:
        if not self.enable_safety_haptics:
            return

        reason = msg.data.strip().upper()

        if not reason or reason == self.last_warning_reason:
            return

        self.last_warning_reason = reason
        now = self.now_seconds()

        if (
            now - self.last_safety_event_time
            < self.safety_event_cooldown_s
        ):
            return

        localization_keywords = (
            'LOCALIZATION',
            'ODOMETRY',
            'ODOM_',
        )

        danger_keywords = (
            'NO_FULLY_VALID_ARC',
            'EMERGENCY',
            'ACTUATOR_TIMEOUT',
            'HARD_',
        )

        if any(
            key in reason
            for key in localization_keywords
        ):
            self.enqueue_event(
                self.HAPTIC_LOCALIZATION_LOST,
                f'LOCALIZACION:{reason}',
            )
            self.last_safety_event_time = now

        elif any(
            key in reason
            for key in danger_keywords
        ):
            self.enqueue_event(
                self.HAPTIC_DANGER,
                f'PELIGRO:{reason}',
            )
            self.last_safety_event_time = now

    # ------------------------------------------------------------------
    # Maniobra
    # ------------------------------------------------------------------
    def start_manoeuvre(self) -> None:
        self.manoeuvre_active = True
        self.turn_cue_sent = False
        self.side_cue_sent = False

        self.turn_sign = 0
        self.expected_object_side = 'UNKNOWN'

        self.saw_return_line = False

        self.reset_turn_candidate()
        self.side_detect_since = None

        self.get_logger().info(
            'Nueva maniobra: esperando dirección real del servo.'
        )

    def finish_manoeuvre(self) -> None:
        self.manoeuvre_active = False
        self.turn_cue_sent = False
        self.side_cue_sent = False

        self.turn_sign = 0
        self.expected_object_side = 'UNKNOWN'

        self.saw_return_line = False

        self.reset_turn_candidate()
        self.side_detect_since = None

    def reset_turn_candidate(self) -> None:
        self.turn_candidate_sign = 0
        self.turn_candidate_since = None

    # ------------------------------------------------------------------
    # Lógica
    # ------------------------------------------------------------------
    def timer_cb(self) -> None:
        now = self.now_seconds()

        self.update_navigation_events(now)
        self.update_event_output(now)

    def update_navigation_events(self, now: float) -> None:

        # --------------------------------------------------------------
        # 1) Aviso de dirección al empezar la evasión
        # --------------------------------------------------------------
        if (
            self.manoeuvre_active
            and self.state == self.AVOID_OPEN
            and not self.turn_cue_sent
        ):
            self.update_turn_cue(now)

        # Si el lado de giro cambia antes de detectar el obstáculo lateral,
        # podemos actualizar qué lado esperamos encontrar.
        elif (
            self.manoeuvre_active
            and self.state == self.AVOID_OPEN
            and self.turn_cue_sent
            and not self.side_cue_sent
            and self.allow_turn_side_update
        ):
            self.update_expected_side_from_current_steer(now)

        # --------------------------------------------------------------
        # 2) Aviso de que el obstáculo ya está al lado
        # --------------------------------------------------------------
        if (
            self.manoeuvre_active
            and self.turn_cue_sent
            and not self.side_cue_sent
            and self.state
            in (
                self.AVOID_OPEN,
                self.PASS_OBSTACLE,
            )
        ):
            self.update_side_cue(now)

    def update_turn_cue(self, now: float) -> None:

        if (
            self.latest_steer_stamp is None
            or (
                now - self.latest_steer_stamp
                > self.steering_timeout_s
            )
        ):
            self.reset_turn_candidate()
            return

        steer = self.latest_steer

        if abs(steer) < self.turn_min_steer_rad:
            self.reset_turn_candidate()
            return

        sign = 1 if steer > 0.0 else -1

        if sign != self.turn_candidate_sign:
            self.turn_candidate_sign = sign
            self.turn_candidate_since = now
            return

        if self.turn_candidate_since is None:
            self.turn_candidate_since = now
            return

        if (
            now - self.turn_candidate_since
            < self.turn_confirm_s
        ):
            return

        self.turn_sign = sign
        self.turn_cue_sent = True

        if sign > 0:
            self.expected_object_side = 'RIGHT'
            self.enqueue_event(
                self.HAPTIC_TURN_LEFT,
                'GIRO_IZQUIERDA',
            )
        else:
            self.expected_object_side = 'LEFT'
            self.enqueue_event(
                self.HAPTIC_TURN_RIGHT,
                'GIRO_DERECHA',
            )

        self.side_detect_since = None

        self.get_logger().info(
            'Inicio evasión confirmado: '
            f'steer={steer:+.3f} rad | '
            f'obstáculo esperado={self.expected_object_side}'
        )

    def update_expected_side_from_current_steer(
        self,
        now: float,
    ) -> None:

        if (
            self.latest_steer_stamp is None
            or (
                now - self.latest_steer_stamp
                > self.steering_timeout_s
            )
        ):
            return

        steer = self.latest_steer

        if abs(steer) < self.turn_min_steer_rad:
            return

        current_sign = 1 if steer > 0.0 else -1

        if current_sign == self.turn_sign:
            return

        # El controlador ha cambiado de lado de evasión.
        self.turn_sign = current_sign
        self.expected_object_side = (
            'RIGHT'
            if current_sign > 0
            else 'LEFT'
        )
        self.side_detect_since = None

        self.get_logger().warn(
            'La dirección de evasión cambió antes de la '
            'confirmación lateral. Nuevo obstáculo esperado: '
            f'{self.expected_object_side}'
        )

        # Avisamos también del NUEVO giro, porque físicamente cambia lo que
        # va a notar la persona en el bastón.
        if current_sign > 0:
            self.enqueue_event(
                self.HAPTIC_TURN_LEFT,
                'CAMBIO_GIRO_IZQUIERDA',
            )
        else:
            self.enqueue_event(
                self.HAPTIC_TURN_RIGHT,
                'CAMBIO_GIRO_DERECHA',
            )

    def update_side_cue(self, now: float) -> None:

        if self.expected_object_side == 'LEFT':
            fresh, valid, distance = self.get_left(now)
            code = self.HAPTIC_OBJECT_LEFT
            label = 'OBSTACULO_LADO_IZQUIERDO'

        elif self.expected_object_side == 'RIGHT':
            fresh, valid, distance = self.get_right(now)
            code = self.HAPTIC_OBJECT_RIGHT
            label = 'OBSTACULO_LADO_DERECHO'

        else:
            self.side_detect_since = None
            return

        detected = (
            fresh
            and valid
            and distance
            <= self.side_detect_distance_m
        )

        if not detected:
            self.side_detect_since = None
            return

        if self.side_detect_since is None:
            self.side_detect_since = now
            return

        if (
            now - self.side_detect_since
            < self.side_detect_confirm_s
        ):
            return

        self.side_cue_sent = True
        self.enqueue_event(
            code,
            label,
        )

        self.get_logger().info(
            f'{label}: d={distance:.2f} m'
        )

    def get_left(
        self,
        now: float,
    ) -> Tuple[bool, bool, float]:

        fresh = (
            self.left_range_stamp is not None
            and (
                now - self.left_range_stamp
                <= self.range_timeout_s
            )
        )

        return (
            fresh,
            self.left_range_valid,
            self.left_range,
        )

    def get_right(
        self,
        now: float,
    ) -> Tuple[bool, bool, float]:

        fresh = (
            self.right_range_stamp is not None
            and (
                now - self.right_range_stamp
                <= self.range_timeout_s
            )
        )

        return (
            fresh,
            self.right_range_valid,
            self.right_range,
        )

    # ------------------------------------------------------------------
    # Cola / rearme de eventos
    # ------------------------------------------------------------------
    def enqueue_event(
        self,
        code: int,
        label: str,
    ) -> None:

        if code <= 0:
            return

        # Evita duplicados consecutivos idénticos.
        if (
            self.active_code == code
            and self.active_label == label
        ):
            return

        if (
            self.event_queue
            and self.event_queue[-1]
            == (code, label)
        ):
            return

        self.event_queue.append(
            (code, label)
        )

    def update_event_output(self, now: float) -> None:

        # Evento activo.
        if self.active_code != self.HAPTIC_NONE:

            if now < self.active_until:
                self.publish_code(
                    self.active_code
                )
                return

            # Final del evento: mandamos 0 para rearmar el ESP32.
            self.get_logger().info(
                f'HAPTIC END {self.active_code}: '
                f'{self.active_label}'
            )

            self.active_code = self.HAPTIC_NONE
            self.active_label = ''
            self.rearm_until = (
                now + self.rearm_zero_s
            )

            self.publish_code(
                self.HAPTIC_NONE
            )
            return

        # Ventana obligatoria de 0 entre eventos.
        if now < self.rearm_until:
            self.publish_code(
                self.HAPTIC_NONE
            )
            return

        # Lanzar siguiente evento.
        if self.event_queue:
            code, label = self.event_queue.popleft()

            self.active_code = code
            self.active_label = label
            self.active_until = (
                now + self.event_code_hold_s
            )

            self.get_logger().info(
                f'HAPTIC START {code}: {label}'
            )

            self.publish_code(code)
            return

        # Reposo.
        self.publish_code(
            self.HAPTIC_NONE
        )

    def publish_code(self, code: int) -> None:
        code = int(max(0, min(7, code)))

        self.pub.publish(
            UInt8(data=code)
        )

        self.last_published = code

    def now_seconds(self) -> float:
        return (
            self.get_clock().now().nanoseconds
            * 1e-9
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = HapticEventCoordinatorNode()

    try:
        rclpy.spin(node)

    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
        pass

    finally:
        try:
            node.pub.publish(UInt8(data=0))
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
