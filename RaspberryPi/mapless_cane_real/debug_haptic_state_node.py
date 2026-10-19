#!/usr/bin/env python3
"""
Nodo de diagnóstico háptico para las pruebas reales del Mapless Cane.

OBJETIVO
--------
Permitir conocer físicamente en qué estado se encuentra la máquina de estados
sin necesidad de observar continuamente el terminal.

Este nodo está pensado EXCLUSIVAMENTE para pruebas y depuración.

No utiliza la posición espacial de los vibradores para codificar información.
Todos los patrones se generan reutilizando el evento háptico 7 del firmware,
que activa los cuatro vibradores mediante una ráfaga de tres pulsos.

La información se codifica mediante el NÚMERO DE RÁFAGAS:

    AVOID_OPEN
        1 ráfaga

    PASS_OBSTACLE
        2 ráfagas

    RETURN_LINE
        3 ráfagas

    RETURN_LINE -> USER_GUIDED
        4 ráfagas
        Maniobra terminada correctamente.

MANUAL_TURN y USER_GUIDED normal permanecen en silencio.

El nodo también imprime información de diagnóstico asociada al cambio:
    - error lateral;
    - error angular;
    - distancia frontal;
    - validez de odometría;
    - último aviso de seguridad.

IMPORTANTE
----------
Este nodo publica en /haptic_pattern_final.

Por ello NO debe ejecutarse simultáneamente con
haptic_event_coordinator_node.

El launch real_cane_complete_v3.launch.py se encarga de arrancar uno u otro
según el argumento:

    debug_haptics:=true
    debug_haptics:=false
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import Bool, Float64, String, UInt8


@dataclass
class DiagnosticSequence:
    label: str
    bursts: int


class DebugHapticStateNode(Node):

    USER_GUIDED = 'USER_GUIDED'
    MANUAL_TURN = 'MANUAL_TURN'
    AVOID_OPEN = 'AVOID_OPEN'
    PASS_OBSTACLE = 'PASS_OBSTACLE'
    RETURN_LINE = 'RETURN_LINE'

    # Se reutiliza el patrón 7 existente del firmware.
    # Dicho patrón activa los cuatro vibradores con tres pulsos rápidos.
    HAPTIC_ALL_MOTORS_BURST = 7
    HAPTIC_NONE = 0

    def __init__(self) -> None:
        super().__init__('debug_haptic_state_node')

        # --------------------------------------------------------------
        # Parámetros
        # --------------------------------------------------------------
        self.declare_parameter(
            'state_topic',
            '/mapless_bypass_state',
        )

        self.declare_parameter(
            'output_topic',
            '/haptic_pattern_final',
        )

        # Tiempo durante el cual ROS mantiene el código 7.
        # El ESP32 genera internamente la ráfaga completa.
        self.declare_parameter(
            'command_hold_s',
            0.18,
        )

        # El evento 7 del firmware dura aproximadamente 0.6 s.
        # Dejamos margen antes de disparar la siguiente ráfaga.
        self.declare_parameter(
            'burst_interval_s',
            0.85,
        )

        # Separación entre dos patrones correspondientes a estados distintos.
        self.declare_parameter(
            'sequence_interval_s',
            1.00,
        )

        self.state_topic = str(
            self.get_parameter('state_topic').value
        )

        self.output_topic = str(
            self.get_parameter('output_topic').value
        )

        self.command_hold_s = float(
            self.get_parameter('command_hold_s').value
        )

        self.burst_interval_s = float(
            self.get_parameter('burst_interval_s').value
        )

        self.sequence_interval_s = float(
            self.get_parameter('sequence_interval_s').value
        )

        # --------------------------------------------------------------
        # Estado recibido del sistema
        # --------------------------------------------------------------
        self.state = ''
        self.previous_state = ''

        self.lateral_error = 0.0
        self.heading_error = 0.0
        self.front_distance = float('inf')

        self.odom_valid = False
        self.system_ready = False

        self.warning_reason = ''

        # --------------------------------------------------------------
        # Cola de patrones
        # --------------------------------------------------------------
        self.sequence_queue: Deque[DiagnosticSequence] = deque()

        self.active_sequence: DiagnosticSequence | None = None
        self.bursts_sent = 0

        # Estados internos del generador:
        # idle
        # hold
        # wait_burst
        # wait_sequence
        self.output_phase = 'idle'
        self.phase_until = 0.0

        self.last_output = -1

        # --------------------------------------------------------------
        # Suscripciones
        # --------------------------------------------------------------
        self.create_subscription(
            String,
            self.state_topic,
            self.state_cb,
            10,
        )

        self.create_subscription(
            Float64,
            '/lateral_error',
            self.lateral_error_cb,
            10,
        )

        self.create_subscription(
            Float64,
            '/heading_error',
            self.heading_error_cb,
            10,
        )

        self.create_subscription(
            Float64,
            '/front_collision_distance',
            self.front_distance_cb,
            10,
        )

        self.create_subscription(
            Bool,
            '/local_odom_valid',
            self.odom_valid_cb,
            10,
        )

        self.create_subscription(
            Bool,
            '/system_ready',
            self.system_ready_cb,
            10,
        )

        self.create_subscription(
            String,
            '/safety_warning_reason',
            self.warning_reason_cb,
            10,
        )

        self.publisher = self.create_publisher(
            UInt8,
            self.output_topic,
            10,
        )

        # 50 Hz únicamente para temporizar los eventos.
        self.timer = self.create_timer(
            0.02,
            self.timer_cb,
        )

        self.get_logger().info(
            '================================================'
        )
        self.get_logger().info(
            'DEBUG HAPTIC STATE NODE ACTIVADO'
        )
        self.get_logger().info(
            'AVOID_OPEN     -> 1 rafaga'
        )
        self.get_logger().info(
            'PASS_OBSTACLE  -> 2 rafagas'
        )
        self.get_logger().info(
            'RETURN_LINE    -> 3 rafagas'
        )
        self.get_logger().info(
            'RETURN COMPLETE-> 4 rafagas'
        )
        self.get_logger().info(
            f'Salida: {self.output_topic}'
        )
        self.get_logger().info(
            '================================================'
        )

    # ==============================================================
    # Callbacks de información
    # ==============================================================

    def lateral_error_cb(self, msg: Float64) -> None:
        self.lateral_error = float(msg.data)

    def heading_error_cb(self, msg: Float64) -> None:
        self.heading_error = float(msg.data)

    def front_distance_cb(self, msg: Float64) -> None:
        self.front_distance = float(msg.data)

    def odom_valid_cb(self, msg: Bool) -> None:
        self.odom_valid = bool(msg.data)

    def system_ready_cb(self, msg: Bool) -> None:
        new_value = bool(msg.data)

        if new_value and not self.system_ready:
            self.get_logger().info(
                'SYSTEM READY: todos los sensores necesarios disponibles.'
            )

        self.system_ready = new_value

    def warning_reason_cb(self, msg: String) -> None:
        reason = msg.data.strip()

        if reason != self.warning_reason:
            self.warning_reason = reason

            if reason:
                self.get_logger().warn(
                    f'SAFETY WARNING: {reason}'
                )

    # ==============================================================
    # Máquina de estados
    # ==============================================================

    def state_cb(self, msg: String) -> None:
        new_state = msg.data.strip()

        if not new_state:
            return

        if new_state == self.state:
            return

        old_state = self.state

        self.previous_state = old_state
        self.state = new_state

        self.log_state_transition(
            old_state,
            new_state,
        )

        # Primer USER_GUIDED al arrancar:
        # no vibramos para no confundirlo con una maniobra terminada.
        if (
            not old_state
            and new_state == self.USER_GUIDED
        ):
            return

        # ----------------------------------------------------------
        # Inicio de evasión
        # ----------------------------------------------------------
        if new_state == self.AVOID_OPEN:
            self.enqueue_sequence(
                'AVOID_OPEN',
                1,
            )
            return

        # ----------------------------------------------------------
        # Paso lateral del obstáculo
        # ----------------------------------------------------------
        if new_state == self.PASS_OBSTACLE:
            self.enqueue_sequence(
                'PASS_OBSTACLE',
                2,
            )
            return

        # ----------------------------------------------------------
        # Inicio del retorno a la línea original
        # ----------------------------------------------------------
        if new_state == self.RETURN_LINE:
            self.enqueue_sequence(
                'RETURN_LINE',
                3,
            )
            return

        # ----------------------------------------------------------
        # Retorno completado
        # ----------------------------------------------------------
        if (
            old_state == self.RETURN_LINE
            and new_state == self.USER_GUIDED
        ):
            self.enqueue_sequence(
                'RETURN_COMPLETE',
                4,
            )
            return

        # MANUAL_TURN queda deliberadamente en silencio.
        if new_state == self.MANUAL_TURN:
            self.get_logger().info(
                'MANUAL_TURN: sin patrón háptico de diagnóstico.'
            )

    # ==============================================================
    # Logging compacto y legible
    # ==============================================================

    def log_state_transition(
        self,
        old_state: str,
        new_state: str,
    ) -> None:

        front_text = (
            f'{self.front_distance:.2f} m'
            if self.front_distance < 1000.0
            else 'inf'
        )

        self.get_logger().info(
            '\n'
            '================================================\n'
            ' CAMBIO DE ESTADO\n'
            f' {old_state or "NONE"} -> {new_state}\n'
            f' e_y      = {self.lateral_error:+.3f} m\n'
            f' e_theta  = {self.heading_error:+.3f} rad\n'
            f' front    = {front_text}\n'
            f' odom     = {self.odom_valid}\n'
            f' ready    = {self.system_ready}\n'
            f' warning  = {self.warning_reason or "NONE"}\n'
            '================================================'
        )

    # ==============================================================
    # Cola de secuencias
    # ==============================================================

    def enqueue_sequence(
        self,
        label: str,
        bursts: int,
    ) -> None:

        if bursts <= 0:
            return

        # Evita duplicados consecutivos.
        if (
            self.active_sequence is not None
            and self.active_sequence.label == label
        ):
            return

        if (
            self.sequence_queue
            and self.sequence_queue[-1].label == label
        ):
            return

        self.sequence_queue.append(
            DiagnosticSequence(
                label=label,
                bursts=bursts,
            )
        )

        self.get_logger().info(
            f'DIAGNOSTICO HAPTICO: {label} -> '
            f'{bursts} rafaga(s)'
        )

    # ==============================================================
    # Generador temporal
    # ==============================================================

    def timer_cb(self) -> None:
        now = self.now_seconds()

        # ----------------------------------------------------------
        # No hay secuencia activa: coger la siguiente
        # ----------------------------------------------------------
        if self.active_sequence is None:

            self.publish_code(
                self.HAPTIC_NONE
            )

            if not self.sequence_queue:
                return

            self.active_sequence = (
                self.sequence_queue.popleft()
            )

            self.bursts_sent = 0
            self.start_burst(now)
            return

        # ----------------------------------------------------------
        # Código 7 mantenido brevemente hacia el bridge
        # ----------------------------------------------------------
        if self.output_phase == 'hold':

            if now < self.phase_until:
                self.publish_code(
                    self.HAPTIC_ALL_MOTORS_BURST
                )
                return

            # Mandamos cero para rearmar el protocolo.
            self.publish_code(
                self.HAPTIC_NONE
            )

            self.output_phase = 'wait_burst'
            self.phase_until = (
                now + self.burst_interval_s
            )
            return

        # ----------------------------------------------------------
        # Esperar a que el ESP32 termine físicamente la ráfaga
        # ----------------------------------------------------------
        if self.output_phase == 'wait_burst':

            self.publish_code(
                self.HAPTIC_NONE
            )

            if now < self.phase_until:
                return

            if (
                self.active_sequence is not None
                and self.bursts_sent
                < self.active_sequence.bursts
            ):
                self.start_burst(now)
                return

            self.get_logger().info(
                f'FIN PATRON: '
                f'{self.active_sequence.label}'
            )

            self.output_phase = 'wait_sequence'
            self.phase_until = (
                now + self.sequence_interval_s
            )
            return

        # ----------------------------------------------------------
        # Pausa antes del siguiente estado
        # ----------------------------------------------------------
        if self.output_phase == 'wait_sequence':

            self.publish_code(
                self.HAPTIC_NONE
            )

            if now < self.phase_until:
                return

            self.active_sequence = None
            self.output_phase = 'idle'
            return

    def start_burst(self, now: float) -> None:

        if self.active_sequence is None:
            return

        self.bursts_sent += 1

        self.get_logger().info(
            f'RAFAGA {self.bursts_sent}/'
            f'{self.active_sequence.bursts} '
            f'[{self.active_sequence.label}]'
        )

        self.publish_code(
            self.HAPTIC_ALL_MOTORS_BURST
        )

        self.output_phase = 'hold'
        self.phase_until = (
            now + self.command_hold_s
        )

    # ==============================================================
    # Publicación
    # ==============================================================

    def publish_code(self, code: int) -> None:

        # No necesitamos inundar el topic con el mismo valor.
        if code == self.last_output:
            return

        msg = UInt8()
        msg.data = int(code)

        self.publisher.publish(msg)

        self.last_output = code

    def now_seconds(self) -> float:
        return (
            self.get_clock()
            .now()
            .nanoseconds
            / 1e9
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = DebugHapticStateNode()

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
