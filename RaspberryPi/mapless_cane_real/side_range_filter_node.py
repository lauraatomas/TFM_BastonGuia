#!/usr/bin/env python3
"""Filtro robusto para lecturas de distancia izquierda/derecha tipo HC-SR04.

Política de seguridad:
* un estado desconocido o sin eco se considera inválido (`valid=False`) y nunca como "despejado";
* una medición que indique una distancia menor de forma repentina se acepta de inmediato;
* una medición que indique una distancia mayor de forma repentina debe persistir antes de ser aceptada, evitando así que un único eco largo indique erróneamente que el obstáculo ha terminado;
* el historial obsoleto se borra tras agotarse el tiempo de espera, para evitar que una referencia antigua afecte a la detección del siguiente objeto.
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

# Convierte un timestamp ROS (segundos + nanosegundos) a segundos en coma
# flotante. Se utiliza para comprobar la antigüedad de cada medida recibida.

def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# ============================================================================
# Estado interno de cada sensor lateral
# ============================================================================
# Para cada HC-SR04 se mantiene:
#   - una ventana de medidas recientes;
#   - el instante de la última medida válida;
#   - el estado lógico clear/occupied;
#   - la posible transición pendiente hacia una distancia mayor.
#
# El estado pending_rise permite distinguir entre una desaparición real del
# obstáculo y un eco largo o espurio aislado.
# ============================================================================

@dataclass
class SideState:
    values: Deque[float]
    stamp: Optional[float] = None
    clear: bool = False
    pending_rise_value: Optional[float] = None
    pending_rise_count: int = 0


# ============================================================================
# Nodo de filtrado de distancias laterales
# ============================================================================
# Este nodo recibe las medidas Range publicadas por el bridge del ESP32 y
# genera una estimación más robusta para cada lateral.
#
# Objetivos principales:
#   - rechazar valores fuera del rango físico esperado;
#   - suavizar ruido mediante una mediana temporal;
#   - aceptar inmediatamente aproximaciones de obstáculos;
#   - exigir persistencia antes de aceptar incrementos bruscos de distancia;
#   - aplicar histéresis para evitar cambios rápidos entre ocupado y libre;
#   - marcar como inválidas las medidas que han quedado obsoletas.
# ============================================================================

class SideRangeFilterNode(Node):
    def __init__(self) -> None:
        super().__init__('side_range_filter_node')

        # ------------------------------------------------------------------
        # Parámetros de configuración
        # ------------------------------------------------------------------
        # Los topics de entrada corresponden a las medidas Range generadas por
        # el bridge hardware para los sensores ultrasónicos izquierdo y derecho.
        self.declare_parameter('left_topic', '/ultrasound_left/range')
        self.declare_parameter('right_topic', '/ultrasound_right/range')
        # window_size define el número máximo de medidas recientes utilizadas
        # para calcular la mediana del sensor.
        self.declare_parameter('window_size', 5)
        # Si no llega una medida válida dentro de timeout_s, el historial se
        # descarta y el sensor pasa a considerarse inválido.
        self.declare_parameter('timeout_s', 0.35)
        # Rango físico aceptado para las medidas ultrasónicas.
        # Cualquier valor fuera de estos límites se ignora.
        self.declare_parameter('min_range_m', 0.05)
        self.declare_parameter('max_range_m', 2.80)
        # Umbrales con histéresis:
        # - por debajo de occupied_threshold_m se considera obstáculo presente;
        # - por encima de clear_threshold_m se considera lateral despejado;
        # - entre ambos valores se conserva el estado anterior.
        self.declare_parameter('occupied_threshold_m', 0.62)
        self.declare_parameter('clear_threshold_m', 0.90)
        # max_rise_jump_m limita los incrementos bruscos de distancia aceptados
        # inmediatamente. Un aumento superior puede indicar un eco largo aislado.
        self.declare_parameter('max_rise_jump_m', 0.45)
        # Número de lecturas coherentes necesarias para confirmar que un salto
        # hacia una distancia mayor corresponde realmente al final del obstáculo.
        self.declare_parameter('rise_confirm_count', 2)
        # Frecuencia a la que se publica el estado filtrado de ambos laterales.
        self.declare_parameter('publish_rate_hz', 20.0)

        # Lectura y almacenamiento de los parámetros ROS en atributos internos.
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

        # Se crea un estado independiente para cada lateral, utilizando una
        # cola de longitud fija como ventana de filtrado.
        self.left = SideState(values=deque(maxlen=self.window_size))
        self.right = SideState(values=deque(maxlen=self.window_size))
        self.last_debug_time = self.get_clock().now()

        # Suscripciones a las medidas ultrasónicas publicadas por el bridge ESP32.
        # Se utiliza qos_profile_sensor_data por tratarse de datos de sensores.
        self.create_subscription(Range, self.left_topic, self.left_callback, qos_profile_sensor_data)
        self.create_subscription(Range, self.right_topic, self.right_callback, qos_profile_sensor_data)

        # Publicadores del estado procesado:
        # distancia filtrada, validez de la medida, estado libre/ocupado y debug.
        self.left_dist_pub = self.create_publisher(Float64, '/side_left_dist', 10)
        self.right_dist_pub = self.create_publisher(Float64, '/side_right_dist', 10)
        self.left_valid_pub = self.create_publisher(Bool, '/side_left_valid', 10)
        self.right_valid_pub = self.create_publisher(Bool, '/side_right_valid', 10)
        self.left_clear_pub = self.create_publisher(Bool, '/side_left_clear', 10)
        self.right_clear_pub = self.create_publisher(Bool, '/side_right_clear', 10)
        self.debug_pub = self.create_publisher(String, '/side_range_debug', 10)

        # Temporizador periódico encargado de publicar el resultado del filtro.
        rate = float(gp('publish_rate_hz'))
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.timer_callback)
        self.get_logger().info(
            f'Side range filter v4 started | left={self.left_topic} right={self.right_topic}'
        )

    # Callback del sensor izquierdo. Delega toda la lógica de filtrado en
    # accept_measurement() utilizando el estado asociado al lateral izquierdo.
    def left_callback(self, msg: Range) -> None:
        self.accept_measurement(msg, self.left)

    # Callback equivalente para el sensor derecho.
    def right_callback(self, msg: Range) -> None:
        self.accept_measurement(msg, self.right)

    # ------------------------------------------------------------------
    # Validación y filtrado de una nueva medida
    # ------------------------------------------------------------------
    # Esta función aplica la política de seguridad del filtro:
    #   1. descarta medidas no finitas o fuera de rango;
    #   2. calcula la mediana del historial;
    #   3. acepta inmediatamente acercamientos hacia obstáculos;
    #   4. exige confirmación temporal para alejamientos bruscos.
    def accept_measurement(self, msg: Range, state: SideState) -> None:
        # Valor de distancia recibido en metros.
        value = float(msg.range)
        lower = max(self.min_range_m, float(msg.min_range) if msg.min_range > 0.0 else 0.0)
        upper = self.max_range_m
        if msg.max_range > 0.0 and math.isfinite(msg.max_range):
            upper = min(upper, float(msg.max_range))
        # Las lecturas inválidas no se incorporan a la ventana de filtrado.
        if not math.isfinite(value) or value < lower or value > upper:
            return

        stamp = stamp_to_seconds(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.now_seconds()

        # La primera medida válida inicializa directamente el historial.
        if not state.values:
            state.values.append(value)
            state.stamp = stamp
            return

        # La mediana reduce la influencia de valores atípicos frente a una media.
        median = float(np.median(np.asarray(state.values, dtype=np.float32)))

        # Closer readings are safety-critical and are never rejected as a jump.
        # Las medidas cercanas o moderadamente mayores que la mediana se
        # consideran compatibles con la evolución normal del sensor y se aceptan.
        if value <= median + self.max_rise_jump_m:
            state.values.append(value)
            state.stamp = stamp
            state.pending_rise_value = None
            state.pending_rise_count = 0
            return

        # A much larger distance can mean that the object ended, but it can also
        # be one specular/long echo. Require persistence before replacing the
        # window with the new level.
        # Un salto grande hacia mayor distancia inicia una posible transición.
        # Si la siguiente lectura no es coherente con la anterior, el proceso
        # de confirmación vuelve a empezar.
        if state.pending_rise_value is None or abs(value - state.pending_rise_value) > 0.20:
            state.pending_rise_value = value
            state.pending_rise_count = 1
            return

        # Cuando varias lecturas consecutivas confirman el mismo nuevo nivel,
        # se acepta el cambio y se reinicia la ventana con esa distancia.
        state.pending_rise_count += 1
        state.pending_rise_value = 0.5 * (state.pending_rise_value + value)
        if state.pending_rise_count >= self.rise_confirm_count:
            state.values.clear()
            state.values.append(float(state.pending_rise_value))
            state.stamp = stamp
            state.pending_rise_value = None
            state.pending_rise_count = 0

    # ------------------------------------------------------------------
    # Publicación periódica del estado filtrado
    # ------------------------------------------------------------------
    # Se obtiene el valor actual de cada lateral, se actualiza la histéresis
    # clear/occupied y se publican los topics consumidos por el controlador.
    def timer_callback(self) -> None:
        now_sec = self.now_seconds()
        left_dist, left_valid = self.current_value(self.left, now_sec)
        right_dist, right_valid = self.current_value(self.right, now_sec)

        # Actualización del estado lógico libre/ocupado con histéresis.
        self.left.clear = self.update_clear(left_dist, left_valid, self.left.clear)
        self.right.clear = self.update_clear(right_dist, right_valid, self.right.clear)

        # Cuando un sensor no es válido se publica 999.0 como valor centinela.
        # La validez real se comunica de forma independiente mediante Bool.
        self.left_dist_pub.publish(Float64(data=left_dist if left_valid else 999.0))
        self.right_dist_pub.publish(Float64(data=right_dist if right_valid else 999.0))
        self.left_valid_pub.publish(Bool(data=left_valid))
        self.right_valid_pub.publish(Bool(data=right_valid))
        self.left_clear_pub.publish(Bool(data=self.left.clear))
        self.right_clear_pub.publish(Bool(data=self.right.clear))

        # Mensaje de depuración con distancia, validez y estado clear de ambos lados.
        debug = (
            f'left={left_dist:.2f} valid={left_valid} clear={self.left.clear} | '
            f'right={right_dist:.2f} valid={right_valid} clear={self.right.clear}'
        )
        self.debug_pub.publish(String(data=debug))
        now = self.get_clock().now()
        if (now - self.last_debug_time).nanoseconds * 1e-9 > 0.75:
            self.get_logger().info(debug)
            self.last_debug_time = now

    # Obtiene la estimación actual del sensor.
    # Si la última medida es demasiado antigua o no existe historial, se borra
    # el estado acumulado y se devuelve (inf, False).
    def current_value(self, state: SideState, now_sec: float) -> tuple[float, bool]:
        if state.stamp is None or (now_sec - state.stamp) > self.timeout_s or not state.values:
            state.values.clear()
            state.pending_rise_value = None
            state.pending_rise_count = 0
            state.clear = False
            return math.inf, False
        return float(np.median(np.asarray(state.values, dtype=np.float32))), True

    # ------------------------------------------------------------------
    # Histéresis de ocupación
    # ------------------------------------------------------------------
    # La histéresis evita oscilaciones cuando la distancia se encuentra cerca
    # de un único umbral:
    #
    #   distancia <= occupied_threshold_m  -> ocupado
    #   distancia >= clear_threshold_m     -> libre
    #   zona intermedia                    -> mantiene estado anterior
    def update_clear(self, distance: float, valid: bool, previous: bool) -> bool:
        if not valid:
            return False
        if distance <= self.occupied_threshold_m:
            return False
        if distance >= self.clear_threshold_m:
            return True
        return previous

    # Conversión del reloj ROS a segundos en coma flotante.
    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


# ============================================================================
# Punto de entrada del nodo
# ============================================================================
# Inicializa ROS 2, crea el filtro y mantiene el nodo activo con rclpy.spin().
# Durante el cierre se contempla además un RuntimeError específico observado
# en ROS 2 Jazzy durante la destrucción de algunas suscripciones tras SIGINT.
# ============================================================================

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
