#!/usr/bin/env python3
"""Controlador robusto de evasión sin mapa para un bastón robótico impulsado
por el propio usuario.

El bastón no dispone de freno. El usuario continúa proporcionando la propulsión y el controlador solo
puede modificar la dirección de la rueda delantera y solicitar avisos hápticos.

La máquina de estados incorpora las salvaguardas que resultaron útiles en el
controlador previo basado en odometría, pero toda la geometría utiliza ahora
/local_odom y arcos locales previamente comprobados frente a colisiones. Las
principales protecciones son:

* un sensor lateral no considera que ha detectado el obstáculo únicamente porque
  su distancia absoluta esté por debajo de un umbral; debe observarse además una
  caída persistente respecto a la línea base tomada al inicio de la maniobra;
* el obstáculo no puede declararse perdido hasta que haya sido realmente detectado
  y la condición de liberación se mantenga durante un tiempo suficiente;
* la transición a PASS_OBSTACLE no se permite únicamente porque un sensor lateral
  pase a ser inválido;
* se utiliza una referencia de obstáculo de corta duración procedente del planner
  RGB-D cuando el obstáculo abandona el campo de visión frontal;
* mientras el obstáculo seguido no haya quedado detrás del robot, se prohíbe girar
  hacia él y se mantiene una orden mínima de alejamiento o paralelismo;
* el retorno a la línea de referencia solo comienza cuando existe evidencia
  positiva de que el robot completo ha superado el obstáculo.
"""

# ============================================================================
# Importaciones
# ============================================================================
# El controlador combina lógica de máquina de estados, geometría planar,
# selección de trayectorias locales y comunicación ROS 2.
# ============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import (
    Bool, Float32MultiArray, Float64, Float64MultiArray, String, UInt8,
)
from std_srvs.srv import Trigger


# Limita un valor al intervalo indicado para mantener órdenes y magnitudes
# dentro de rangos seguros.

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Normaliza un ángulo al intervalo [-pi, pi].

def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# Extrae el ángulo de guiñada de un cuaternión de orientación ROS.

def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================================
# Representación de un arco candidato
# ============================================================================
# Cada candidato recibido del planner contiene dirección, validez, clearance,
# cobertura observada, distancia de colisión y métricas adicionales usadas en
# la selección de la trayectoria.
# ============================================================================

@dataclass
class ArcCandidate:
    steer: float
    valid: bool
    clearance: float
    observed_ratio: float
    collision_distance: float
    score: float
    far_clearance: float
    tail_clearance: float


# ============================================================================
# Controlador principal de evasión
# ============================================================================
# Implementa una máquina de estados que mantiene siempre la prioridad del
# usuario y solo interviene sobre la dirección cuando existe riesgo u obstáculo.
# ============================================================================

class MaplessBypassControllerNode(Node):
    USER_GUIDED = 'USER_GUIDED'
    MANUAL_TURN = 'MANUAL_TURN'
    AVOID_OPEN = 'AVOID_OPEN'
    PASS_OBSTACLE = 'PASS_OBSTACLE'
    RETURN_LINE = 'RETURN_LINE'

    HAPTIC_NONE = 0
    HAPTIC_LEFT = 1
    HAPTIC_RIGHT = 2
    HAPTIC_SLOW = 3
    HAPTIC_DANGER = 4
    HAPTIC_RETURN_COMPLETE = 5
    HAPTIC_LOCALIZATION_LOST = 6

    CANDIDATE_WIDTH = 8
    TRACK_WIDTH = 5  # valid, x_local, y_local, radius, quality

    def __init__(self) -> None:
        super().__init__('mapless_bypass_controller_node')

        # ------------------------------------------------------------------
        # Interfaces ROS y temporización
        # ------------------------------------------------------------------
        # Define topics de odometría, dirección, interacción humana, latidos de
        # hardware y temporizaciones de control y vigilancia.
        # Interfaces and timing.
        self.declare_parameter('odom_topic', '/local_odom')
        self.declare_parameter('steering_output_topic', '/assist_steering_cmd')
        self.declare_parameter('human_steer_topic', '/hardware/human_steer_cmd')
        self.declare_parameter('human_push_topic', '/hardware/user_push_hint')
        self.declare_parameter('manual_turn_active_topic', '/hardware/manual_turn_active')
        self.declare_parameter('reset_local_odom_service', '/reset_local_odom')
        self.declare_parameter('control_rate_hz', 25.0)
        self.declare_parameter('sensor_timeout_s', 1.20)
        self.declare_parameter('odom_timeout_s', 0.80)
        self.declare_parameter('track_timeout_s', 1.00)
        self.declare_parameter('startup_grace_s', 4.0)
        self.declare_parameter('min_odom_confidence', 0.30)
        # The strict validity bit from visual odometry may flicker for one or two
        # frames even though the pose is still usable over a short manoeuvre.
        # Keep a recent reliable pose snapshot and add temporal hysteresis before
        # abandoning the captured line.
        self.declare_parameter('line_capture_min_confidence', 0.24)
        self.declare_parameter('line_snapshot_max_age_s', 1.60)
        self.declare_parameter('localization_bad_grace_s', 1.20)
        self.declare_parameter('localization_line_invalidate_s', 4.00)
        self.declare_parameter('localization_line_abandon_s', 12.0)
        self.declare_parameter('human_override_deadband_rad', 0.16)
        self.declare_parameter('manual_turn_release_deadband_rad', 0.06)
        self.declare_parameter('manual_turn_stable_time_s', 0.65)
        self.declare_parameter('manual_turn_min_forward_speed_m_s', 0.05)
        self.declare_parameter('manual_turn_max_yaw_rate_rad_s', 0.14)
        self.declare_parameter('manual_turn_post_reset_settle_s', 0.40)
        self.declare_parameter('manual_turn_reset_timeout_s', 1.50)
        self.declare_parameter('reset_odom_after_manual_turn', True)
        self.declare_parameter('initial_route_capture_enabled', True)
        self.declare_parameter('initial_route_capture_delay_s', 0.35)

        # ------------------------------------------------------------------
        # Geometría del robot y límites del servo
        # ------------------------------------------------------------------
        # Estos parámetros fijan dimensiones, margen de seguridad y límites de
        # magnitud y velocidad de cambio de la dirección.
        # Robot and servo.
        self.declare_parameter('wheelbase_m', 0.275)
        self.declare_parameter('robot_width_m', 0.30)
        self.declare_parameter('safety_margin_m', 0.12)
        self.declare_parameter('max_steer_rad', 0.50)
        self.declare_parameter('max_steer_rate_rad_s', 1.30)
        self.declare_parameter('steer_filter_tau_s', 0.08)

        # ------------------------------------------------------------------
        # Estimación de trayectoria estable del usuario
        # ------------------------------------------------------------------
        # Antes de una evasión se identifica un tramo de avance suficientemente
        # recto para definir una línea de referencia persistente.
        # Stable user trajectory before the obstacle.
        self.declare_parameter('stable_heading_yaw_rate_rad_s', 0.18)
        self.declare_parameter('stable_heading_min_speed_m_s', 0.04)
        self.declare_parameter('stable_heading_time_s', 0.28)
        self.declare_parameter('stable_heading_max_age_s', 2.0)

        # ------------------------------------------------------------------
        # Detección frontal e inicio de evasión
        # ------------------------------------------------------------------
        # Regula cuándo se activa la asistencia y cómo se abre progresivamente
        # la trayectoria para separarse del obstáculo.
        # Detection and opening.
        self.declare_parameter('minimum_trigger_distance_m', 0.75)
        self.declare_parameter('emergency_distance_m', 0.38)
        self.declare_parameter('open_initial_steer_rad', 0.18)
        self.declare_parameter('open_steer_rad', 0.28)
        self.declare_parameter('open_hold_steer_rad', 0.07)
        self.declare_parameter('open_ramp_time_s', 0.70)
        self.declare_parameter('open_ramp_progress_m', 0.30)
        self.declare_parameter('open_edge_follow_when_seen', True)
        self.declare_parameter('open_candidate_extra_steer_rad', 0.08)
        self.declare_parameter('open_candidate_desired_weight', 6.5)
        self.declare_parameter('open_min_time_s', 0.45)
        self.declare_parameter('open_min_progress_m', 0.24)
        self.declare_parameter('open_lateral_min_m', 0.34)
        self.declare_parameter('open_heading_min_rad', 0.24)
        self.declare_parameter('open_lateral_max_m', 0.72)
        self.declare_parameter('open_heading_max_rad', 0.62)
        self.declare_parameter('front_clear_for_pass_m', 1.05)
        self.declare_parameter('front_clear_confirm_s', 0.24)
        self.declare_parameter('pass_entry_obstacle_x_m', 0.45)

        # ------------------------------------------------------------------
        # Selección del lado de evasión
        # ------------------------------------------------------------------
        # La decisión izquierda/derecha se basa en la calidad global de cada
        # corredor y puede cambiar antes de que la maniobra quede comprometida.
        # Compare complete left/right corridor quality.  The selected side may
        # still change before commitment when the opposite gap is clearly wider.
        self.declare_parameter('side_summary_timeout_s', 1.20)
        self.declare_parameter('side_choice_score_margin', 0.18)
        self.declare_parameter('side_choice_switch_margin', 0.55)
        self.declare_parameter('side_choice_switch_confirm_s', 0.18)
        self.declare_parameter('side_choice_switch_max_time_s', 3.0)
        self.declare_parameter('side_choice_max_switches', 1)
        self.declare_parameter('side_choice_commit_lateral_m', 0.32)
        self.declare_parameter('side_choice_commit_heading_rad', 0.46)
        self.declare_parameter('side_choice_sensor_soft_m', 0.52)
        self.declare_parameter('side_choice_sensor_hard_m', 0.30)
        self.declare_parameter('side_choice_line_bias_weight', 0.55)
        self.declare_parameter('side_choice_line_bias_score_window', 0.85)

        # ------------------------------------------------------------------
        # Detección lateral relativa a línea base
        # ------------------------------------------------------------------
        # Los sensores laterales deben mostrar una caída real respecto a la
        # distancia inicial para confirmar que el obstáculo está siendo bordeado.
        # Baseline-relative side detection.  The release threshold is deliberately
        # above the absolute detection threshold, creating real hysteresis.
        self.declare_parameter('side_seen_drop_m', 0.18)
        self.declare_parameter('side_seen_absolute_m', 0.72)
        self.declare_parameter('side_seen_confirm_s', 0.12)
        self.declare_parameter('side_release_absolute_m', 0.92)
        self.declare_parameter('side_release_drop_m', 0.22)
        self.declare_parameter('side_lost_confirm_s', 0.30)
        # A short-range side sensor often loses its echo immediately after the
        # rear edge of a compact obstacle.  Version 5 accepts that dropout only
        # when the obstacle was genuinely seen, the distance reached a minimum,
        # then rose by a meaningful amount before the echo disappeared.
        self.declare_parameter('side_exit_rise_from_min_m', 0.16)
        self.declare_parameter('side_exit_min_last_distance_m', 0.48)
        self.declare_parameter('side_exit_invalid_confirm_s', 0.25)
        self.declare_parameter('side_exit_last_valid_max_age_s', 0.85)
        self.declare_parameter('side_target_distance_m', 0.46)
        self.declare_parameter('side_follow_gain', 0.48)
        self.declare_parameter('parallel_heading_gain', 0.52)
        self.declare_parameter('side_hard_distance_m', 0.24)
        self.declare_parameter('side_soft_distance_m', 0.38)
        self.declare_parameter('side_keepaway_steer_rad', 0.12)

        # ------------------------------------------------------------------
        # Superación del obstáculo
        # ------------------------------------------------------------------
        # Durante PASS_OBSTACLE se mantiene separación lateral y se utiliza la
        # memoria geométrica de corta duración como evidencia complementaria.
        # Passing with short-lived obstacle memory.
        self.declare_parameter('pass_hold_away_steer_rad', 0.11)
        self.declare_parameter('pass_max_toward_object_steer_rad', 0.02)
        self.declare_parameter('pass_rear_clearance_m', 0.30)
        self.declare_parameter('pass_min_total_progress_m', 0.75)
        self.declare_parameter('pass_unconfirmed_warning_progress_m', 1.45)
        self.declare_parameter('tracked_obstacle_min_quality', 0.25)

        # ------------------------------------------------------------------
        # Retorno a la línea de referencia
        # ------------------------------------------------------------------
        # Una vez superado el obstáculo se calcula una reentrada progresiva hacia
        # la línea previamente capturada.
        # Return to the frozen local line.
        self.declare_parameter('return_lookahead_min_m', 0.50)
        self.declare_parameter('return_lookahead_speed_gain_s', 0.65)
        self.declare_parameter('return_lookahead_max_m', 0.95)
        self.declare_parameter('return_heading_gain', 0.28)
        self.declare_parameter('return_stanley_gain', 1.15)
        self.declare_parameter('return_speed_softening_m_s', 0.18)
        self.declare_parameter('return_min_corrective_steer_rad', 0.12)
        self.declare_parameter('return_candidate_desired_weight', 7.5)
        self.declare_parameter('return_wrong_direction_penalty', 35.0)
        self.declare_parameter('return_progress_timeout_s', 2.6)
        self.declare_parameter('return_progress_epsilon_m', 0.025)
        self.declare_parameter('return_recovery_gain_multiplier', 1.35)
        self.declare_parameter('return_lateral_tolerance_m', 0.10)
        self.declare_parameter('return_heading_tolerance_rad', 0.18)
        self.declare_parameter('return_stable_time_s', 0.25)
        self.declare_parameter('return_replan_front_m', 1.05)
        self.declare_parameter('return_side_guard_soft_m', 0.50)
        self.declare_parameter('return_side_guard_hard_m', 0.28)
        self.declare_parameter('return_side_away_steer_rad', 0.12)
        # Version 9 uses a real lookahead point on the frozen route line and
        # evaluates the predicted endpoint of every candidate arc.  A blocked
        # return therefore becomes a bounded detour, not an arbitrary
        # least-risk arc that may continue increasing lateral error.
        self.declare_parameter('return_pure_pursuit_weight', 0.78)
        self.declare_parameter('return_prediction_min_m', 0.55)
        self.declare_parameter('return_prediction_speed_gain_s', 0.65)
        self.declare_parameter('return_prediction_max_m', 1.05)
        self.declare_parameter('return_future_lateral_weight', 8.0)
        self.declare_parameter('return_future_heading_weight', 2.2)
        self.declare_parameter('return_error_growth_penalty', 28.0)
        self.declare_parameter('return_max_error_growth_m', 0.16)
        self.declare_parameter('return_detour_line_bias_weight', 3.2)
        self.declare_parameter('return_detour_max_gap_disadvantage', 1.10)
        self.declare_parameter('return_detour_max_extra_lateral_m', 0.30)
        self.declare_parameter('return_detour_parallel_steer_rad', 0.06)

        # ------------------------------------------------------------------
        # Límite de giro autónomo
        # ------------------------------------------------------------------
        # El bastón no debe ejecutar por sí solo un giro amplio o cambio de rumbo.
        # Si la desviación angular aumenta demasiado, la asistencia se limita.
        # A passive cane must never autonomously perform a U-turn.  The guard
        # only limits the servo; it does not brake or remove propulsion from
        # the user.  Larger route changes remain exclusive to MANUAL_TURN.
        self.declare_parameter('autonomous_turn_soft_limit_rad', 1.00)
        self.declare_parameter('autonomous_turn_hard_limit_rad', 1.30)
        self.declare_parameter('autonomous_turn_unwind_steer_rad', 0.18)
        self.declare_parameter('autonomous_turn_front_guard_m', 0.48)

        # ------------------------------------------------------------------
        # Selección de arcos candidatos
        # ------------------------------------------------------------------
        # Combina cercanía al comando deseado, clearance, observación, distancia
        # de colisión y continuidad respecto al comando anterior.
        # Candidate selection.
        self.declare_parameter('desired_steer_weight', 2.8)
        self.declare_parameter('clearance_reward_weight', 0.90)
        self.declare_parameter('observed_reward_weight', 0.34)
        self.declare_parameter('collision_reward_weight', 0.30)
        self.declare_parameter('steer_change_weight', 0.38)
        self.declare_parameter('wrong_side_penalty', 40.0)
        self.declare_parameter('min_straight_observed_ratio', 0.18)
        self.declare_parameter('min_fallback_observed_ratio', 0.10)
        self.declare_parameter('last_safe_steer_hold_s', 1.20)
        self.declare_parameter('slow_speed_threshold_m_s', 0.45)
        self.declare_parameter('emergency_side_soft_m', 0.48)
        self.declare_parameter('emergency_side_hard_m', 0.26)
        self.declare_parameter('emergency_min_escape_steer_rad', 0.20)

        # El latido de hardware puede exigirse en el sistema real y desactivarse
        # durante simulación.
        # Hardware heartbeat is optional in simulation.
        self.declare_parameter('require_hardware_alive', False)
        self.declare_parameter('hardware_timeout_s', 0.50)

        # ------------------------------------------------------------------
        # Lectura y almacenamiento de parámetros ROS
        # ------------------------------------------------------------------
        # Read parameters.
        gp = lambda name: self.get_parameter(name).value
        self.odom_topic = str(gp('odom_topic'))
        self.steering_output_topic = str(gp('steering_output_topic'))
        self.human_steer_topic = str(gp('human_steer_topic'))
        self.human_push_topic = str(gp('human_push_topic'))
        self.manual_turn_active_topic = str(gp('manual_turn_active_topic'))
        self.reset_local_odom_service = str(gp('reset_local_odom_service'))
        self.control_rate_hz = float(gp('control_rate_hz'))
        self.sensor_timeout_s = float(gp('sensor_timeout_s'))
        self.odom_timeout_s = float(gp('odom_timeout_s'))
        self.track_timeout_s = float(gp('track_timeout_s'))
        self.startup_grace_s = float(gp('startup_grace_s'))
        self.min_odom_confidence = float(gp('min_odom_confidence'))
        self.line_capture_min_confidence = float(gp('line_capture_min_confidence'))
        self.line_snapshot_max_age_s = float(gp('line_snapshot_max_age_s'))
        self.localization_bad_grace_s = float(gp('localization_bad_grace_s'))
        self.localization_line_invalidate_s = float(gp('localization_line_invalidate_s'))
        self.localization_line_abandon_s = float(gp('localization_line_abandon_s'))
        self.human_override_deadband_rad = float(gp('human_override_deadband_rad'))
        self.manual_turn_release_deadband_rad = float(gp('manual_turn_release_deadband_rad'))
        self.manual_turn_stable_time_s = float(gp('manual_turn_stable_time_s'))
        self.manual_turn_min_forward_speed_m_s = float(gp('manual_turn_min_forward_speed_m_s'))
        self.manual_turn_max_yaw_rate_rad_s = float(gp('manual_turn_max_yaw_rate_rad_s'))
        self.manual_turn_post_reset_settle_s = float(gp('manual_turn_post_reset_settle_s'))
        self.manual_turn_reset_timeout_s = float(gp('manual_turn_reset_timeout_s'))
        self.reset_odom_after_manual_turn = bool(gp('reset_odom_after_manual_turn'))
        self.initial_route_capture_enabled = bool(gp('initial_route_capture_enabled'))
        self.initial_route_capture_delay_s = float(gp('initial_route_capture_delay_s'))

        self.wheelbase_m = float(gp('wheelbase_m'))
        self.robot_width_m = float(gp('robot_width_m'))
        self.safety_margin_m = float(gp('safety_margin_m'))
        self.max_steer_rad = float(gp('max_steer_rad'))
        self.max_steer_rate_rad_s = float(gp('max_steer_rate_rad_s'))
        self.steer_filter_tau_s = float(gp('steer_filter_tau_s'))

        self.stable_heading_yaw_rate_rad_s = float(gp('stable_heading_yaw_rate_rad_s'))
        self.stable_heading_min_speed_m_s = float(gp('stable_heading_min_speed_m_s'))
        self.stable_heading_time_s = float(gp('stable_heading_time_s'))
        self.stable_heading_max_age_s = float(gp('stable_heading_max_age_s'))

        self.minimum_trigger_distance_m = float(gp('minimum_trigger_distance_m'))
        self.emergency_distance_m = float(gp('emergency_distance_m'))
        self.open_initial_steer_rad = float(gp('open_initial_steer_rad'))
        self.open_steer_rad = float(gp('open_steer_rad'))
        self.open_hold_steer_rad = float(gp('open_hold_steer_rad'))
        self.open_ramp_time_s = float(gp('open_ramp_time_s'))
        self.open_ramp_progress_m = float(gp('open_ramp_progress_m'))
        self.open_edge_follow_when_seen = bool(gp('open_edge_follow_when_seen'))
        self.open_candidate_extra_steer_rad = float(gp('open_candidate_extra_steer_rad'))
        self.open_candidate_desired_weight = float(gp('open_candidate_desired_weight'))
        self.open_min_time_s = float(gp('open_min_time_s'))
        self.open_min_progress_m = float(gp('open_min_progress_m'))
        self.open_lateral_min_m = float(gp('open_lateral_min_m'))
        self.open_heading_min_rad = float(gp('open_heading_min_rad'))
        self.open_lateral_max_m = float(gp('open_lateral_max_m'))
        self.open_heading_max_rad = float(gp('open_heading_max_rad'))
        self.front_clear_for_pass_m = float(gp('front_clear_for_pass_m'))
        self.front_clear_confirm_s = float(gp('front_clear_confirm_s'))
        self.pass_entry_obstacle_x_m = float(gp('pass_entry_obstacle_x_m'))
        self.side_summary_timeout_s = float(gp('side_summary_timeout_s'))
        self.side_choice_score_margin = float(gp('side_choice_score_margin'))
        self.side_choice_switch_margin = float(gp('side_choice_switch_margin'))
        self.side_choice_switch_confirm_s = float(gp('side_choice_switch_confirm_s'))
        self.side_choice_switch_max_time_s = float(gp('side_choice_switch_max_time_s'))
        self.side_choice_max_switches = max(0, int(gp('side_choice_max_switches')))
        self.side_choice_commit_lateral_m = float(gp('side_choice_commit_lateral_m'))
        self.side_choice_commit_heading_rad = float(gp('side_choice_commit_heading_rad'))
        self.side_choice_sensor_soft_m = float(gp('side_choice_sensor_soft_m'))
        self.side_choice_sensor_hard_m = float(gp('side_choice_sensor_hard_m'))
        self.side_choice_line_bias_weight = float(gp('side_choice_line_bias_weight'))
        self.side_choice_line_bias_score_window = float(gp('side_choice_line_bias_score_window'))

        self.side_seen_drop_m = float(gp('side_seen_drop_m'))
        self.side_seen_absolute_m = float(gp('side_seen_absolute_m'))
        self.side_seen_confirm_s = float(gp('side_seen_confirm_s'))
        self.side_release_absolute_m = float(gp('side_release_absolute_m'))
        self.side_release_drop_m = float(gp('side_release_drop_m'))
        self.side_lost_confirm_s = float(gp('side_lost_confirm_s'))
        self.side_exit_rise_from_min_m = float(gp('side_exit_rise_from_min_m'))
        self.side_exit_min_last_distance_m = float(gp('side_exit_min_last_distance_m'))
        self.side_exit_invalid_confirm_s = float(gp('side_exit_invalid_confirm_s'))
        self.side_exit_last_valid_max_age_s = float(gp('side_exit_last_valid_max_age_s'))
        self.side_target_distance_m = float(gp('side_target_distance_m'))
        self.side_follow_gain = float(gp('side_follow_gain'))
        self.parallel_heading_gain = float(gp('parallel_heading_gain'))
        self.side_hard_distance_m = float(gp('side_hard_distance_m'))
        self.side_soft_distance_m = float(gp('side_soft_distance_m'))
        self.side_keepaway_steer_rad = float(gp('side_keepaway_steer_rad'))

        self.pass_hold_away_steer_rad = float(gp('pass_hold_away_steer_rad'))
        self.pass_max_toward_object_steer_rad = float(gp('pass_max_toward_object_steer_rad'))
        self.pass_rear_clearance_m = float(gp('pass_rear_clearance_m'))
        self.pass_min_total_progress_m = float(gp('pass_min_total_progress_m'))
        self.pass_unconfirmed_warning_progress_m = float(gp('pass_unconfirmed_warning_progress_m'))
        self.tracked_obstacle_min_quality = float(gp('tracked_obstacle_min_quality'))

        self.return_lookahead_min_m = float(gp('return_lookahead_min_m'))
        self.return_lookahead_speed_gain_s = float(gp('return_lookahead_speed_gain_s'))
        self.return_lookahead_max_m = float(gp('return_lookahead_max_m'))
        self.return_heading_gain = float(gp('return_heading_gain'))
        self.return_stanley_gain = float(gp('return_stanley_gain'))
        self.return_speed_softening_m_s = float(gp('return_speed_softening_m_s'))
        self.return_min_corrective_steer_rad = float(gp('return_min_corrective_steer_rad'))
        self.return_candidate_desired_weight = float(gp('return_candidate_desired_weight'))
        self.return_wrong_direction_penalty = float(gp('return_wrong_direction_penalty'))
        self.return_progress_timeout_s = float(gp('return_progress_timeout_s'))
        self.return_progress_epsilon_m = float(gp('return_progress_epsilon_m'))
        self.return_recovery_gain_multiplier = float(gp('return_recovery_gain_multiplier'))
        self.return_lateral_tolerance_m = float(gp('return_lateral_tolerance_m'))
        self.return_heading_tolerance_rad = float(gp('return_heading_tolerance_rad'))
        self.return_stable_time_s = float(gp('return_stable_time_s'))
        self.return_replan_front_m = float(gp('return_replan_front_m'))
        self.return_side_guard_soft_m = float(gp('return_side_guard_soft_m'))
        self.return_side_guard_hard_m = float(gp('return_side_guard_hard_m'))
        self.return_side_away_steer_rad = float(gp('return_side_away_steer_rad'))
        self.return_pure_pursuit_weight = float(gp('return_pure_pursuit_weight'))
        self.return_prediction_min_m = float(gp('return_prediction_min_m'))
        self.return_prediction_speed_gain_s = float(gp('return_prediction_speed_gain_s'))
        self.return_prediction_max_m = float(gp('return_prediction_max_m'))
        self.return_future_lateral_weight = float(gp('return_future_lateral_weight'))
        self.return_future_heading_weight = float(gp('return_future_heading_weight'))
        self.return_error_growth_penalty = float(gp('return_error_growth_penalty'))
        self.return_max_error_growth_m = float(gp('return_max_error_growth_m'))
        self.return_detour_line_bias_weight = float(gp('return_detour_line_bias_weight'))
        self.return_detour_max_gap_disadvantage = float(gp('return_detour_max_gap_disadvantage'))
        self.return_detour_max_extra_lateral_m = float(gp('return_detour_max_extra_lateral_m'))
        self.return_detour_parallel_steer_rad = float(gp('return_detour_parallel_steer_rad'))
        self.autonomous_turn_soft_limit_rad = float(gp('autonomous_turn_soft_limit_rad'))
        self.autonomous_turn_hard_limit_rad = float(gp('autonomous_turn_hard_limit_rad'))
        self.autonomous_turn_unwind_steer_rad = float(gp('autonomous_turn_unwind_steer_rad'))
        self.autonomous_turn_front_guard_m = float(gp('autonomous_turn_front_guard_m'))

        self.desired_steer_weight = float(gp('desired_steer_weight'))
        self.clearance_reward_weight = float(gp('clearance_reward_weight'))
        self.observed_reward_weight = float(gp('observed_reward_weight'))
        self.collision_reward_weight = float(gp('collision_reward_weight'))
        self.steer_change_weight = float(gp('steer_change_weight'))
        self.wrong_side_penalty = float(gp('wrong_side_penalty'))
        self.min_straight_observed_ratio = float(gp('min_straight_observed_ratio'))
        self.min_fallback_observed_ratio = float(gp('min_fallback_observed_ratio'))
        self.last_safe_steer_hold_s = float(gp('last_safe_steer_hold_s'))
        self.slow_speed_threshold_m_s = float(gp('slow_speed_threshold_m_s'))
        self.emergency_side_soft_m = float(gp('emergency_side_soft_m'))
        self.emergency_side_hard_m = float(gp('emergency_side_hard_m'))
        self.emergency_min_escape_steer_rad = float(gp('emergency_min_escape_steer_rad'))
        self.require_hardware_alive = bool(gp('require_hardware_alive'))
        self.hardware_timeout_s = float(gp('hardware_timeout_s'))

        # ------------------------------------------------------------------
        # Estado de pose, entradas humanas y percepción
        # ------------------------------------------------------------------
        # Pose and input state.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.have_odom = False
        self.odom_valid = False
        self.odom_confidence = 0.0
        self.odom_stamp: Optional[float] = None
        self.human_steer = 0.0
        self.human_push = 0.0
        self.manual_turn_active = False

        self.candidates: list[ArcCandidate] = []
        self.candidates_stamp: Optional[float] = None
        self.front_collision_distance = math.inf
        self.trigger_distance = self.minimum_trigger_distance_m
        self.arc_emergency = False

        # Resumen agregado de la calidad de los corredores izquierdo y derecho.
        # Planner side summary: left/right aggregate scores, far and tail gap
        # clearances, best steering and viable fractions.
        self.side_summary = [-1e6, -1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.side_summary_stamp: Optional[float] = None
        self.side_switch_candidate_sign = 0
        self.side_switch_candidate_since: Optional[float] = None
        self.side_switch_count = 0
        self.last_side_choice_left_score = -1e6
        self.last_side_choice_right_score = -1e6

        self.side_left_dist = math.inf
        self.side_right_dist = math.inf
        self.side_left_valid = False
        self.side_right_valid = False
        self.side_left_clear = False
        self.side_right_clear = False

        # Estado del obstáculo seguido por el planner y copia bloqueada al inicio
        # de la maniobra para mantener una referencia consistente.
        # Current obstacle track from planner and locked track for the manoeuvre.
        self.track_valid = False
        self.track_x_local = 0.0
        self.track_y_local = 0.0
        self.track_radius = 0.0
        self.track_quality = 0.0
        self.track_stamp: Optional[float] = None
        self.locked_obstacle_valid = False
        self.locked_obstacle_x_local = 0.0
        self.locked_obstacle_y_local = 0.0
        self.locked_obstacle_radius = 0.0

        # Línea de referencia del tramo actual y memoria de rumbo estable.
        # Reference line and stable heading.
        self.line_valid = False
        self.line_x0 = 0.0
        self.line_y0 = 0.0
        self.line_theta = 0.0
        self.line_progress = 0.0
        self.lateral_error = 0.0
        self.heading_error = 0.0
        self.stable_heading = 0.0
        self.stable_heading_since: Optional[float] = None
        self.stable_heading_stamp: Optional[float] = None
        self.reference_snapshot_valid = False
        self.reference_snapshot_x = 0.0
        self.reference_snapshot_y = 0.0
        self.reference_snapshot_yaw = 0.0
        self.reference_snapshot_stamp: Optional[float] = None
        self.localization_bad_since: Optional[float] = None
        self.route_segment_id = 0
        self.route_line_source = 'NONE'
        self.initial_route_captured = False
        self.initial_route_ready_since: Optional[float] = None

        # Estado dedicado al giro manual: durante esta fase el usuario tiene
        # prioridad exclusiva y la línea previa deja de ser válida.
        # Explicit manual-turn/rebase state.  A route line is never replaced by
        # an obstacle; only startup and a completed human turn may create one.
        self.manual_turn_started_at: Optional[float] = None
        self.manual_turn_release_since: Optional[float] = None
        self.manual_turn_reset_future = None
        self.manual_turn_reset_requested_at: Optional[float] = None
        self.manual_turn_post_reset_until: Optional[float] = None
        self.manual_turn_waiting_for_fresh_odom = False

        # Estado actual de la máquina y memoria temporal específica de la maniobra.
        # State machine and per-manoeuvre memory.
        self.state = self.USER_GUIDED
        self.state_start_time = self.now_seconds()
        self.state_start_x = 0.0
        self.state_start_y = 0.0
        self.avoidance_start_x = 0.0
        self.avoidance_start_y = 0.0
        self.avoidance_start_progress = 0.0
        self.avoidance_start_lateral = 0.0
        self.avoidance_start_heading = 0.0
        self.avoid_sign = 0
        self.object_side = 'UNKNOWN'
        self.reactive_only_mode = False

        self.side_baseline_valid = False
        self.side_baseline_dist = math.inf
        self.side_min_dist = math.inf
        self.side_seen = False
        self.side_seen_since: Optional[float] = None
        self.side_lost_since: Optional[float] = None
        self.side_last_valid_dist = math.inf
        self.side_last_valid_stamp: Optional[float] = None
        self.side_invalid_since: Optional[float] = None
        self.side_exit_confirmed = False
        self.side_exit_reason = ''
        self.front_clear_since: Optional[float] = None
        self.return_stable_since: Optional[float] = None
        self.return_best_abs_error = math.inf
        self.return_last_improvement_stamp: Optional[float] = None
        self.return_path_blocked = False
        self.return_recovery_active = False
        self.avoidance_from_return = False
        self.return_target_valid = False
        self.return_target_x = 0.0
        self.return_target_y = 0.0
        self.return_target_theta = 0.0
        self.return_target_lookahead = 0.0
        self.autonomous_heading_anchor_valid = False
        self.autonomous_heading_anchor = 0.0
        self.autonomous_turn_deviation = 0.0
        self.autonomous_turn_guard_active = False

        self.hardware_alive = True
        self.hardware_alive_stamp: Optional[float] = None

        self.theta_cmd = 0.0
        self.last_safe_steer = 0.0
        self.last_safe_steer_stamp: Optional[float] = None
        self.last_control_time = self.get_clock().now()
        self.last_debug_time = self.get_clock().now()
        self.startup_stamp = self.now_seconds()
        self.warning_active = False
        self.warning_reason = ''
        self.last_warning_reason = ''
        self.haptic_latch_until = 0.0
        self.haptic_latch_pattern = self.HAPTIC_NONE
        self.last_haptic_reported = -1

        # ------------------------------------------------------------------
        # Suscripciones ROS 2
        # ------------------------------------------------------------------
        # Reciben odometría, percepción del planner, sensores laterales, órdenes
        # humanas y estado del hardware.
        # Subscriptions.
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(Bool, '/local_odom_valid', self.odom_valid_callback, 10)
        self.create_subscription(Float64, '/local_odom_confidence', self.odom_confidence_callback, 10)
        self.create_subscription(Float64, self.human_steer_topic, self.human_steer_callback, 10)
        self.create_subscription(Float64, self.human_push_topic, self.human_push_callback, 10)
        self.create_subscription(Bool, self.manual_turn_active_topic, self.manual_turn_active_callback, 10)
        self.create_subscription(Float32MultiArray, '/local_arc_candidates', self.candidates_callback, 10)
        self.create_subscription(Float32MultiArray, '/arc_side_summary', self.side_summary_callback, 10)
        self.create_subscription(Float64, '/front_collision_distance', self.front_collision_callback, 10)
        self.create_subscription(Float64, '/avoidance_trigger_distance', self.trigger_callback, 10)
        self.create_subscription(Bool, '/arc_planner_emergency', self.emergency_callback, 10)
        self.create_subscription(Float32MultiArray, '/tracked_front_obstacle', self.track_callback, 10)
        self.create_subscription(Float64, '/side_left_dist', self.side_left_callback, 10)
        self.create_subscription(Float64, '/side_right_dist', self.side_right_callback, 10)
        self.create_subscription(Bool, '/side_left_valid', self.side_left_valid_callback, 10)
        self.create_subscription(Bool, '/side_right_valid', self.side_right_valid_callback, 10)
        self.create_subscription(Bool, '/side_left_clear', self.side_left_clear_callback, 10)
        self.create_subscription(Bool, '/side_right_clear', self.side_right_clear_callback, 10)
        self.create_subscription(Bool, '/hardware_alive', self.hardware_alive_callback, 10)

        # ------------------------------------------------------------------
        # Publicadores ROS 2
        # ------------------------------------------------------------------
        # Se publican dirección asistida, háptica, avisos de seguridad, estado de
        # la máquina y distintas variables de diagnóstico geométrico.
        # Publishers.
        self.steering_pub = self.create_publisher(Float64, self.steering_output_topic, 10)
        self.haptic_pub = self.create_publisher(UInt8, '/haptic_pattern', 10)
        self.stop_pub = self.create_publisher(Bool, '/stop_requested', 10)
        self.warning_pub = self.create_publisher(Bool, '/safety_warning', 10)
        self.warning_reason_pub = self.create_publisher(String, '/safety_warning_reason', 10)
        self.assist_pub = self.create_publisher(Bool, '/assist_active', 10)
        self.state_pub = self.create_publisher(String, '/mapless_bypass_state', 10)
        self.lateral_pub = self.create_publisher(Float64, '/lateral_error', 10)
        self.heading_pub = self.create_publisher(Float64, '/heading_error', 10)
        self.debug_pub = self.create_publisher(String, '/mapless_bypass_debug', 10)
        self.planner_clear_memory_pub = self.create_publisher(Bool, '/planner/clear_obstacle_memory', 10)
        self.route_segment_pub = self.create_publisher(UInt8, '/route_segment_id', 10)
        self.route_line_debug_pub = self.create_publisher(
            Float64MultiArray, '/debug/route_line', 10
        )
        self.return_target_debug_pub = self.create_publisher(
            Float64MultiArray, '/debug/return_target', 10
        )
        self.locked_obstacle_debug_pub = self.create_publisher(
            Float64MultiArray, '/debug/locked_obstacle', 10
        )
        self.autonomous_turn_debug_pub = self.create_publisher(
            Float64MultiArray, '/debug/autonomous_turn', 10
        )
        self.odom_reset_client = self.create_client(Trigger, self.reset_local_odom_service)

        self.timer = self.create_timer(1.0 / max(self.control_rate_hz, 1.0), self.control_callback)
        self.get_logger().info(
            'Mapless bypass controller v10 started in NO-BRAKE mode. '
            'Side detection is baseline-relative, rear-edge dropout is validated '
            'from a minimum-then-rising signature, and RGB-D obstacle memory is '
            'used as complementary evidence rather than a permanent return lock.'
        )

    # =========================================================================
    # Callbacks de entrada
    # =========================================================================
    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry) -> None:
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.speed = max(0.0, abs(float(msg.twist.twist.linear.x)))
        self.yaw_rate = float(msg.twist.twist.angular.z)
        self.have_odom = True
        self.odom_stamp = self.now_seconds()

    def odom_valid_callback(self, msg: Bool) -> None:
        self.odom_valid = bool(msg.data)

    def odom_confidence_callback(self, msg: Float64) -> None:
        self.odom_confidence = float(msg.data)

    def human_steer_callback(self, msg: Float64) -> None:
        self.human_steer = float(msg.data)

    def human_push_callback(self, msg: Float64) -> None:
        self.human_push = float(msg.data)

    def manual_turn_active_callback(self, msg: Bool) -> None:
        self.manual_turn_active = bool(msg.data)

    # Convierte el array plano recibido del planner en estructuras ArcCandidate.
    def candidates_callback(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) % self.CANDIDATE_WIDTH != 0:
            self.get_logger().warn(
                f'Invalid candidate array length: {len(data)}',
                throttle_duration_sec=2.0,
            )
            return
        parsed: list[ArcCandidate] = []
        for index in range(0, len(data), self.CANDIDATE_WIDTH):
            parsed.append(
                ArcCandidate(
                    steer=float(data[index]),
                    valid=bool(data[index + 1] > 0.5),
                    clearance=float(data[index + 2]),
                    observed_ratio=float(data[index + 3]),
                    collision_distance=float(data[index + 4]),
                    score=float(data[index + 5]),
                    far_clearance=float(data[index + 6]),
                    tail_clearance=float(data[index + 7]),
                )
            )
        self.candidates = parsed
        self.candidates_stamp = self.now_seconds()

    def side_summary_callback(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 10:
            return
        self.side_summary = [float(value) for value in data[:10]]
        self.side_summary_stamp = self.now_seconds()

    def front_collision_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        self.front_collision_distance = value if value > 0.0 else math.inf

    def trigger_callback(self, msg: Float64) -> None:
        self.trigger_distance = max(self.minimum_trigger_distance_m, float(msg.data))

    def emergency_callback(self, msg: Bool) -> None:
        self.arc_emergency = bool(msg.data)

    # Actualiza la referencia del obstáculo frontal seguido por el planner.
    def track_callback(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < self.TRACK_WIDTH:
            self.track_valid = False
            return
        self.track_valid = bool(data[0] > 0.5)
        self.track_x_local = float(data[1])
        self.track_y_local = float(data[2])
        self.track_radius = max(0.0, float(data[3]))
        self.track_quality = clamp(float(data[4]), 0.0, 1.0)
        self.track_stamp = self.now_seconds()

    def side_left_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        self.side_left_dist = math.inf if value >= 900.0 else value

    def side_right_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        self.side_right_dist = math.inf if value >= 900.0 else value

    def side_left_valid_callback(self, msg: Bool) -> None:
        self.side_left_valid = bool(msg.data)

    def side_right_valid_callback(self, msg: Bool) -> None:
        self.side_right_valid = bool(msg.data)

    def side_left_clear_callback(self, msg: Bool) -> None:
        self.side_left_clear = bool(msg.data)

    def side_right_clear_callback(self, msg: Bool) -> None:
        self.side_right_clear = bool(msg.data)

    def hardware_alive_callback(self, msg: Bool) -> None:
        self.hardware_alive = bool(msg.data)
        self.hardware_alive_stamp = self.now_seconds()

    # =========================================================================
    # Bucle principal de control
    # =========================================================================
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    # Ejecuta el ciclo de control:
    #   1. actualiza referencias y errores;
    #   2. comprueba frescura de percepción, odometría y hardware;
    #   3. respeta prioridad manual;
    #   4. ejecuta la lógica de la máquina de estados;
    #   5. limita y filtra la dirección;
    #   6. publica dirección, háptica y avisos.
    def control_callback(self) -> None:
        now = self.get_clock().now()
        now_sec = self.now_seconds()
        dt = clamp((now - self.last_control_time).nanoseconds * 1e-9, 1e-3, 0.12)
        self.last_control_time = now
        self.warning_active = False
        self.warning_reason = ''

        self.update_stable_heading(now_sec)
        self.update_line_errors()

        # Verificación de frescura de la percepción procedente del planner.
        perception_fresh = (
            self.candidates_stamp is not None
            and (now_sec - self.candidates_stamp) <= self.sensor_timeout_s
            and bool(self.candidates)
        )
        odom_fresh = (
            self.odom_stamp is not None
            and (now_sec - self.odom_stamp) <= self.odom_timeout_s
        )
        # Odometría considerada plenamente fiable para control geométrico.
        localization_good = (
            self.have_odom
            and odom_fresh
            and self.odom_valid
            and self.odom_confidence >= self.min_odom_confidence
        )
        # Umbral más permisivo usado para conservar referencias de trayectoria.
        localization_usable = (
            self.have_odom
            and odom_fresh
            and self.odom_confidence >= self.line_capture_min_confidence
        )
        assist_manoeuvre_active = self.state in (
            self.AVOID_OPEN,
            self.PASS_OBSTACLE,
            self.RETURN_LINE,
        )

        # Keep a recent pose snapshot for startup or explicit manual-turn
        # recovery.  It is never used to replace an existing route line when an
        # obstacle appears.
        if self.state == self.USER_GUIDED and localization_usable:
            self.reference_snapshot_valid = True
            self.reference_snapshot_x = self.x
            self.reference_snapshot_y = self.y
            self.reference_snapshot_yaw = self.yaw
            self.reference_snapshot_stamp = now_sec

        if localization_good:
            self.localization_bad_since = None
            if self.line_valid:
                self.reactive_only_mode = False
        elif assist_manoeuvre_active and self.localization_bad_since is None:
            self.localization_bad_since = now_sec

        hardware_fresh = (
            self.hardware_alive_stamp is not None
            and (now_sec - self.hardware_alive_stamp) <= self.hardware_timeout_s
            and self.hardware_alive
        )

        # Un giro manual explícito o una orden humana suficientemente grande
        # fuerza la transición al estado MANUAL_TURN.
        explicit_turn = self.manual_turn_active
        fallback_turn = (
            self.state != self.MANUAL_TURN
            and abs(self.human_steer) >= self.human_override_deadband_rad
        )

        # A u/o command starts a dedicated MANUAL_TURN phase.  Assistance stays
        # disabled until the user has centred the steering, moved forward with a
        # stable heading and the local odometry has been rebased.  Obstacles seen
        # during the corner cannot create a new line or restart avoidance.
        if explicit_turn or fallback_turn:
            if self.state != self.MANUAL_TURN:
                self.begin_manual_turn(now_sec)
            target = 0.0
            assist_active = False
            decision = 'MANUAL_TURN_HUMAN_HAS_PRIORITY'
            if self.arc_emergency or (
                math.isfinite(self.front_collision_distance)
                and self.front_collision_distance < self.emergency_distance_m
            ):
                self.set_warning('MANUAL_TURN_OBSTACLE_CLOSE_HUMAN_REDUCE_SPEED')
                self.latch_haptic(self.HAPTIC_DANGER, 0.30)
        elif self.state == self.MANUAL_TURN:
            target, decision, assist_active = self.handle_manual_turn(
                now_sec,
                localization_usable,
            )
        else:
            self.ensure_initial_route_line(now_sec, localization_usable)

            if self.require_hardware_alive and not hardware_fresh:
                self.set_warning('ESP32_OR_ACTUATOR_TIMEOUT')
                target = (
                    self.fallback_guidance_command(now_sec)
                    if assist_manoeuvre_active else 0.0
                )
                assist_active = assist_manoeuvre_active
                decision = 'WARN_ACTUATOR_TIMEOUT_NO_BRAKE'
                self.latch_haptic(self.HAPTIC_DANGER, 0.45)
            elif (
                not perception_fresh
                and (now_sec - self.startup_stamp) > self.startup_grace_s
            ):
                self.set_warning('PERCEPTION_TIMEOUT')
                target = (
                    self.fallback_guidance_command(now_sec)
                    if assist_manoeuvre_active else 0.0
                )
                assist_active = assist_manoeuvre_active
                decision = (
                    'WARN_PERCEPTION_TIMEOUT_HOLD_LAST_GUIDANCE'
                    if assist_manoeuvre_active
                    else 'WARN_PERCEPTION_TIMEOUT_USER_GUIDED'
                )
                self.latch_haptic(self.HAPTIC_DANGER, 0.45)
            elif assist_manoeuvre_active and not localization_good:
                bad_duration = (
                    0.0
                    if self.localization_bad_since is None
                    else now_sec - self.localization_bad_since
                )
                retain_line = (
                    self.line_valid
                    and odom_fresh
                    and bad_duration <= self.localization_line_invalidate_s
                )
                if retain_line:
                    target, inner_decision, assist_active = self.compute_state_control(
                        now_sec, True
                    )
                    if bad_duration <= self.localization_bad_grace_s:
                        decision = (
                            f'ODOM_VALIDITY_FLICKER_ROUTE_LINE_RETAINED::'
                            f'{inner_decision}'
                        )
                    else:
                        self.set_warning('LOCAL_ODOMETRY_DEGRADED_ROUTE_LINE_RETAINED')
                        decision = (
                            f'WARN_ODOM_DEGRADED_ROUTE_LINE_RETAINED::'
                            f'{inner_decision}'
                        )
                        self.latch_haptic(self.HAPTIC_LOCALIZATION_LOST, 0.25)
                else:
                    self.set_warning('LOCAL_ODOMETRY_UNRELIABLE_REACTIVE_ONLY')
                    if (
                        bad_duration > self.localization_line_abandon_s
                        or not odom_fresh
                    ):
                        self.reactive_only_mode = True
                    target = self.compute_reactive_safe_command(now_sec)
                    assist_active = True
                    decision = (
                        'WARN_LOCAL_ODOMETRY_LOST_REACTIVE_GUIDANCE_'
                        'ROUTE_LINE_SUSPENDED'
                    )
                    self.latch_haptic(self.HAPTIC_LOCALIZATION_LOST, 0.50)
            else:
                target, decision, assist_active = self.compute_state_control(
                    now_sec, localization_good
                )

        # Limitación final del objetivo y aplicación de salvaguarda frente a
        # giros autónomos excesivos.
        target = clamp(target, -self.max_steer_rad, self.max_steer_rad)
        target = self.apply_autonomous_turn_guard(target, assist_active, now_sec)
        self.theta_cmd = self.rate_limited_filter(target, self.theta_cmd, dt)

        haptic = self.current_haptic(now_sec)
        if self.warning_active and haptic == self.HAPTIC_NONE:
            haptic = self.HAPTIC_DANGER
        elif self.speed > self.slow_speed_threshold_m_s and assist_manoeuvre_active:
            haptic = max(haptic, self.HAPTIC_SLOW)

        self.steering_pub.publish(Float64(data=float(self.theta_cmd)))
        self.haptic_pub.publish(UInt8(data=int(haptic)))
        self.stop_pub.publish(Bool(data=False))
        self.warning_pub.publish(Bool(data=bool(self.warning_active)))
        self.warning_reason_pub.publish(String(data=self.warning_reason))
        self.assist_pub.publish(Bool(data=bool(assist_active)))
        self.state_pub.publish(String(data=self.state))
        self.route_segment_pub.publish(UInt8(data=int(self.route_segment_id % 256)))
        self.lateral_pub.publish(Float64(data=float(self.lateral_error)))
        self.heading_pub.publish(Float64(data=float(self.heading_error)))
        self.publish_geometry_debug()

        self.report_haptic_change(haptic)
        self.report_warning_change()

        side_valid, side_dist, side_clear = self.relevant_side_measurement()
        obs_rel = self.locked_obstacle_relative()
        obs_text = (
            'none'
            if obs_rel is None
            else f'x={obs_rel[0]:.2f},y={obs_rel[1]:.2f},r={obs_rel[2]:.2f}'
        )
        debug = (
            f'state={self.state} v={self.speed:.2f} front={self.front_collision_distance:.2f} '
            f'trigger={self.trigger_distance:.2f} odom_good={localization_good} '
            f'usable={localization_usable} conf={self.odom_confidence:.2f} '
            f'route_segment={self.route_segment_id} route_source={self.route_line_source} '
            f'line={self.line_valid} e_y={self.lateral_error:.2f} '
            f'e_th={self.heading_error:.2f} human_turn={self.manual_turn_active} '
            f'avoid_sign={self.avoid_sign:+d} object_side={self.object_side} '
            f'side={side_dist:.2f} valid={side_valid} clear={side_clear} '
            f'baseline={self.side_baseline_dist:.2f} seen={self.side_seen} '
            f'min_side={self.side_min_dist:.2f} last_side={self.side_last_valid_dist:.2f} '
            f'rise={self.side_rise_from_min():.2f} '
            f'side_exit={self.side_exit_confirmed}:{self.side_exit_reason or "none"} '
            f'locked_obs={obs_text} '
            f'gap_scores(L={self.last_side_choice_left_score:.2f},'
            f'R={self.last_side_choice_right_score:.2f}) '
            f'return_blocked={self.return_path_blocked} '
            f'return_recovery={self.return_recovery_active} '
            f'from_return={self.avoidance_from_return} '
            f'return_target=({self.return_target_x:.2f},{self.return_target_y:.2f}) '
            f'auto_turn={self.autonomous_turn_deviation:.2f} '
            f'guard={self.autonomous_turn_guard_active} '
            f'cmd={self.theta_cmd:.2f} warning={self.warning_active} stop=False | '
            f'{decision}'
        )
        self.debug_pub.publish(String(data=debug))
        if (now - self.last_debug_time).nanoseconds * 1e-9 > 0.65:
            self.get_logger().info(debug)
            self.last_debug_time = now

    # =========================================================================
    # Máquina de estados
    # =========================================================================
    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    # Implementa la lógica principal de transición y control para los estados:
    # USER_GUIDED, AVOID_OPEN, PASS_OBSTACLE y RETURN_LINE.
    def compute_state_control(self, now_sec: float, localization_good: bool) -> tuple[float, str, bool]:
        straight = self.straight_candidate()
        straight_observed = straight is not None and straight.observed_ratio >= self.min_straight_observed_ratio
        obstacle_near = (
            math.isfinite(self.front_collision_distance)
            and self.front_collision_distance < self.trigger_distance
        )
        emergency = self.arc_emergency or (
            math.isfinite(self.front_collision_distance)
            and self.front_collision_distance < self.emergency_distance_m
        )

        # USER_GUIDED:
        # el usuario controla dirección y velocidad; la asistencia solo se activa
        # si aparece un obstáculo dentro de la distancia de disparo.
        if self.state == self.USER_GUIDED:
            if not straight_observed and not obstacle_near:
                self.set_warning('STRAIGHT_CORRIDOR_INSUFFICIENTLY_OBSERVED')
                self.latch_haptic(self.HAPTIC_DANGER, 0.35)
                return 0.0, 'WARN_CORRIDOR_NOT_OBSERVED_USER_CONTINUES', False

            if obstacle_near:
                # Obstacles never create or replace a route line.  The current
                # segment was established at startup or after an explicit human
                # turn and is reused by every avoidance on that street.
                self.reactive_only_mode = not self.line_valid
                if not self.line_valid:
                    self.get_logger().warn(
                        'Avoidance starts reactively because no route line has '
                        'been established for the current street segment.'
                    )
                if not self.start_avoidance(now_sec):
                    self.set_warning('NO_ESCAPE_SIDE_FOUND')
                    self.latch_haptic(self.HAPTIC_DANGER, 0.50)
                    return self.least_risk_candidate(), 'WARN_NO_ESCAPE_SIDE_LEAST_RISK', True
                desired = self.compute_progressive_open_command(now_sec)
                return self.select_open_candidate(
                    desired, emergency=emergency
                ), 'START_AVOID_OPEN_PROGRESSIVE', True

            return 0.0, 'USER_CONTROLS_DIRECTION_AND_SPEED', False

        # No-brake fallback.  We continue steering to the least risky arc, but
        # do not advance the state machine on an unvalidated trajectory.
        if not self.valid_candidates():
            self.set_warning('NO_FULLY_VALID_ARC_EMERGENCY_ESCAPE')
            self.latch_haptic(self.HAPTIC_DANGER, 0.40)
            return self.compute_emergency_escape_command(), 'WARN_NO_VALID_ARC_EMERGENCY_ESCAPE', True

        # AVOID_OPEN:
        # apertura progresiva de la trayectoria hacia el lado elegido hasta
        # confirmar que el obstáculo está siendo correctamente bordeado.
        if self.state == self.AVOID_OPEN:
            self.update_side_seen(now_sec)
            self.update_side_exit_evidence(now_sec)
            self.update_front_clear_timer(now_sec)
            self.maybe_switch_avoidance_side(now_sec, emergency)
            elapsed = now_sec - self.state_start_time
            progress = self.distance_from_state_start()
            relative_lateral = abs(self.lateral_error - self.avoidance_start_lateral)
            relative_heading = abs(wrap_angle(self.heading_error - self.avoidance_start_heading))
            line_open_enough = self.line_valid and (
                relative_lateral >= self.open_lateral_min_m
                or relative_heading >= self.open_heading_min_rad
            )
            front_clear_confirmed = self.front_clear_confirmed(now_sec)
            track_entry_ready = self.tracked_obstacle_ready_for_pass_entry()
            # A real side contact is itself positive evidence that the opening
            # phase succeeded.  In v5, line=False forced relative errors to zero,
            # so the controller could see and completely pass the obstacle while
            # remaining trapped forever in AVOID_OPEN.
            open_enough = line_open_enough or self.side_seen or track_entry_ready

            # The side profile may already contain the complete rear edge before
            # PASS_OBSTACLE is entered.  Do not wait for an artificial extra state:
            # start the return immediately when the obstacle is demonstrably over.
            direct_return_ready = (
                self.side_exit_confirmed
                and front_clear_confirmed
                and progress >= self.pass_min_total_progress_m
            )
            if direct_return_ready:
                if self.line_valid:
                    self.change_state(self.RETURN_LINE, now_sec)
                    return self.compute_return_command(), 'SIDE_REAR_EDGE_DIRECT_START_RETURN', True
                self.release_to_user(now_sec, reason='REACTIVE_AVOIDANCE_COMPLETE_NO_LINE')
                self.latch_haptic(self.HAPTIC_RETURN_COMPLETE, 0.30)
                return 0.0, 'SIDE_REAR_EDGE_COMPLETE_RELEASE_NO_LINE', False

            can_enter_pass = (
                elapsed >= self.open_min_time_s
                and progress >= self.open_min_progress_m
                and front_clear_confirmed
                and open_enough
                and (self.side_seen or track_entry_ready)
            )
            if can_enter_pass:
                self.change_state(self.PASS_OBSTACLE, now_sec)
                return self.compute_pass_command(now_sec), 'ENTER_PASS_WITH_CONFIRMED_OBJECT_EVIDENCE', True

            # Do not continue increasing lateral displacement indefinitely while
            # waiting for side evidence.  Hold a small away command, never a
            # command towards the obstacle.
            at_open_limit = (
                relative_lateral >= self.open_lateral_max_m
                or relative_heading >= self.open_heading_max_rad
            )
            if self.side_seen and self.open_edge_follow_when_seen:
                desired = self.compute_edge_follow_command()
                phase = 'AVOID_OPEN_EDGE_FOLLOW'
            elif at_open_limit:
                desired = self.avoid_sign * self.open_hold_steer_rad
                phase = 'AVOID_OPEN_HOLD_LIMIT_WAIT_OBJECT_EVIDENCE'
            else:
                desired = self.compute_progressive_open_command(now_sec)
                phase = 'AVOID_OPEN_PROGRESSIVE'
            if emergency:
                self.set_warning('EMERGENCY_CLEARANCE_SERVO_ESCAPE')
                self.latch_haptic(self.HAPTIC_DANGER, 0.35)
            if self.side_seen and self.open_edge_follow_when_seen:
                command = self.select_candidate(
                    desired,
                    forbid_toward_object=True,
                    desired_weight=self.open_candidate_desired_weight,
                )
            else:
                command = self.select_open_candidate(
                    desired,
                    emergency=emergency,
                )
            return command, phase, True

        # PASS_OBSTACLE:
        # mantiene una trayectoria paralela o alejada hasta confirmar que la parte
        # trasera del obstáculo ha quedado completamente detrás del robot.
        if self.state == self.PASS_OBSTACLE:
            self.update_side_seen(now_sec)
            self.update_side_exit_evidence(now_sec)
            self.update_front_clear_timer(now_sec)
            side_lost = self.side_object_lost(now_sec)
            front_clear_confirmed = self.front_clear_confirmed(now_sec)
            geometry_passed = self.obstacle_passed_by_geometry()
            total_progress = self.distance_from_avoidance_start()

            # Either consistent local geometry or a strong side-sensor
            # rear-edge signature may confirm that the obstacle has ended.
            # Requiring both caused a permanent PASS_OBSTACLE lock whenever local
            # odometry drifted while the side echo disappeared after rising.
            side_passed = self.side_seen and (side_lost or self.side_exit_confirmed)
            completed = (
                total_progress >= self.pass_min_total_progress_m
                and front_clear_confirmed
                and (geometry_passed or side_passed)
            )
            if completed:
                if self.line_valid and not self.reactive_only_mode:
                    self.change_state(self.RETURN_LINE, now_sec)
                    return self.compute_return_command(), 'OBSTACLE_CONFIRMED_BEHIND_START_RETURN', True
                self.release_to_user(now_sec, reason='REACTIVE_AVOIDANCE_COMPLETE_NO_LINE')
                self.latch_haptic(self.HAPTIC_RETURN_COMPLETE, 0.30)
                return 0.0, 'REACTIVE_AVOIDANCE_COMPLETE_RELEASE_TO_USER', False

            if total_progress >= self.pass_unconfirmed_warning_progress_m and not self.side_seen and not self.locked_obstacle_valid:
                self.set_warning('PASS_NOT_CONFIRMED_KEEP_AWAY_DO_NOT_RETURN')
                self.latch_haptic(self.HAPTIC_DANGER, 0.40)

            return self.compute_pass_command(now_sec), 'PASS_HOLD_UNTIL_OBJECT_BEHIND', True

        # RETURN_LINE:
        # corrige progresivamente error lateral y angular hasta recuperar la línea
        # de referencia original del tramo.
        if self.state == self.RETURN_LINE:
            if obstacle_near and self.front_collision_distance < self.return_replan_front_m:
                if self.start_avoidance(now_sec, keep_reference_line=True):
                    desired = self.compute_progressive_open_command(now_sec)
                    return self.select_open_candidate(desired, emergency=emergency), 'NEW_OBSTACLE_INTERRUPTS_RETURN', True
                self.set_warning('RETURN_BLOCKED_NO_DETOUR_SIDE_RETURN_AWARE_ESCAPE')
                return (
                    self.compute_return_command(),
                    'WARN_RETURN_BLOCKED_RETURN_AWARE_ESCAPE',
                    True,
                )

            if not self.line_valid:
                self.release_to_user(now_sec, reason='REFERENCE_LINE_INVALID')
                return 0.0, 'REFERENCE_LINE_INVALID_RELEASE_CONTROL', False

            within = (
                abs(self.lateral_error) <= self.return_lateral_tolerance_m
                and abs(self.heading_error) <= self.return_heading_tolerance_rad
            )
            if within:
                if self.return_stable_since is None:
                    self.return_stable_since = now_sec
                elif (now_sec - self.return_stable_since) >= self.return_stable_time_s:
                    self.release_to_user(
                        now_sec, reason='RETURN_COMPLETE', preserve_route_line=True
                    )
                    self.latch_haptic(self.HAPTIC_RETURN_COMPLETE, 0.30)
                    return 0.0, 'RETURN_COMPLETE_RELEASE_TO_USER', False
            else:
                self.return_stable_since = None

            self.update_return_progress(now_sec)
            command = self.compute_return_command()
            if self.return_path_blocked:
                self.set_warning('RETURN_PATH_TOWARD_LINE_BLOCKED')
                return command, 'RETURN_PATH_BLOCKED_KEEP_REFERENCE', True
            if self.return_recovery_active:
                self.set_warning('RETURN_PROGRESS_STALLED_STRONGER_REENTRY')
                return command, 'RETURN_STALLED_STRONGER_REENTRY', True
            return command, 'RETURN_TO_CAPTURED_LINE', True

        self.release_to_user(now_sec, reason='UNKNOWN_STATE')
        return 0.0, 'UNKNOWN_STATE_RESET', False

    # =========================================================================
    # Línea de referencia y seguimiento de obstáculo
    # =========================================================================
    # ------------------------------------------------------------------
    # Reference line and obstacle track
    # ------------------------------------------------------------------
    # Detecta periodos de avance estable y actualiza una estimación suavizada
    # del rumbo del usuario.
    def update_stable_heading(self, now_sec: float) -> None:
        if (
            self.state == self.USER_GUIDED
            and not self.manual_turn_active
            and abs(self.human_steer) <= self.manual_turn_release_deadband_rad
            and self.speed >= self.stable_heading_min_speed_m_s
            and abs(self.yaw_rate) <= self.stable_heading_yaw_rate_rad_s
        ):
            if self.stable_heading_since is None:
                self.stable_heading_since = now_sec
            if (now_sec - self.stable_heading_since) >= self.stable_heading_time_s:
                if self.stable_heading_stamp is None:
                    self.stable_heading = self.yaw
                else:
                    self.stable_heading = wrap_angle(
                        self.stable_heading
                        + 0.25 * wrap_angle(self.yaw - self.stable_heading)
                    )
                self.stable_heading_stamp = now_sec
        else:
            self.stable_heading_since = None

    def clear_stable_reference_memory(self) -> None:
        self.stable_heading = 0.0
        self.stable_heading_since = None
        self.stable_heading_stamp = None
        self.reference_snapshot_valid = False
        self.reference_snapshot_stamp = None
        self.initial_route_ready_since = None

    # Crea la línea persistente del tramo actual a partir de la pose y el rumbo
    # estable disponibles.
    def capture_route_line(
        self,
        now_sec: float,
        source: str,
        prefer_stable_heading: bool = True,
    ) -> bool:
        """Create the route line for a street segment.

        This function is intentionally called only during initial startup or
        after a completed human turn.  Obstacle avoidance never calls it.
        """
        current_usable = (
            self.have_odom
            and self.odom_stamp is not None
            and (now_sec - self.odom_stamp) <= self.odom_timeout_s
            and self.odom_confidence >= self.line_capture_min_confidence
        )
        snapshot_fresh = (
            self.reference_snapshot_valid
            and self.reference_snapshot_stamp is not None
            and (now_sec - self.reference_snapshot_stamp)
            <= self.line_snapshot_max_age_s
        )

        if current_usable:
            px, py, pose_yaw = self.x, self.y, self.yaw
            pose_source = 'CURRENT_USABLE_ODOM'
        elif snapshot_fresh:
            px = self.reference_snapshot_x
            py = self.reference_snapshot_y
            pose_yaw = self.reference_snapshot_yaw
            pose_source = 'RECENT_REFERENCE_SNAPSHOT'
        else:
            return False

        theta = pose_yaw
        stable_fresh = (
            self.stable_heading_stamp is not None
            and (now_sec - self.stable_heading_stamp)
            <= self.stable_heading_max_age_s
        )
        if prefer_stable_heading and stable_fresh:
            theta = self.stable_heading

        self.line_x0 = px
        self.line_y0 = py
        self.line_theta = theta
        self.line_valid = True
        self.reactive_only_mode = False
        self.route_segment_id += 1
        self.route_line_source = source
        self.initial_route_captured = True
        self.update_line_errors()
        self.get_logger().info(
            f'Route line created [{source}/{pose_source}] '
            f'segment={self.route_segment_id} '
            f'p0=({self.line_x0:.2f},{self.line_y0:.2f}) '
            f'theta={self.line_theta:.2f} conf={self.odom_confidence:.2f}'
        )
        return True

    # Captura automáticamente la primera línea de ruta tras un tramo inicial
    # suficientemente estable.
    def ensure_initial_route_line(
        self,
        now_sec: float,
        localization_usable: bool,
    ) -> None:
        if (
            not self.initial_route_capture_enabled
            or self.initial_route_captured
            or self.line_valid
            or self.state != self.USER_GUIDED
            or not localization_usable
        ):
            self.initial_route_ready_since = None
            return

        stable_fresh = (
            self.stable_heading_stamp is not None
            and (now_sec - self.stable_heading_stamp)
            <= self.stable_heading_max_age_s
        )
        if not stable_fresh:
            self.initial_route_ready_since = None
            return

        if self.initial_route_ready_since is None:
            self.initial_route_ready_since = now_sec
            return
        if (
            now_sec - self.initial_route_ready_since
            < self.initial_route_capture_delay_s
        ):
            return

        self.capture_route_line(
            now_sec,
            source='INITIAL_STRAIGHT_SEGMENT',
            prefer_stable_heading=True,
        )

    def clear_manoeuvre_memory(self) -> None:
        self.avoid_sign = 0
        self.object_side = 'UNKNOWN'
        self.side_seen = False
        self.side_seen_since = None
        self.side_lost_since = None
        self.side_min_dist = math.inf
        self.side_last_valid_dist = math.inf
        self.side_last_valid_stamp = None
        self.side_invalid_since = None
        self.side_exit_confirmed = False
        self.side_exit_reason = ''
        self.front_clear_since = None
        self.return_stable_since = None
        self.locked_obstacle_valid = False
        self.localization_bad_since = None
        self.side_switch_candidate_sign = 0
        self.side_switch_candidate_since = None
        self.side_switch_count = 0
        self.return_path_blocked = False
        self.return_recovery_active = False
        self.avoidance_from_return = False
        self.return_target_valid = False
        self.autonomous_heading_anchor_valid = False
        self.autonomous_turn_deviation = 0.0
        self.autonomous_turn_guard_active = False

    # Inicia un giro manual: invalida la línea anterior y entrega prioridad total
    # al usuario.
    def begin_manual_turn(self, now_sec: float) -> None:
        previous = self.state
        self.clear_manoeuvre_memory()
        self.line_valid = False
        self.route_line_source = 'PENDING_MANUAL_TURN'
        self.reactive_only_mode = False
        self.clear_stable_reference_memory()
        self.manual_turn_started_at = now_sec
        self.manual_turn_release_since = None
        self.manual_turn_reset_future = None
        self.manual_turn_reset_requested_at = None
        self.manual_turn_post_reset_until = None
        self.manual_turn_waiting_for_fresh_odom = False
        self.planner_clear_memory_pub.publish(Bool(data=True))
        self.change_state(self.MANUAL_TURN, now_sec)
        self.get_logger().info(
            f'Manual turn started from {previous}: old route line invalidated; '
            'human has exclusive steering priority.'
        )

    # Supervisa cuándo el giro manual ha terminado, solicita opcionalmente un
    # reset de odometría y crea una nueva línea de referencia.
    def handle_manual_turn(
        self,
        now_sec: float,
        localization_usable: bool,
    ) -> tuple[float, str, bool]:
        if (
            self.manual_turn_active
            or abs(self.human_steer) > self.manual_turn_release_deadband_rad
        ):
            self.manual_turn_release_since = None
            return 0.0, 'MANUAL_TURN_WAITING_FOR_STEERING_CENTRE', False

        if self.manual_turn_waiting_for_fresh_odom:
            settle_done = (
                self.manual_turn_post_reset_until is not None
                and now_sec >= self.manual_turn_post_reset_until
            )
            if not settle_done or not localization_usable:
                return 0.0, 'MANUAL_TURN_WAITING_FOR_REBASED_ODOM', False
            if self.capture_route_line(
                now_sec,
                source='HUMAN_TURN_REBASED_SEGMENT',
                prefer_stable_heading=False,
            ):
                self.state = self.USER_GUIDED
                self.state_start_time = now_sec
                self.state_start_x = self.x
                self.state_start_y = self.y
                self.clear_stable_reference_memory()
                self.get_logger().info(
                    'Manual turn complete: new persistent route segment is active.'
                )
                return 0.0, 'MANUAL_TURN_COMPLETE_NEW_ROUTE_LINE', False
            return 0.0, 'MANUAL_TURN_REBASED_ODOM_NOT_YET_USABLE', False

        if self.manual_turn_reset_future is not None:
            if self.manual_turn_reset_future.done():
                try:
                    result = self.manual_turn_reset_future.result()
                    reset_ok = bool(result.success)
                    reset_message = str(result.message)
                except Exception as exc:  # service transport failure
                    reset_ok = False
                    reset_message = str(exc)
                self.manual_turn_reset_future = None
                if reset_ok:
                    self.clear_stable_reference_memory()
                    self.planner_clear_memory_pub.publish(Bool(data=True))
                    self.manual_turn_waiting_for_fresh_odom = True
                    self.manual_turn_post_reset_until = (
                        now_sec + self.manual_turn_post_reset_settle_s
                    )
                    self.get_logger().info(
                        f'Local odometry rebased after manual turn: {reset_message}'
                    )
                    return 0.0, 'MANUAL_TURN_ODOM_RESET_ACCEPTED', False
                self.get_logger().warn(
                    f'Local odometry reset failed; capturing segment in current '
                    f'local frame instead: {reset_message}'
                )
                if self.capture_route_line(
                    now_sec,
                    source='HUMAN_TURN_CURRENT_ODOM_FALLBACK',
                    prefer_stable_heading=False,
                ):
                    self.state = self.USER_GUIDED
                    return 0.0, 'MANUAL_TURN_COMPLETE_RESET_FALLBACK', False
            elif (
                self.manual_turn_reset_requested_at is not None
                and now_sec - self.manual_turn_reset_requested_at
                > self.manual_turn_reset_timeout_s
            ):
                self.get_logger().warn(
                    'Local odometry reset timed out; capturing the new route '
                    'segment in the current local frame.'
                )
                self.manual_turn_reset_future = None
                if self.capture_route_line(
                    now_sec,
                    source='HUMAN_TURN_RESET_TIMEOUT_FALLBACK',
                    prefer_stable_heading=False,
                ):
                    self.state = self.USER_GUIDED
                    return 0.0, 'MANUAL_TURN_COMPLETE_RESET_TIMEOUT_FALLBACK', False
            return 0.0, 'MANUAL_TURN_WAITING_FOR_ODOM_RESET_SERVICE', False

        stable_after_turn = (
            localization_usable
            and self.speed >= self.manual_turn_min_forward_speed_m_s
            and abs(self.yaw_rate) <= self.manual_turn_max_yaw_rate_rad_s
        )
        if not stable_after_turn:
            self.manual_turn_release_since = None
            return 0.0, 'MANUAL_TURN_WAITING_FOR_STABLE_FORWARD_MOTION', False

        if self.manual_turn_release_since is None:
            self.manual_turn_release_since = now_sec
            return 0.0, 'MANUAL_TURN_STABILITY_TIMER_STARTED', False
        if (
            now_sec - self.manual_turn_release_since
            < self.manual_turn_stable_time_s
        ):
            return 0.0, 'MANUAL_TURN_CONFIRMING_NEW_HEADING', False

        if self.reset_odom_after_manual_turn:
            if self.odom_reset_client.service_is_ready():
                self.manual_turn_reset_future = self.odom_reset_client.call_async(
                    Trigger.Request()
                )
                self.manual_turn_reset_requested_at = now_sec
                return 0.0, 'MANUAL_TURN_REQUESTED_LOCAL_ODOM_RESET', False
            self.get_logger().warn(
                'Reset-local-odom service is not ready; using current local '
                'coordinates for the new segment.'
            )

        if self.capture_route_line(
            now_sec,
            source='HUMAN_TURN_STABLE_SEGMENT',
            prefer_stable_heading=False,
        ):
            self.state = self.USER_GUIDED
            self.state_start_time = now_sec
            self.state_start_x = self.x
            self.state_start_y = self.y
            self.clear_stable_reference_memory()
            return 0.0, 'MANUAL_TURN_COMPLETE_NEW_ROUTE_LINE', False
        return 0.0, 'MANUAL_TURN_NEW_LINE_CAPTURE_FAILED', False

    # Calcula avance longitudinal, error lateral y error angular respecto a la
    # línea de referencia actual.
    def update_line_errors(self) -> None:
        if not self.line_valid:
            self.line_progress = 0.0
            self.lateral_error = 0.0
            self.heading_error = 0.0
            return
        dx = self.x - self.line_x0
        dy = self.y - self.line_y0
        ct = math.cos(self.line_theta)
        st = math.sin(self.line_theta)
        self.line_progress = ct * dx + st * dy
        self.lateral_error = -st * dx + ct * dy
        self.heading_error = wrap_angle(self.line_theta - self.yaw)

    # Fija una copia del obstáculo seguido al comenzar la evasión para conservar
    # su identidad aunque abandone el campo de visión frontal.
    def lock_current_obstacle_track(self, now_sec: float) -> None:
        fresh = self.track_stamp is not None and (now_sec - self.track_stamp) <= self.track_timeout_s
        if self.track_valid and fresh and self.track_quality >= self.tracked_obstacle_min_quality:
            self.locked_obstacle_valid = True
            self.locked_obstacle_x_local = self.track_x_local
            self.locked_obstacle_y_local = self.track_y_local
            self.locked_obstacle_radius = max(0.08, self.track_radius)
            self.get_logger().info(
                f'Obstacle track locked: local=({self.locked_obstacle_x_local:.2f},'
                f'{self.locked_obstacle_y_local:.2f}) radius={self.locked_obstacle_radius:.2f} '
                f'quality={self.track_quality:.2f}'
            )
        else:
            self.locked_obstacle_valid = False
            self.locked_obstacle_radius = 0.0
            self.get_logger().warn('No reliable RGB-D obstacle track available at avoidance start.')

    def locked_obstacle_relative(self) -> Optional[tuple[float, float, float]]:
        if not self.locked_obstacle_valid or not self.have_odom:
            return None
        dx = self.locked_obstacle_x_local - self.x
        dy = self.locked_obstacle_y_local - self.y
        x_robot = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        y_robot = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        return x_robot, y_robot, self.locked_obstacle_radius

    def tracked_obstacle_ready_for_pass_entry(self) -> bool:
        relative = self.locked_obstacle_relative()
        if relative is None:
            return False
        x_robot, y_robot, radius = relative
        required_lateral = 0.5 * self.robot_width_m + self.safety_margin_m + radius
        correct_side = (self.object_side == 'LEFT' and y_robot > 0.0) or (
            self.object_side == 'RIGHT' and y_robot < 0.0
        )
        return correct_side and x_robot <= self.pass_entry_obstacle_x_m and abs(y_robot) >= required_lateral

    # Determina geométricamente si la parte trasera del obstáculo ha quedado
    # detrás del robot.
    def obstacle_passed_by_geometry(self) -> bool:
        relative = self.locked_obstacle_relative()
        if relative is None:
            return False
        x_robot, _y_robot, radius = relative
        # The rear of the obstacle, not only its centre, must be behind the robot.
        return (x_robot + radius) <= -self.pass_rear_clearance_m

    # =========================================================================
    # Inicio de evasión y evidencia lateral
    # =========================================================================
    # ------------------------------------------------------------------
    # Start and side evidence
    # ------------------------------------------------------------------
    # Inicializa una maniobra de evasión, selecciona lado, registra línea base de
    # sensores laterales y bloquea el obstáculo actual.
    def start_avoidance(self, now_sec: float, keep_reference_line: bool = False) -> bool:
        previous_state = self.state
        if not keep_reference_line and not self.line_valid and not self.reactive_only_mode:
            return False
        self.avoidance_from_return = bool(
            keep_reference_line or previous_state == self.RETURN_LINE
        )
        sign = self.choose_avoidance_side()
        if sign == 0:
            self.avoidance_from_return = False
            return False
        self.avoid_sign = sign
        self.object_side = 'RIGHT' if sign > 0 else 'LEFT'
        if not self.autonomous_heading_anchor_valid:
            self.autonomous_heading_anchor = self.yaw
            self.autonomous_heading_anchor_valid = True
        self.side_switch_candidate_sign = 0
        self.side_switch_candidate_since = None
        self.side_switch_count = 0
        self.avoidance_start_x = self.x
        self.avoidance_start_y = self.y
        self.avoidance_start_progress = self.line_progress
        self.avoidance_start_lateral = self.lateral_error
        self.avoidance_start_heading = self.heading_error
        self.side_seen = False
        self.side_seen_since = None
        self.side_lost_since = None
        self.side_min_dist = math.inf
        self.side_last_valid_dist = math.inf
        self.side_last_valid_stamp = None
        self.side_invalid_since = None
        self.side_exit_confirmed = False
        self.side_exit_reason = ''
        valid, distance, _clear = self.relevant_side_measurement()
        self.side_baseline_valid = valid and math.isfinite(distance)
        self.side_baseline_dist = distance if self.side_baseline_valid else math.inf
        self.front_clear_since = None
        self.return_stable_since = None
        self.lock_current_obstacle_track(now_sec)
        if keep_reference_line:
            self.planner_clear_memory_pub.publish(Bool(data=True))
        self.change_state(self.AVOID_OPEN, now_sec)
        self.latch_haptic(self.HAPTIC_LEFT if sign > 0 else self.HAPTIC_RIGHT, 0.35)
        self.get_logger().info(
            f'Avoidance started: sign={sign:+d} object_side={self.object_side} '
            f'side_baseline={self.side_baseline_dist:.2f} valid={self.side_baseline_valid} '
            f'gap_scores(L={self.last_side_choice_left_score:.2f},'
            f'R={self.last_side_choice_right_score:.2f}) '
            f'from_return={self.avoidance_from_return}'
        )
        return True

    # Confirma que el obstáculo ha alcanzado el lateral usando una caída
    # persistente respecto a la línea base inicial.
    def update_side_seen(self, now_sec: float) -> None:
        """Update side-object detection and retain the complete distance profile.

        Version 4 returned immediately after ``side_seen`` became true, so
        ``side_min_dist`` remained equal to the first detection (0.66 m in the
        reported log) even though the sensor later reached roughly 0.32 m.
        Keeping the full profile is essential for recognising the rear edge.
        """
        valid, distance, _clear = self.relevant_side_measurement()
        if valid and math.isfinite(distance):
            self.side_last_valid_dist = distance
            self.side_last_valid_stamp = now_sec
            self.side_min_dist = min(self.side_min_dist, distance)
            self.side_invalid_since = None
        else:
            if not self.side_seen:
                self.side_seen_since = None
            return

        if self.side_seen:
            return

        if self.side_baseline_valid:
            detected_now = (
                distance <= self.side_seen_absolute_m
                and distance <= (self.side_baseline_dist - self.side_seen_drop_m)
            )
        else:
            detected_now = distance <= self.side_seen_absolute_m

        if detected_now:
            if self.side_seen_since is None:
                self.side_seen_since = now_sec
            elif (now_sec - self.side_seen_since) >= self.side_seen_confirm_s:
                self.side_seen = True
                self.side_lost_since = None
                self.side_invalid_since = None
                self.get_logger().info(
                    f'Object confirmed by {self.object_side} side sensor: '
                    f'd={distance:.2f}, baseline={self.side_baseline_dist:.2f}'
                )
        else:
            self.side_seen_since = None

    def side_rise_from_min(self) -> float:
        if not (
            math.isfinite(self.side_min_dist)
            and math.isfinite(self.side_last_valid_dist)
        ):
            return 0.0
        return max(0.0, self.side_last_valid_dist - self.side_min_dist)

    # Detecta el borde trasero del obstáculo mediante aumento de distancia,
    # condición clear o una secuencia mínimo -> subida -> pérdida de eco.
    def update_side_exit_evidence(self, now_sec: float) -> None:
        """Confirm a compact obstacle rear edge from valid rise or echo dropout."""
        if not self.side_seen or self.side_exit_confirmed:
            return

        valid, distance, clear = self.relevant_side_measurement()
        if valid and math.isfinite(distance):
            self.side_last_valid_dist = distance
            self.side_last_valid_stamp = now_sec
            self.side_min_dist = min(self.side_min_dist, distance)
            self.side_invalid_since = None

            release_threshold = self.side_release_absolute_m
            if self.side_baseline_valid:
                release_threshold = max(
                    release_threshold,
                    self.side_baseline_dist - self.side_release_drop_m,
                )

            rise = self.side_rise_from_min()
            conventional_release = clear and distance >= release_threshold
            rear_edge_rise = (
                rise >= self.side_exit_rise_from_min_m
                and distance >= self.side_exit_min_last_distance_m
            )
            if conventional_release or rear_edge_rise:
                if self.side_lost_since is None:
                    self.side_lost_since = now_sec
                elif (now_sec - self.side_lost_since) >= self.side_lost_confirm_s:
                    self.side_exit_confirmed = True
                    self.side_exit_reason = (
                        'VALID_CLEAR' if conventional_release else 'VALID_MIN_THEN_RISE'
                    )
                    self.get_logger().info(
                        f'Side rear edge confirmed ({self.side_exit_reason}): '
                        f'min={self.side_min_dist:.2f} last={distance:.2f} rise={rise:.2f}'
                    )
            else:
                self.side_lost_since = None
            return

        # Invalid is unknown unless a recent valid profile already showed a
        # pronounced minimum followed by a rise.
        recent_valid = (
            self.side_last_valid_stamp is not None
            and (now_sec - self.side_last_valid_stamp)
            <= self.side_exit_last_valid_max_age_s
        )
        dropout_signature = (
            recent_valid
            and self.side_rise_from_min() >= self.side_exit_rise_from_min_m
            and self.side_last_valid_dist >= self.side_exit_min_last_distance_m
        )
        if dropout_signature:
            if self.side_invalid_since is None:
                self.side_invalid_since = now_sec
            elif (now_sec - self.side_invalid_since) >= self.side_exit_invalid_confirm_s:
                self.side_exit_confirmed = True
                self.side_exit_reason = 'MIN_THEN_RISE_THEN_DROPOUT'
                self.get_logger().info(
                    f'Side rear edge confirmed ({self.side_exit_reason}): '
                    f'min={self.side_min_dist:.2f} last={self.side_last_valid_dist:.2f} '
                    f'rise={self.side_rise_from_min():.2f}'
                )
        else:
            self.side_invalid_since = None

    def side_object_lost(self, now_sec: float) -> bool:
        self.update_side_exit_evidence(now_sec)
        return self.side_exit_confirmed

    def relevant_side_measurement(self) -> tuple[bool, float, bool]:
        if self.object_side == 'LEFT':
            return self.side_left_valid, self.side_left_dist, self.side_left_clear
        if self.object_side == 'RIGHT':
            return self.side_right_valid, self.side_right_dist, self.side_right_clear
        return False, math.inf, False

    def update_front_clear_timer(self, now_sec: float) -> None:
        clear_now = self.front_collision_distance >= self.front_clear_for_pass_m
        if clear_now:
            if self.front_clear_since is None:
                self.front_clear_since = now_sec
        else:
            self.front_clear_since = None

    def front_clear_confirmed(self, now_sec: float) -> bool:
        return self.front_clear_since is not None and (now_sec - self.front_clear_since) >= self.front_clear_confirm_s

    # =========================================================================
    # Generación de comandos
    # =========================================================================
    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    # Genera una apertura suave cuya magnitud crece con tiempo y progreso en vez
    # de aplicar de forma instantánea el giro máximo.
    def compute_progressive_open_command(self, now_sec: float) -> float:
        """Open smoothly instead of immediately commanding the full steering.

        The first part of the avoidance uses a small angle and ramps with both
        elapsed time and travelled distance.  Once the obstacle is detected by
        the side sensor, the controller switches to edge following rather than
        continuing to rotate away from the original route.
        """
        if self.side_seen and self.open_edge_follow_when_seen:
            return self.compute_edge_follow_command()

        elapsed = max(0.0, now_sec - self.state_start_time)
        progress = self.distance_from_state_start()
        time_ratio = clamp(
            elapsed / max(self.open_ramp_time_s, 1e-3), 0.0, 1.0
        )
        progress_ratio = clamp(
            progress / max(self.open_ramp_progress_m, 1e-3), 0.0, 1.0
        )
        ramp = clamp(0.55 * time_ratio + 0.45 * progress_ratio, 0.0, 1.0)

        if (
            math.isfinite(self.front_collision_distance)
            and self.front_collision_distance
            <= max(self.emergency_distance_m * 1.15, 0.42)
        ):
            ramp = 1.0

        magnitude = self.open_initial_steer_rad + ramp * (
            self.open_steer_rad - self.open_initial_steer_rad
        )

        relative_heading = abs(
            wrap_angle(self.heading_error - self.avoidance_start_heading)
        )
        if relative_heading >= self.open_heading_max_rad:
            magnitude = min(magnitude, self.open_hold_steer_rad)

        if self.avoidance_from_return and self.line_valid:
            extra_error = (
                abs(self.lateral_error) - abs(self.avoidance_start_lateral)
            )
            preferred_sign = (
                -1 if self.lateral_error > self.return_lateral_tolerance_m
                else 1 if self.lateral_error < -self.return_lateral_tolerance_m
                else 0
            )
            if (
                preferred_sign != 0
                and self.avoid_sign != preferred_sign
                and extra_error >= self.return_detour_max_extra_lateral_m
            ):
                magnitude = min(
                    magnitude, self.return_detour_parallel_steer_rad
                )

        return self.avoid_sign * clamp(
            magnitude,
            0.0,
            self.open_steer_rad,
        )

    # Genera una orden aproximadamente paralela al obstáculo usando distancia
    # lateral y error de rumbo.
    def compute_edge_follow_command(self) -> float:
        """Generate a near-parallel command while bordering the obstacle."""
        valid, distance, _clear = self.relevant_side_measurement()
        if self.side_seen and valid and math.isfinite(distance):
            object_sign = 1.0 if self.object_side == 'LEFT' else -1.0
            side_component = (
                object_sign
                * self.side_follow_gain
                * (distance - self.side_target_distance_m)
            )
            desired = self.parallel_heading_gain * self.heading_error + side_component
        else:
            desired = self.avoid_sign * self.open_hold_steer_rad

        desired = self.apply_side_keepaway(desired)
        desired = self.forbid_toward_object_value(desired)
        return clamp(desired, -0.25, 0.25)

    # Mantiene separación respecto al obstáculo durante la fase de paso.
    def compute_pass_command(self, now_sec: float) -> float:
        del now_sec
        valid, distance, _clear = self.relevant_side_measurement()
        obstacle_passed = self.obstacle_passed_by_geometry()

        if self.side_seen and valid and math.isfinite(distance):
            desired = self.compute_edge_follow_command()
        elif not obstacle_passed:
            # The obstacle left the frontal image before the side sensor saw it.
            # Keep moving away/parallel instead of progressively centring the wheel.
            desired = self.avoid_sign * self.pass_hold_away_steer_rad
        else:
            desired = self.parallel_heading_gain * self.heading_error

        desired = self.apply_side_keepaway(desired)
        if not obstacle_passed:
            desired = self.forbid_toward_object_value(desired)
        desired = clamp(desired, -0.28, 0.28)
        return self.select_candidate(desired, forbid_toward_object=not obstacle_passed)

    def update_return_progress(self, now_sec: float) -> None:
        error = abs(self.lateral_error)
        if error + self.return_progress_epsilon_m < self.return_best_abs_error:
            self.return_best_abs_error = error
            self.return_last_improvement_stamp = now_sec
            self.return_recovery_active = False
            return
        if self.return_last_improvement_stamp is None:
            self.return_last_improvement_stamp = now_sec
            self.return_best_abs_error = error
            return
        self.return_recovery_active = (
            error > self.return_lateral_tolerance_m
            and (now_sec - self.return_last_improvement_stamp) >= self.return_progress_timeout_s
        )

    # Calcula un punto objetivo adelantado sobre la línea de referencia para
    # realizar una reentrada diagonal y estable.
    def compute_return_target(self) -> tuple[float, float, float, float]:
        """Return a forward target on the frozen route line.

        The target is placed ahead of the orthogonal projection of the current
        pose, so the cane converges diagonally instead of becoming parallel to
        the line while preserving a large lateral offset.
        """
        lookahead = clamp(
            self.return_lookahead_min_m
            + self.return_lookahead_speed_gain_s * self.speed,
            self.return_lookahead_min_m,
            self.return_lookahead_max_m,
        )
        target_progress = self.line_progress + lookahead
        target_x = self.line_x0 + target_progress * math.cos(self.line_theta)
        target_y = self.line_y0 + target_progress * math.sin(self.line_theta)
        self.return_target_valid = True
        self.return_target_x = target_x
        self.return_target_y = target_y
        self.return_target_theta = self.line_theta
        self.return_target_lookahead = lookahead
        return target_x, target_y, self.line_theta, lookahead

    # Predice el error lateral y angular futuro de un arco candidato al final
    # de una distancia de anticipación.
    def predict_candidate_line_error(
        self,
        candidate: ArcCandidate,
        distance: float,
    ) -> tuple[float, float, float, float]:
        """Predict route errors at a candidate arc endpoint."""
        curvature = math.tan(candidate.steer) / max(self.wheelbase_m, 1e-3)
        if abs(curvature) < 1e-6:
            dx_body = distance
            dy_body = 0.0
            delta_yaw = 0.0
        else:
            dx_body = math.sin(curvature * distance) / curvature
            dy_body = (1.0 - math.cos(curvature * distance)) / curvature
            delta_yaw = curvature * distance

        ct = math.cos(self.yaw)
        st = math.sin(self.yaw)
        future_x = self.x + ct * dx_body - st * dy_body
        future_y = self.y + st * dx_body + ct * dy_body
        future_yaw = wrap_angle(self.yaw + delta_yaw)

        dx = future_x - self.line_x0
        dy = future_y - self.line_y0
        line_ct = math.cos(self.line_theta)
        line_st = math.sin(self.line_theta)
        future_lateral = -line_st * dx + line_ct * dy
        future_heading = wrap_angle(self.line_theta - future_yaw)
        return future_lateral, future_heading, future_x, future_y

    # Selecciona el arco de retorno que mejor reduce el error futuro sin
    # incrementar excesivamente la desviación lateral.
    def select_return_candidate(self, desired: float) -> float:
        options = self.valid_candidates() or self.fallback_candidates()
        if not options:
            self.return_path_blocked = True
            return self.fallback_guidance_command(self.now_seconds())

        prediction_distance = clamp(
            self.return_prediction_min_m
            + self.return_prediction_speed_gain_s * self.speed,
            self.return_prediction_min_m,
            self.return_prediction_max_m,
        )
        current_abs = abs(self.lateral_error)
        required_sign = 0
        if current_abs > self.return_lateral_tolerance_m:
            required_sign = -1 if self.lateral_error > 0.0 else 1

        ranked: list[tuple[float, ArcCandidate, float, float]] = []
        corrective_exists = False
        bounded_exists = False
        for candidate in options:
            future_lateral, future_heading, _fx, _fy = (
                self.predict_candidate_line_error(candidate, prediction_distance)
            )
            future_abs = abs(future_lateral)
            growth = future_abs - current_abs
            if future_abs + self.return_progress_epsilon_m < current_abs:
                corrective_exists = True
            if growth <= self.return_max_error_growth_m:
                bounded_exists = True

            cost = self.return_candidate_desired_weight * abs(
                candidate.steer - desired
            )
            cost += self.return_future_lateral_weight * future_abs
            cost += self.return_future_heading_weight * abs(future_heading)
            cost += self.steer_change_weight * abs(
                candidate.steer - self.theta_cmd
            )
            cost -= self.clearance_reward_weight * min(
                candidate.clearance, 1.5
            )
            cost -= self.observed_reward_weight * candidate.observed_ratio
            cost -= self.collision_reward_weight * min(
                candidate.collision_distance, 2.0
            )
            if growth > self.return_max_error_growth_m:
                cost += self.return_error_growth_penalty * (
                    growth - self.return_max_error_growth_m
                )
            if (
                required_sign != 0
                and candidate.steer * required_sign < 0.015
                and future_abs >= current_abs - self.return_progress_epsilon_m
            ):
                cost += self.return_wrong_direction_penalty
            if not candidate.valid:
                cost += 3.0
            ranked.append((cost, candidate, future_lateral, future_heading))

        self.return_path_blocked = not corrective_exists
        if bounded_exists:
            bounded = [
                item for item in ranked
                if abs(item[2]) - current_abs <= self.return_max_error_growth_m
            ]
            if bounded:
                ranked = bounded
        best = min(ranked, key=lambda item: item[0])[1]
        return self.remember_safe_steer(best.steer)

    # Combina Pure Pursuit y Stanley para generar la dirección deseada de retorno
    # y después la contrasta con los arcos realmente transitables del planner.
    def compute_return_command(self) -> float:
        if not self.line_valid:
            self.return_path_blocked = True
            self.return_target_valid = False
            return 0.0

        target_x, target_y, _target_theta, lookahead = (
            self.compute_return_target()
        )
        dx = target_x - self.x
        dy = target_y - self.y
        ct = math.cos(self.yaw)
        st = math.sin(self.yaw)
        x_robot = ct * dx + st * dy
        y_robot = -st * dx + ct * dy
        target_distance_sq = max(x_robot * x_robot + y_robot * y_robot, 0.04)
        pure_pursuit = math.atan(
            2.0 * self.wheelbase_m * y_robot / target_distance_sq
        )

        gain = self.return_stanley_gain
        if self.return_recovery_active:
            gain *= self.return_recovery_gain_multiplier
        stanley = -math.atan2(
            gain * self.lateral_error,
            max(self.speed + self.return_speed_softening_m_s, 0.05),
        )
        blend = clamp(self.return_pure_pursuit_weight, 0.0, 1.0)
        desired = (
            blend * pure_pursuit
            + (1.0 - blend) * stanley
            + self.return_heading_gain * self.heading_error
        )

        required_sign = 0
        if abs(self.lateral_error) > self.return_lateral_tolerance_m:
            required_sign = -1 if self.lateral_error > 0.0 else 1
            if desired * required_sign < self.return_min_corrective_steer_rad:
                desired = required_sign * max(
                    abs(desired), self.return_min_corrective_steer_rad
                )

        desired = self.apply_return_side_guard(desired)
        desired = clamp(desired, -self.max_steer_rad, self.max_steer_rad)
        command = self.select_return_candidate(desired)
        if self.return_path_blocked:
            # Keep moving through the safest bounded detour.  The route line and
            # target remain alive and the controller retries convergence every
            # cycle instead of abandoning the line or drifting indefinitely.
            command = clamp(
                command,
                -self.max_steer_rad,
                self.max_steer_rad,
            )
        return command

    # Estrategia reactiva usada cuando la odometría no es suficientemente fiable.
    def compute_reactive_safe_command(self, now_sec: float) -> float:
        if not self.valid_candidates():
            return self.compute_emergency_escape_command()
        if self.state == self.PASS_OBSTACLE:
            return self.compute_pass_command(now_sec)
        return self.least_risk_candidate(forbid_toward_object=True)

    # Impone separación mínima frente al lateral donde se encuentra el obstáculo.
    def apply_side_keepaway(self, desired: float) -> float:
        valid, distance, _clear = self.relevant_side_measurement()
        if not valid or not math.isfinite(distance):
            return desired
        toward = self.is_steering_toward_object(desired)
        if distance < self.side_hard_distance_m:
            return self.avoid_sign * max(abs(desired), self.side_keepaway_steer_rad)
        if distance < self.side_soft_distance_m and toward:
            return self.avoid_sign * max(self.pass_hold_away_steer_rad, 0.06)
        return desired

    # Evita que el retorno a línea ordene un giro hacia un obstáculo lateral cercano.
    def apply_return_side_guard(self, desired: float) -> float:
        left_hard = self.side_left_valid and self.side_left_dist < self.return_side_guard_hard_m
        right_hard = self.side_right_valid and self.side_right_dist < self.return_side_guard_hard_m
        left_soft = self.side_left_valid and self.side_left_dist < self.return_side_guard_soft_m
        right_soft = self.side_right_valid and self.side_right_dist < self.return_side_guard_soft_m
        if left_hard and right_hard:
            return 0.0
        if left_hard:
            return min(desired, -self.return_side_away_steer_rad)
        if right_hard:
            return max(desired, self.return_side_away_steer_rad)
        if left_soft and desired > 0.0:
            return 0.0
        if right_soft and desired < 0.0:
            return 0.0
        return desired

    def is_steering_toward_object(self, steer: float) -> bool:
        return (self.object_side == 'LEFT' and steer > 0.025) or (
            self.object_side == 'RIGHT' and steer < -0.025
        )

    def forbid_toward_object_value(self, desired: float) -> float:
        limit = self.pass_max_toward_object_steer_rad
        if self.object_side == 'LEFT':
            return min(desired, limit)
        if self.object_side == 'RIGHT':
            return max(desired, -limit)
        return desired

    # Selecciona una dirección de escape cuando todos los arcos son inválidos.
    # Mantiene el lado de evasión ya comprometido salvo peligro lateral inmediato.
    def compute_emergency_escape_command(self) -> float:
        """Choose the least-risk non-zero escape when every arc is invalid.

        The manoeuvre side already committed by AVOID_OPEN/PASS_OBSTACLE is
        preserved while the obstacle is still relevant.  A lateral sensor may
        override that commitment only when it detects an immediate side hazard;
        otherwise a transient all-invalid RGB-D frame must not flip the cane
        toward the obstacle it was already bypassing.
        """
        options = self.fallback_candidates()
        if not options:
            options = [
                candidate for candidate in self.candidates
                if candidate.collision_distance > 0.01
            ]

        left = self.side_left_dist if self.side_left_valid else math.inf
        right = self.side_right_dist if self.side_right_valid else math.inf

        # Immediate lateral danger has highest priority.  The sign points AWAY
        # from the close side sensor.
        forced_sign = 0
        if left < self.emergency_side_hard_m and right >= self.emergency_side_hard_m:
            forced_sign = -1
        elif right < self.emergency_side_hard_m and left >= self.emergency_side_hard_m:
            forced_sign = 1
        elif left < self.emergency_side_soft_m or right < self.emergency_side_soft_m:
            if left + 0.04 < right:
                forced_sign = -1
            elif right + 0.04 < left:
                forced_sign = 1

        # REAL-SYSTEM FIX: once an avoidance side has been committed, an
        # all-invalid planner frame must not arbitrarily reverse it.  PASS may
        # release the constraint once geometry says the obstacle is already
        # behind, because from there steering back toward the route can be valid.
        committed_sign = 0
        obstacle_still_relevant = (
            self.state == self.AVOID_OPEN
            or (
                self.state == self.PASS_OBSTACLE
                and not self.obstacle_passed_by_geometry()
            )
        )
        if obstacle_still_relevant and self.avoid_sign != 0:
            committed_sign = int(self.avoid_sign)

        required_sign = forced_sign if forced_sign != 0 else committed_sign

        if not options:
            # Reuse the last safe steering only if it does not violate the
            # currently required escape side.
            if abs(self.last_safe_steer) >= 0.03:
                if required_sign == 0 or self.last_safe_steer * required_sign >= -0.015:
                    return self.last_safe_steer

            sign = required_sign
            if sign == 0:
                sign = self.avoid_sign
            if sign == 0:
                sign = -1 if left < right else 1
            return sign * self.emergency_min_escape_steer_rad

        # Prefer only candidates that respect the required side.  If no such
        # sampled arc exists, command a bounded minimum escape on that side
        # rather than crossing through zero and steering toward the obstacle.
        allowed_options = options
        if required_sign != 0:
            same_side = [
                candidate for candidate in options
                if candidate.steer * required_sign >= -0.015
            ]
            if same_side:
                allowed_options = same_side
            else:
                steer = required_sign * self.emergency_min_escape_steer_rad
                return self.remember_safe_steer(
                    clamp(steer, -self.max_steer_rad, self.max_steer_rad)
                )

        def score(candidate: ArcCandidate) -> float:
            value = (
                3.2 * min(candidate.collision_distance, 2.0)
                + 1.4 * min(candidate.clearance, 1.5)
                + 1.0 * min(candidate.far_clearance, 1.5)
                + 0.7 * min(candidate.tail_clearance, 1.5)
                + 0.35 * candidate.observed_ratio
                - 0.16 * abs(candidate.steer - self.theta_cmd)
            )
            return value

        best = max(allowed_options, key=score)
        steer = float(best.steer)

        if abs(steer) < self.emergency_min_escape_steer_rad:
            sign = required_sign
            if sign == 0:
                sign = 1 if steer > 0.0 else -1 if steer < 0.0 else self.avoid_sign
            if sign == 0:
                sign = -1 if left < right else 1
            steer = sign * self.emergency_min_escape_steer_rad

        return self.remember_safe_steer(
            clamp(steer, -self.max_steer_rad, self.max_steer_rad)
        )

    # =========================================================================
    # Selección de arcos
    # =========================================================================
    # ------------------------------------------------------------------
    # Arc selection
    # ------------------------------------------------------------------
    def valid_candidates(self) -> list[ArcCandidate]:
        return [candidate for candidate in self.candidates if candidate.valid]

    def fallback_candidates(self) -> list[ArcCandidate]:
        return [
            candidate for candidate in self.candidates
            if candidate.observed_ratio >= self.min_fallback_observed_ratio
            and candidate.collision_distance > 0.05
        ]

    def straight_candidate(self) -> Optional[ArcCandidate]:
        return min(self.candidates, key=lambda candidate: abs(candidate.steer)) if self.candidates else None

    def side_summary_fresh(self) -> bool:
        return (
            self.side_summary_stamp is not None
            and (self.now_seconds() - self.side_summary_stamp) <= self.side_summary_timeout_s
        )

    # Calcula las puntuaciones globales izquierda/derecha combinando información
    # del planner, sensores laterales y relación con la línea de referencia.
    def side_choice_scores(self) -> tuple[float, float]:
        if self.side_summary_fresh():
            left_score = float(self.side_summary[0])
            right_score = float(self.side_summary[1])
        else:
            options = self.valid_candidates() or self.fallback_candidates()
            left = [c for c in options if c.steer > 0.035]
            right = [c for c in options if c.steer < -0.035]
            left_score = max((self.side_choice_score(c) for c in left), default=-1e9)
            right_score = max((self.side_choice_score(c) for c in right), default=-1e9)

        # A side sensor is not used to invent free space, but a close measured
        # boundary is strong negative evidence for steering into that side.
        if self.side_left_valid and math.isfinite(self.side_left_dist):
            if self.side_left_dist <= self.side_choice_sensor_hard_m:
                left_score -= 12.0
            elif self.side_left_dist <= self.side_choice_sensor_soft_m:
                left_score -= 4.0 * (
                    self.side_choice_sensor_soft_m - self.side_left_dist
                ) / max(
                    self.side_choice_sensor_soft_m - self.side_choice_sensor_hard_m,
                    1e-3,
                )
        if self.side_right_valid and math.isfinite(self.side_right_dist):
            if self.side_right_dist <= self.side_choice_sensor_hard_m:
                right_score -= 12.0
            elif self.side_right_dist <= self.side_choice_sensor_soft_m:
                right_score -= 4.0 * (
                    self.side_choice_sensor_soft_m - self.side_right_dist
                ) / max(
                    self.side_choice_sensor_soft_m - self.side_choice_sensor_hard_m,
                    1e-3,
                )

        # A new obstacle encountered during RETURN_LINE is a temporary detour,
        # not a new route.  Prefer the gap that reduces |e_y|, but only when its
        # raw corridor score is not substantially worse than the safer side.
        if self.line_valid and self.avoidance_from_return:
            preferred_sign = 0
            if self.lateral_error > self.return_lateral_tolerance_m:
                preferred_sign = -1
            elif self.lateral_error < -self.return_lateral_tolerance_m:
                preferred_sign = 1
            if preferred_sign > 0:
                if left_score >= right_score - self.return_detour_max_gap_disadvantage:
                    left_score += self.return_detour_line_bias_weight
                    right_score -= 0.35 * self.return_detour_line_bias_weight
            elif preferred_sign < 0:
                if right_score >= left_score - self.return_detour_max_gap_disadvantage:
                    right_score += self.return_detour_line_bias_weight
                    left_score -= 0.35 * self.return_detour_line_bias_weight
        elif self.line_valid and abs(left_score - right_score) <= self.side_choice_line_bias_score_window:
            # For an ordinary avoidance, the line remains only a weak tie-breaker.
            if self.lateral_error > self.return_lateral_tolerance_m:
                right_score += self.side_choice_line_bias_weight
            elif self.lateral_error < -self.return_lateral_tolerance_m:
                left_score += self.side_choice_line_bias_weight

        self.last_side_choice_left_score = left_score
        self.last_side_choice_right_score = right_score
        return left_score, right_score

    # Elige el lado de evasión con mayor puntuación y aplica criterios de desempate.
    def choose_avoidance_side(self) -> int:
        left_score, right_score = self.side_choice_scores()
        if left_score <= -1e5 and right_score <= -1e5:
            return 0
        if abs(left_score - right_score) < self.side_choice_score_margin:
            # Prefer the planner's best aggregate steer in a near tie; then keep
            # the previous command sign to avoid random left/right oscillation.
            if self.side_summary_fresh():
                left_best = float(self.side_summary[6])
                right_best = float(self.side_summary[7])
                if abs(left_best) > 0.035 or abs(right_best) > 0.035:
                    return 1 if left_score >= right_score else -1
            if abs(self.theta_cmd) > 0.035:
                return 1 if self.theta_cmd > 0.0 else -1
        return 1 if left_score >= right_score else -1

    def side_choice_score(self, candidate: ArcCandidate) -> float:
        return (
            1.25 * min(candidate.far_clearance, 1.5)
            + 1.00 * min(candidate.tail_clearance, 1.5)
            + 0.70 * min(candidate.clearance, 1.5)
            + 0.65 * candidate.observed_ratio
            + 0.45 * min(candidate.collision_distance, 2.0)
            - 0.24 * abs(candidate.steer) / max(self.max_steer_rad, 1e-3)
        )

    # Permite cambiar una única vez de lado antes de quedar comprometido, solo si
    # el corredor opuesto resulta claramente mejor.
    def maybe_switch_avoidance_side(self, now_sec: float, emergency: bool) -> None:
        if (
            self.state != self.AVOID_OPEN
            or self.avoid_sign == 0
            or self.side_seen
            or self.side_switch_count >= self.side_choice_max_switches
        ):
            self.side_switch_candidate_sign = 0
            self.side_switch_candidate_since = None
            return

        elapsed = now_sec - self.state_start_time
        relative_lateral = abs(self.lateral_error - self.avoidance_start_lateral)
        relative_heading = abs(wrap_angle(self.heading_error - self.avoidance_start_heading))
        committed = (
            relative_lateral >= self.side_choice_commit_lateral_m
            or relative_heading >= self.side_choice_commit_heading_rad
            or elapsed >= self.side_choice_switch_max_time_s
        )

        left_score, right_score = self.side_choice_scores()
        current_score = left_score if self.avoid_sign > 0 else right_score
        opposite_score = right_score if self.avoid_sign > 0 else left_score
        opposite_sign = -self.avoid_sign

        current_side_hard = (
            (self.avoid_sign > 0 and self.side_left_valid and self.side_left_dist < self.emergency_side_hard_m)
            or (self.avoid_sign < 0 and self.side_right_valid and self.side_right_dist < self.emergency_side_hard_m)
        )
        switch_allowed = (not committed) or (emergency and current_side_hard)
        superior = opposite_score >= current_score + self.side_choice_switch_margin
        if not switch_allowed or not superior:
            self.side_switch_candidate_sign = 0
            self.side_switch_candidate_since = None
            return

        if self.side_switch_candidate_sign != opposite_sign:
            self.side_switch_candidate_sign = opposite_sign
            self.side_switch_candidate_since = now_sec
            return
        if self.side_switch_candidate_since is None or (
            now_sec - self.side_switch_candidate_since
        ) < self.side_choice_switch_confirm_s:
            return

        old_sign = self.avoid_sign
        self.avoid_sign = opposite_sign
        self.object_side = 'RIGHT' if self.avoid_sign > 0 else 'LEFT'
        self.side_switch_count += 1
        self.state_start_time = now_sec
        self.state_start_x = self.x
        self.state_start_y = self.y
        self.avoidance_start_lateral = self.lateral_error
        self.avoidance_start_heading = self.heading_error
        self.reset_side_evidence_for_current_sign(now_sec)
        self.side_switch_candidate_sign = 0
        self.side_switch_candidate_since = None
        self.latch_haptic(
            self.HAPTIC_LEFT if self.avoid_sign > 0 else self.HAPTIC_RIGHT,
            0.35,
        )
        self.get_logger().warn(
            f'Avoidance side switched {old_sign:+d}->{self.avoid_sign:+d}: '
            f'left_score={left_score:.2f} right_score={right_score:.2f}'
        )

    def reset_side_evidence_for_current_sign(self, now_sec: float) -> None:
        self.side_seen = False
        self.side_seen_since = None
        self.side_lost_since = None
        self.side_min_dist = math.inf
        self.side_last_valid_dist = math.inf
        self.side_last_valid_stamp = None
        self.side_invalid_since = None
        self.side_exit_confirmed = False
        self.side_exit_reason = ''
        valid, distance, _clear = self.relevant_side_measurement()
        self.side_baseline_valid = valid and math.isfinite(distance)
        self.side_baseline_dist = distance if self.side_baseline_valid else math.inf
        self.front_clear_since = None

    def select_open_candidate(self, desired: float, emergency: bool = False) -> float:
        """Select an opening arc without allowing an unnecessary hard turn."""
        options = self.valid_candidates() or self.fallback_candidates()
        if not options:
            return self.fallback_guidance_command(self.now_seconds())

        if emergency:
            return self.select_candidate(
                desired,
                forbid_toward_object=True,
                desired_weight=self.open_candidate_desired_weight,
            )

        sign = 1 if self.avoid_sign > 0 else -1
        max_magnitude = min(
            self.max_steer_rad,
            abs(desired) + self.open_candidate_extra_steer_rad,
        )
        bounded = [
            candidate
            for candidate in options
            if candidate.steer * sign >= -0.01
            and abs(candidate.steer) <= max_magnitude + 1e-6
            and not self.is_steering_toward_object(candidate.steer)
        ]
        if bounded:
            original_candidates = self.candidates
            try:
                self.candidates = bounded
                return self.select_candidate(
                    desired,
                    forbid_toward_object=True,
                    desired_weight=self.open_candidate_desired_weight,
                )
            finally:
                self.candidates = original_candidates

        return self.select_candidate(
            desired,
            forbid_toward_object=True,
            desired_weight=self.open_candidate_desired_weight,
        )

    # Selección genérica del arco que minimiza un coste ponderado respecto al
    # comando deseado, clearance, observación y distancia de colisión.
    def select_candidate(
        self,
        desired: float,
        forbid_toward_object: bool = False,
        required_sign: int = 0,
        desired_weight: Optional[float] = None,
        wrong_direction_penalty: float = 0.0,
    ) -> float:
        options = self.valid_candidates() or self.fallback_candidates()
        if not options:
            return self.fallback_guidance_command(self.now_seconds())
        best: Optional[ArcCandidate] = None
        best_cost = math.inf
        for candidate in options:
            weight = self.desired_steer_weight if desired_weight is None else desired_weight
            cost = weight * abs(candidate.steer - desired)
            cost += self.steer_change_weight * abs(candidate.steer - self.theta_cmd)
            cost -= self.clearance_reward_weight * min(candidate.clearance, 1.5)
            cost -= self.observed_reward_weight * candidate.observed_ratio
            cost -= self.collision_reward_weight * min(candidate.collision_distance, 2.0)
            if not candidate.valid:
                cost += 3.0
            if forbid_toward_object and self.is_steering_toward_object(candidate.steer):
                cost += self.wrong_side_penalty
            if required_sign != 0 and candidate.steer * required_sign < 0.025:
                cost += wrong_direction_penalty
            if cost < best_cost:
                best = candidate
                best_cost = cost
        if best is None:
            return self.fallback_guidance_command(self.now_seconds())
        return self.remember_safe_steer(best.steer)

    # Devuelve el arco menos arriesgado cuando no existe una solución ideal.
    def least_risk_candidate(self, forbid_toward_object: bool = False) -> float:
        options = self.valid_candidates() or self.fallback_candidates()
        if not options:
            return self.fallback_guidance_command(self.now_seconds())
        filtered = [c for c in options if not (forbid_toward_object and self.is_steering_toward_object(c.steer))]
        if not filtered:
            filtered = options
        best = max(
            filtered,
            key=lambda c: (
                1.50 * c.collision_distance
                + 1.00 * c.clearance
                + 0.55 * c.observed_ratio
                - 0.22 * abs(c.steer - self.theta_cmd)
                - (1.0 if not c.valid else 0.0)
            ),
        )
        return self.remember_safe_steer(best.steer)

    def remember_safe_steer(self, steer: float) -> float:
        self.last_safe_steer = float(steer)
        self.last_safe_steer_stamp = self.now_seconds()
        return float(steer)

    def fallback_guidance_command(self, now_sec: float) -> float:
        if self.candidates and self.fallback_candidates():
            return self.least_risk_candidate(forbid_toward_object=self.state != self.RETURN_LINE)
        if self.last_safe_steer_stamp is not None and (now_sec - self.last_safe_steer_stamp) <= self.last_safe_steer_hold_s:
            return float(self.last_safe_steer)
        return float(self.theta_cmd)

    # =========================================================================
    # Funciones auxiliares de estado
    # =========================================================================
    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def distance_from_state_start(self) -> float:
        return math.hypot(self.x - self.state_start_x, self.y - self.state_start_y)

    def distance_from_avoidance_start(self) -> float:
        return math.hypot(self.x - self.avoidance_start_x, self.y - self.avoidance_start_y)

    # Centraliza los cambios de estado y reinicia las variables temporales
    # correspondientes a cada transición.
    def change_state(self, new_state: str, now_sec: float) -> None:
        if new_state != self.state:
            self.get_logger().info(f'State: {self.state} -> {new_state}')
        self.state = new_state
        self.state_start_time = now_sec
        self.state_start_x = self.x
        self.state_start_y = self.y
        if new_state != self.RETURN_LINE:
            self.return_stable_since = None
        if new_state == self.PASS_OBSTACLE:
            self.side_lost_since = None
            if not self.side_exit_confirmed:
                self.side_invalid_since = None
        if new_state == self.RETURN_LINE:
            self.front_clear_since = None
            self.return_target_valid = False
            self.return_best_abs_error = abs(self.lateral_error)
            self.return_last_improvement_stamp = now_sec
            self.return_path_blocked = False
            self.return_recovery_active = False

    # Devuelve el control al usuario al finalizar la asistencia o ante una
    # condición que invalida la maniobra.
    def release_to_user(
        self,
        now_sec: float,
        reason: str,
        preserve_route_line: bool = True,
    ) -> None:
        previous = self.state
        self.state = self.USER_GUIDED
        self.state_start_time = now_sec
        self.state_start_x = self.x
        self.state_start_y = self.y
        if not preserve_route_line:
            self.line_valid = False
            self.route_line_source = 'NONE'
        self.reactive_only_mode = False
        self.clear_manoeuvre_memory()
        self.get_logger().info(
            f'State: {previous} -> USER_GUIDED ({reason}) '
            f'route_line_preserved={preserve_route_line and self.line_valid}'
        )

    # Limita desviaciones angulares excesivas durante asistencia automática y
    # evita que el sistema derive hacia una maniobra equivalente a un giro en U.
    def apply_autonomous_turn_guard(
        self,
        target: float,
        assist_active: bool,
        now_sec: float,
    ) -> float:
        del now_sec
        self.autonomous_turn_guard_active = False
        if (
            not assist_active
            or self.state in (self.USER_GUIDED, self.MANUAL_TURN)
            or self.manual_turn_active
        ):
            self.autonomous_heading_anchor_valid = False
            self.autonomous_turn_deviation = 0.0
            return target

        if not self.autonomous_heading_anchor_valid:
            self.autonomous_heading_anchor = self.yaw
            self.autonomous_heading_anchor_valid = True

        deviation = wrap_angle(
            self.yaw - self.autonomous_heading_anchor
        )
        self.autonomous_turn_deviation = deviation
        abs_deviation = abs(deviation)
        increasing = target * deviation > 0.0
        if abs_deviation < self.autonomous_turn_soft_limit_rad or not increasing:
            return target

        self.autonomous_turn_guard_active = True
        self.set_warning('AUTONOMOUS_TURN_LIMIT_USER_REDUCE_SPEED')
        self.latch_haptic(self.HAPTIC_DANGER, 0.35)
        unwind_sign = -1 if deviation > 0.0 else 1

        options = self.valid_candidates() or self.fallback_candidates()
        unwind_options = [
            candidate for candidate in options
            if candidate.steer * unwind_sign >= 0.015
        ]
        unwind_command = 0.0
        if unwind_options:
            preferred_unwind = unwind_sign * self.autonomous_turn_unwind_steer_rad
            unwind_candidate = max(
                unwind_options,
                key=lambda candidate: (
                    1.5 * candidate.collision_distance
                    + 0.9 * candidate.clearance
                    + 0.4 * candidate.observed_ratio
                    - 0.35 * abs(candidate.steer - preferred_unwind)
                ),
            )
            unwind_command = self.remember_safe_steer(unwind_candidate.steer)
        front_too_close = (
            math.isfinite(self.front_collision_distance)
            and self.front_collision_distance < self.autonomous_turn_front_guard_m
        )
        if abs_deviation >= self.autonomous_turn_hard_limit_rad:
            if unwind_options:
                return unwind_command
            # With no safe unwind arc, centre rather than increasing the U-turn.
            return 0.0

        span = max(
            self.autonomous_turn_hard_limit_rad
            - self.autonomous_turn_soft_limit_rad,
            1e-3,
        )
        ratio = clamp(
            (abs_deviation - self.autonomous_turn_soft_limit_rad) / span,
            0.0,
            1.0,
        )
        if front_too_close and unwind_options:
            return unwind_command
        # Progressively replace any command that increases the autonomous turn
        # with a verified arc that unwinds it. If no unwind arc is currently
        # observed, centre the steering rather than continuing toward a U-turn.
        return (1.0 - ratio) * target + ratio * unwind_command

    def publish_geometry_debug(self) -> None:
        self.route_line_debug_pub.publish(
            Float64MultiArray(
                data=[
                    1.0 if self.line_valid else 0.0,
                    float(self.line_x0),
                    float(self.line_y0),
                    float(self.line_theta),
                    float(self.route_segment_id),
                    float(self.line_progress),
                    float(self.lateral_error),
                    float(self.heading_error),
                ]
            )
        )
        self.return_target_debug_pub.publish(
            Float64MultiArray(
                data=[
                    1.0 if self.return_target_valid else 0.0,
                    float(self.return_target_x),
                    float(self.return_target_y),
                    float(self.return_target_theta),
                    float(self.return_target_lookahead),
                ]
            )
        )
        self.locked_obstacle_debug_pub.publish(
            Float64MultiArray(
                data=[
                    1.0 if self.locked_obstacle_valid else 0.0,
                    float(self.locked_obstacle_x_local),
                    float(self.locked_obstacle_y_local),
                    float(self.locked_obstacle_radius),
                ]
            )
        )
        self.autonomous_turn_debug_pub.publish(
            Float64MultiArray(
                data=[
                    1.0 if self.autonomous_heading_anchor_valid else 0.0,
                    float(self.autonomous_heading_anchor),
                    float(self.autonomous_turn_deviation),
                    1.0 if self.autonomous_turn_guard_active else 0.0,
                ]
            )
        )

    # Filtro de primer orden con límite de velocidad angular del servo para
    # suavizar los cambios de dirección.
    def rate_limited_filter(self, target: float, current: float, dt: float) -> float:
        alpha = dt / max(self.steer_filter_tau_s, dt)
        filtered = current + alpha * (target - current)
        max_change = self.max_steer_rate_rad_s * dt
        return current + clamp(filtered - current, -max_change, max_change)

    # =========================================================================
    # Háptica y avisos de seguridad
    # =========================================================================
    # ------------------------------------------------------------------
    # Haptics and warnings
    # ------------------------------------------------------------------
    # Mantiene un patrón háptico activo durante un intervalo mínimo.
    def latch_haptic(self, pattern: int, duration_s: float) -> None:
        self.haptic_latch_pattern = int(pattern)
        self.haptic_latch_until = self.now_seconds() + duration_s

    def current_haptic(self, now_sec: float) -> int:
        return self.haptic_latch_pattern if now_sec <= self.haptic_latch_until else self.HAPTIC_NONE

    # Activa un aviso de seguridad y registra su causa.
    def set_warning(self, reason: str) -> None:
        self.warning_active = True
        self.warning_reason = str(reason)

    def report_warning_change(self) -> None:
        if self.warning_reason == self.last_warning_reason:
            return
        if self.warning_reason:
            self.get_logger().warn(
                f'SAFETY WARNING (no brake): {self.warning_reason}. '
                'Human push remains active; servo guidance continues.'
            )
        elif self.last_warning_reason:
            self.get_logger().info('Safety warning cleared.')
        self.last_warning_reason = self.warning_reason

    def report_haptic_change(self, pattern: int) -> None:
        if int(pattern) == int(self.last_haptic_reported):
            return
        names = {
            self.HAPTIC_NONE: 'NONE',
            self.HAPTIC_LEFT: 'TURN_LEFT',
            self.HAPTIC_RIGHT: 'TURN_RIGHT',
            self.HAPTIC_SLOW: 'REDUCE_SPEED',
            self.HAPTIC_DANGER: 'DANGER_REDUCE_SPEED',
            self.HAPTIC_RETURN_COMPLETE: 'ASSISTANCE_COMPLETE',
            self.HAPTIC_LOCALIZATION_LOST: 'LOCALIZATION_LOST_REACTIVE_GUIDANCE',
        }
        name = names.get(int(pattern), f'UNKNOWN_{int(pattern)}')
        if int(pattern) == self.HAPTIC_NONE:
            self.get_logger().info('HAPTIC_SIM: NONE')
        else:
            self.get_logger().warn(f'HAPTIC_SIM: {name}')
        self.last_haptic_reported = int(pattern)

    # Conversión del reloj ROS a segundos en coma flotante.
    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


# ============================================================================
# Punto de entrada del nodo
# ============================================================================
# Inicializa ROS 2, crea el controlador y mantiene activo el bucle de callbacks.
# ============================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MaplessBypassControllerNode()
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
