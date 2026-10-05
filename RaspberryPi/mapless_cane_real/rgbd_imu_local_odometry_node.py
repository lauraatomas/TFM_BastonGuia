#!/usr/bin/env python3
"""Odometría local de corto alcance basada en datos RGB-D alineados y una IMU.

*   la traslación RGB-D puede estimarse manteniendo fija la rotación de la IMU,
    lo que evita que pequeños errores visuales de rotación se conviertan en deriva de traslación;
*   se compensa el brazo de palanca rígido entre la cámara y la base, de modo que
    la rotación de una cámara montada frente a la base no se interprete erróneamente
    como una distancia adicional recorrida;
*   la distancia recorrida se proyecta sobre el arco plano cinemáticamente esperado
    en lugar de utilizar la norma (siempre positiva) del movimiento ruidoso
    hacia adelante o lateral;
*   la escala de traslación se adapta a la curvatura: se mantiene la calibración
    para línea recta, mientras que la escala se reduce durante giros pronunciados;
*   los fotogramas clave (*keyframes*) se registran con menor frecuencia y solo
    si tienen calidad suficiente, reduciendo así el número de pequeños incrementos
    sesgados que se encadenan permanentemente;
*   un presupuesto de deriva explícito y nuevas herramientas de diagnóstico hacen
    visible la incertidumbre acumulada sin necesidad de retroalimentar el controlador
    con datos de referencia de Gazebo.

*   el giroscopio proporciona una referencia absoluta de guiñada (*yaw*) a corto plazo,
    en lugar de añadir un pequeño error visual de guiñada en cada imagen;
*   el movimiento RGB-D se estima con respecto a un fotograma clave, mejorando
    la relación señal-ruido cuando el desplazamiento entre fotogramas es de
    apenas unos milímetros;
*   la traslación se proyecta sobre una trayectoria no holónoma y se integra
    utilizando la orientación del giroscopio, evitando así grandes movimientos
    laterales ficticios;
*   la indicación heredada de empuje del usuario sigue siendo opcional para la
    simulación, pero está desactivada en el bastón real, ya que la propulsión
    y la detención dependen totalmente del usuario humano;
*   la confianza incluye los residuos, la proporción de valores válidos (*inliers*),
    la plausibilidad del movimiento y... Concordancia de guiñada entre RGB-D y giroscopio.

"""

# ============================================================================
# Importaciones
# ============================================================================
# Este nodo combina procesamiento de imagen, álgebra numérica y comunicaciones
# ROS 2 para estimar la odometría local del bastón a partir de la RealSense D435
# y de la IMU.
# ============================================================================

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Bool, Float64, Int32, String
from std_srvs.srv import Trigger

from .image_utils import image_to_bgr8, image_to_depth_metres


# Limita un valor al intervalo especificado. Se utiliza de forma recurrente
# para mantener escalas, probabilidades, velocidades y factores de mezcla dentro
# de rangos físicamente razonables.

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Normaliza un ángulo al intervalo [-pi, pi], evitando discontinuidades al
# integrar o comparar orientaciones de guiñada.

def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

# Conversión de marcas temporales ROS a segundos en coma flotante.

def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# Extrae el ángulo de guiñada (yaw) de un cuaternión.

def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# Genera un cuaternión equivalente a una rotación planar definida únicamente
# por el ángulo yaw.

def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)
    return q


# ============================================================================
# Nodo de odometría local RGB-D + IMU
# ============================================================================
# El objetivo del nodo es estimar el movimiento local del bastón sin utilizar
# odometría de ruedas. Para ello combina:
#
#   - características visuales ORB extraídas de la imagen RGB;
#   - profundidad alineada de la RealSense;
#   - rotación de corto plazo proporcionada por la IMU;
#   - restricciones cinemáticas del movimiento plano del bastón.
#
# La pose resultante se publica en /local_odom y se acompaña de una medida de
# confianza y diversos indicadores de diagnóstico.
# ============================================================================

class RgbdImuLocalOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__('rgbd_imu_local_odometry_node')

        # ------------------------------------------------------------------
        # Entradas y salidas principales
        # ------------------------------------------------------------------
        # Topics de imagen RGB, profundidad alineada, información intrínseca de
        # la cámara e IMU. El motion_hint es opcional y no se utiliza por defecto
        # en el bastón real.
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('imu_topic', '/imu/data_raw')
        self.declare_parameter('motion_hint_topic', '/hardware/user_push_hint')
        self.declare_parameter('use_imu', True)
        self.declare_parameter('use_motion_hint', False)
        self.declare_parameter('odom_topic', '/local_odom')
        self.declare_parameter('publish_debug_images', True)
        self.declare_parameter('debug_features_topic', '/debug/vo_features_image')

        # ------------------------------------------------------------------
        # Parámetros de procesamiento visual
        # ------------------------------------------------------------------
        # Controlan frecuencia de ejecución, sincronización RGB-D, escalado de
        # imagen, número de características ORB, filtrado CLAHE y criterios de
        # correspondencia entre descriptores.
        self.declare_parameter('process_rate_hz', 15.0)
        self.declare_parameter('max_sync_error_s', 0.080)
        self.declare_parameter('image_scale', 0.75)
        self.declare_parameter('orb_features', 1200)
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('ratio_test', 0.76)
        self.declare_parameter('min_matches', 24)
        self.declare_parameter('min_inliers', 16)
        self.declare_parameter('min_inlier_ratio', 0.32)
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 5.50)

        # ------------------------------------------------------------------
        # Estimación métrica 3D-3D
        # ------------------------------------------------------------------
        # Cuando existe profundidad válida en ambos fotogramas, se reconstruyen
        # pares de puntos 3D y se estima directamente la transformación rígida.
        # Preferred metric 3-D/3-D estimator.
        self.declare_parameter('use_rgbd_3d3d', True)
        self.declare_parameter('min_3d3d_matches', 18)
        self.declare_parameter('rigid_ransac_iterations', 100)
        self.declare_parameter('rigid_inlier_threshold_m', 0.050)
        self.declare_parameter('rigid_max_rmse_m', 0.050)

        # ------------------------------------------------------------------
        # Método PnP de respaldo
        # ------------------------------------------------------------------
        # Si no hay suficientes pares 3D-3D, se utiliza una estimación PnP entre
        # puntos 3D del keyframe y sus correspondencias 2D en la imagen actual.
        # PnP fallback.
        self.declare_parameter('pnp_reprojection_error_px', 3.5)
        self.declare_parameter('pnp_max_rmse_px', 3.5)

        # ------------------------------------------------------------------
        # Gestión de keyframes
        # ------------------------------------------------------------------
        # Los keyframes actúan como referencias temporales estables para evitar
        # integrar continuamente desplazamientos visuales muy pequeños y ruidosos.
        # Keyframe motion accumulation.
        self.declare_parameter('keyframe_min_translation_m', 0.060)
        self.declare_parameter('keyframe_min_yaw_rad', 0.060)
        self.declare_parameter('keyframe_max_age_s', 0.90)
        self.declare_parameter('keyframe_min_confidence', 0.55)
        self.declare_parameter('keyframe_min_inliers_for_commit', 30)
        self.declare_parameter('keyframe_force_max_age_s', 1.35)

        # ------------------------------------------------------------------
        # Escalado métrico de la traslación
        # ------------------------------------------------------------------
        # La escala de traslación puede adaptarse a la curvatura de la trayectoria:
        # se permite una calibración diferente para marcha recta y para giros.
        # Metric translation. ``translation_scale`` is retained as a legacy
        # fallback when adaptive scaling is disabled.
        self.declare_parameter('translation_scale', 1.0)
        self.declare_parameter('adaptive_translation_scale_enabled', True)
        self.declare_parameter('translation_scale_straight', 1.48)
        self.declare_parameter('translation_scale_turn', 1.20)
        self.declare_parameter('scale_curvature_start_rad_m', 0.20)
        self.declare_parameter('scale_curvature_full_rad_m', 1.20)
        self.declare_parameter('use_arc_projection', True)
        self.declare_parameter('arc_lateral_residual_scale_m', 0.060)

        # ------------------------------------------------------------------
        # Traslación condicionada por la rotación de la IMU
        # ------------------------------------------------------------------
        # Si la orientación inercial es reciente, la rotación relativa se fija
        # mediante la IMU y el bloque RGB-D estima únicamente la traslación.
        # Esto reduce errores de traslación inducidos por pequeñas rotaciones
        # visuales espurias.
        # When a fresh IMU heading exists, estimate only translation from the
        # RGB-D correspondences while keeping the relative rotation fixed.
        self.declare_parameter('imu_constrained_translation', True)
        self.declare_parameter('fixed_rotation_inlier_threshold_m', 0.045)
        self.declare_parameter('fixed_rotation_max_rmse_m', 0.045)

        # ------------------------------------------------------------------
        # Geometría de montaje de la cámara
        # ------------------------------------------------------------------
        # Posición rígida de la RealSense respecto a base_footprint. Esta
        # información permite compensar el movimiento aparente de la cámara
        # debido a su separación respecto al centro de giro del bastón.
        # Camera origin expressed in base_footprint.  This translation is needed
        # to remove the apparent camera motion caused purely by base rotation.
        self.declare_parameter('camera_x_m', 0.13)
        self.declare_parameter('camera_y_m', 0.0)
        self.declare_parameter('camera_z_m', 0.15)

        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('backward_noise_tolerance_m', 0.015)

        # ------------------------------------------------------------------
        # Criterios de plausibilidad dinámica
        # ------------------------------------------------------------------
        # Límites utilizados para rechazar incrementos visuales incompatibles
        # con las velocidades, aceleraciones y giros esperables del sistema.
        # Motion plausibility.
        self.declare_parameter('max_translation_per_frame_m', 0.35)
        self.declare_parameter('max_yaw_per_frame_rad', 0.40)
        self.declare_parameter('max_speed_m_s', 1.15)
        self.declare_parameter('max_yaw_rate_rad_s', 2.2)
        self.declare_parameter('max_acceleration_m_s2', 5.0)
        self.declare_parameter('velocity_filter_tau_s', 0.20)
        self.declare_parameter('confidence_filter_tau_s', 0.22)
        self.declare_parameter('valid_confidence_threshold', 0.42)
        self.declare_parameter('valid_enter_confidence', 0.40)
        self.declare_parameter('valid_exit_confidence', 0.25)

        # ------------------------------------------------------------------
        # Configuración de la IMU
        # ------------------------------------------------------------------
        # Se admite orientación absoluta cuando está disponible. En caso
        # contrario se integra la velocidad angular del giroscopio.
        # También se incluye calibración automática del bias de yaw.
        # IMU and mounting.  Fused orientation is preferred when the IMU
        # publishes it; raw gyro integration remains the fallback for hardware
        # that only provides angular velocity.
        self.declare_parameter('imu_yaw_weight', 0.98)
        self.declare_parameter('use_imu_orientation', False)
        # Some simulation bridges publish a valid unit quaternion but mark the
        # orientation covariance as unavailable.  This opt-in accepts that
        # quaternion; keep it false for a raw real IMU that does not fuse yaw.
        self.declare_parameter('accept_imu_orientation_without_covariance', False)
        self.declare_parameter('imu_orientation_yaw_offset_rad', 0.0)
        self.declare_parameter('max_gyro_integration_dt_s', 0.75)
        self.declare_parameter('imu_timeout_s', 0.25)
        self.declare_parameter('imu_disagreement_scale_rad', 0.12)
        self.declare_parameter('gyro_bias_z', 0.0)
        self.declare_parameter('auto_calibrate_gyro', True)
        self.declare_parameter('gyro_calibration_samples', 150)
        self.declare_parameter('camera_pitch_down_rad', 0.0)
        self.declare_parameter('camera_yaw_offset_rad', 0.0)

        # ------------------------------------------------------------------
        # Supresión de deriva en reposo
        # ------------------------------------------------------------------
        # La pose solo se congela cuando coinciden evidencia visual, baja
        # velocidad angular y, opcionalmente, ausencia prolongada de movimiento.
        # Stationary drift suppression.  The push command only indicates that a
        # user is no longer requesting movement.  Visual speed and gyro must also
        # be small before the pose is frozen, so coasting is not discarded.
        self.declare_parameter('motion_hint_timeout_s', 0.40)
        self.declare_parameter('zero_command_threshold', 0.02)
        self.declare_parameter('zero_command_hold_s', 0.65)
        self.declare_parameter('stationary_visual_speed_m_s', 0.035)
        self.declare_parameter('stationary_gyro_rad_s', 0.035)
        self.declare_parameter('stationary_keyframe_refresh_s', 0.35)

        # ------------------------------------------------------------------
        # Dead reckoning acotado ante pérdidas visuales breves
        # ------------------------------------------------------------------
        # Durante interrupciones muy cortas de la odometría visual se propaga la
        # posición usando la última velocidad visual y el yaw de la IMU.
        # La duración está expresamente limitada para evitar deriva inercial.
        # Bridge short visual drop-outs with bounded dead reckoning.  It uses
        # only the last visually measured speed and the current IMU heading; it
        # is deliberately limited to less than one second and therefore is not
        # a substitute for wheel odometry.
        self.declare_parameter('visual_loss_dead_reckon_enabled', True)
        self.declare_parameter('visual_loss_dead_reckon_max_s', 0.55)
        self.declare_parameter('visual_loss_speed_decay_tau_s', 0.45)
        self.declare_parameter('visual_loss_min_speed_m_s', 0.025)

        # ------------------------------------------------------------------
        # Presupuesto explícito de deriva
        # ------------------------------------------------------------------
        # La incertidumbre acumulada no corrige la pose, pero se incorpora a la
        # covarianza y a los diagnósticos para reflejar que recorridos largos son
        # menos fiables que trayectos cortos.
        # Honest local uncertainty budget.  This does not correct the pose; it
        # grows covariance and provides diagnostics so long route segments are
        # not presented as equally certain as short ones.
        self.declare_parameter('drift_base_fraction_per_m', 0.010)
        self.declare_parameter('drift_quality_fraction_per_m', 0.070)
        self.declare_parameter('drift_residual_gain', 0.020)

        self.declare_parameter('frame_id', 'local_odom')
        self.declare_parameter('child_frame_id', 'base_footprint')

        # ------------------------------------------------------------------
        # Lectura y almacenamiento de parámetros
        # ------------------------------------------------------------------
        # Los valores declarados anteriormente se copian a atributos internos
        # para evitar consultas repetitivas al servidor de parámetros.
        gp = lambda name: self.get_parameter(name).value
        self.color_topic = str(gp('color_topic'))
        self.depth_topic = str(gp('depth_topic'))
        self.camera_info_topic = str(gp('camera_info_topic'))
        self.imu_topic = str(gp('imu_topic'))
        self.motion_hint_topic = str(gp('motion_hint_topic'))
        self.use_imu = bool(gp('use_imu'))
        self.use_motion_hint = bool(gp('use_motion_hint'))
        self.odom_topic = str(gp('odom_topic'))
        self.publish_debug_images = bool(gp('publish_debug_images'))
        self.debug_features_topic = str(gp('debug_features_topic'))

        self.process_rate_hz = float(gp('process_rate_hz'))
        self.max_sync_error_s = float(gp('max_sync_error_s'))
        self.image_scale = float(gp('image_scale'))
        self.ratio_test = float(gp('ratio_test'))
        self.min_matches = int(gp('min_matches'))
        self.min_inliers = int(gp('min_inliers'))
        self.min_inlier_ratio = float(gp('min_inlier_ratio'))
        self.min_depth_m = float(gp('min_depth_m'))
        self.max_depth_m = float(gp('max_depth_m'))

        self.use_rgbd_3d3d = bool(gp('use_rgbd_3d3d'))
        self.min_3d3d_matches = int(gp('min_3d3d_matches'))
        self.rigid_ransac_iterations = int(gp('rigid_ransac_iterations'))
        self.rigid_inlier_threshold_m = float(gp('rigid_inlier_threshold_m'))
        self.rigid_max_rmse_m = float(gp('rigid_max_rmse_m'))
        self.pnp_reprojection_error_px = float(gp('pnp_reprojection_error_px'))
        self.pnp_max_rmse_px = float(gp('pnp_max_rmse_px'))

        self.keyframe_min_translation_m = float(gp('keyframe_min_translation_m'))
        self.keyframe_min_yaw_rad = float(gp('keyframe_min_yaw_rad'))
        self.keyframe_max_age_s = float(gp('keyframe_max_age_s'))
        self.keyframe_min_confidence = float(gp('keyframe_min_confidence'))
        self.keyframe_min_inliers_for_commit = int(
            gp('keyframe_min_inliers_for_commit')
        )
        self.keyframe_force_max_age_s = float(gp('keyframe_force_max_age_s'))

        self.translation_scale = float(gp('translation_scale'))
        self.adaptive_translation_scale_enabled = bool(
            gp('adaptive_translation_scale_enabled')
        )
        self.translation_scale_straight = float(gp('translation_scale_straight'))
        self.translation_scale_turn = float(gp('translation_scale_turn'))
        self.scale_curvature_start_rad_m = float(
            gp('scale_curvature_start_rad_m')
        )
        self.scale_curvature_full_rad_m = float(
            gp('scale_curvature_full_rad_m')
        )
        self.use_arc_projection = bool(gp('use_arc_projection'))
        self.arc_lateral_residual_scale_m = float(
            gp('arc_lateral_residual_scale_m')
        )
        self.imu_constrained_translation = bool(
            gp('imu_constrained_translation')
        )
        self.fixed_rotation_inlier_threshold_m = float(
            gp('fixed_rotation_inlier_threshold_m')
        )
        self.fixed_rotation_max_rmse_m = float(
            gp('fixed_rotation_max_rmse_m')
        )
        self.camera_x_m = float(gp('camera_x_m'))
        self.camera_y_m = float(gp('camera_y_m'))
        self.camera_z_m = float(gp('camera_z_m'))

        self.allow_reverse = bool(gp('allow_reverse'))
        self.backward_noise_tolerance_m = float(gp('backward_noise_tolerance_m'))

        self.max_translation_per_frame_m = float(gp('max_translation_per_frame_m'))
        self.max_yaw_per_frame_rad = float(gp('max_yaw_per_frame_rad'))
        self.max_speed_m_s = float(gp('max_speed_m_s'))
        self.max_yaw_rate_rad_s = float(gp('max_yaw_rate_rad_s'))
        self.max_acceleration_m_s2 = float(gp('max_acceleration_m_s2'))
        self.velocity_filter_tau_s = float(gp('velocity_filter_tau_s'))
        self.confidence_filter_tau_s = float(gp('confidence_filter_tau_s'))
        self.valid_confidence_threshold = float(gp('valid_confidence_threshold'))
        self.valid_enter_confidence = float(gp('valid_enter_confidence'))
        self.valid_exit_confidence = float(gp('valid_exit_confidence'))

        self.imu_yaw_weight = float(gp('imu_yaw_weight'))
        self.use_imu_orientation = bool(gp('use_imu_orientation'))
        self.accept_imu_orientation_without_covariance = bool(
            gp('accept_imu_orientation_without_covariance')
        )
        self.imu_orientation_yaw_offset_rad = float(gp('imu_orientation_yaw_offset_rad'))
        self.max_gyro_integration_dt_s = float(gp('max_gyro_integration_dt_s'))
        self.imu_timeout_s = float(gp('imu_timeout_s'))
        self.imu_disagreement_scale_rad = float(gp('imu_disagreement_scale_rad'))
        self.gyro_bias_z = float(gp('gyro_bias_z'))
        self.auto_calibrate_gyro = bool(gp('auto_calibrate_gyro'))
        self.gyro_calibration_samples = int(gp('gyro_calibration_samples'))
        self.camera_pitch_down_rad = float(gp('camera_pitch_down_rad'))
        self.camera_yaw_offset_rad = float(gp('camera_yaw_offset_rad'))

        self.motion_hint_timeout_s = float(gp('motion_hint_timeout_s'))
        self.zero_command_threshold = float(gp('zero_command_threshold'))
        self.zero_command_hold_s = float(gp('zero_command_hold_s'))
        self.stationary_visual_speed_m_s = float(gp('stationary_visual_speed_m_s'))
        self.stationary_gyro_rad_s = float(gp('stationary_gyro_rad_s'))
        self.stationary_keyframe_refresh_s = float(gp('stationary_keyframe_refresh_s'))
        self.visual_loss_dead_reckon_enabled = bool(gp('visual_loss_dead_reckon_enabled'))
        self.visual_loss_dead_reckon_max_s = float(gp('visual_loss_dead_reckon_max_s'))
        self.visual_loss_speed_decay_tau_s = float(gp('visual_loss_speed_decay_tau_s'))
        self.visual_loss_min_speed_m_s = float(gp('visual_loss_min_speed_m_s'))
        self.drift_base_fraction_per_m = float(gp('drift_base_fraction_per_m'))
        self.drift_quality_fraction_per_m = float(
            gp('drift_quality_fraction_per_m')
        )
        self.drift_residual_gain = float(gp('drift_residual_gain'))

        self.frame_id = str(gp('frame_id'))
        self.child_frame_id = str(gp('child_frame_id'))

        # ------------------------------------------------------------------
        # Inicialización de herramientas de visión
        # ------------------------------------------------------------------
        # CLAHE mejora el contraste local; ORB extrae características y
        # BFMatcher realiza la correspondencia binaria entre descriptores.
        self.clahe = cv2.createCLAHE(
            clipLimit=float(gp('clahe_clip_limit')),
            tileGridSize=(8, 8),
        )
        self.use_clahe = bool(gp('use_clahe'))
        self.orb = cv2.ORB_create(
            nfeatures=int(gp('orb_features')),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=20,
            fastThreshold=12,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.rng = np.random.default_rng(42)

        # ------------------------------------------------------------------
        # Estado de las entradas RGB-D
        # ------------------------------------------------------------------
        # Se almacenan los últimos mensajes recibidos hasta disponer de una pareja
        # suficientemente sincronizada para ejecutar una estimación.
        self.camera_info: Optional[CameraInfo] = None
        self.latest_color: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_color_stamp = 0.0
        self.latest_depth_stamp = 0.0
        self.last_processed_stamp = -1.0

        # ------------------------------------------------------------------
        # Estado del keyframe actual
        # ------------------------------------------------------------------
        # Incluye profundidad, puntos ORB, descriptores y pose asociada.
        # Keyframe RGB-D data.  The old prev_* names are retained inside the
        # estimator, but they now refer to the current keyframe.
        self.prev_depth: Optional[np.ndarray] = None
        self.prev_keypoints = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_frame_stamp: Optional[float] = None
        self.keyframe_x = 0.0
        self.keyframe_y = 0.0
        self.keyframe_yaw = 0.0

        # Estado estimado de la odometría local: posición, orientación,
        # velocidades, confianza y validez de la solución.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.last_raw_speed = 0.0
        self.confidence = 0.0
        self.valid = False

        self.last_output_x = 0.0
        self.last_output_y = 0.0
        self.last_output_yaw = 0.0
        self.last_output_stamp: Optional[float] = None

        # Estado interno de la IMU: yaw integrado, referencia inicial,
        # disponibilidad de orientación absoluta y calibración del giroscopio.
        self.imu_yaw_integrated = 0.0
        self.imu_origin_yaw = 0.0
        self.imu_origin_set = False
        self.imu_orientation_available = False
        self.imu_orientation_yaw = 0.0
        self.imu_orientation_origin_yaw = 0.0
        self.imu_orientation_origin_set = False
        self.imu_orientation_stamp: Optional[float] = None
        self.imu_orientation_accept_logged = False
        self.imu_orientation_reject_logged = False
        self.last_imu_stamp: Optional[float] = None
        self.last_gyro_z = 0.0
        self.gyro_calibration_values: list[float] = []
        self.gyro_calibrated = not self.auto_calibrate_gyro

        # Estado opcional asociado a la indicación de movimiento del usuario.
        self.motion_hint = 0.0
        self.motion_hint_stamp: Optional[float] = None
        self.zero_command_since: Optional[float] = None

        # Variables utilizadas para limitar la propagación durante pérdidas
        # visuales breves.
        self.last_visual_success_stamp: Optional[float] = None
        self.last_dead_reckon_stamp: Optional[float] = None
        self.dead_reckon_speed_m_s = 0.0

        # Variables de diagnóstico y métricas de calidad de la odometría visual.
        self.last_debug_time = self.get_clock().now()
        self.debug_feature_image: Optional[np.ndarray] = None
        self.last_feature_count = 0
        self.last_match_count = 0
        self.last_inlier_count = 0
        self.last_estimation_method = 'NONE'
        self.last_estimation_residual = math.inf
        self.last_effective_translation_scale = self.translation_scale_straight
        self.last_motion_curvature = 0.0
        self.last_arc_lateral_residual = 0.0
        self.distance_since_reset_m = 0.0
        self.estimated_drift_variance_m2 = 0.0

        # ------------------------------------------------------------------
        # Suscripciones ROS 2
        # ------------------------------------------------------------------
        # RGB, profundidad y CameraInfo son siempre necesarios. IMU y motion_hint
        # se activan en función de los parámetros configurados.
        self.create_subscription(Image, self.color_topic, self.color_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        if self.use_imu:
            self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data)
        if self.use_motion_hint:
            self.create_subscription(Float64, self.motion_hint_topic, self.motion_hint_callback, 10)

        # ------------------------------------------------------------------
        # Publicadores ROS 2
        # ------------------------------------------------------------------
        # Además de la odometría se publican confianza, validez y numerosas
        # señales de diagnóstico para evaluar el comportamiento del estimador.
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.valid_pub = self.create_publisher(Bool, '/local_odom_valid', 10)
        self.confidence_pub = self.create_publisher(Float64, '/local_odom_confidence', 10)
        self.debug_pub = self.create_publisher(String, '/local_odom_debug', 10)
        self.reset_event_pub = self.create_publisher(Bool, '/local_odom_reset_event', 10)
        self.feature_debug_pub = self.create_publisher(
            Image, self.debug_features_topic, 2
        )
        self.feature_count_pub = self.create_publisher(
            Int32, '/debug/vo_feature_count', 10
        )
        self.match_count_pub = self.create_publisher(
            Int32, '/debug/vo_match_count', 10
        )
        self.inlier_count_pub = self.create_publisher(
            Int32, '/debug/vo_inlier_count', 10
        )
        self.imu_source_pub = self.create_publisher(
            String, '/debug/imu_yaw_source', 10
        )
        self.imu_yaw_pub = self.create_publisher(
            Float64, '/debug/imu_relative_yaw', 10
        )
        self.translation_scale_pub = self.create_publisher(
            Float64, '/debug/vo_translation_scale', 10
        )
        self.motion_curvature_pub = self.create_publisher(
            Float64, '/debug/vo_motion_curvature', 10
        )
        self.arc_lateral_residual_pub = self.create_publisher(
            Float64, '/debug/vo_arc_lateral_residual', 10
        )
        self.distance_since_reset_pub = self.create_publisher(
            Float64, '/debug/vo_distance_since_reset', 10
        )
        self.estimated_drift_pub = self.create_publisher(
            Float64, '/debug/vo_estimated_drift', 10
        )
        # Servicio que permite redefinir el origen local de la odometría.
        # Resulta útil tras giros manuales o cambios de segmento de trayectoria.
        self.reset_srv = self.create_service(Trigger, '/reset_local_odom', self.reset_callback)
        self.timer = self.create_timer(
            1.0 / max(self.process_rate_hz, 1.0),
            self.process_latest_pair,
        )

        self.get_logger().info(
            f'RGB-D/IMU local odometry v10 started | color={self.color_topic} '
            f'depth={self.depth_topic} use_imu={self.use_imu} '
            f'keyframes=quality_gated imu_fixed_translation={self.imu_constrained_translation} motion_hint={self.use_motion_hint}'
        )

    # =========================================================================
    # Entradas de sensores
    # =========================================================================
    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def color_callback(self, msg: Image) -> None:
        self.latest_color = msg
        self.latest_color_stamp = stamp_to_seconds(msg.header.stamp) or self.now_seconds()

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth = msg
        self.latest_depth_stamp = stamp_to_seconds(msg.header.stamp) or self.now_seconds()

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    # Actualiza la indicación opcional de movimiento y controla cuánto tiempo
    # lleva el sistema sin recibir una petición significativa de avance.
    def motion_hint_callback(self, msg: Float64) -> None:
        now = self.now_seconds()
        self.motion_hint = float(msg.data)
        self.motion_hint_stamp = now
        if abs(self.motion_hint) <= self.zero_command_threshold:
            if self.zero_command_since is None:
                self.zero_command_since = now
        else:
            self.zero_command_since = None

    # ------------------------------------------------------------------
    # Procesamiento de la IMU
    # ------------------------------------------------------------------
    # Si existe orientación absoluta válida se utiliza como referencia de yaw.
    # En caso contrario se integra la velocidad angular del eje Z, compensando
    # previamente el bias estimado durante la calibración inicial.
    def imu_callback(self, msg: Imu) -> None:
        stamp = stamp_to_seconds(msg.header.stamp) or self.now_seconds()
        gyro_raw = float(msg.angular_velocity.z)

        # Gazebo and many fused IMU drivers publish an absolute orientation.
        # Using it prevents missed gyro samples under CPU load from accumulating
        # a large yaw lag.  Raw IMUs usually mark orientation unavailable with a
        # negative covariance or an invalid zero quaternion, in which case the
        # code transparently falls back to gyro integration.
        q = msg.orientation
        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        covariance_available = (
            len(msg.orientation_covariance) == 9
            and float(msg.orientation_covariance[0]) >= 0.0
        )
        quaternion_finite = all(
            math.isfinite(value) for value in (q.x, q.y, q.z, q.w)
        )
        orientation_allowed = covariance_available or (
            self.accept_imu_orientation_without_covariance
            and quaternion_finite
            and 0.70 <= q_norm <= 1.30
        )
        if self.use_imu_orientation and quaternion_finite and q_norm > 0.5 and orientation_allowed:
            # Normalise before extracting yaw.  A few bridges preserve the
            # quaternion direction but not its exact norm.
            inv_norm = 1.0 / max(q_norm, 1e-9)
            class _Q:
                pass
            qn = _Q()
            qn.x = q.x * inv_norm
            qn.y = q.y * inv_norm
            qn.z = q.z * inv_norm
            qn.w = q.w * inv_norm
            absolute_yaw = wrap_angle(
                yaw_from_quaternion(qn) + self.imu_orientation_yaw_offset_rad
            )
            self.imu_orientation_yaw = absolute_yaw
            self.imu_orientation_stamp = stamp
            self.imu_orientation_available = True
            if not self.imu_orientation_origin_set:
                self.imu_orientation_origin_yaw = absolute_yaw
                self.imu_orientation_origin_set = True
            if not self.imu_orientation_accept_logged:
                mode = 'covariance-valid' if covariance_available else 'opt-in-no-covariance'
                self.get_logger().info(
                    f'Using absolute IMU orientation ({mode}, q_norm={q_norm:.3f}).'
                )
                self.imu_orientation_accept_logged = True
        elif self.use_imu_orientation and not self.imu_orientation_reject_logged:
            self.get_logger().warn(
                'Absolute IMU orientation not accepted; using gyro integration. '
                f'q_norm={q_norm:.3f} covariance_available={covariance_available} '
                f'accept_without_covariance={self.accept_imu_orientation_without_covariance}'
            )
            self.imu_orientation_reject_logged = True

        if self.auto_calibrate_gyro and not self.gyro_calibrated:
            self.gyro_calibration_values.append(gyro_raw)
            self.last_imu_stamp = stamp
            if len(self.gyro_calibration_values) >= self.gyro_calibration_samples:
                self.gyro_bias_z = float(np.median(self.gyro_calibration_values))
                self.gyro_calibrated = True
                self.imu_yaw_integrated = 0.0
                self.imu_origin_yaw = 0.0
                self.imu_origin_set = True
                self.get_logger().info(
                    f'Gyro bias calibrated: {self.gyro_bias_z:.6f} rad/s'
                )
            return

        gyro = gyro_raw - self.gyro_bias_z
        self.last_gyro_z = gyro
        if self.last_imu_stamp is not None:
            dt = stamp - self.last_imu_stamp
            if 0.0 < dt < self.max_gyro_integration_dt_s:
                self.imu_yaw_integrated = wrap_angle(
                    self.imu_yaw_integrated + gyro * dt
                )
        self.last_imu_stamp = stamp
        if not self.imu_origin_set:
            self.imu_origin_yaw = self.imu_yaw_integrated
            self.imu_origin_set = True

    # =========================================================================
    # Procesamiento principal RGB-D
    # =========================================================================
    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    # Esta función constituye el ciclo principal del estimador:
    #   1. verifica disponibilidad y sincronización de RGB y profundidad;
    #   2. convierte y preprocesa las imágenes;
    #   3. detecta características ORB;
    #   4. estima el movimiento respecto al keyframe;
    #   5. fusiona la rotación visual con la IMU;
    #   6. aplica restricciones cinemáticas y de plausibilidad;
    #   7. actualiza pose, confianza, covarianza y keyframe.
    def process_latest_pair(self) -> None:
        if self.camera_info is None or self.latest_color is None or self.latest_depth is None:
            self.publish_state('WAITING_FOR_RGBD_OR_CAMERA_INFO')
            return

        sync_error = abs(self.latest_color_stamp - self.latest_depth_stamp)
        pair_stamp = max(self.latest_color_stamp, self.latest_depth_stamp)
        if sync_error > self.max_sync_error_s:
            self.degrade_confidence(0.78)
            self.publish_state(f'RGBD_NOT_SYNCHRONIZED dt={sync_error:.3f}s')
            return
        if pair_stamp <= self.last_processed_stamp:
            return
        self.last_processed_stamp = pair_stamp

        color = image_to_bgr8(self.latest_color)
        depth_m = image_to_depth_metres(self.latest_depth)
        if color is None or depth_m is None or color.shape[:2] != depth_m.shape[:2]:
            self.valid = False
            self.publish_state('IMAGE_CONVERSION_OR_ALIGNMENT_ERROR')
            return

        # La imagen RGB se convierte a escala de grises y opcionalmente se
        # mejora mediante CLAHE antes de extraer características ORB.
        gray_full = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            gray_full = self.clahe.apply(gray_full)
        gray, depth_scaled, k_scaled = self.resize_inputs(gray_full, depth_m)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        self.last_feature_count = len(keypoints)
        self.last_match_count = 0
        self.last_inlier_count = 0
        self.last_estimation_method = 'NONE'
        self.last_estimation_residual = math.inf
        if self.publish_debug_images:
            scaled_color = cv2.resize(
                color, (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            self.debug_feature_image = cv2.drawKeypoints(
                scaled_color, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        # Si la imagen actual no contiene suficientes características fiables,
        # no se intenta estimar una transformación visual.
        if descriptors is None or len(keypoints) < self.min_matches:
            self.propagate_during_visual_loss(pair_stamp)
            self.degrade_confidence(0.72)
            self.publish_state('TOO_FEW_CURRENT_FEATURES')
            return

        # El primer par RGB-D válido se instala como keyframe inicial.
        if self.prev_descriptors is None or self.prev_keypoints is None or self.prev_depth is None:
            self.install_keyframe(
                depth_scaled,
                keypoints,
                descriptors,
                pair_stamp,
                self.x,
                self.y,
                self.current_absolute_yaw(),
            )
            self.yaw = self.current_absolute_yaw()
            self.confidence = 0.0
            self.valid = False
            self.update_output_velocity(pair_stamp)
            self.publish_state('INITIALIZED_FIRST_KEYFRAME')
            return

        keyframe_stamp = pair_stamp if self.prev_frame_stamp is None else self.prev_frame_stamp
        keyframe_age = max(1e-3, pair_stamp - float(keyframe_stamp))
        imu_fresh = self.imu_is_fresh(pair_stamp)
        imu_delta_yaw = None
        if imu_fresh and self.imu_constrained_translation:
            imu_delta_yaw = wrap_angle(
                self.current_absolute_yaw() - self.keyframe_yaw
            )

        # Estimación del movimiento relativo entre el keyframe y la imagen actual.
        estimate = self.estimate_increment(
            keypoints,
            descriptors,
            depth_scaled,
            k_scaled,
            imu_delta_yaw=imu_delta_yaw,
        )

        # Ante un fallo temporal del estimador visual se conserva el yaw de la
        # IMU y, durante un intervalo breve, se permite una propagación acotada
        # de la traslación.
        if estimate is None:
            # Keep the pose bounded during a short visual gap.  Heading comes
            # from the IMU and translation is propagated only for the configured
            # short window using the last visual speed.
            self.propagate_during_visual_loss(pair_stamp)
            self.angular_speed = self.last_gyro_z if imu_fresh else 0.0
            self.degrade_confidence(0.68)
            if keyframe_age >= 1.8 * self.keyframe_max_age_s:
                self.install_keyframe(
                    depth_scaled,
                    keypoints,
                    descriptors,
                    pair_stamp,
                    self.x,
                    self.y,
                    self.yaw,
                )
            self.update_output_velocity(pair_stamp)
            self.publish_state('VISUAL_ODOMETRY_REJECTED_HOLD_TRANSLATION')
            return

        (
            delta_forward,
            delta_left,
            delta_yaw_vo,
            estimate_confidence,
            match_count,
            inlier_count,
            residual,
            method,
        ) = estimate
        self.last_match_count = int(match_count)
        self.last_inlier_count = int(inlier_count)
        self.last_estimation_method = str(method)
        self.last_estimation_residual = float(residual)

        # ------------------------------------------------------------------
        # Fusión de yaw visual e inercial
        # ------------------------------------------------------------------
        # El yaw obtenido por RGB-D se corrige hacia la referencia de la IMU
        # mediante un peso configurable, evitando integrar sesgos visuales.
        # Absolute short-term yaw reference.  Using an absolute reference avoids
        # repeatedly integrating a small fraction of visual yaw bias.
        vo_yaw_candidate = wrap_angle(self.keyframe_yaw + delta_yaw_vo)
        if imu_fresh:
            imu_yaw_candidate = self.current_absolute_yaw()
            yaw_candidate = wrap_angle(
                vo_yaw_candidate
                + clamp(self.imu_yaw_weight, 0.0, 1.0)
                * wrap_angle(imu_yaw_candidate - vo_yaw_candidate)
            )
            yaw_disagreement = abs(
                wrap_angle(vo_yaw_candidate - imu_yaw_candidate)
            )
            yaw_consistency = math.exp(
                -yaw_disagreement / max(self.imu_disagreement_scale_rad, 1e-3)
            )
        else:
            yaw_candidate = vo_yaw_candidate
            yaw_disagreement = 0.0
            yaw_consistency = 0.65 if self.use_imu else 1.0

        delta_yaw_path = wrap_angle(yaw_candidate - self.keyframe_yaw)

        # ------------------------------------------------------------------
        # Proyección sobre una trayectoria plana no holónoma
        # ------------------------------------------------------------------
        # El movimiento estimado se proyecta sobre el eje medio del arco esperado,
        # reduciendo desplazamientos laterales ficticios generados por ruido RGB-D.
        # For a planar non-holonomic base, the endpoint chord is aligned with
        # the mean heading of the arc.  Projecting onto that direction rejects
        # the always-positive bias introduced by hypot(df, dl) when lateral
        # translation is mostly RGB-D noise.
        half_yaw = 0.5 * delta_yaw_path
        arc_axis_x = math.cos(half_yaw)
        arc_axis_y = math.sin(half_yaw)
        projected_chord = (
            delta_forward * arc_axis_x + delta_left * arc_axis_y
        )
        arc_lateral_residual = (
            -delta_forward * arc_axis_y + delta_left * arc_axis_x
        )
        if self.use_arc_projection:
            raw_chord_signed = projected_chord
        else:
            raw_chord_signed = math.copysign(
                math.hypot(delta_forward, delta_left),
                delta_forward if abs(delta_forward) > 1e-6 else 1.0,
            )

        direction_sign = 1.0
        if raw_chord_signed < -self.backward_noise_tolerance_m:
            if self.allow_reverse:
                direction_sign = -1.0
            else:
                raw_chord_signed = 0.0
        elif raw_chord_signed < 0.0:
            raw_chord_signed = 0.0

        raw_chord = abs(raw_chord_signed)
        curvature = abs(delta_yaw_path) / max(raw_chord, 0.025)
        effective_scale = self.effective_translation_scale(curvature)
        chord = effective_scale * raw_chord

        if abs(delta_yaw_path) > 1e-3:
            denominator = 2.0 * math.sin(0.5 * abs(delta_yaw_path))
            if abs(denominator) > 1e-4:
                chord *= clamp(abs(delta_yaw_path) / denominator, 1.0, 1.12)

        path_distance = direction_sign * chord
        self.last_effective_translation_scale = effective_scale
        self.last_motion_curvature = curvature
        self.last_arc_lateral_residual = arc_lateral_residual
        raw_speed = abs(path_distance) / keyframe_age
        raw_yaw_rate = abs(delta_yaw_path) / keyframe_age
        raw_accel = abs(raw_speed - self.last_raw_speed) / keyframe_age

        # Evaluación de plausibilidad del incremento estimado según velocidad,
        # tasa de giro y aceleración.
        plausibility = 1.0
        if raw_speed > self.max_speed_m_s:
            plausibility *= max(
                0.0,
                1.0 - (raw_speed - self.max_speed_m_s) / max(self.max_speed_m_s, 1e-3),
            )
        if raw_yaw_rate > self.max_yaw_rate_rad_s:
            plausibility *= max(
                0.0,
                1.0 - (raw_yaw_rate - self.max_yaw_rate_rad_s)
                / max(self.max_yaw_rate_rad_s, 1e-3),
            )
        if raw_accel > self.max_acceleration_m_s2 and abs(path_distance) > 0.020:
            plausibility *= 0.60

        if (
            chord > self.max_translation_per_frame_m
            or abs(delta_yaw_path) > self.max_yaw_per_frame_rad
            or plausibility < 0.20
        ):
            self.degrade_confidence(0.50)
            if keyframe_age >= 1.8 * self.keyframe_max_age_s:
                self.install_keyframe(
                    depth_scaled,
                    keypoints,
                    descriptors,
                    pair_stamp,
                    self.x,
                    self.y,
                    yaw_candidate,
                )
            self.update_output_velocity(pair_stamp)
            self.publish_state(
                f'IMPLAUSIBLE_KEYFRAME_INCREMENT method={method} '
                f'speed={raw_speed:.2f} accel={raw_accel:.2f} '
                f'yaw_rate={raw_yaw_rate:.2f}'
            )
            return

        # Si existen evidencias suficientes de reposo, la traslación se anula
        # para evitar deriva acumulativa mientras el bastón permanece inmóvil.
        stationary = self.stationary_evidence(pair_stamp, raw_speed)
        if stationary:
            candidate_x = self.keyframe_x
            candidate_y = self.keyframe_y
            path_distance = 0.0
            raw_speed = 0.0
        else:
            yaw_mid = wrap_angle(
                self.keyframe_yaw + 0.5 * delta_yaw_path
            )
            candidate_x = self.keyframe_x + path_distance * math.cos(yaw_mid)
            candidate_y = self.keyframe_y + path_distance * math.sin(yaw_mid)

        # La distancia finalmente aceptada se utiliza también para actualizar
        # el presupuesto acumulado de deriva.
        accepted_step = math.hypot(
            float(candidate_x) - self.x,
            float(candidate_y) - self.y,
        )
        self.x = float(candidate_x)
        self.y = float(candidate_y)
        self.yaw = float(yaw_candidate)
        if accepted_step > 0.0:
            self.distance_since_reset_m += accepted_step
            step_sigma = accepted_step * (
                self.drift_base_fraction_per_m
                + self.drift_quality_fraction_per_m
                * (1.0 - clamp(estimate_confidence, 0.0, 1.0))
            )
            step_sigma += self.drift_residual_gain * max(residual, 0.0)
            self.estimated_drift_variance_m2 += step_sigma * step_sigma
        self.last_visual_success_stamp = pair_stamp
        self.last_dead_reckon_stamp = pair_stamp
        self.dead_reckon_speed_m_s = max(raw_speed, self.linear_speed)

        lateral_consistency = math.exp(
            -abs(arc_lateral_residual)
            / max(self.arc_lateral_residual_scale_m, 1e-3)
        )
        # La confianza combina calidad geométrica, plausibilidad dinámica,
        # coherencia del yaw y consistencia lateral respecto al arco esperado.
        confidence_raw = clamp(
            estimate_confidence
            * plausibility
            * yaw_consistency
            * lateral_consistency,
            0.0,
            1.0,
        )
        if stationary:
            confidence_raw = max(confidence_raw, 0.72 * estimate_confidence)

        confidence_alpha = keyframe_age / max(
            self.confidence_filter_tau_s,
            keyframe_age,
        )
        self.confidence += confidence_alpha * (confidence_raw - self.confidence)
        self.update_valid_state()
        self.last_raw_speed = raw_speed

        self.update_output_velocity(pair_stamp)

        # Se instala un nuevo keyframe cuando el movimiento es suficiente,
        # la calidad es adecuada o la referencia actual ha envejecido demasiado.
        motion_requests_keyframe = (
            abs(path_distance) >= self.keyframe_min_translation_m
            or abs(delta_yaw_path) >= self.keyframe_min_yaw_rad
            or keyframe_age >= self.keyframe_max_age_s
        )
        keyframe_quality_ok = (
            estimate_confidence >= self.keyframe_min_confidence
            and inlier_count >= self.keyframe_min_inliers_for_commit
            and lateral_consistency >= 0.30
        )
        commit_keyframe = (
            (stationary and keyframe_age >= self.stationary_keyframe_refresh_s)
            or (motion_requests_keyframe and keyframe_quality_ok)
            or keyframe_age >= self.keyframe_force_max_age_s
        )
        if commit_keyframe:
            self.install_keyframe(
                depth_scaled,
                keypoints,
                descriptors,
                pair_stamp,
                self.x,
                self.y,
                self.yaw,
            )

        self.publish_state(
            f'OK method={method} matches={match_count} inliers={inlier_count} '
            f'residual={residual:.4f} chord={chord:.3f} '
            f'df={delta_forward:.3f} dl={delta_left:.3f} '
            f'arc_lat={arc_lateral_residual:.3f} curvature={curvature:.2f} '
            f'scale={effective_scale:.2f} '
            f'dyaw_vo={delta_yaw_vo:.3f} yaw_disagree={yaw_disagreement:.3f} '
            f'imu_source={"ORIENTATION" if self.imu_orientation_available else "GYRO"} '
            f'key_age={keyframe_age:.2f} committed={commit_keyframe} '
            f'stationary={stationary} plausibility={plausibility:.2f}'
        )

    # =========================================================================
    # Estimación del movimiento
    # =========================================================================
    # ------------------------------------------------------------------
    # Motion estimation
    # ------------------------------------------------------------------
    # Realiza el emparejamiento de descriptores ORB y construye las
    # correspondencias 3D-3D y 3D-2D necesarias para los estimadores.
    def estimate_increment(
        self,
        current_keypoints,
        current_descriptors: np.ndarray,
        current_depth: np.ndarray,
        k_matrix: np.ndarray,
        imu_delta_yaw: Optional[float] = None,
    ) -> Optional[Tuple[float, float, float, float, int, int, float, str]]:
        try:
            knn = self.matcher.knnMatch(
                self.prev_descriptors,
                current_descriptors,
                k=2,
            )
        except cv2.error:
            return None

        good = [
            pair[0]
            for pair in knn
            if len(pair) == 2
            and pair[0].distance < self.ratio_test * pair[1].distance
        ]
        self.last_match_count = len(good)
        if len(good) < self.min_matches or self.prev_depth is None:
            return None

        fx, fy = float(k_matrix[0, 0]), float(k_matrix[1, 1])
        cx, cy = float(k_matrix[0, 2]), float(k_matrix[1, 2])
        h, w = self.prev_depth.shape
        prev_3d: list[tuple[float, float, float]] = []
        cur_3d: list[tuple[float, float, float]] = []
        pnp_object: list[tuple[float, float, float]] = []
        pnp_image: list[tuple[float, float]] = []

        for match in good:
            up, vp = self.prev_keypoints[match.queryIdx].pt
            uc, vc = current_keypoints[match.trainIdx].pt
            ip, jp = int(round(up)), int(round(vp))
            ic, jc = int(round(uc)), int(round(vc))
            if not (
                0 <= ip < w
                and 0 <= jp < h
                and 0 <= ic < w
                and 0 <= jc < h
            ):
                continue

            zp = float(self.prev_depth[jp, ip])
            zc = float(current_depth[jc, ic])
            if math.isfinite(zp) and self.min_depth_m <= zp <= self.max_depth_m:
                pp = ((up - cx) * zp / fx, (vp - cy) * zp / fy, zp)
                pnp_object.append(pp)
                pnp_image.append((uc, vc))
                if math.isfinite(zc) and self.min_depth_m <= zc <= self.max_depth_m:
                    pc = ((uc - cx) * zc / fx, (vc - cy) * zc / fy, zc)
                    prev_3d.append(pp)
                    cur_3d.append(pc)

        # Prioridad de estimadores:
        #   1. 3D-3D con rotación fijada por la IMU;
        #   2. transformación rígida 3D-3D libre;
        #   3. PnP como método de respaldo.
        if self.use_rgbd_3d3d and len(prev_3d) >= self.min_3d3d_matches:
            previous_points = np.asarray(prev_3d, dtype=np.float64)
            current_points = np.asarray(cur_3d, dtype=np.float64)

            # Preferred v10 estimator.  The IMU supplies the relative planar
            # rotation and RGB-D solves only translation.  This removes a major
            # source of translation bias during curved avoidance manoeuvres.
            if imu_delta_yaw is not None:
                fixed = self.estimate_translation_fixed_rotation(
                    previous_points,
                    current_points,
                    float(imu_delta_yaw),
                )
                if fixed is not None:
                    r_cur_prev, t_cur_prev, inliers, rmse = fixed
                    result = self.convert_camera_transform(
                        r_cur_prev,
                        t_cur_prev,
                        len(good),
                        inliers,
                        rmse,
                        'RGBD_IMU_FIXED_ROTATION',
                    )
                    if result is not None:
                        self.last_inlier_count = int(inliers)
                        self.last_estimation_method = 'RGBD_IMU_FIXED_ROTATION'
                        self.last_estimation_residual = float(rmse)
                        return result

            # Free rigid transform remains the fallback for stale / absent IMU
            # and for frames where fixed-rotation residuals are inconsistent.
            rigid = self.estimate_rigid_ransac(
                previous_points,
                current_points,
            )
            if rigid is not None:
                r_cur_prev, t_cur_prev, inliers, rmse = rigid
                result = self.convert_camera_transform(
                    r_cur_prev,
                    t_cur_prev,
                    len(good),
                    inliers,
                    rmse,
                    'RGBD_3D3D_KEYFRAME',
                )
                if result is not None:
                    self.last_inlier_count = int(inliers)
                    self.last_estimation_method = 'RGBD_3D3D_KEYFRAME'
                    self.last_estimation_residual = float(rmse)
                    return result

        if len(pnp_object) < self.min_matches:
            return None
        result = self.estimate_pnp(
            np.asarray(pnp_object, dtype=np.float32),
            np.asarray(pnp_image, dtype=np.float32),
            k_matrix,
            len(good),
        )
        if result is not None:
            self.last_inlier_count = int(result[5])
            self.last_estimation_method = str(result[7])
            self.last_estimation_residual = float(result[6])
        return result

    # Estima únicamente la traslación suponiendo conocida la rotación relativa
    # proporcionada por la IMU. Utiliza una mediana robusta y refinamientos por
    # inliers para reducir el efecto de correspondencias erróneas.
    def estimate_translation_fixed_rotation(
        self,
        prev_points: np.ndarray,
        cur_points: np.ndarray,
        delta_yaw_base: float,
    ) -> Optional[tuple[np.ndarray, np.ndarray, int, float]]:
        """Estimate camera translation while holding base yaw rotation fixed.

        ``prev_points`` and ``cur_points`` contain corresponding points in the
        previous and current optical camera frames.  The relative base yaw comes
        from the IMU.  Translation is the robust median of the offsets after
        applying that fixed rotation, followed by two inlier refinements.
        """
        count = prev_points.shape[0]
        if count < self.min_3d3d_matches:
            return None

        c = math.cos(delta_yaw_base)
        s = math.sin(delta_yaw_base)
        rotation_base_prev_cur = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        r_base_camera = self.camera_to_base_rotation()
        # Current camera orientation in previous camera coordinates.
        rotation_camera_prev_cur = (
            r_base_camera.T
            @ rotation_base_prev_cur
            @ r_base_camera
        )
        # Point mapping used by the matcher: previous camera -> current camera.
        rotation_cur_prev = rotation_camera_prev_cur.T

        rotated = (rotation_cur_prev @ prev_points.T).T
        offsets = cur_points - rotated
        translation_cur_prev = np.median(offsets, axis=0)

        threshold = max(self.fixed_rotation_inlier_threshold_m, 1e-3)
        mask = np.ones(count, dtype=bool)
        for _ in range(2):
            residuals = np.linalg.norm(
                offsets - translation_cur_prev,
                axis=1,
            )
            mask = residuals <= threshold
            inlier_count = int(np.count_nonzero(mask))
            if inlier_count < self.min_inliers:
                return None
            inlier_ratio = inlier_count / max(count, 1)
            if inlier_ratio < self.min_inlier_ratio:
                return None
            translation_cur_prev = np.median(offsets[mask], axis=0)

        residuals = np.linalg.norm(
            offsets[mask] - translation_cur_prev,
            axis=1,
        )
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        if rmse > self.fixed_rotation_max_rmse_m:
            return None
        return (
            rotation_cur_prev,
            translation_cur_prev,
            int(np.count_nonzero(mask)),
            rmse,
        )

    # Estimación de transformación rígida 3D-3D mediante RANSAC.
    # Se selecciona el modelo con mayor número de inliers y menor error residual.
    def estimate_rigid_ransac(
        self,
        prev_points: np.ndarray,
        cur_points: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, int, float]]:
        count = prev_points.shape[0]
        if count < self.min_3d3d_matches:
            return None

        best_mask = None
        best_count = 0
        best_rmse = math.inf
        for _ in range(self.rigid_ransac_iterations):
            indices = self.rng.choice(count, size=3, replace=False)
            transform = self.rigid_svd(
                prev_points[indices],
                cur_points[indices],
            )
            if transform is None:
                continue
            rotation, translation = transform
            predicted = (rotation @ prev_points.T).T + translation
            residuals = np.linalg.norm(predicted - cur_points, axis=1)
            mask = residuals <= self.rigid_inlier_threshold_m
            inliers = int(np.count_nonzero(mask))
            if inliers < 3:
                continue
            rmse = float(np.sqrt(np.mean(np.square(residuals[mask]))))
            if inliers > best_count or (
                inliers == best_count and rmse < best_rmse
            ):
                best_mask = mask
                best_count = inliers
                best_rmse = rmse

        if best_mask is None or best_count < self.min_inliers:
            return None
        inlier_ratio = best_count / max(count, 1)
        if inlier_ratio < self.min_inlier_ratio:
            return None

        refined = self.rigid_svd(
            prev_points[best_mask],
            cur_points[best_mask],
        )
        if refined is None:
            return None
        rotation, translation = refined
        predicted = (rotation @ prev_points[best_mask].T).T + translation
        residual_vector = np.linalg.norm(
            predicted - cur_points[best_mask],
            axis=1,
        )
        rmse = float(np.sqrt(np.mean(np.square(residual_vector))))
        if rmse > self.rigid_max_rmse_m:
            return None
        return rotation, translation, best_count, rmse

    # Cálculo de la transformación rígida óptima entre dos nubes de puntos
    # mediante descomposición en valores singulares (SVD).
    @staticmethod
    def rigid_svd(
        source: np.ndarray,
        target: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if (
            source.shape[0] < 3
            or np.linalg.matrix_rank(source - np.mean(source, axis=0)) < 2
        ):
            return None
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        covariance = (
            source - source_centroid
        ).T @ (
            target - target_centroid
        )
        u, _s, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        translation = target_centroid - rotation @ source_centroid
        return rotation, translation

    # Método PnP de respaldo. Estima la pose a partir de puntos 3D del keyframe
    # y sus posiciones correspondientes en la imagen actual.
    def estimate_pnp(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        k_matrix: np.ndarray,
        match_count: int,
    ) -> Optional[Tuple[float, float, float, float, int, int, float, str]]:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            k_matrix,
            np.zeros((4, 1), dtype=np.float64),
            iterationsCount=120,
            reprojectionError=self.pnp_reprojection_error_px,
            confidence=0.995,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            return None

        inlier_indices = inliers.reshape(-1)
        inlier_count = int(inlier_indices.size)
        inlier_ratio = inlier_count / max(object_points.shape[0], 1)
        if (
            inlier_count < self.min_inliers
            or inlier_ratio < self.min_inlier_ratio
        ):
            return None

        refined, refined_rvec, refined_tvec = cv2.solvePnP(
            object_points[inlier_indices],
            image_points[inlier_indices],
            k_matrix,
            np.zeros((4, 1), dtype=np.float64),
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if refined:
            rvec = refined_rvec
            tvec = refined_tvec

        projected, _ = cv2.projectPoints(
            object_points[inlier_indices],
            rvec,
            tvec,
            k_matrix,
            np.zeros((4, 1), dtype=np.float64),
        )
        residuals = np.linalg.norm(
            projected.reshape(-1, 2) - image_points[inlier_indices],
            axis=1,
        )
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        if rmse > self.pnp_max_rmse_px:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        return self.convert_camera_transform(
            rotation,
            np.asarray(tvec, dtype=np.float64).reshape(3),
            match_count,
            inlier_count,
            rmse / max(self.pnp_max_rmse_px, 1e-3),
            'PNP_KEYFRAME_FALLBACK',
        )

    # Convierte la transformación estimada en el sistema óptico de la cámara
    # al sistema base_footprint. También compensa el brazo de palanca causado por
    # la posición adelantada de la RealSense respecto al origen del robot.
    def convert_camera_transform(
        self,
        rotation_cur_prev: np.ndarray,
        translation_cur_prev: np.ndarray,
        match_count: int,
        inlier_count: int,
        residual: float,
        method: str,
    ) -> Optional[Tuple[float, float, float, float, int, int, float, str]]:
        # The estimated transform maps keyframe points into the current camera.
        # Inverting it yields the current camera pose in the keyframe camera.
        rotation_prev_cur = rotation_cur_prev.T
        translation_prev_cur = -rotation_cur_prev.T @ translation_cur_prev

        r_base_camera = self.camera_to_base_rotation()
        rotation_base_delta = (
            r_base_camera @ rotation_prev_cur @ r_base_camera.T
        )

        # Full rigid extrinsic:
        # T_Bprev_Bcur = T_BC * T_Cprev_Ccur * T_CB.
        # The two lever-arm terms remove apparent camera translation generated
        # only because a forward-mounted camera rotates around the base origin.
        camera_position_base = np.array(
            [self.camera_x_m, self.camera_y_m, self.camera_z_m],
            dtype=np.float64,
        )
        translation_base = (
            camera_position_base
            + r_base_camera @ translation_prev_cur
            - rotation_base_delta @ camera_position_base
        )

        delta_forward = float(translation_base[0])
        delta_left = float(translation_base[1])
        delta_yaw = float(
            math.atan2(
                rotation_base_delta[1, 0],
                rotation_base_delta[0, 0],
            )
        )

        if math.hypot(delta_forward, delta_left) > self.max_translation_per_frame_m:
            return None
        if abs(delta_yaw) > self.max_yaw_per_frame_rad:
            return None

        inlier_ratio = inlier_count / max(match_count, 1)
        count_score = min(
            inlier_count / max(2.0 * self.min_inliers, 1.0),
            1.0,
        )
        if method.startswith('RGBD_3D3D'):
            residual_score = math.exp(
                -residual / max(self.rigid_inlier_threshold_m, 1e-3)
            )
        elif method.startswith('RGBD_IMU_FIXED'):
            residual_score = math.exp(
                -residual
                / max(self.fixed_rotation_inlier_threshold_m, 1e-3)
            )
        else:
            residual_score = math.exp(-residual)
        confidence = clamp(
            0.44 * inlier_ratio
            + 0.24 * count_score
            + 0.32 * residual_score,
            0.0,
            1.0,
        )
        return (
            delta_forward,
            delta_left,
            delta_yaw,
            confidence,
            match_count,
            inlier_count,
            residual,
            method,
        )

    # Calcula la escala de traslación efectiva en función de la curvatura.
    # La interpolación suave evita cambios bruscos entre calibración recta y giro.
    def effective_translation_scale(self, curvature_rad_m: float) -> float:
        """Blend straight and turning scale using a smooth curvature gate."""
        if not self.adaptive_translation_scale_enabled:
            return max(0.05, self.translation_scale)
        start = max(0.0, self.scale_curvature_start_rad_m)
        full = max(start + 1e-3, self.scale_curvature_full_rad_m)
        ratio = clamp((curvature_rad_m - start) / (full - start), 0.0, 1.0)
        smooth = ratio * ratio * (3.0 - 2.0 * ratio)
        return max(
            0.05,
            self.translation_scale_straight
            + smooth * (
                self.translation_scale_turn
                - self.translation_scale_straight
            ),
        )

    # =========================================================================
    # Funciones auxiliares y publicación
    # =========================================================================
    # ------------------------------------------------------------------
    # Helpers and publication
    # ------------------------------------------------------------------
    # Reescala imagen, profundidad y matriz intrínseca de la cámara de forma
    # coherente para reducir coste computacional sin alterar la geometría.
    def resize_inputs(
        self,
        gray: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.camera_info is not None
        scale = clamp(self.image_scale, 0.35, 1.0)
        if abs(scale - 1.0) < 1e-3:
            gray_scaled, depth_scaled = gray, depth
        else:
            size = (
                max(64, int(round(gray.shape[1] * scale))),
                max(48, int(round(gray.shape[0] * scale))),
            )
            gray_scaled = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            depth_scaled = cv2.resize(
                depth,
                size,
                interpolation=cv2.INTER_NEAREST,
            )

        k = np.array(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        k[0, 0] *= scale
        k[1, 1] *= scale
        k[0, 2] *= scale
        k[1, 2] *= scale
        return gray_scaled, depth_scaled, k

    # Construye la matriz de rotación entre el frame óptico de la cámara y
    # base_footprint, incluyendo pitch y yaw de montaje.
    def camera_to_base_rotation(self) -> np.ndarray:
        optical_to_level_base = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        p = self.camera_pitch_down_rad
        cp, sp = math.cos(p), math.sin(p)
        pitch = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float64,
        )
        y = self.camera_yaw_offset_rad
        cy, sy = math.cos(y), math.sin(y)
        yaw = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return yaw @ pitch @ optical_to_level_base

    # Sustituye el keyframe actual por una nueva referencia RGB-D y almacena
    # simultáneamente la pose local asociada.
    def install_keyframe(
        self,
        depth,
        keypoints,
        descriptors,
        stamp: float,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> None:
        self.prev_depth = depth
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame_stamp = stamp
        self.keyframe_x = float(pose_x)
        self.keyframe_y = float(pose_y)
        self.keyframe_yaw = float(pose_yaw)

    # Propagación limitada durante pérdidas visuales breves. El yaw procede de
    # la IMU y la traslación utiliza la última velocidad visual con decaimiento.
    def propagate_during_visual_loss(self, stamp: float) -> None:
        """Bounded IMU-heading dead reckoning for brief feature drop-outs.

        Translation is never extrapolated beyond ``visual_loss_dead_reckon_max_s``
        from the last accepted visual estimate.  This prevents the multi-second
        frozen-pose behaviour seen near the bench while still avoiding unbounded
        inertial drift.
        """
        imu_fresh = self.imu_is_fresh(stamp)
        if imu_fresh:
            self.yaw = self.current_absolute_yaw()

        if (
            not self.visual_loss_dead_reckon_enabled
            or self.last_visual_success_stamp is None
            or self.last_output_stamp is None
        ):
            return

        age = stamp - self.last_visual_success_stamp
        if age < 0.0 or age > self.visual_loss_dead_reckon_max_s:
            return

        previous_stamp = self.last_dead_reckon_stamp or self.last_output_stamp
        dt = clamp(stamp - previous_stamp, 0.0, 0.20)
        self.last_dead_reckon_stamp = stamp
        if dt <= 0.0:
            return

        decay = math.exp(
            -dt / max(self.visual_loss_speed_decay_tau_s, 1e-3)
        )
        self.dead_reckon_speed_m_s *= decay
        if self.dead_reckon_speed_m_s < self.visual_loss_min_speed_m_s:
            return

        distance = self.dead_reckon_speed_m_s * dt
        self.x += distance * math.cos(self.yaw)
        self.y += distance * math.sin(self.yaw)

    # Devuelve la mejor estimación disponible de yaw:
    # orientación absoluta de IMU, integración del giroscopio o yaw actual.
    def current_absolute_yaw(self) -> float:
        if not self.use_imu:
            return self.yaw
        if self.imu_orientation_available and self.imu_orientation_origin_set:
            return wrap_angle(
                self.imu_orientation_yaw - self.imu_orientation_origin_yaw
            )
        if not self.imu_origin_set:
            return self.yaw
        return wrap_angle(self.imu_yaw_integrated - self.imu_origin_yaw)

    # Comprueba si la información inercial disponible es suficientemente reciente
    # respecto al instante de la pareja RGB-D procesada.
    def imu_is_fresh(self, reference_stamp: float) -> bool:
        if not self.use_imu:
            return False
        orientation_fresh = (
            self.imu_orientation_available
            and self.imu_orientation_stamp is not None
            and abs(reference_stamp - self.imu_orientation_stamp) <= self.imu_timeout_s
        )
        gyro_fresh = (
            self.last_imu_stamp is not None
            and self.gyro_calibrated
            and abs(reference_stamp - self.last_imu_stamp) <= self.imu_timeout_s
        )
        return orientation_fresh or gyro_fresh

    # Determina si existe evidencia suficiente de reposo combinando, cuando está
    # habilitado, motion_hint, velocidad visual y velocidad angular de la IMU.
    def stationary_evidence(self, now: float, visual_speed: float) -> bool:
        if not self.use_motion_hint:
            return False
        if self.motion_hint_stamp is None or self.zero_command_since is None:
            return False
        if now - self.motion_hint_stamp > self.motion_hint_timeout_s:
            return False
        if now - self.zero_command_since < self.zero_command_hold_s:
            return False
        return (
            visual_speed <= self.stationary_visual_speed_m_s
            and abs(self.last_gyro_z) <= self.stationary_gyro_rad_s
        )

    # Calcula y filtra las velocidades lineal y angular a partir de la evolución
    # temporal de la pose estimada.
    def update_output_velocity(self, stamp: float) -> None:
        if self.last_output_stamp is None:
            self.last_output_stamp = stamp
            self.last_output_x = self.x
            self.last_output_y = self.y
            self.last_output_yaw = self.yaw
            self.linear_speed = 0.0
            self.angular_speed = 0.0
            return

        dt = clamp(stamp - self.last_output_stamp, 1e-3, 0.30)
        position_delta = math.hypot(
            self.x - self.last_output_x,
            self.y - self.last_output_y,
        )
        raw_linear_speed = position_delta / dt
        raw_angular_speed = wrap_angle(
            self.yaw - self.last_output_yaw
        ) / dt

        alpha = dt / max(self.velocity_filter_tau_s, dt)
        self.linear_speed += alpha * (
            raw_linear_speed - self.linear_speed
        )
        self.angular_speed += alpha * (
            raw_angular_speed - self.angular_speed
        )

        self.last_output_stamp = stamp
        self.last_output_x = self.x
        self.last_output_y = self.y
        self.last_output_yaw = self.yaw

    # Actualiza la validez de la odometría mediante histéresis sobre la confianza,
    # evitando oscilaciones rápidas entre estados válido/no válido.
    def update_valid_state(self) -> None:
        enter = max(self.valid_enter_confidence, self.valid_exit_confidence)
        exit_ = min(self.valid_enter_confidence, self.valid_exit_confidence)
        if self.valid:
            self.valid = self.confidence >= exit_
        else:
            self.valid = self.confidence >= enter

    # Reduce la confianza cuando existe un fallo temporal de percepción o una
    # estimación insuficientemente fiable.
    def degrade_confidence(self, factor: float) -> None:
        self.confidence *= factor
        self.update_valid_state()

    # Servicio de reinicio de la odometría local. Restablece origen, velocidades,
    # keyframe, métricas de deriva y referencias inerciales relativas.
    def reset_callback(self, request, response):
        del request
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.last_raw_speed = 0.0
        self.confidence = 0.0
        self.valid = False

        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame_stamp = None
        self.keyframe_x = 0.0
        self.keyframe_y = 0.0
        self.keyframe_yaw = 0.0

        self.last_output_x = 0.0
        self.last_output_y = 0.0
        self.last_output_yaw = 0.0
        self.last_output_stamp = None
        self.last_visual_success_stamp = None
        self.last_dead_reckon_stamp = None
        self.dead_reckon_speed_m_s = 0.0
        self.last_feature_count = 0
        self.last_match_count = 0
        self.last_inlier_count = 0
        self.last_estimation_method = 'RESET'
        self.last_estimation_residual = math.inf
        self.last_effective_translation_scale = self.translation_scale_straight
        self.last_motion_curvature = 0.0
        self.last_arc_lateral_residual = 0.0
        self.distance_since_reset_m = 0.0
        self.estimated_drift_variance_m2 = 0.0

        if self.use_imu:
            self.imu_origin_yaw = self.imu_yaw_integrated
            self.imu_origin_set = True
            if self.imu_orientation_available:
                self.imu_orientation_origin_yaw = self.imu_orientation_yaw
                self.imu_orientation_origin_set = True

        self.reset_event_pub.publish(Bool(data=True))
        response.success = True
        response.message = (
            'Local odometry reset; the next RGB-D frame initializes a new '
            'keyframe and local origin.'
        )
        return response

    # Publica la pose, velocidades, covarianzas, confianza, validez y métricas
    # de diagnóstico asociadas al estado actual del estimador.
    def publish_state(self, reason: str) -> None:
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.orientation = yaw_to_quaternion(self.yaw)
        odom.twist.twist.linear.x = float(self.linear_speed)
        odom.twist.twist.angular.z = float(self.angular_speed)

        estimated_drift_m = math.sqrt(
            max(self.estimated_drift_variance_m2, 0.0)
        )
        pose_variance = (
            0.015
            + 0.45 * (1.0 - self.confidence)
            + estimated_drift_m * estimated_drift_m
        )
        yaw_variance = 0.008 + 0.25 * (1.0 - self.confidence)
        odom.pose.covariance[0] = pose_variance
        odom.pose.covariance[7] = pose_variance
        odom.pose.covariance[35] = yaw_variance
        odom.twist.covariance[0] = pose_variance
        odom.twist.covariance[35] = yaw_variance

        self.odom_pub.publish(odom)
        self.valid_pub.publish(Bool(data=bool(self.valid)))
        self.confidence_pub.publish(Float64(data=float(self.confidence)))
        self.feature_count_pub.publish(Int32(data=int(self.last_feature_count)))
        self.match_count_pub.publish(Int32(data=int(self.last_match_count)))
        self.inlier_count_pub.publish(Int32(data=int(self.last_inlier_count)))
        imu_source = (
            'ORIENTATION' if self.imu_orientation_available else 'GYRO'
            if self.use_imu else 'NONE'
        )
        self.imu_source_pub.publish(String(data=imu_source))
        self.imu_yaw_pub.publish(Float64(data=float(self.current_absolute_yaw())))
        self.translation_scale_pub.publish(
            Float64(data=float(self.last_effective_translation_scale))
        )
        self.motion_curvature_pub.publish(
            Float64(data=float(self.last_motion_curvature))
        )
        self.arc_lateral_residual_pub.publish(
            Float64(data=float(self.last_arc_lateral_residual))
        )
        self.distance_since_reset_pub.publish(
            Float64(data=float(self.distance_since_reset_m))
        )
        self.estimated_drift_pub.publish(
            Float64(data=float(estimated_drift_m))
        )
        self.publish_feature_debug(reason)

        debug = (
            f'valid={self.valid} conf={self.confidence:.2f} '
            f'pose=({self.x:.2f},{self.y:.2f},{self.yaw:.2f}) '
            f'v={self.linear_speed:.2f} wz={self.angular_speed:.2f} '
            f'features={self.last_feature_count} matches={self.last_match_count} '
            f'inliers={self.last_inlier_count} method={self.last_estimation_method} '
            f'scale={self.last_effective_translation_scale:.2f} '
            f'curv={self.last_motion_curvature:.2f} '
            f'arc_lat={self.last_arc_lateral_residual:+.3f} '
            f'drift_est={estimated_drift_m:.2f}m dist={self.distance_since_reset_m:.1f}m | {reason}'
        )
        self.debug_pub.publish(String(data=debug))
        now = self.get_clock().now()
        if (now - self.last_debug_time).nanoseconds * 1e-9 > 0.75:
            self.get_logger().info(debug)
            self.last_debug_time = now

    # Genera una imagen de depuración con las características ORB y un resumen
    # textual del estado de la odometría visual.
    def publish_feature_debug(self, reason: str) -> None:
        if not self.publish_debug_images or self.debug_feature_image is None:
            return
        image = self.debug_feature_image.copy()
        source = (
            'ORIENTATION' if self.imu_orientation_available else 'GYRO'
            if self.use_imu else 'NONE'
        )
        estimated_drift_m = math.sqrt(
            max(self.estimated_drift_variance_m2, 0.0)
        )
        lines = [
            f'VO {self.last_estimation_method}  valid={self.valid} conf={self.confidence:.2f}',
            f'features={self.last_feature_count} matches={self.last_match_count} inliers={self.last_inlier_count}',
            f'scale={self.last_effective_translation_scale:.2f} curv={self.last_motion_curvature:.2f} arc_lat={self.last_arc_lateral_residual:+.3f}',
            f'dist={self.distance_since_reset_m:.1f}m drift_est={estimated_drift_m:.2f}m imu={source} yaw={self.current_absolute_yaw():+.2f}',
            f'{reason[:72]}',
        ]
        for index, text in enumerate(lines):
            y = 24 + 24 * index
            cv2.putText(
                image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.child_frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        self.feature_debug_pub.publish(msg)

    # Conversión del reloj ROS a segundos en coma flotante.
    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


# ============================================================================
# Punto de entrada del nodo
# ============================================================================
# Inicializa ROS 2, crea el estimador de odometría y mantiene el nodo activo.
# El cierre contempla además un RuntimeError específico observado en ROS 2 Jazzy
# durante la destrucción de algunas suscripciones tras SIGINT.
# ============================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbdImuLocalOdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        # Jazzy may raise this exact pybind conversion error while a
        # subscription is being torn down after SIGINT.
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
