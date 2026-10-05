#!/usr/bin/env python3

# ============================================================================
# LAUNCH PRINCIPAL DEL BASTÓN ASISTIVO REAL
# ============================================================================
#
# Este archivo arranca los principales componentes ROS 2 necesarios para el
# funcionamiento del prototipo físico:
#
#   1. Intel RealSense D435
#   2. Transformaciones TF y modelo URDF del bastón
#   3. Comunicación con el ESP32-S3
#   4. Filtrado de sensores ultrasónicos laterales
#   5. Odometría local RGB-D + IMU
#   6. Planner de transitabilidad / evitación de obstáculos
#   7. Controlador de evasión
#   8. Mux final de órdenes de dirección
#   9. Comprobación del estado del hardware
#  10. Visualizador de depuración opcional
#  11. RViz opcional
#
# IMPORTANTE:
# Este archivo contiene también la geometría física calibrada de la D435
# respecto al robot:
#
#   X     = 0.172 m
#   Y     = 0.000 m
#   Z     = 0.136 m
#   Pitch = 0.000 rad
#   Yaw   = 0.000 rad
#
# ============================================================================


# ============================================================================
# IMPORTACIONES ROS 2 LAUNCH
# ============================================================================

from launch import LaunchDescription

# Permite declarar argumentos del launch e incluir otros archivos launch.
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription

# Permite activar nodos o launches solo si se cumple una condición.
from launch.conditions import IfCondition

# Permite incluir archivos launch escritos en Python.
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Utilidades para:
# - ejecutar comandos (Command)
# - leer argumentos del launch (LaunchConfiguration)
# - construir rutas (PathJoinSubstitution)
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution

# Acción utilizada para arrancar nodos ROS 2.
from launch_ros.actions import Node

# Permite indicar explícitamente el tipo de un parámetro.
from launch_ros.parameter_descriptions import ParameterValue

# Permite localizar la carpeta share de un paquete ROS 2 instalado.
from launch_ros.substitutions import FindPackageShare


# ============================================================================
# FUNCIÓN PRINCIPAL DEL LAUNCH
# ============================================================================

def generate_launch_description() -> LaunchDescription:

    # ------------------------------------------------------------------------
    # ARGUMENTOS GENERALES
    # ------------------------------------------------------------------------
    #
    # LaunchConfiguration no contiene todavía el valor del parámetro:
    # representa el valor que ROS resolverá cuando se ejecute el launch.
    # Esto permite sobrescribir los valores desde terminal.
    #

    # Archivo YAML que contiene parámetros generales del sistema.
    params_file = LaunchConfiguration('params_file')
    # Puerto serie utilizado para comunicarse con el ESP32-S3.
    serial_device = LaunchConfiguration('serial_device')
    # Topic donde el bridge publica los datos de la MPU-6050.
    imu_topic = LaunchConfiguration('imu_topic')
    # Permite activar/desactivar el uso de IMU en la odometría.
    use_imu = LaunchConfiguration('use_imu')
    # Indica si la IMU proporciona una orientación absoluta utilizable.
    # En nuestro caso se deja normalmente en false porque se utiliza
    # principalmente velocidad angular/aceleración.
    imu_has_orientation = LaunchConfiguration('imu_has_orientation')
    # Permite arrancar o no la RealSense desde este launch.
    start_realsense = LaunchConfiguration('start_realsense')
    # Permite habilitar la nube de puntos de la RealSense.
    # Se mantiene desactivada normalmente para ahorrar recursos.
    enable_pointcloud = LaunchConfiguration('enable_pointcloud')
    # Activa el nodo visualizador de depuración.
    run_debug_visualizer = LaunchConfiguration('run_debug_visualizer')
    # Permite arrancar RViz automáticamente.
    rviz = LaunchConfiguration('rviz')
    # Número de serie de la RealSense.
    # Útil si hubiera más de una cámara conectada.
    camera_serial_no = LaunchConfiguration('camera_serial_no')
    # Resolución y frecuencia utilizadas por la cámara.
    camera_profile = LaunchConfiguration('camera_profile')

    # ------------------------------------------------------------------------
    # GEOMETRÍA FÍSICA DE LA CÁMARA
    # ------------------------------------------------------------------------
    #
    # Posición/orientación de la RealSense D435 respecto al frame
    # base_footprint del bastón.
    #
    # Estos valores se utilizan posteriormente tanto en:
    #   - URDF / TF
    #   - odometría RGB-D
    #   - planner de obstáculos
    #

    camera_x_m = LaunchConfiguration('camera_x_m')
    camera_y_m = LaunchConfiguration('camera_y_m')
    camera_z_m = LaunchConfiguration('camera_z_m')
    camera_pitch_down_rad = LaunchConfiguration('camera_pitch_down_rad')
    camera_yaw_offset_rad = LaunchConfiguration('camera_yaw_offset_rad')


    # ------------------------------------------------------------------------
    # LOCALIZACIÓN DE ARCHIVOS DEL PAQUETE
    # ------------------------------------------------------------------------

    # Busca la carpeta share del paquete mapless_cane_real.
    pkg_share = FindPackageShare('mapless_cane_real')

    # Configuración de RViz utilizada cuando rviz:=true.
    rviz_file = PathJoinSubstitution(
        [pkg_share, 'rviz', 'mapless_real_debug.rviz']
    )

    # Modelo URDF/Xacro físico del bastón.
    xacro_file = PathJoinSubstitution(
        [pkg_share, 'urdf', 'mapless_cane_real.urdf.xacro']
    )


    # ------------------------------------------------------------------------
    # ROBOT DESCRIPTION / URDF
    # ------------------------------------------------------------------------
    #
    # Ejecuta Xacro pasando la geometría calibrada de la cámara.
    #
    # El resultado será utilizado por robot_state_publisher para publicar
    # la estructura TF definida en el URDF.
    #

    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' camera_x:=', camera_x_m,
            ' camera_y:=', camera_y_m,
            ' camera_z:=', camera_z_m,
            ' camera_pitch:=', camera_pitch_down_rad,
            ' camera_yaw:=', camera_yaw_offset_rad,
        ]),
        value_type=str,
    )


    # =========================================================================
    # INTEL REALSENSE D435
    # =========================================================================
    #
    # En vez de arrancar directamente realsense2_camera_node se incluye
    # rs_launch.py, que es el launch oficial del paquete realsense2_camera.
    #
    # Configuración:
    #
    #   - cámara D435
    #   - color activado
    #   - profundidad activada
    #   - infrarrojos desactivados
    #   - sincronización color/profundidad activada
    #   - depth alineado con color
    #   - pointcloud opcional
    #   - resolución 640x480 @ 30 FPS por defecto
    #

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py']
            )
        ),

        # Solo se ejecuta si start_realsense:=true.
        condition=IfCondition(start_realsense),

        launch_arguments={

            # Namespace ROS de la cámara.
            'camera_namespace': 'camera',
            # Nombre del nodo/cámara.
            'camera_name': 'camera',
            # Permite seleccionar una cámara concreta por número de serie.
            'serial_no': camera_serial_no,
            # Modelo de RealSense utilizado.
            'device_type': 'd435',
            # Stream RGB.
            'enable_color': 'true',
            # Stream de profundidad.
            'enable_depth': 'true',
            # Streams infrarrojos desactivados.
            'enable_infra1': 'false',
            'enable_infra2': 'false',
            # Sincroniza los frames de profundidad y color.
            'enable_sync': 'true',
            # Alinea la imagen de profundidad con la imagen RGB.
            'align_depth.enable': 'true',
            # Nube de puntos opcional.
            'pointcloud.enable': enable_pointcloud,
            # Perfil de profundidad.
            'depth_module.depth_profile': camera_profile,
            # Perfil RGB.
            'rgb_camera.color_profile': camera_profile,
            # Publicación periódica de diagnósticos.
            'diagnostics_period': '1.0',
            # Tiempo antes de intentar reconectar la cámara.
            'reconnect_timeout': '3.0',
            # Espera indefinidamente a que aparezca la cámara si no está
            # disponible durante el arranque.
            'wait_for_device_timeout': '-1.0',

        }.items(),
    )


    # ------------------------------------------------------------------------
    # PARÁMETROS COMUNES
    # ------------------------------------------------------------------------
    #
    # Muchos nodos leen su configuración desde mapless_real.yaml.
    #
    # use_sim_time=False indica que utilizamos tiempo real del sistema y no
    # el reloj simulado de Gazebo.
    #

    common = [params_file, {'use_sim_time': False}]

    # Conversión explícita a booleano de argumentos que llegan como texto.
    bool_use_imu = ParameterValue(use_imu, value_type=bool)
    bool_imu_orientation = ParameterValue(
        imu_has_orientation,
        value_type=bool
    )


    # =========================================================================
    # DESCRIPCIÓN COMPLETA DEL SISTEMA
    # =========================================================================

    return LaunchDescription([


        # =====================================================================
        # ARGUMENTOS DEL LAUNCH
        # =====================================================================

        # Archivo principal YAML de configuración.
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'mapless_real.yaml']
            ),
        ),

        # Puerto USB del ESP32-S3.
        DeclareLaunchArgument(
            'serial_device',
            default_value='/dev/ttyACM0'
        ),

        # Topic donde se publican los datos de la IMU.
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/imu/data_raw'
        ),

        # La odometría utilizará la IMU.
        DeclareLaunchArgument(
            'use_imu',
            default_value='true'
        ),

        # La orientación absoluta de la IMU no se considera válida.
        DeclareLaunchArgument(
            'imu_has_orientation',
            default_value='false'
        ),

        # Arranca la D435 automáticamente.
        DeclareLaunchArgument(
            'start_realsense',
            default_value='true'
        ),

        # Por defecto no publicamos pointcloud para reducir carga.
        DeclareLaunchArgument(
            'enable_pointcloud',
            default_value='false'
        ),

        # Visualizador de depuración desactivado normalmente.
        DeclareLaunchArgument(
            'run_debug_visualizer',
            default_value='false'
        ),

        # RViz no arranca automáticamente.
        DeclareLaunchArgument(
            'rviz',
            default_value='false'
        ),

        # Si queda vacío se utiliza la RealSense encontrada.
        DeclareLaunchArgument(
            'camera_serial_no',
            default_value=''
        ),

        # Resolución 640x480 a 30 FPS.
        DeclareLaunchArgument(
            'camera_profile',
            default_value='640x480x30'
        ),

        # ---------------------------------------------------------------------
        # GEOMETRÍA CALIBRADA DE LA REALSENSE
        # ---------------------------------------------------------------------

        # La cámara se encuentra 17.2 cm por delante del origen del robot.
        DeclareLaunchArgument(
            'camera_x_m',
            default_value='0.172'
        ),

        # Cámara centrada lateralmente.
        DeclareLaunchArgument(
            'camera_y_m',
            default_value='0.0'
        ),

        # Altura óptica de la cámara respecto al suelo: 13.6 cm.
        DeclareLaunchArgument(
            'camera_z_m',
            default_value='0.136'
        ),

        # Cámara sin inclinación adicional hacia abajo.
        DeclareLaunchArgument(
            'camera_pitch_down_rad',
            default_value='0.0'
        ),

        # Cámara sin desviación angular lateral.
        DeclareLaunchArgument(
            'camera_yaw_offset_rad',
            default_value='0.0'
        ),


        # =====================================================================
        # 1. INTEL REALSENSE D435
        # =====================================================================
        #
        # Arranca el launch oficial de la cámara definido anteriormente.
        #

        realsense,


        # =====================================================================
        # 2. TRANSFORMACIÓN BASE_FOOTPRINT -> CAMERA_LINK
        # =====================================================================
        #
        # Publica una transformación TF estática que indica dónde está
        # físicamente la cámara respecto a la base del bastón.
        #
        # Esta transformación utiliza directamente los valores calibrados:
        #
        # base_footprint
        #       |
        #       | X = camera_x_m
        #       | Y = camera_y_m
        #       | Z = camera_z_m
        #       |
        #       v
        # camera_link
        #

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_realsense_tf',

            arguments=[
                '--x', camera_x_m,
                '--y', camera_y_m,
                '--z', camera_z_m,

                '--roll', '0.0',
                '--pitch', camera_pitch_down_rad,
                '--yaw', camera_yaw_offset_rad,

                '--frame-id', 'base_footprint',
                '--child-frame-id', 'camera_link',
            ],

            output='screen',
        ),


        # =====================================================================
        # 3. ROBOT STATE PUBLISHER
        # =====================================================================
        #
        # Publica en TF la estructura del robot definida en el URDF/Xacro.
        #
        # Recibe robot_description, que se ha generado anteriormente pasando
        # al Xacro los parámetros físicos de montaje de la cámara.
        #

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',

            parameters=[
                {
                    'robot_description': robot_description,
                    'use_sim_time': False
                }
            ],

            output='screen',
        ),


        # =====================================================================
        # 4. BRIDGE ESP32-S3 <-> ROS 2
        # =====================================================================
        #
        # Es el enlace entre la Raspberry Pi y el microcontrolador.
        #
        # A través del puerto serie permite intercambiar información con
        # el hardware físico del bastón.
        #
        # El ESP32 se encarga, entre otras cosas, de:
        #
        #   - MPU-6050
        #   - sensores HC-SR04
        #   - botones
        #   - servo de dirección
        #   - vibradores hápticos
        #
        # Este nodo utiliza mapless_real.yaml y sobrescribe el puerto serie
        # con el argumento serial_device del launch.
        #

        Node(
            package='mapless_cane_real',
            executable='esp32_hardware_bridge_node',
            name='esp32_hardware_bridge_node',

            parameters=[
                params_file,
                {
                    'serial_device': serial_device,
                    'use_sim_time': False
                }
            ],

            output='screen',
        ),


        # =====================================================================
        # 5. FILTRO DE SENSORES LATERALES
        # =====================================================================
        #
        # Procesa las medidas procedentes de los HC-SR04 laterales.
        #
        # Su función es evitar trabajar directamente con medidas ultrasónicas
        # ruidosas o inestables y generar una representación más fiable del
        # espacio lateral del bastón.
        #
        # Sus parámetros específicos se cargan desde mapless_real.yaml.
        #

        Node(
            package='mapless_cane_real',
            executable='side_range_filter_node',
            name='side_range_filter_node',

            parameters=common,

            output='screen',
        ),


        # =====================================================================
        # 6. ODOMETRÍA LOCAL RGB-D + IMU
        # =====================================================================
        #
        # Estima el movimiento local del bastón utilizando:
        #
        #   RealSense D435
        #       +
        #   MPU-6050
        #
        # La combinación de información visual/de profundidad e inercial
        # permite estimar el desplazamiento y orientación del sistema.
        #
        # Los parámetros de posición/orientación de la cámara son esenciales
        # para transformar correctamente las mediciones de la D435 al sistema
        # de referencia del bastón.
        #

        Node(
            package='mapless_cane_real',
            executable='rgbd_imu_local_odometry_node',
            name='rgbd_imu_local_odometry_node',

            parameters=[

                # Parámetros generales desde YAML.
                params_file,

                {
                    # Topic de la MPU-6050.
                    'imu_topic': imu_topic,

                    # Activa/desactiva uso de IMU.
                    'use_imu': bool_use_imu,

                    # Decide si se utiliza la orientación absoluta de la IMU.
                    'use_imu_orientation': bool_imu_orientation,

                    # No se acepta una orientación sin covarianza válida.
                    'accept_imu_orientation_without_covariance': False,

                    # Permite calibrar automáticamente el offset del giroscopio
                    # al arrancar.
                    'auto_calibrate_gyro': True,

                    # Posición X de la cámara.
                    'camera_x_m': ParameterValue(
                        camera_x_m,
                        value_type=float
                    ),

                    # Posición lateral de la cámara.
                    'camera_y_m': ParameterValue(
                        camera_y_m,
                        value_type=float
                    ),

                    # Altura de la cámara.
                    'camera_z_m': ParameterValue(
                        camera_z_m,
                        value_type=float
                    ),

                    # Inclinación vertical de la cámara.
                    'camera_pitch_down_rad': ParameterValue(
                        camera_pitch_down_rad,
                        value_type=float
                    ),

                    # Desviación horizontal de la cámara.
                    'camera_yaw_offset_rad': ParameterValue(
                        camera_yaw_offset_rad,
                        value_type=float
                    ),

                    # Se utiliza tiempo real.
                    'use_sim_time': False,
                },
            ],

            output='screen',
        ),


        # =====================================================================
        # 7. PLANNER DE TRANSITABILIDAD
        # =====================================================================
        #
        # Analiza la información RGB-D y determina por qué zonas puede avanzar
        # el bastón y dónde existen obstáculos.
        #
        # Su objetivo es generar la información necesaria para decidir qué
        # trayectoria o arco de dirección resulta más seguro.
        #
        # Necesita conocer la posición física de la cámara porque los puntos
        # detectados por la D435 deben interpretarse respecto al suelo y al
        # propio robot.
        #

        Node(
            package='mapless_cane_real',
            executable='traversability_arc_planner_node',
            name='traversability_arc_planner_node',

            parameters=[

                # Configuración general del planner desde YAML.
                params_file,

                {
                    # Posición longitudinal de la cámara.
                    'camera_x_m': ParameterValue(
                        camera_x_m,
                        value_type=float
                    ),

                    # Posición lateral de la cámara.
                    'camera_y_m': ParameterValue(
                        camera_y_m,
                        value_type=float
                    ),

                    # Altura de la cámara sobre el suelo.
                    'camera_height_m': ParameterValue(
                        camera_z_m,
                        value_type=float
                    ),

                    # Inclinación vertical.
                    'camera_pitch_down_rad': ParameterValue(
                        camera_pitch_down_rad,
                        value_type=float
                    ),

                    # Desviación horizontal.
                    'camera_yaw_offset_rad': ParameterValue(
                        camera_yaw_offset_rad,
                        value_type=float
                    ),

                    'use_sim_time': False,
                },
            ],

            output='screen',
        ),


        # =====================================================================
        # 8. CONTROLADOR DE EVASIÓN DE OBSTÁCULOS
        # =====================================================================
        #
        # Recibe la información calculada por el planner y decide la acción de
        # dirección necesaria para rodear un obstáculo.
        #
        # En nuestro sistema:
        #
        #   - el usuario proporciona el desplazamiento físico del bastón;
        #   - ROS no impulsa el robot;
        #   - ROS únicamente modifica la DIRECCIÓN mediante el servo.
        #
        # Sus umbrales, ganancias y demás parámetros se cargan desde
        # mapless_real.yaml.
        #

        Node(
            package='mapless_cane_real',
            executable='mapless_bypass_controller_node',
            name='mapless_bypass_controller_node',

            parameters=common,

            output='screen',
        ),


        # =====================================================================
        # 9. STEERING COMMAND MUX
        # =====================================================================
        #
        # Es el multiplexor final de dirección.
        #
        # Puede recibir diferentes posibles fuentes de órdenes de giro y decide
        # cuál tiene prioridad en cada momento.
        #
        # Finalmente genera la orden que acabará llegando al servo mediante el
        # bridge del ESP32.
        #
        # Conceptualmente:
        #
        #      órdenes humanas
        #             +
        #      asistencia automática
        #             +
        #      otras estrategias
        #             |
        #             v
        #     steering_command_mux
        #             |
        #             v
        #      comando final servo
        #

        Node(
            package='mapless_cane_real',
            executable='steering_command_mux_node',
            name='steering_command_mux_node',

            parameters=common,

            output='screen',
        ),


        # =====================================================================
        # 10. HARDWARE READINESS
        # =====================================================================
        #
        # Supervisa que los componentes necesarios estén disponibles antes de
        # considerar que el sistema está preparado.
        #
        # Por ejemplo, comprueba la presencia/actividad de la IMU cuando
        # use_imu está activado.
        #

        Node(
            package='mapless_cane_real',
            executable='hardware_readiness_node',
            name='hardware_readiness_node',

            parameters=[
                params_file,
                {
                    'imu_topic': imu_topic,

                    # Si usamos IMU, se exige que esté disponible.
                    'require_imu': bool_use_imu,

                    'use_sim_time': False
                },
            ],

            output='screen',
        ),


        # =====================================================================
        # 11. VISUALIZADOR DE DEPURACIÓN
        # =====================================================================
        #
        # Nodo opcional utilizado para inspeccionar durante el desarrollo la
        # información interna del sistema.
        #
        # Por defecto:
        #
        #   run_debug_visualizer := false
        #
        # por lo que NO se ejecuta durante el uso normal del bastón.
        #

        Node(
            package='mapless_cane_real',
            executable='mapless_debug_visualizer_node',
            name='mapless_debug_visualizer_node',

            condition=IfCondition(run_debug_visualizer),

            parameters=common,

            output='screen',
        ),


        # =====================================================================
        # 12. RVIZ
        # =====================================================================
        #
        # Interfaz gráfica de ROS 2 utilizada para visualizar:
        #
        #   - TF
        #   - modelo del robot
        #   - sensores
        #   - información del planner
        #   - depuración
        #
        # Está desactivado por defecto para evitar gastar recursos en la
        # Raspberry durante las pruebas autónomas.
        #
        # Puede activarse manualmente mediante:
        #
        #   rviz:=true
        #

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',

            condition=IfCondition(rviz),

            # Carga nuestra configuración RViz preparada para el bastón.
            arguments=['-d', rviz_file],

            parameters=[
                {'use_sim_time': False}
            ],

            output='screen',
        ),

    ])