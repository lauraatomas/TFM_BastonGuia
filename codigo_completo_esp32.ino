/*
  MAPLESS CANE - ESP32-S3 FULL HARDWARE v4
  SERVO CALIBRATED + HAPTICA LOCAL SUPERIOR/LATERAL
  =========================================

  Distribución:
    HC-SR04 IZQUIERDO : TRIG GPIO4  / ECHO GPIO5
    HC-SR04 DERECHO   : TRIG GPIO6  / ECHO GPIO7
    MPU-6050 I2C      : SDA GPIO8   / SCL GPIO9
    HC-SR04 SUPERIOR  : TRIG GPIO10 / ECHO GPIO11
    BOTÓN IZQUIERDA   : GPIO12
    BOTÓN DERECHA     : GPIO13
    ERM SUPERIOR      : GPIO14
    ERM INFERIOR      : GPIO15
    ERM IZQUIERDO     : GPIO16
    SERVO              : GPIO17
    ERM DERECHO       : GPIO18

  IMPORTANTE:
  Esta asignación supone que físicamente has conectado los motores así.
  Si tus cuatro motores están en otros GPIO, cambia SOLO las constantes
  VIB_TOP_PIN / VIB_BOTTOM_PIN / VIB_LEFT_PIN / VIB_RIGHT_PIN.

  IMPORTANTE ELÉCTRICO:
  - Los ECHO del HC-SR04 son de 5 V: usar divisor resistivo/level shifter
    hacia GPIO del ESP32 (3,3 V).
  - Los motores ERM NO se conectan directamente a GPIO. Cada motor necesita
    transistor/MOSFET, diodo de rueda libre si corresponde y alimentación
    acorde a la tensión nominal del motor. Masa común con ESP32.
  - El servo usa alimentación externa regulada de 6 V; solo la señal va al
    GPIO17. Masa común ESP32 <-> fuente del servo.
  - El MPU-6050 se conecta a 3,3 V si el breakout lo permite; SDA/SCL son 3,3 V.

  Arquitectura:
  - Raspberry: percepción, navegación y ángulo final.
  - ESP32: sensores, botones, IMU, haptics y PWM del servo.
  - La persona empuja y detiene físicamente el bastón. No existe freno.

  SEGURIDAD HÁPTICA LOCAL:
  - HC-SR04 superior:
      HC-SR04 superior -> ESP32 -> ERM SUPERIOR
    Es 100 % local, no se publica a ROS y no interviene en navegación.
    Se avisa UNA VEZ al entrar en zona de peligro y, si el obstáculo se
    acerca mucho más, UNA segunda vez con una ráfaga más rápida.
    No repite continuamente mientras el mismo obstáculo permanezca delante.

  - HC-SR04 laterales:
      HC-SR04 izquierdo/derecho -> ESP32 -> ERM izquierdo/derecho
    Además de seguir enviándose a Raspberry para navegación, actúan como
    capa de seguridad local permanente. Si aparece algo muy cerca del lado
    (< 40 cm, configurable), se emite un doble pulso UNA VEZ.
    No vuelve a avisar hasta que el lado queda libre (> 55 cm, configurable).

  - Los eventos hápticos semánticos de Raspberry (giro, obstáculo durante
    evasión, fin de maniobra) se mantienen sin cambios.

  Protocolo serie ESP32 -> Raspberry:
    TEL2,seq,uptime_ms,
         left_mm,right_mm,
         left_valid,right_valid,
         button_left,button_right,imu_valid,
         ax,ay,az,gx,gy,gz,
         servo_us,haptic

  Unidades IMU:
    aceleración: m/s^2
    giro: rad/s

  Raspberry -> ESP32:
    CMD,seq,steer_rad,haptic

  Convención:
    steer > 0 = izquierda
    steer < 0 = derecha

  Servo calibrado para el rango final de dirección:
    -0.50 rad -> 1500 us
    -0.25 rad -> 1750 us
     0.00 rad -> 2000 us
    +0.25 rad -> 2250 us
    +0.50 rad -> 2500 us

  IMPORTANTE:
  El servo se controla con la librería ESP32Servo, exactamente igual que en
  el sketch simple que sí mueve físicamente el servo.
*/

#include <Arduino.h>
#include <Wire.h>
#include <ESP32Servo.h>

// -----------------------------------------------------------------------------
// Pines
// -----------------------------------------------------------------------------
static constexpr uint8_t US_LEFT_TRIG  = 4;
static constexpr uint8_t US_LEFT_ECHO  = 5;

static constexpr uint8_t US_RIGHT_TRIG = 6;
static constexpr uint8_t US_RIGHT_ECHO = 7;

static constexpr uint8_t I2C_SDA_PIN   = 8;
static constexpr uint8_t I2C_SCL_PIN   = 9;

static constexpr uint8_t US_UPPER_TRIG = 10;
static constexpr uint8_t US_UPPER_ECHO = 11;

static constexpr uint8_t BUTTON_LEFT_PIN  = 12;
static constexpr uint8_t BUTTON_RIGHT_PIN = 13;

// Distribución física alrededor del mango.
// Si el cableado real es distinto, cambia únicamente estos cuatro GPIO.
static constexpr uint8_t VIB_TOP_PIN    = 14;
static constexpr uint8_t VIB_BOTTOM_PIN = 15;
static constexpr uint8_t VIB_LEFT_PIN   = 16;
static constexpr uint8_t SERVO_PIN      = 17;
static constexpr uint8_t VIB_RIGHT_PIN  = 18;

// -----------------------------------------------------------------------------
// PWM
// -----------------------------------------------------------------------------
// SERVO: usamos ESP32Servo, porque es exactamente el método ya verificado
// físicamente con este ESP32-S3 + GPIO17 + servo.
static constexpr uint32_t SERVO_PWM_HZ = 50;


static constexpr uint16_t SERVO_MIN_US = 1500;
static constexpr uint16_t SERVO_CENTER_US = 2080;
static constexpr uint16_t SERVO_MAX_US = 2500;

// FINAL SERVO CALIBRATION
// -0.50 rad = 1500 us  (máximo derecha)
// -0.25 rad = 1750 us
//  0.00 rad = 2000 us  (centro)
// +0.25 rad = 2250 us
// +0.50 rad = 2500 us  (máximo izquierda)
static constexpr float MAX_STEER_RAD = 0.50f;
static constexpr float SERVO_DIRECTION = 1.0f;

// -----------------------------------------------------------------------------
// Timings
// -----------------------------------------------------------------------------
static constexpr uint32_t COMMAND_WATCHDOG_MS = 1000;
static constexpr uint32_t TELEMETRY_PERIOD_MS = 80;
static constexpr uint32_t BUTTON_DEBOUNCE_MS = 20;
static constexpr uint32_t ULTRASONIC_TIMEOUT_US = 18000;

// HC-SR04 usable range for this project.
static constexpr uint16_t ULTRASONIC_MIN_MM = 50;
static constexpr uint16_t ULTRASONIC_MAX_MM = 2800;

// -----------------------------------------------------------------------------
// Seguridad háptica LOCAL.
//
// SUPERIOR:
//   - entra en alerta a <= 0,80 m tras 3 lecturas válidas consecutivas;
//   - se rearma al quedar >= 0,95 m (o sin eco) durante 4 lecturas;
//   - si después se acerca hasta <= 0,45 m, da UNA segunda advertencia rápida.
//
// Esto evita la vibración casi continua que tenía la versión anterior.
//
// LATERALES:
//   - seguridad permanente independiente de la máquina de estados;
//   - alerta a <= 0,40 m tras 2 lecturas válidas consecutivas;
//   - se rearma al quedar >= 0,55 m (o sin eco) durante 3 lecturas;
//   - cada entrada genera solo UN doble pulso en el lado correspondiente.
//
// Los umbrales son parámetros de partida para pruebas reales.
// -----------------------------------------------------------------------------
static constexpr uint16_t UPPER_ALERT_ON_MM = 800;
static constexpr uint16_t UPPER_ALERT_OFF_MM = 950;
static constexpr uint16_t UPPER_NEAR_MM = 450;

static constexpr uint8_t UPPER_ALERT_CONFIRM_COUNT = 3;
static constexpr uint8_t UPPER_NEAR_CONFIRM_COUNT = 2;
static constexpr uint8_t UPPER_CLEAR_CONFIRM_COUNT = 4;

static constexpr uint16_t LATERAL_ALERT_ON_MM = 400;
static constexpr uint16_t LATERAL_ALERT_OFF_MM = 550;

static constexpr uint8_t LATERAL_ALERT_CONFIRM_COUNT = 2;
static constexpr uint8_t LATERAL_CLEAR_CONFIRM_COUNT = 3;

// -----------------------------------------------------------------------------
// MPU-6050
// -----------------------------------------------------------------------------
static constexpr uint8_t MPU_ADDR = 0x68;
static constexpr uint8_t MPU_REG_PWR_MGMT_1 = 0x6B;
static constexpr uint8_t MPU_REG_CONFIG = 0x1A;
static constexpr uint8_t MPU_REG_GYRO_CONFIG = 0x1B;
static constexpr uint8_t MPU_REG_ACCEL_CONFIG = 0x1C;
static constexpr uint8_t MPU_REG_ACCEL_XOUT_H = 0x3B;

static constexpr float G_MPS2 = 9.80665f;
// ±4 g -> 8192 LSB/g
static constexpr float ACCEL_LSB_PER_G = 8192.0f;
// ±500 deg/s -> 65.5 LSB/(deg/s)
static constexpr float GYRO_LSB_PER_DPS = 65.5f;
static constexpr float DEG_TO_RAD_F = 0.01745329251994329577f;

// Configuración que DEBE tener siempre el MPU-6050.
static constexpr uint8_t MPU_EXPECTED_GYRO_CONFIG = 0x08;   // ±500 deg/s
static constexpr uint8_t MPU_EXPECTED_ACCEL_CONFIG = 0x08;  // ±4 g
static constexpr uint8_t MPU_EXPECTED_CONFIG = 0x03;        // DLPF
static constexpr uint8_t MPU_CONFIG_MAX_ATTEMPTS = 5;
static constexpr uint32_t MPU_CONFIG_CHECK_PERIOD_MS = 1000;

static bool mpuConfigValid = false;
static uint32_t lastMpuConfigCheckMs = 0;

// -----------------------------------------------------------------------------
// Servo
// -----------------------------------------------------------------------------
Servo steeringServo;

// -----------------------------------------------------------------------------
// Estado
// -----------------------------------------------------------------------------
struct RangeReading {
  uint16_t mm;
  bool valid;
};

struct ImuReading {
  bool valid;
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
};

struct DebouncedButton {
  uint8_t pin;
  bool stablePressed;
  bool lastRawPressed;
  uint32_t lastChangeMs;
};

static DebouncedButton buttonLeft {
  BUTTON_LEFT_PIN, false, false, 0
};

static DebouncedButton buttonRight {
  BUTTON_RIGHT_PIN, false, false, 0
};

static RangeReading usLeft  {0, false};
static RangeReading usUpper {0, false};
static RangeReading usRight {0, false};
static ImuReading imu {false, 0, 0, 0, 0, 0, 0};

static uint32_t telemetrySeq = 0;
static uint32_t lastTelemetryMs = 0;
static uint32_t lastCommandMs = 0;

static float requestedSteerRad = 0.0f;
static uint8_t requestedHaptic = 0;

static uint16_t currentServoUs = SERVO_CENTER_US;
static uint8_t currentHaptic = 0;

// Los eventos hápticos enviados por Raspberry son de una sola ejecución.
// Se disparan cuando el código cambia de 0 -> evento o de un evento -> otro.
// Para repetir el mismo evento más adelante, Raspberry debe enviar antes 0.
static uint8_t lastHapticCommandCode = 0;
static uint8_t activeHapticEvent = 0;
static uint32_t hapticEventStartMs = 0;

// -----------------------------------------------------------------------------
// Estado de seguridad háptica LOCAL.
// -----------------------------------------------------------------------------

// Superior.
static bool upperObstacleAlert = false;
static bool upperNearNotified = false;
static uint8_t upperNearCount = 0;
static uint8_t upperVeryNearCount = 0;
static uint8_t upperClearCount = 0;

// Laterales: latch = ya se avisó del obstáculo actual.
// No se vuelve a avisar hasta que el sensor quede claramente libre.
static bool leftSafetyLatched = false;
static bool rightSafetyLatched = false;

static uint8_t leftSafetyNearCount = 0;
static uint8_t rightSafetyNearCount = 0;

static uint8_t leftSafetyClearCount = 0;
static uint8_t rightSafetyClearCount = 0;

// Cola mínima de eventos locales.
// Los bits pendientes permiten no perder una alerta si en ese instante
// Raspberry está reproduciendo otro evento háptico.
static constexpr uint8_t LOCAL_PENDING_UPPER      = 0x01;
static constexpr uint8_t LOCAL_PENDING_UPPER_NEAR = 0x02;
static constexpr uint8_t LOCAL_PENDING_LEFT       = 0x04;
static constexpr uint8_t LOCAL_PENDING_RIGHT      = 0x08;

static uint8_t pendingLocalHaptics = 0;

enum LocalHapticEvent : uint8_t {
  LOCAL_HAPTIC_NONE = 0,
  LOCAL_HAPTIC_UPPER,
  LOCAL_HAPTIC_UPPER_NEAR,
  LOCAL_HAPTIC_LEFT,
  LOCAL_HAPTIC_RIGHT,
  LOCAL_HAPTIC_BOTH_SIDES
};

static LocalHapticEvent activeLocalHaptic = LOCAL_HAPTIC_NONE;
static uint32_t localHapticStartMs = 0;

// Buffer de entrada serie.
static char serialBuffer[128];
static size_t serialLength = 0;

// -----------------------------------------------------------------------------
// PWM servo + vibradores
// -----------------------------------------------------------------------------
static bool attachServoPwm() {
  steeringServo.setPeriodHertz(
    SERVO_PWM_HZ
  );

  steeringServo.attach(
    SERVO_PIN,
    SERVO_MIN_US,
    SERVO_MAX_US
  );

  return steeringServo.attached();
}

static void attachVibrationOutput(uint8_t pin) {
  // NO usamos LEDC para los ERM.
  // ESP32Servo ya utiliza LEDC internamente para generar los 50 Hz del servo.
  // Los vibradores se controlan únicamente ON/OFF a través de sus MOSFET.
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
}


static void writeVibrationOutput(
  uint8_t pin,
  uint8_t requestedPower
) {
  digitalWrite(
    pin,
    requestedPower > 0 ? HIGH : LOW
  );
}

static void writeServoMicroseconds(uint16_t pulseUs) {
  pulseUs = constrain(
    pulseUs,
    SERVO_MIN_US,
    SERVO_MAX_US
  );

  steeringServo.writeMicroseconds(
    pulseUs
  );

  currentServoUs = pulseUs;
}

static uint16_t steeringToPulse(float steerRad) {
  steerRad = constrain(
    steerRad,
    -MAX_STEER_RAD,
    MAX_STEER_RAD
  );

  if (fabsf(steerRad) < 0.001f) {
    return SERVO_CENTER_US;
  }

  if (steerRad > 0.0f) {
    const float fraction =
      steerRad / MAX_STEER_RAD;

    return static_cast<uint16_t>(
      lroundf(
        static_cast<float>(SERVO_CENTER_US)
        + fraction
        * static_cast<float>(
            SERVO_MAX_US - SERVO_CENTER_US
          )
      )
    );
  }

  const float fraction =
    (-steerRad) / MAX_STEER_RAD;

  return static_cast<uint16_t>(
    lroundf(
      static_cast<float>(SERVO_CENTER_US)
      - fraction
      * static_cast<float>(
          SERVO_CENTER_US - SERVO_MIN_US
        )
    )
  );
}

static void writeSteeringRadians(float steerRad) {
  steerRad *= SERVO_DIRECTION;

  steerRad = constrain(
    steerRad,
    -MAX_STEER_RAD,
    MAX_STEER_RAD
  );

  const uint16_t pulse =
    steeringToPulse(
      steerRad
    );

  writeServoMicroseconds(
    pulse
  );
}

static void setVibrationPower(
  uint8_t top,
  uint8_t bottom,
  uint8_t left,
  uint8_t right
) {
  // Actualmente cada salida es ON/OFF:
  // cualquier valor > 0 activa el motor al 100 %.
  writeVibrationOutput(VIB_TOP_PIN, top);
  writeVibrationOutput(VIB_BOTTOM_PIN, bottom);
  writeVibrationOutput(VIB_LEFT_PIN, left);
  writeVibrationOutput(VIB_RIGHT_PIN, right);
}

// -----------------------------------------------------------------------------
// Lenguaje háptico
//
// La Raspberry sigue enviando:
//   CMD,seq,steer_rad,haptic
//
// haptic = 0  -> sin evento
// haptic = 1  -> INICIO GIRO IZQUIERDA: 1 pulso claro a la izquierda
// haptic = 2  -> INICIO GIRO DERECHA:   1 pulso claro a la derecha
// haptic = 3  -> OBSTÁCULO AL LADO IZQ: 2 pulsos cortos a la izquierda
// haptic = 4  -> OBSTÁCULO AL LADO DER: 2 pulsos cortos a la derecha
// haptic = 5  -> MANIOBRA TERMINADA:    2 pulsos cortos abajo
// haptic = 6  -> LOCALIZACIÓN PERDIDA:  alternancia izquierda/derecha
// haptic = 7  -> PELIGRO GENERAL:       3 pulsos de los cuatro
//
// SEGURIDAD LOCAL adicional:
// - superior: aviso local único + segundo aviso si se acerca mucho;
// - laterales: doble pulso local al entrar a <= 40 cm, independiente
//   de la máquina de estados, con rearme al quedar > 55 cm.
//
// Los eventos 1..7 de Raspberry siguen conservando exactamente su significado.
//
// IMPORTANTE:
// Los eventos 1..7 son "one-shot". Si Raspberry mantiene, por ejemplo,
// haptic=1 durante 2 segundos, NO vibra continuamente. Solo se ejecuta una vez.
// Para volver a disparar el mismo evento, Raspberry debe enviar haptic=0
// entre ambos eventos.
// -----------------------------------------------------------------------------

static void startHapticEvent(
  uint8_t eventCode,
  uint32_t nowMs
) {
  activeHapticEvent = eventCode;
  hapticEventStartMs = nowMs;
  currentHaptic = eventCode;
}

static void stopHapticEvent() {
  activeHapticEvent = 0;
  currentHaptic = 0;
  setVibrationPower(0, 0, 0, 0);
}

// -----------------------------------------------------------------------------
// Eventos hápticos LOCALES.
//
// PRIORIDAD:
//   1) evento local que ya está reproduciéndose -> se termina correctamente;
//   2) aviso SUPERIOR pendiente -> seguridad prioritaria;
//   3) evento semántico de Raspberry;
//   4) aviso lateral local pendiente.
//
// Así un aviso lateral no corta a mitad una señal de "voy a girar", pero
// tampoco se pierde: queda pendiente y se reproduce después.
//
// El superior puede adelantarse a un evento de Raspberry porque protege
// cabeza/parte alta y se considera la advertencia local más prioritaria.
// -----------------------------------------------------------------------------

static void queueLocalHaptic(uint8_t pendingBit) {
  pendingLocalHaptics |= pendingBit;
}

static bool hasPendingUpperHaptic() {
  return (
    pendingLocalHaptics
    & (
      LOCAL_PENDING_UPPER
      | LOCAL_PENDING_UPPER_NEAR
    )
  ) != 0;
}

static bool hasPendingSideHaptic() {
  return (
    pendingLocalHaptics
    & (
      LOCAL_PENDING_LEFT
      | LOCAL_PENDING_RIGHT
    )
  ) != 0;
}

static void startNextUpperLocalHaptic(uint32_t nowMs) {
  if (
    pendingLocalHaptics
    & LOCAL_PENDING_UPPER_NEAR
  ) {
    pendingLocalHaptics &=
      ~LOCAL_PENDING_UPPER_NEAR;

    activeLocalHaptic =
      LOCAL_HAPTIC_UPPER_NEAR;

    localHapticStartMs = nowMs;
    return;
  }

  if (
    pendingLocalHaptics
    & LOCAL_PENDING_UPPER
  ) {
    pendingLocalHaptics &=
      ~LOCAL_PENDING_UPPER;

    activeLocalHaptic =
      LOCAL_HAPTIC_UPPER;

    localHapticStartMs = nowMs;
  }
}

static void startNextSideLocalHaptic(uint32_t nowMs) {
  const bool leftPending =
    (
      pendingLocalHaptics
      & LOCAL_PENDING_LEFT
    ) != 0;

  const bool rightPending =
    (
      pendingLocalHaptics
      & LOCAL_PENDING_RIGHT
    ) != 0;

  if (
    leftPending
    && rightPending
  ) {
    pendingLocalHaptics &=
      ~(
        LOCAL_PENDING_LEFT
        | LOCAL_PENDING_RIGHT
      );

    activeLocalHaptic =
      LOCAL_HAPTIC_BOTH_SIDES;

    localHapticStartMs = nowMs;
    return;
  }

  if (leftPending) {
    pendingLocalHaptics &=
      ~LOCAL_PENDING_LEFT;

    activeLocalHaptic =
      LOCAL_HAPTIC_LEFT;

    localHapticStartMs = nowMs;
    return;
  }

  if (rightPending) {
    pendingLocalHaptics &=
      ~LOCAL_PENDING_RIGHT;

    activeLocalHaptic =
      LOCAL_HAPTIC_RIGHT;

    localHapticStartMs = nowMs;
  }
}

static bool updateLocalHaptic(uint32_t nowMs) {
  if (
    activeLocalHaptic
    == LOCAL_HAPTIC_NONE
  ) {
    return false;
  }

  const uint32_t elapsed =
    nowMs - localHapticStartMs;

  switch (activeLocalHaptic) {
    case LOCAL_HAPTIC_UPPER: {
      // Tres pulsos reconocibles, una sola vez.
      // 130 ON / 120 OFF / 130 ON / 120 OFF / 130 ON.
      const bool on =
        (elapsed < 130)
        || (
          elapsed >= 250
          && elapsed < 380
        )
        || (
          elapsed >= 500
          && elapsed < 630
        );

      if (elapsed < 630) {
        setVibrationPower(
          on ? 255 : 0,
          0,
          0,
          0
        );
      } else {
        activeLocalHaptic =
          LOCAL_HAPTIC_NONE;

        setVibrationPower(
          0,
          0,
          0,
          0
        );
      }

      return true;
    }

    case LOCAL_HAPTIC_UPPER_NEAR: {
      // Obstáculo superior ya muy cercano:
      // misma semántica, pero ráfaga más rápida.
      const bool on =
        (elapsed < 110)
        || (
          elapsed >= 175
          && elapsed < 285
        )
        || (
          elapsed >= 350
          && elapsed < 460
        );

      if (elapsed < 460) {
        setVibrationPower(
          on ? 255 : 0,
          0,
          0,
          0
        );
      } else {
        activeLocalHaptic =
          LOCAL_HAPTIC_NONE;

        setVibrationPower(
          0,
          0,
          0,
          0
        );
      }

      return true;
    }

    case LOCAL_HAPTIC_LEFT:
    case LOCAL_HAPTIC_RIGHT:
    case LOCAL_HAPTIC_BOTH_SIDES: {
      // Seguridad lateral:
      // doble pulso corto, una sola vez al entrar en zona de peligro.
      const bool on =
        (elapsed < 120)
        || (
          elapsed >= 220
          && elapsed < 340
        );

      if (elapsed < 340) {
        const uint8_t left =
          (
            activeLocalHaptic
            == LOCAL_HAPTIC_LEFT
            || activeLocalHaptic
               == LOCAL_HAPTIC_BOTH_SIDES
          )
          ? (on ? 255 : 0)
          : 0;

        const uint8_t right =
          (
            activeLocalHaptic
            == LOCAL_HAPTIC_RIGHT
            || activeLocalHaptic
               == LOCAL_HAPTIC_BOTH_SIDES
          )
          ? (on ? 255 : 0)
          : 0;

        setVibrationPower(
          0,
          0,
          left,
          right
        );
      } else {
        activeLocalHaptic =
          LOCAL_HAPTIC_NONE;

        setVibrationPower(
          0,
          0,
          0,
          0
        );
      }

      return true;
    }

    default: {
      activeLocalHaptic =
        LOCAL_HAPTIC_NONE;

      setVibrationPower(
        0,
        0,
        0,
        0
      );
      return false;
    }
  }
}

static void updateRemoteHaptic(uint32_t nowMs) {
  if (activeHapticEvent == 0) {
    setVibrationPower(0, 0, 0, 0);
    currentHaptic = 0;
    return;
  }

  const uint32_t elapsed =
    nowMs - hapticEventStartMs;

  switch (activeHapticEvent) {
    case 1: { // INICIO GIRO IZQUIERDA: un pulso claro
      if (elapsed < 320) {
        setVibrationPower(
          0,
          0,
          255,
          0
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    case 2: { // INICIO GIRO DERECHA: un pulso claro
      if (elapsed < 320) {
        setVibrationPower(
          0,
          0,
          0,
          255
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    case 3: { // OBSTÁCULO AL LADO IZQUIERDO
      const bool on =
        (elapsed < 120)
        || (
          elapsed >= 220
          && elapsed < 340
        );

      if (elapsed < 340) {
        setVibrationPower(
          0,
          0,
          on ? 255 : 0,
          0
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    case 4: { // OBSTÁCULO AL LADO DERECHO
      const bool on =
        (elapsed < 120)
        || (
          elapsed >= 220
          && elapsed < 340
        );

      if (elapsed < 340) {
        setVibrationPower(
          0,
          0,
          0,
          on ? 255 : 0
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    case 5: { // MANIOBRA TERMINADA: doble pulso abajo
      const bool on =
        (elapsed < 140)
        || (
          elapsed >= 250
          && elapsed < 390
        );

      if (elapsed < 390) {
        setVibrationPower(
          0,
          on ? 255 : 0,
          0,
          0
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    case 6: { // LOCALIZACIÓN PERDIDA
      if (elapsed < 180) {
        setVibrationPower(
          0,
          0,
          255,
          0
        );
      } else if (elapsed < 280) {
        setVibrationPower(
          0,
          0,
          0,
          0
        );
      } else if (elapsed < 460) {
        setVibrationPower(
          0,
          0,
          0,
          255
        );
      } else if (elapsed < 560) {
        setVibrationPower(
          0,
          0,
          0,
          0
        );
      } else if (elapsed < 740) {
        setVibrationPower(
          0,
          0,
          255,
          0
        );
      } else if (elapsed < 840) {
        setVibrationPower(
          0,
          0,
          0,
          0
        );
      } else if (elapsed < 1020) {
        setVibrationPower(
          0,
          0,
          0,
          255
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    case 7: { // PELIGRO GENERAL
      const bool on =
        (elapsed < 140)
        || (
          elapsed >= 230
          && elapsed < 370
        )
        || (
          elapsed >= 460
          && elapsed < 600
        );

      if (elapsed < 600) {
        const uint8_t v =
          on ? 255 : 0;

        setVibrationPower(
          v,
          v,
          v,
          v
        );
      } else {
        stopHapticEvent();
      }
      break;
    }

    default: {
      stopHapticEvent();
      break;
    }
  }
}

static void updateHaptics(uint32_t nowMs) {
  // 1) Si un evento local ya empezó, lo terminamos sin cortarlo.
  if (
    activeLocalHaptic
    != LOCAL_HAPTIC_NONE
  ) {
    updateLocalHaptic(nowMs);
    return;
  }

  // 2) El aviso superior pendiente tiene prioridad sobre Raspberry.
  if (hasPendingUpperHaptic()) {
    startNextUpperLocalHaptic(nowMs);
    updateLocalHaptic(nowMs);
    return;
  }

  // 3) Eventos semánticos de navegación enviados por Raspberry.
  if (activeHapticEvent != 0) {
    updateRemoteHaptic(nowMs);
    return;
  }

  // 4) Seguridad lateral permanente.
  if (hasPendingSideHaptic()) {
    startNextSideLocalHaptic(nowMs);
    updateLocalHaptic(nowMs);
    return;
  }

  setVibrationPower(
    0,
    0,
    0,
    0
  );
  currentHaptic = 0;
}

// -----------------------------------------------------------------------------
// Botones
// -----------------------------------------------------------------------------
static void updateButton(
  DebouncedButton &button,
  uint32_t nowMs
) {
  const bool rawPressed =
    digitalRead(button.pin) == LOW;

  if (rawPressed != button.lastRawPressed) {
    button.lastRawPressed = rawPressed;
    button.lastChangeMs = nowMs;
  }

  if (
    nowMs - button.lastChangeMs
    >= BUTTON_DEBOUNCE_MS
  ) {
    button.stablePressed = rawPressed;
  }
}

// -----------------------------------------------------------------------------
// Ultrasonidos
// -----------------------------------------------------------------------------
static RangeReading readUltrasonic(
  uint8_t trigPin,
  uint8_t echoPin
) {
  RangeReading result {0, false};

  digitalWrite(trigPin, LOW);
  delayMicroseconds(3);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  const unsigned long durationUs = pulseIn(
    echoPin,
    HIGH,
    ULTRASONIC_TIMEOUT_US
  );

  if (durationUs == 0) {
    return result;
  }

  // HC-SR04: distancia cm ~= duración_us / 58.
  // En mm: duración_us * 10 / 58.
  const uint32_t mm =
    (durationUs * 10UL) / 58UL;

  if (
    mm < ULTRASONIC_MIN_MM
    || mm > ULTRASONIC_MAX_MM
  ) {
    return result;
  }

  result.mm = static_cast<uint16_t>(mm);
  result.valid = true;
  return result;
}

static void updateUpperObstacleState() {
  // ------------------------------------------------------------
  // ENTRADA a zona superior de peligro.
  // ------------------------------------------------------------
  if (!upperObstacleAlert) {
    if (
      usUpper.valid
      && usUpper.mm <= UPPER_ALERT_ON_MM
    ) {
      if (
        upperNearCount
        < UPPER_ALERT_CONFIRM_COUNT
      ) {
        ++upperNearCount;
      }

      upperClearCount = 0;

      if (
        upperNearCount
        >= UPPER_ALERT_CONFIRM_COUNT
      ) {
        upperObstacleAlert = true;
        upperNearCount =
          UPPER_ALERT_CONFIRM_COUNT;

        upperVeryNearCount = 0;

        // Si ya entró extremadamente cerca, usamos directamente
        // la advertencia rápida y no reproducimos dos ráfagas seguidas.
        if (
          usUpper.mm
          <= UPPER_NEAR_MM
        ) {
          upperNearNotified = true;

          queueLocalHaptic(
            LOCAL_PENDING_UPPER_NEAR
          );
        } else {
          upperNearNotified = false;

          queueLocalHaptic(
            LOCAL_PENDING_UPPER
          );
        }
      }

      return;
    }

    upperNearCount = 0;
    upperClearCount = 0;
    upperVeryNearCount = 0;
    return;
  }

  // ------------------------------------------------------------
  // Ya existe un obstáculo superior activo.
  // NO repetimos la alerta normal continuamente.
  //
  // Solo generamos una segunda advertencia si pasa a <= 45 cm.
  // ------------------------------------------------------------
  if (
    usUpper.valid
    && usUpper.mm < UPPER_ALERT_OFF_MM
  ) {
    upperClearCount = 0;

    if (
      !upperNearNotified
      && usUpper.mm <= UPPER_NEAR_MM
    ) {
      if (
        upperVeryNearCount
        < UPPER_NEAR_CONFIRM_COUNT
      ) {
        ++upperVeryNearCount;
      }

      if (
        upperVeryNearCount
        >= UPPER_NEAR_CONFIRM_COUNT
      ) {
        upperNearNotified = true;

        queueLocalHaptic(
          LOCAL_PENDING_UPPER_NEAR
        );
      }
    } else if (
      usUpper.mm > UPPER_NEAR_MM
    ) {
      upperVeryNearCount = 0;
    }

    return;
  }

  // ------------------------------------------------------------
  // REARME.
  // Se exige que el obstáculo esté claramente fuera de la zona
  // durante varias muestras para no rearmar por un eco aislado.
  // ------------------------------------------------------------
  if (
    upperClearCount
    < UPPER_CLEAR_CONFIRM_COUNT
  ) {
    ++upperClearCount;
  }

  if (
    upperClearCount
    >= UPPER_CLEAR_CONFIRM_COUNT
  ) {
    upperObstacleAlert = false;
    upperNearNotified = false;

    upperNearCount = 0;
    upperVeryNearCount = 0;
    upperClearCount = 0;
  }
}

static void updateOneLateralSafetyState(
  const RangeReading &reading,
  bool &latched,
  uint8_t &nearCount,
  uint8_t &clearCount,
  uint8_t pendingBit
) {
  // ------------------------------------------------------------
  // ENTRADA a zona lateral de seguridad.
  // ------------------------------------------------------------
  if (!latched) {
    if (
      reading.valid
      && reading.mm <= LATERAL_ALERT_ON_MM
    ) {
      if (
        nearCount
        < LATERAL_ALERT_CONFIRM_COUNT
      ) {
        ++nearCount;
      }

      clearCount = 0;

      if (
        nearCount
        >= LATERAL_ALERT_CONFIRM_COUNT
      ) {
        latched = true;

        nearCount =
          LATERAL_ALERT_CONFIRM_COUNT;

        queueLocalHaptic(
          pendingBit
        );
      }

      return;
    }

    nearCount = 0;
    clearCount = 0;
    return;
  }

  // ------------------------------------------------------------
  // El mismo obstáculo sigue al lado:
  // permanecemos latched y NO repetimos vibración.
  // ------------------------------------------------------------
  if (
    reading.valid
    && reading.mm < LATERAL_ALERT_OFF_MM
  ) {
    nearCount =
      LATERAL_ALERT_CONFIRM_COUNT;

    clearCount = 0;
    return;
  }

  // ------------------------------------------------------------
  // REARME:
  // debe desaparecer/quedar lejos durante varias muestras.
  // ------------------------------------------------------------
  nearCount = 0;

  if (
    clearCount
    < LATERAL_CLEAR_CONFIRM_COUNT
  ) {
    ++clearCount;
  }

  if (
    clearCount
    >= LATERAL_CLEAR_CONFIRM_COUNT
  ) {
    latched = false;
    clearCount = 0;
  }
}

static void updateLateralSafetyState() {
  updateOneLateralSafetyState(
    usLeft,
    leftSafetyLatched,
    leftSafetyNearCount,
    leftSafetyClearCount,
    LOCAL_PENDING_LEFT
  );

  updateOneLateralSafetyState(
    usRight,
    rightSafetyLatched,
    rightSafetyNearCount,
    rightSafetyClearCount,
    LOCAL_PENDING_RIGHT
  );
}

static void updateUltrasonics() {
  // Lectura secuencial para reducir crosstalk.
  //
  // IZQ/DER:
  //   1) siguen transmitiéndose a ROS para navegación;
  //   2) además alimentan la seguridad háptica local.
  //
  // SUPERIOR:
  //   solo alimenta la seguridad háptica local.

  usLeft = readUltrasonic(
    US_LEFT_TRIG,
    US_LEFT_ECHO
  );
  delay(2);

  usUpper = readUltrasonic(
    US_UPPER_TRIG,
    US_UPPER_ECHO
  );
  updateUpperObstacleState();
  delay(2);

  usRight = readUltrasonic(
    US_RIGHT_TRIG,
    US_RIGHT_ECHO
  );

  // Se actualizan juntos una vez disponibles ambos laterales.
  updateLateralSafetyState();
}

// -----------------------------------------------------------------------------
// MPU-6050 sin librería externa
// -----------------------------------------------------------------------------
static bool i2cWriteByte(
  uint8_t address,
  uint8_t reg,
  uint8_t value
) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

static bool i2cReadByte(
  uint8_t address,
  uint8_t reg,
  uint8_t &value
) {
  Wire.beginTransmission(address);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const uint8_t received = Wire.requestFrom(
    address,
    static_cast<uint8_t>(1),
    true
  );

  if (received != 1 || !Wire.available()) {
    while (Wire.available()) {
      Wire.read();
    }
    return false;
  }

  value = static_cast<uint8_t>(Wire.read());
  return true;
}

static bool readMpuConfigRegisters(
  uint8_t &configValue,
  uint8_t &gyroConfigValue,
  uint8_t &accelConfigValue
) {
  return
    i2cReadByte(MPU_ADDR, MPU_REG_CONFIG, configValue)
    && i2cReadByte(MPU_ADDR, MPU_REG_GYRO_CONFIG, gyroConfigValue)
    && i2cReadByte(MPU_ADDR, MPU_REG_ACCEL_CONFIG, accelConfigValue);
}

static bool mpuConfigRegistersAreCorrect(
  uint8_t configValue,
  uint8_t gyroConfigValue,
  uint8_t accelConfigValue
) {
  return
    configValue == MPU_EXPECTED_CONFIG
    && gyroConfigValue == MPU_EXPECTED_GYRO_CONFIG
    && accelConfigValue == MPU_EXPECTED_ACCEL_CONFIG;
}

static void printMpuConfigStatus(
  const char *tag,
  bool readOk,
  uint8_t configValue,
  uint8_t gyroConfigValue,
  uint8_t accelConfigValue,
  bool configOk
) {
  Serial.print("INFO,MPU_CONFIG,");
  Serial.print(tag);
  Serial.print(",READ_OK=");
  Serial.print(readOk ? 1 : 0);
  Serial.print(",CONFIG=0x");
  Serial.print(configValue, HEX);
  Serial.print(",GYRO_CONFIG=0x");
  Serial.print(gyroConfigValue, HEX);
  Serial.print(",ACCEL_CONFIG=0x");
  Serial.print(accelConfigValue, HEX);
  Serial.print(",OK=");
  Serial.println(configOk ? 1 : 0);
}

static bool configureAndVerifyMpu6050(
  bool printEveryAttempt
) {
  for (
    uint8_t attempt = 1;
    attempt <= MPU_CONFIG_MAX_ATTEMPTS;
    ++attempt
  ) {
    bool writeOk = true;

    // Selecciona el reloj interno y despierta el MPU-6050.
    writeOk &= i2cWriteByte(
      MPU_ADDR,
      MPU_REG_PWR_MGMT_1,
      0x00
    );
    delay(20);

    // DLPF ~44 Hz gyro / ~42 Hz accel.
    writeOk &= i2cWriteByte(
      MPU_ADDR,
      MPU_REG_CONFIG,
      MPU_EXPECTED_CONFIG
    );

    // Gyro ±500 deg/s -> 65.5 LSB/(deg/s).
    writeOk &= i2cWriteByte(
      MPU_ADDR,
      MPU_REG_GYRO_CONFIG,
      MPU_EXPECTED_GYRO_CONFIG
    );

    // Accel ±4 g -> 8192 LSB/g.
    writeOk &= i2cWriteByte(
      MPU_ADDR,
      MPU_REG_ACCEL_CONFIG,
      MPU_EXPECTED_ACCEL_CONFIG
    );

    // Damos tiempo al sensor antes de leer de vuelta los registros.
    delay(10);

    uint8_t configValue = 0xFF;
    uint8_t gyroConfigValue = 0xFF;
    uint8_t accelConfigValue = 0xFF;

    const bool readOk = readMpuConfigRegisters(
      configValue,
      gyroConfigValue,
      accelConfigValue
    );

    const bool configOk =
      writeOk
      && readOk
      && mpuConfigRegistersAreCorrect(
        configValue,
        gyroConfigValue,
        accelConfigValue
      );

    if (printEveryAttempt || configOk || attempt == MPU_CONFIG_MAX_ATTEMPTS) {
      Serial.print("INFO,MPU_CONFIG_ATTEMPT=");
      Serial.print(attempt);
      Serial.print(",WRITE_OK=");
      Serial.println(writeOk ? 1 : 0);

      printMpuConfigStatus(
        configOk ? "VERIFIED" : "INVALID",
        readOk,
        configValue,
        gyroConfigValue,
        accelConfigValue,
        configOk
      );
    }

    if (configOk) {
      mpuConfigValid = true;
      return true;
    }

    delay(25);
  }

  mpuConfigValid = false;
  return false;
}

static bool initMpu6050() {
  // Al encender, algunos módulos necesitan un pequeño margen antes de
  // aceptar de forma fiable las escrituras de configuración.
  delay(100);

  mpuConfigValid = configureAndVerifyMpu6050(true);
  lastMpuConfigCheckMs = millis();
  return mpuConfigValid;
}

static bool checkAndRecoverMpuConfig(
  uint32_t nowMs
) {
  if (
    mpuConfigValid
    && nowMs - lastMpuConfigCheckMs < MPU_CONFIG_CHECK_PERIOD_MS
  ) {
    return true;
  }

  lastMpuConfigCheckMs = nowMs;

  uint8_t configValue = 0xFF;
  uint8_t gyroConfigValue = 0xFF;
  uint8_t accelConfigValue = 0xFF;

  const bool readOk = readMpuConfigRegisters(
    configValue,
    gyroConfigValue,
    accelConfigValue
  );

  const bool configOk =
    readOk
    && mpuConfigRegistersAreCorrect(
      configValue,
      gyroConfigValue,
      accelConfigValue
    );

  if (configOk) {
    mpuConfigValid = true;
    return true;
  }

  // Si el MPU se ha reseteado o ha perdido la escala, NO usamos sus datos.
  // Primero intentamos restaurar y verificar la configuración correcta.
  printMpuConfigStatus(
    "LOST",
    readOk,
    configValue,
    gyroConfigValue,
    accelConfigValue,
    false
  );

  mpuConfigValid = false;

  Serial.println("WARN,MPU_CONFIG_LOST,RECONFIGURING");
  const bool recovered = configureAndVerifyMpu6050(false);

  Serial.print("INFO,MPU_RECOVERY,OK=");
  Serial.println(recovered ? 1 : 0);

  return recovered;
}

static int16_t readInt16BE(
  const uint8_t *data
) {
  return static_cast<int16_t>(
    (static_cast<uint16_t>(data[0]) << 8)
    | data[1]
  );
}

static ImuReading readMpu6050() {
  ImuReading result {
    false,
    0, 0, 0,
    0, 0, 0
  };

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_REG_ACCEL_XOUT_H);

  if (
    Wire.endTransmission(false) != 0
  ) {
    return result;
  }

  const uint8_t wanted = 14;
  const uint8_t received = Wire.requestFrom(
    MPU_ADDR,
    wanted,
    true
  );

  if (received != wanted) {
    while (Wire.available()) {
      Wire.read();
    }
    return result;
  }

  uint8_t data[wanted];
  for (uint8_t i = 0; i < wanted; ++i) {
    data[i] = Wire.read();
  }

  const int16_t axRaw = readInt16BE(&data[0]);
  const int16_t ayRaw = readInt16BE(&data[2]);
  const int16_t azRaw = readInt16BE(&data[4]);

  // data[6:8] = temperatura, no necesaria.
  const int16_t gxRaw = readInt16BE(&data[8]);
  const int16_t gyRaw = readInt16BE(&data[10]);
  const int16_t gzRaw = readInt16BE(&data[12]);

  result.ax =
    (static_cast<float>(axRaw) / ACCEL_LSB_PER_G)
    * G_MPS2;
  result.ay =
    (static_cast<float>(ayRaw) / ACCEL_LSB_PER_G)
    * G_MPS2;
  result.az =
    (static_cast<float>(azRaw) / ACCEL_LSB_PER_G)
    * G_MPS2;

  result.gx =
    (static_cast<float>(gxRaw) / GYRO_LSB_PER_DPS)
    * DEG_TO_RAD_F;
  result.gy =
    (static_cast<float>(gyRaw) / GYRO_LSB_PER_DPS)
    * DEG_TO_RAD_F;
  result.gz =
    (static_cast<float>(gzRaw) / GYRO_LSB_PER_DPS)
    * DEG_TO_RAD_F;

  result.valid = true;
  return result;
}

// -----------------------------------------------------------------------------
// Serie
// -----------------------------------------------------------------------------
static void processCommandLine(
  const char *line
) {
  unsigned long seq = 0;
  float steerRad = 0.0f;
  unsigned int haptic = 0;

  const int parsed = sscanf(
    line,
    "CMD,%lu,%f,%u",
    &seq,
    &steerRad,
    &haptic
  );

  if (parsed != 3) {
    return;
  }

  if (!isfinite(steerRad)) {
    return;
  }

  requestedSteerRad = constrain(
    steerRad,
    -MAX_STEER_RAD,
    MAX_STEER_RAD
  );
  requestedHaptic = static_cast<uint8_t>(
    constrain(
      static_cast<int>(haptic),
      0,
      255
    )
  );

  const uint32_t nowMs = millis();

  // Eventos hápticos one-shot:
  // - 0 rearma el sistema para permitir repetir el mismo código más adelante.
  // - un código distinto de 0 dispara solo al aparecer/cambiar.
  if (requestedHaptic == 0) {
    lastHapticCommandCode = 0;
  } else if (requestedHaptic != lastHapticCommandCode) {
    startHapticEvent(requestedHaptic, nowMs);
    lastHapticCommandCode = requestedHaptic;
  }

  lastCommandMs = nowMs;
  writeSteeringRadians(requestedSteerRad);

  // ACK útil para prueba directa desde el Monitor Serie.
  Serial.print("ACK,CMD,seq=");
  Serial.print(seq);
  Serial.print(",steer=");
  Serial.print(requestedSteerRad, 4);
  Serial.print(",servo_us=");
  Serial.print(currentServoUs);
  Serial.print(",haptic=");
  Serial.println(requestedHaptic);
}

static void readSerialCommands() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(
      Serial.read()
    );

    if (c == '\r') {
      continue;
    }

    if (c == '\n') {
      serialBuffer[serialLength] = '\0';

      if (serialLength > 0) {
        processCommandLine(serialBuffer);
      }

      serialLength = 0;
      continue;
    }

    if (
      serialLength
      < sizeof(serialBuffer) - 1
    ) {
      serialBuffer[serialLength++] = c;
    } else {
      // Línea corrupta/demasiado larga.
      serialLength = 0;
    }
  }
}

static void applyCommandWatchdog(
  uint32_t nowMs
) {
  if (
    lastCommandMs == 0
    || nowMs - lastCommandMs
       > COMMAND_WATCHDOG_MS
  ) {
    requestedSteerRad = 0.0f;
    requestedHaptic = 0;
    lastHapticCommandCode = 0;

    if (activeHapticEvent != 0) {
      stopHapticEvent();
    }

    if (currentServoUs != SERVO_CENTER_US) {
      writeSteeringRadians(0.0f);
    }
  }
}

static void sendTelemetry(
  uint32_t nowMs
) {
  ++telemetrySeq;

  // El HC-SR04 superior NO forma parte de esta trama.
  Serial.print("TEL2,");
  Serial.print(telemetrySeq);
  Serial.print(',');
  Serial.print(nowMs);

  Serial.print(',');
  Serial.print(usLeft.mm);
  Serial.print(',');
  Serial.print(usRight.mm);

  Serial.print(',');
  Serial.print(usLeft.valid ? 1 : 0);
  Serial.print(',');
  Serial.print(usRight.valid ? 1 : 0);

  Serial.print(',');
  Serial.print(buttonLeft.stablePressed ? 1 : 0);
  Serial.print(',');
  Serial.print(buttonRight.stablePressed ? 1 : 0);

  Serial.print(',');
  Serial.print(imu.valid ? 1 : 0);

  Serial.print(',');
  Serial.print(imu.ax, 4);
  Serial.print(',');
  Serial.print(imu.ay, 4);
  Serial.print(',');
  Serial.print(imu.az, 4);

  Serial.print(',');
  Serial.print(imu.gx, 5);
  Serial.print(',');
  Serial.print(imu.gy, 5);
  Serial.print(',');
  Serial.print(imu.gz, 5);

  Serial.print(',');
  Serial.print(currentServoUs);

  Serial.print(',');
  Serial.println(currentHaptic);
}

// -----------------------------------------------------------------------------
// Setup / loop
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(300);

  // HC-SR04: laterales + superior local
  pinMode(US_LEFT_TRIG, OUTPUT);
  pinMode(US_LEFT_ECHO, INPUT);

  pinMode(US_UPPER_TRIG, OUTPUT);
  pinMode(US_UPPER_ECHO, INPUT);

  pinMode(US_RIGHT_TRIG, OUTPUT);
  pinMode(US_RIGHT_ECHO, INPUT);

  digitalWrite(US_LEFT_TRIG, LOW);
  digitalWrite(US_UPPER_TRIG, LOW);
  digitalWrite(US_RIGHT_TRIG, LOW);

  // Botones a GND al pulsar.
  pinMode(BUTTON_LEFT_PIN, INPUT_PULLUP);
  pinMode(BUTTON_RIGHT_PIN, INPUT_PULLUP);

  // PWM servo: mismo método que el sketch simple verificado.
  const bool servoPwmOk = attachServoPwm();
  //delay(300);

  /* AUTOPRUEBA DEL SERVO: debe verse físicamente SIN enviar ningún CMD.
  // Pulsos conservadores para evitar topes mecánicos.
  writeServoMicroseconds(2000);
  delay(500);
  writeServoMicroseconds(2250);
  delay(700);
  writeServoMicroseconds(2000);
  delay(500);
  writeServoMicroseconds(1750);
  delay(700);
  writeServoMicroseconds(2000);
  delay(500);*/

  writeServoMicroseconds(SERVO_CENTER_US);
  delay(1500);

  // ERM por MOSFET, sin reservar canales/timers LEDC.
  attachVibrationOutput(VIB_TOP_PIN);
  attachVibrationOutput(VIB_BOTTOM_PIN);
  attachVibrationOutput(VIB_LEFT_PIN);
  attachVibrationOutput(VIB_RIGHT_PIN);

  writeSteeringRadians(0.0f);
  setVibrationPower(0, 0, 0, 0);

  // IMU.
  Wire.begin(
    I2C_SDA_PIN,
    I2C_SCL_PIN,
    400000
  );

  const bool imuStarted = initMpu6050();

  Serial.println("INFO,SERVO_BOOT_SWEEP_DONE");

  Serial.print("INFO,SERVO_ESP32SERVO,");
  Serial.print(servoPwmOk ? "OK" : "ERROR");
  Serial.print(",pin=");
  Serial.print(SERVO_PIN);
  Serial.print(",freq=");
  Serial.print(SERVO_PWM_HZ);
  Serial.print(",min_us=");
  Serial.print(SERVO_MIN_US);
  Serial.print(",center_us=");
  Serial.print(SERVO_CENTER_US);
  Serial.print(",max_us=");
  Serial.println(SERVO_MAX_US);

  Serial.print("INFO,MAPLESS_CANE_FULL_READY,IMU=");
  Serial.println(imuStarted ? 1 : 0);

  lastTelemetryMs = millis();
}

void loop() {
  const uint32_t nowMs = millis();

  readSerialCommands();
  applyCommandWatchdog(nowMs);

  updateButton(buttonLeft, nowMs);
  updateButton(buttonRight, nowMs);

  // Haptics no bloqueantes.
  updateHaptics(nowMs);

  if (
    nowMs - lastTelemetryMs
    >= TELEMETRY_PERIOD_MS
  ) {
    lastTelemetryMs = nowMs;

    updateUltrasonics();

    // Aplicar inmediatamente cualquier nuevo evento local superior/lateral.
    updateHaptics(millis());

    // Verifica periódicamente que las escalas del MPU sigan siendo las
    // esperadas. Si no lo son, intenta recuperarlas antes de aceptar datos.
    if (checkAndRecoverMpuConfig(millis())) {
      imu = readMpu6050();
    } else {
      imu = ImuReading {
        false,
        0, 0, 0,
        0, 0, 0
      };
    }

    // Los botones se actualizan otra vez por si las lecturas ultrasónicas
    // consumieron varias decenas de ms.
    const uint32_t afterSensorsMs = millis();
    updateButton(buttonLeft, afterSensorsMs);
    updateButton(buttonRight, afterSensorsMs);

    sendTelemetry(afterSensorsMs);
  }

  delay(1);
}
