import RPi.GPIO as GPIO
import time

# Define GPIO pins connected to DM542
PUL_PIN = 5  # Replace with the actual GPIO pin number connected to PUL+
DIR_PIN = 6  # Replace with the actual GPIO pin number connected to DIR+
# ENA_PIN = 16 # Optional: Replace with the actual GPIO pin number connected to ENA+

# Define motor control parameters
NEMA_23_SPR = 200
DM542_PPR = 400
GEARBOX_RATIO = 46
STEPS_PER_REVOLUTION = NEMA_23_SPR * DM542_PPR * GEARBOX_RATIO
MICROSTEP_RESOLUTION = DM542_PPR / NEMA_23_SPR
TOTAL_STEPS = STEPS_PER_REVOLUTION
MAX_STEP_DELAY = 0.001  # Starting delay for acceleration (slower)
MIN_STEP_DELAY = 0.00005  # Target delay (faster) - lower = more speed but requires more torque

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup GPIO pins as outputs
GPIO.setup(PUL_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)
# GPIO.setup(ENA_PIN, GPIO.OUT) # If using enable pin

def set_direction(direction):
    """Sets the direction of the stepper motor.
    direction: True for clockwise, False for counter-clockwise.
    """
    GPIO.output(DIR_PIN, direction)
    time.sleep(0.000005) # Ensure DIR signal is ahead of PUL by at least 5µs [1, 2]

def send_pulse():
    """Sends a single pulse to the stepper motor driver."""
    GPIO.output(PUL_PIN, GPIO.HIGH)
    time.sleep(0.0000025) # Pulse width not less than 2.5µs [2]
    GPIO.output(PUL_PIN, GPIO.LOW)
    time.sleep(0.0000025) # Low level width not less than 2.5µs [2]

def move_steps(steps, direction=True, use_acceleration=True):
    """Moves the stepper motor a specified number of steps with optional acceleration.
    steps: The number of microsteps to move.
    direction: True for clockwise, False for counter-clockwise (default is True).
    use_acceleration: Whether to use acceleration and deceleration ramping.
    """
    set_direction(direction)

    if not use_acceleration:
        # Constant speed movement
        for _ in range(steps):
            send_pulse()
            time.sleep(MIN_STEP_DELAY)
        return

    # Calculate acceleration and deceleration phases
    accel_steps = min(steps // 3, 1000)  # Accelerate for 1/3 of movement or 1000 steps max
    decel_steps = min(steps // 3, 1000)  # Decelerate for 1/3 of movement or 1000 steps max
    constant_steps = steps - accel_steps - decel_steps

    # Acceleration phase
    for i in range(accel_steps):
        send_pulse()
        delay = MAX_STEP_DELAY - (i * (MAX_STEP_DELAY - MIN_STEP_DELAY) / accel_steps)
        time.sleep(max(delay, MIN_STEP_DELAY))

    # Constant speed phase
    for _ in range(constant_steps):
        send_pulse()
        time.sleep(MIN_STEP_DELAY)

    # Deceleration phase
    for i in range(decel_steps):
        send_pulse()
        delay = MIN_STEP_DELAY + (i * (MAX_STEP_DELAY - MIN_STEP_DELAY) / decel_steps)
        time.sleep(delay)

if __name__ == '__main__':
    try:
        # Optional: Enable the driver (if using ENA pin, typically high to enable for NPN control)
        # GPIO.output(ENA_PIN, GPIO.HIGH)

        print("Moving clockwise for one revolution with acceleration...")
        move_steps(TOTAL_STEPS, True, use_acceleration=True)
        time.sleep(1)

        print("Moving counter-clockwise for half a revolution at constant speed...")
        move_steps(TOTAL_STEPS // 2, False, use_acceleration=False)
        time.sleep(1)

    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        # Optional: Disable the driver on exit
        # GPIO.output(ENA_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("GPIO cleanup done.")
