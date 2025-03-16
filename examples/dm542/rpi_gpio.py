import RPi.GPIO as GPIO
import time

# Define GPIO pins connected to DM542
PUL_PIN = 20  # Replace with the actual GPIO pin number connected to PUL+
DIR_PIN = 21  # Replace with the actual GPIO pin number connected to DIR+
# ENA_PIN = 16 # Optional: Replace with the actual GPIO pin number connected to ENA+

# Define motor control parameters
STEPS_PER_REVOLUTION = 200  # Adjust based on your motor
MICROSTEP_RESOLUTION = 16  # Example: Assuming 16 microsteps per full step (configured on DM542)
TOTAL_STEPS = STEPS_PER_REVOLUTION * MICROSTEP_RESOLUTION
STEP_DELAY = 0.001  # Adjust to control motor speed (seconds between pulses)

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

def move_steps(steps, direction=True):
    """Moves the stepper motor a specified number of steps.
    steps: The number of microsteps to move.
    direction: True for clockwise, False for counter-clockwise (default is True).
    """
    set_direction(direction)
    for _ in range(steps):
        send_pulse()
        time.sleep(STEP_DELAY)

if __name__ == '__main__':
    try:
        # Optional: Enable the driver (if using ENA pin, typically high to enable for NPN control)
        # GPIO.output(ENA_PIN, GPIO.HIGH)

        print("Moving clockwise for one revolution...")
        move_steps(TOTAL_STEPS, True)
        time.sleep(1)

        print("Moving counter-clockwise for half a revolution...")
        move_steps(TOTAL_STEPS // 2, False)
        time.sleep(1)

    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        # Optional: Disable the driver on exit
        # GPIO.output(ENA_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("GPIO cleanup done.")