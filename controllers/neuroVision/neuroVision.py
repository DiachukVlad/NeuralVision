import struct

import keras
import numpy as np
from controller import Robot


class PDController:

    def __init__(self, p, d, sampling_period):
        self.target = 0.0
        self.response = 0.0
        self.old_error = 0.0
        self.p = p
        self.d = d
        self.sampling_period = sampling_period

    def process_measurement(self, measurement):
        error = self.target - measurement
        derivative = (error - self.old_error) / self.sampling_period
        self.old_error = error
        self.response = self.p * error + self.d * derivative
        return self.response


class MotorController:

    def __init__(self, name, pd):
        self.name = name
        self.pd = pd
        self.motor = None
        self.velocity = 0.0

    def enable(self):
        self.motor = robot.getDevice(self.name)
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0.0)

    def update(self):
        self.velocity += self.pd.process_measurement(self.motor.getVelocity())
        self.motor.setVelocity(self.velocity)

    def set_target(self, target):
        self.pd.target = target


class NeuralController:
    def __init__(self, name, timestep):
        self.name = name
        self.image_byte_size = None
        self.device = None
        self.model = None
        self.device = robot.getDevice(self.name)
        self.device.enable(timestep)
        self.image_byte_size = self.device.getWidth() * self.device.getHeight() * 4
        self.model = keras.models.load_model("model.h5")
        self.model.summary()

    def get_key(self):
        frame = self.device.getImage()
        if frame is None:
            return self.frame
        frame = struct.unpack(f'{self.image_byte_size}B', frame)
        frame = np.array(frame,
                         dtype=np.uint8).reshape(self.device.getHeight(),
                                                 self.device.getWidth(),
                                                 4)
        frame = frame[:, :, 0:3]
        prediction = np.argmax(self.model.predict(frame.reshape((-1, 100, 150, 3))))

        return KEYS[prediction]


KEYS = [ord('D'), ord('A'), ord('W'), ord('S'), ]

SPEED = 4.0
TURN_SPEED = SPEED / 2.0
TIME_STEP = 64
TIME_STEP_SECONDS = TIME_STEP / 1000

motor_commands = {
    ord('W'): (SPEED, SPEED),
    ord('S'): (-SPEED, -SPEED),
    ord('A'): (-TURN_SPEED, TURN_SPEED),
    ord('D'): (TURN_SPEED, -TURN_SPEED)
}

# ROBOT PARTS INIT
robot = Robot()

left_motor = MotorController('left wheel', PDController(0.12, 0.001, TIME_STEP_SECONDS))
right_motor = MotorController('right wheel', PDController(0.12, 0.001, TIME_STEP_SECONDS))
left_motor.enable()
right_motor.enable()

inteligence = NeuralController('kinect color', TIME_STEP)

PHOTO_PERIOD = 0.5
last_photo_time = 0
time_passed = 0
photo_mode = True

# MAIN LOOP
while robot.step(TIME_STEP) != -1:
    left_motor.update()
    right_motor.update()

    time_passed += TIME_STEP

    key = inteligence.get_key()
    if key in motor_commands.keys():
        left_motor.set_target(motor_commands[key][0])
        right_motor.set_target(motor_commands[key][1])
    else:
        left_motor.set_target(0.0)
        right_motor.set_target(0.0)
