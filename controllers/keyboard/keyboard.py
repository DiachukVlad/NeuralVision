import os
import struct

import cv2
import numpy as np
from controller import Keyboard
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
        self.pd = pd
        self.velocity = 0.0
        self.motor = robot.getDevice(name)
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0.0)

    def update(self):
        self.velocity += self.pd.process_measurement(self.motor.getVelocity())
        self.motor.setVelocity(self.velocity)

    def set_target(self, target):
        self.pd.target = target


class Camera:
    def __init__(self, name, time_step, folder=None):
        self.name = name
        self.frame = None
        self.image_byte_size = None
        self.folder = folder
        self.image_left_id = 0
        self.image_right_id = 0
        self.image_forward_id = 0
        self.image_reverse_id = 0
        self.device = robot.getDevice(self.name)
        self.device.enable(time_step)
        self.image_byte_size = self.device.getWidth() * self.device.getHeight() * 4
        self.set_number('left', self.image_left_id)
        self.set_number('right', self.image_right_id)
        self.set_number('forward', self.image_forward_id)
        self.set_number('reverse', self.image_reverse_id)

    def set_number(self, category, cat_id):
        if cat_id is not None:
            for filename in os.listdir(f'{self.folder}/{category}'):
                if filename.endswith('.png'):
                    try:
                        image_id = int(filename.split('.')[0])
                        if image_id > cat_id:
                            cat_id = image_id
                    except Exception:
                        pass
                cat_id += 1

            if category == 'left':
                self.image_left_id = cat_id
            elif category == 'right':
                self.image_right_id = cat_id
            elif category == 'forward':
                self.image_forward_id = cat_id
            elif category == 'reverse':
                self.image_reverse_id = cat_id

    def update(self):
        frame = self.device.getImage()
        if frame is None:
            return self.frame
        frame = struct.unpack(f'{self.image_byte_size}B', frame)
        frame = np.array(frame,
                         dtype=np.uint8).reshape(self.device.getHeight(),
                                                 self.device.getWidth(),
                                                 4)
        frame = frame[:, :, 0:3]
        self.frame = frame
        return frame

    def show(self, scale=1.0):
        scaled = cv2.resize(self.frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow(self.name, scaled)

    def save(self, motor_left, motor_right):
        check_left = motor_left.pd.target
        check_right = motor_right.pd.target

        category = ''
        if check_left > 0 and check_right > 0:
            category = 'forward'
        elif check_left < 0 and check_right < 0:
            category = 'reverse'
        elif check_left > 0 > check_right:
            category = 'right'
        elif check_left < 0 < check_right:
            category = 'left'

        if len(category) > 0:
            self.save_to(category, self.frame)
            self.aug_noise(category, self.frame)

    def save_to(self, category, img):

        path = f'{self.folder}/' + category + '/'

        if category == 'left':
            cv2.imwrite(path + f'{self.image_left_id}.png', img)
            print("Photo saved as -> " + category + f'/{self.image_left_id}.png')
            self.image_left_id += 1

        elif category == 'right':
            cv2.imwrite(path + f'{self.image_right_id}.png', img)
            print("Photo saved as -> " + category + f'/{self.image_right_id}.png')
            self.image_right_id += 1

        elif category == 'forward':
            cv2.imwrite(path + f'{self.image_forward_id}.png', img)
            print("Photo saved as -> " + category + f'/{self.image_forward_id}.png')
            self.image_forward_id += 1

        elif category == 'reverse':
            cv2.imwrite(path + f'{self.image_reverse_id}.png', img)
            print("Photo saved as -> " + category + f'/{self.image_reverse_id}.png')
            self.image_reverse_id += 1

    def aug_noise(self, category, noised_img):
        noise = np.random.normal(0.0, NOISE_VAL + 0.01, size=noised_img.shape) * 255
        noise2 = np.random.normal(0.0, NOISE_VAL - 0.01, size=noised_img.shape) * 255
        noised_img1 = noised_img + noise
        noised_img2 = noised_img + noise2
        self.save_to(category, noised_img1)
        self.save_to(category, noised_img2)


NOISE_VAL = 0.05
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

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

camera = Camera('kinect color', TIME_STEP, '../../dataset')

left_motor = MotorController('left wheel', PDController(0.12, 0.001, TIME_STEP_SECONDS))
right_motor = MotorController('right wheel', PDController(0.12, 0.001, TIME_STEP_SECONDS))

PHOTO_PERIOD = 0.5
last_photo_time = 0
time_passed = 0
photo_mode = True

# MAIN LOOP
while robot.step(TIME_STEP) != -1:
    left_motor.update()
    right_motor.update()

    camera.update()
    camera.show(scale=4.0)

    time_passed += TIME_STEP

    if photo_mode and time_passed - last_photo_time >= PHOTO_PERIOD * 1000:
        last_photo_time = time_passed
        camera.save(left_motor, right_motor)

    key = keyboard.getKey()
    if key == ord('R'):#record
        print('RECORDING')
        photo_mode = True
    elif key == ord('C'): #cancel recording
        print('NOT RECORDING')
        photo_mode = False
    elif key in motor_commands.keys():
        left_motor.set_target(motor_commands[key][0])
        right_motor.set_target(motor_commands[key][1])
    else:
        left_motor.set_target(0.0)
        right_motor.set_target(0.0)
