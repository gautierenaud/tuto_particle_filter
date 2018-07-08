from pf.world import world_size, landmarks
from pf.proba import gaussian
import random
from math import pi, sqrt, cos, sin


class Robot:
    """
    Model the robot's behavior
    """

    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi

        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def __str__(self):
        return ';'.join([str(self.x), str(self.y), str(self.orientation*180/pi)])

    def set(self, new_x, new_y, new_orientation):
        """
        set robot's position and orientation
        :param new_x: x to set [0, world_size]
        :param new_y: y to set [0, world_size]
        :param new_orientation: new orientation
        """

        if not 0 <= new_x <= world_size:
            raise ValueError('X out of bound')
        if not 0 <= new_y <= world_size:
            raise ValueError('Y out of bound')
        if not 0 <= new_orientation <= 2*pi:
            raise ValueError('Orientation out of bound')

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_forward_noise, new_turn_noise, new_sensor_noise):
        """
        Set Noise parameters
        :param new_forward_noise:
        :param new_turn_noise:
        :param new_sensor_noise:
        """

        self.forward_noise = float(new_forward_noise)
        self.turn_noise = float(new_turn_noise)
        self.sense_noise = float(new_sensor_noise)

    def sense(self):
        """
        Sense the environment and send distance between each landmarks, with some noises
        :return: list of measured distances
        """

        return [sqrt((self.x-landmark[0])**2 + (self.y-landmark[1])**2) + random.gauss(0.0, self.sense_noise)
                for landmark in landmarks]

    def move(self, turn, forward):
        """
        move robot
        :param turn:
        :param forward:
        :return: robot's state after the move
        """

        if forward < 0:
            raise ValueError('Robot can not move backwards')

        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2*pi

        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + cos(orientation)*dist
        y = self.y + sin(orientation) * dist

        # cyclic truncate
        x %= world_size
        y %= world_size

        # set particles
        res = Robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)

        return res

    def measurement_prob(self, measurement):
        """
        Compute the likeliness of a measurement
        :param measurement: current measurement
        :return: probability
        """

        prob = 1.0

        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0])**2 + (self.y - landmarks[i][1])**2)
            prob *= gaussian(dist, self.sense_noise, measurement[i])
        return prob
