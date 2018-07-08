from math import exp, sqrt, pi
from pf.world import world_size


def gaussian(mu, sigma, x):
    """
    compute the probability of x for 1-dim Gaussian mean mu and var. sigma
    :param mu: distance to the landmark
    :param sigma: standard deviation
    :param x: distance to landmark measured by the robot
    :return: gaussian value
    """

    return exp(-((mu-x)**2) / (sigma**2) / 2.0) / sqrt(2.0*pi*(sigma**2))


def evaluation(r, p):
    """
    Calculate the mean error of the system
    :param r: current robot object
    :param p: particle set
    :return: mean error of the system
    """
    sum = 0.
    for i in range(len(p)):
        dx = (p[i].x-r.x + world_size/2.) % world_size - world_size/2.
        dy = (p[i].y-r.y + world_size/2.) % world_size - world_size/2.
        err = sqrt(dx**2 + dy**2)
        sum += err
    return sum/float(len(p))
