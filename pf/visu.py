import matplotlib.pyplot as plt
from pf.world import landmarks, world_size
from math import sin, cos


def visualization(robot, step, p, pr, weights):
    plt.figure("Robot in the world", figsize = (15., 15.))
    plt.title('Particle filter, step ' + str(step))

    # draw coordinate grid for plotting
    grid = [0, world_size, 0, world_size]
    plt.axis(grid)
    plt.grid(b=True, which='major', color='0.75', linestyle='--')
    plt.xticks([i for i in range(0, int(world_size), 5)])
    plt.yticks([i for i in range(0, int(world_size), 5)])

    # draw particles
    for ind in range(len(p)):
        # particle
        circle = plt.Circle((p[ind].x, p[ind].y), 1., facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
        plt.gca().add_patch(circle)

        # particle's orientation
        arrow = plt.Arrow(p[ind].x, p[ind].y, 2 * cos(p[ind].orientation), 2 * sin(p[ind].orientation), alpha=1.,
                          facecolor='#994c00', edgecolor='#994c00')
        plt.gca().add_patch(arrow)

    # draw resampled particles
    for ind in range(len(pr)):
        # particle
        circle = plt.Circle((pr[ind].x, pr[ind].y), 1., facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
        plt.gca().add_patch(circle)

        # particle's orientation
        arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2 * cos(pr[ind].orientation), 2 * sin(pr[ind].orientation), alpha=1.,
                          facecolor='#006600', edgecolor='#006600')
        plt.gca().add_patch(arrow)

    # fixed landmarks of known locations
    for lm in landmarks:
        circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
        plt.gca().add_patch(circle)

    # robot's location
    circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
    plt.gca().add_patch(circle)

    # robot's orientation
    arrow = plt.Arrow(robot.x, robot.y, 2 * cos(robot.orientation), 2 * sin(robot.orientation), alpha=0.5,
                      facecolor='#000000', edgecolor='#000000')
    plt.gca().add_patch(arrow)

    plt.savefig("figure " + str(step) + ".png")
    plt.close()