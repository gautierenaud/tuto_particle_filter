from pf.robot import Robot
from pf.proba import evaluation
from pf.visu import visualization
import random


myrobot = Robot()
myrobot = myrobot.move(.1, .5)

# create a set of particles
n = 1000
p = []

for i in range(n):
    x = Robot()
    x.set_noise(0.05, 0.05, 5.0)
    p.append(x)

# steps of particle filter
steps = 50
for t in range(steps):
    # save initial particles
    pr = p

    # move robot and probe environment
    myrobot = myrobot.move(.1, 5.)
    z = myrobot.sense()

    # simulate motion for each particles
    p2 = []
    for i in range(n):
        p2.append(p[i].move(.1, 5.))
    p = p2

    # generate particle weights depending on the robot's measurement
    w = []
    for i in range(n):
        w.append(p[i].measurement_prob(z))

    # resample particles according to weights
    p3 = []
    index = int(random.random()*n)
    beta = 0.
    mw = max(w)

    for i in range(n):
        beta += random.random() * 2. * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index+1) % n
        p3.append(p[index])

    p = p3
    print('Step :' + str(t) + '; Eval = ' + str(evaluation(myrobot, p)))

    visualization(myrobot, t, p, pr, w)
