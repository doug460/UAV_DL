'''
Created on Apr 8, 2018

@author: DougBrownWin
'''

import GlobalVariables as vars
from UAV import UAV
from Target import Target
import random
import math
import numpy as np
import random
import GlobalVariables as vars
from UAV import UAV
from Target import Target
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

if __name__ == '__main__':
    pass

    # create UAV and target\
    uavPos = np.array([vars.uavTurn_radius,0])
    uavDir = 90*2*math.pi/360
    uav = UAV(uavPos, uavDir)
    
    targetPos = np.array([vars.uavTurn_radius,0])
    targetDir = 90*2*math.pi/360
    target = Target(targetPos, targetDir)
    
    # want to run simulation for 60 seconds
    seconds = 60    
    
    plt.ion()
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    
    # run simulation
    for iteration in range(seconds * vars.fps):
        # target is detected
        if(np.linalg.norm(uav.position - target.position) < vars.uav_fov/2):
            # TODO: vector noise
            measuredPos = target.position + random.random()
            target.measure(measuredPos)
        else:
            target.predict()
        
        # move target and UAV
        target.step()
        uav.step(vars.A_LEFT)
        
        # plot positions
        ax.plot(target.position[0],target.position[1],'x')
        ax.plot(uav.position[0], uav.position[1],'*')
        ax.axis([-vars.search_radius, vars.search_radius, -vars.search_radius, vars.search_radius])
        # plot circle around uav
        radius = vars.uav_fov/2
        t = np.arange(0, 2 * math.pi, 0.01)
        x = np.cos(t) * radius + uav.position[0]
        y = np.sin(t) * radius + uav.position[1]
        line_fov, = ax.plot(x,y,'g:', label = 'Field of View')
        # plot predicted
        ax.plot(target.ePosition[0],target.ePosition[1],'.')
        # plot standard deviation
        sigX = target.uncertainty[0,0]
        sigY = target.uncertainty[1,1]
        ell = Ellipse(target.ePosition, width = sigX, height = sigY, angle = 0, edgecolor='b', lw=2, facecolor='none')
        ax.add_artist(ell)
        
        
        plt.show()
        plt.pause(0.001)
        ax.cla()

    