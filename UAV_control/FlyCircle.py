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
import matplotlib.animation as animation


def runAnime(frame):
    # save animation of flight
    # only run limited number of steps in simulation
    # this updates the figure with was passed to the animation object
    print('%d of %d'% ( frame, seconds * animeFps))
    for iteration in range(math.floor(vars.fps/animeFps)):
        stepSim()

def stepSim():
    # take a single step in the simulation
    vars.stepTime()
    
    # clear plot
    ax.cla()
    
    # target is detected
    if(np.linalg.norm(uav.position - target.position) < vars.uav_fov/2):
        # add some noise to the measurement
        measured = np.array([target.position]).T + np.random.normal(vars.noiseMean, vars.noiseStd, (2,1))
        target.measure(measured)
    else:
        target.predict()
    
    # move target and UAV
    target.step()
    action = np.array([0,1,0])
    uav.step(action)
    
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
    sigX = math.sqrt(target.uncertainty[0,0])
    sigY = math.sqrt(target.uncertainty[1,1])
    ell = Ellipse(target.ePosition, width = 3*sigX, height = 3*sigY, angle = 0, edgecolor='b', lw=2, facecolor='none')
    ax.add_artist(ell)
    plt.title('3$\sigma$ of target position')
    

        
    


if __name__ == '__main__':
    pass
    print('Setting up stuff')
    
    # limiting search radius (m)
    vars.search_radius = 50
    # minimum turning radius of UAV (m)
    vars.uavTurn_radius = 25
    # field of view diameter (m)
    vars.uav_fov = 40
    vars.target_num = 1
    
    # control visualizations
    anime = True
    animeFps = 15
    saveAnime = False
    viewLive = True
    
    # want to run simulation for 60 seconds
    seconds = 20   
    
    # create UAV and target\
    uavPos = np.array([vars.uavTurn_radius,0])
    uavDir = 90*2*math.pi/360
    uav = UAV(uavPos, uavDir)
    
    targetPos = np.array([vars.uavTurn_radius,0])
    targetDir = 180*2*math.pi/360
    target = Target(targetPos, targetDir)
     
    
    #plt.ion()
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    
    # animate
    ani = animation.FuncAnimation(fig, func = runAnime, interval = 1, frames = seconds*animeFps, repeat = False)
    
    if viewLive:
        plt.show()
        
    if saveAnime:
        # save name
        saveName = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/NMSU Research Dr. Sun/Programming/Python/Data/flySims/'
        saveName += 'flyCircle.mp4'
        
        # writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps = 15, metadata=dict(artist='Me'), bitrate=1800)
         
         
         
        print('Saving Animation...')
        ani.save(saveName, writer = writer)
        print('Save Complete!')

        
    
    
        

    