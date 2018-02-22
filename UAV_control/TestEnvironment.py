'''
Created on Feb 20, 2018

@author: dabrown

Test the environment and visualize what is going on
'''

import numpy as np
import matplotlib.pyplot as plt
from Environment import Environement
import GlobalVariables as vars 
import time

if __name__ == '__main__':
    pass


    # create environment
    env = Environement()
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = 0
    y = 0
    uavLine, = ax.plot(x,y,'x')
    targetsLines = []
    for index in range(vars.target_num):
        line, = ax.plot(x,y,'*')
        targetsLines.append(line)

    
    start_time = time.time()
    
    plt.axis([-vars.search_radius, vars.search_radius, -vars.search_radius, vars.search_radius])
    
    
    
    for indx in range(300):
#         plt.clf()
        # move forward
        
        if(indx < 100):
            uav, targets, cost = env.step(vars.A_FORWARD)
            
        elif(indx < 200):
            uav, targets, cost = env.step(vars.A_LEFT)
            
        else:
            uav, targets, cost = env.step(vars.A_RIGHT)
            
        # plot uav
#         plt.plot(uav.position[0], uav.position[1], '*')
        uavLine.set_xdata(uav.position[0])
        uavLine.set_ydata(uav.position[1])
        
        # plot target stuff
        for index, target in enumerate(targets):
            line = targetsLines[index]
            line.set_xdata(target.position[0])
            line.set_ydata(target.position[1])
#             plt.plot(target.position[0], target.position[1], 'x')
        
        fig.canvas.draw()
        plt.pause(0.001)
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
    

