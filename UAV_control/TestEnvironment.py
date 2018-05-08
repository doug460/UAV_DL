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
import math
from matplotlib.patches import Ellipse


if __name__ == '__main__':
    pass


    # create environment
    env = Environement()
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = 0
    y = 0
    
    start_time = time.time()
    action = np.zeros(vars.A_num)
    
    viewRate = vars.fps/5
    
    
    
    
    for indx in range(700):
#         plt.clf()
        # move forward
        plt.axis([-vars.search_radius, vars.search_radius, -vars.search_radius, vars.search_radius])
        
        if(indx < 100):
            action[0] = 1
            state, reward, terminal = env.step(action)
            
        elif(indx < 200):
            action[0] = 0
            action[1] = 1
            state, reward, terminal = env.step(action)
            
        else:
            action[1] = 0
            action[2] = 1
            state, reward, terminal = env.step(action)
            
            
        # if at view rate, view stuff
        if indx % viewRate == 0:
            plt.title('Reward %.2f Term = %s  t = %6.2f' % (reward, terminal, indx / vars.fps))
            plt.xlabel('UAV: (%.1f %.1f) Tar: (%.1f %.1f) $3*\sigma$: %.2f' % (state[0], state[1], state[2], state[3], state[4]))
            uav = env.uav
            targets = env.targets
                
            # plot uav
            plt.plot(uav.position[0], uav.position[1], '*')
            
            # plot target stuff
            for index, target in enumerate(targets):
                plt.plot(target.position[0], target.position[1], '.')
                
                # plot predicted
                ax.plot(target.ePosition[0],target.ePosition[1],'x')
                # plot standard deviation (3 sigma)
                sigX = math.sqrt(target.uncertainty[0,0])*3
                sigY = math.sqrt(target.uncertainty[1,1])*3
                ell = Ellipse(target.ePosition, width = 3*sigX, height = 3*sigY, angle = 0, edgecolor='b', lw=2, facecolor='none')
                ax.add_artist(ell)
    
            # plot circle around uav
            radius = vars.uav_dfov/2
            t = np.arange(0, 2 * math.pi, 0.01)
            x = np.cos(t) * radius + uav.position[0]
            y = np.sin(t) * radius + uav.position[1]
            line_fov, = ax.plot(x,y,'g:', label = 'Field of View')
            
            
            fig.canvas.draw()
            plt.pause(0.001)
            plt.cla()
        
        if terminal:
            env.reset()
        
    print("--- %s seconds ---" % (time.time() - start_time))






















        
    

