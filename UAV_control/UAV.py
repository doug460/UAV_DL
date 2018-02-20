'''
Created on Feb 19, 2018

@author: DougBrownWin
'''

import numpy as np
import GlobalVariables as vars
from math import cos, sin


class UAV(object):
    '''
    classdocs
    '''


    def __init__(self, position, direction):
        '''
        Constructor
        
        Input:
            position:    cartesian length 2 array
            direction:    radians (+x-axis is 0)
        '''
        
        # position and direction of UAV
        self.position = position
        self.direction = direction
        
        

    def step(self, action):
        '''
        make a single step
        based on specific action
        '''
        
        if(action == vars.A_FORWARD):
            # direction remains unchanged
            self.position += vars.uavForward_distance * np.array([cos(self.direction, sin(self.direction))])
        elif(action == vars.A_LEFT):
            # update direction
            self.direction -= vars.uavTurn_angle
            self.position += vars.uavTurn_distance * np.array([cos(self.direction, sin(self.direction))])
        elif(action == vars.A_RIGHT):
            # update direction
            self.direction += vars.uavTurn_angle
            self.position += vars.uavTurn_distance * np.array([cos(self.direction, sin(self.direction))])
        else:
            raise ValueError('UAV action not understood!')
        
            
            
        
        
        
    