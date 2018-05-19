'''
Created on Feb 19, 2018

@author: DougBrownWin
'''

import numpy as np
import GlobalVariables as vars
from math import cos, sin, pi


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
        
    def angleBound(self, angle):
        while(angle > pi):
            angle -= 2*pi
        while(angle < -pi):
            angle += 2*pi
        
        return angle
        

    def step(self, action):
        '''
        make a single step
        based on specific action
        '''
        
        if(action[0] == 1):
            # direction remains unchanged
            self.position += vars.uavForward_distance * np.array([cos(self.direction), sin(self.direction)])
        elif(action[1] == 1):
            # update direction
            self.direction += vars.uavTurn_angle   
            self.direction = self.angleBound(self.direction)
            self.position = self.position + vars.uavTurn_distance * np.array([cos(self.direction), sin(self.direction)])
        elif(action[2] == 1):
            # update direction
            self.direction -= vars.uavTurn_angle
            self.direction = self.angleBound(self.direction)
            self.position = self.position + vars.uavTurn_distance * np.array([cos(self.direction), sin(self.direction)])
        else:
            print('Tried Action %d\n' % (action))
            raise ValueError('UAV action not available!!')
        
            
            
        
        
        
    