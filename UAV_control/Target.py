'''
Created on Feb 19, 2018

@author: DougBrownWin
'''
import random
import GlobalVariables as vars
from math import cos, sin, pi, asin
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

class Target(object):
    '''
    classdocs
    '''


    def __init__(self, position, direction):
        '''
        Constructor
        
        INPUT:
            position:     length 2 array cartesian
            direction:    radians
        '''
        
        self.position = position
        self.direction = direction 
        
        # need to have uncertainty and estimated position and direction
        self.uncertainty = 0
        self.ePosition = position
        
        # Kalman filter for target
        # four states. position, velocity, Cartesian coordinates
        # two measurements, position
        self.ekf = EKF(dim_x = 4, dim_z = 2)
        
    def step(self, uav):
        '''
        just move uav based speed and direction
        there is a probability of random direction every second
        
        update uncertainty based on if it is visualized by UAV
        '''
        
        # generate random direction if need be
        if(random.random() < vars.targetRand_dir):
            self.direction = random.random() * 2 * pi
            
        # move target
        newPosition = self.position + vars.targetSpeed * np.array([cos(self.direction), sin(self.direction)]) / vars.fps
        
        # check bounds, if out of bounds move target towards center
        if(np.linalg.norm(newPosition) > vars.search_radius):
            self.direction = asin(self.position[1]/np.linalg.norm(self.position))
            newPosition = self.position + vars.targetSpeed * np.array([cos(self.direction), sin(self.direction)]) / vars/fps
        
        # update position
        self.position = newPosition
        
    def updateEstimates(self, measuredPosition):
        '''
        update uncertainties of position and direction
        
        INPUT:
            visualized:    Cartesian coordinates for the measured position of the target
        '''
        
        #TODO: do stuff here that accounts for detection, EKF...
        self.ekf.predict_update(z = measuredPosition, )
        
        
        self.uncertainty = 10
        self.ePosition = self.position
        
        
        
        