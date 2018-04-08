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
        dim_x = 4
        self.ekf = EKF(dim_x = dim_x, dim_z = dim_z)
        
        # transition model
        self.ekf.F = np.array([[1, 0, vars.dt, 0],
                              [0, 1, 0, vars.dt],
                              [ 0, 0, 1, 0], 
                              [0, 0, 0, 1]])
        
        # noise range
        range_std = 1;
        self.ekf.R = np.eye(dim_x) * range_std * range_std
        
        # covariance of process noise
        processNoise = 0.1
        self.ekf.Q = np.eye(dim_x)*processNoise
        
        # uncertainty covariance
        self.ekf.P = np.eye(dim_x)
        
        
        
        
        
        
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
            visualized:    Numpy array, Cartesian coordinates for the measured position of the target
        '''
        
        #TODO: do stuff here that accounts for detection, EKF...
        self.ekf.predict_update(z = measuredPosition, HJacobian = Hjacobian, Hx = hx)
        
        
        self.uncertainty = 10
        self.ePosition = self.position
        
    # Stuff for EKF
    # Jacobian of measurement
    def HJacobian(x):
        array = np.zeros((4,4))
        array[0,0] = 1
        array[1,1] = 1
        return array
    
    # measurement
    def hx(x):
        return np.array([x[0], x[1]])
    
        
        
        
        
        