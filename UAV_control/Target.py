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
        
        # keep track of target velocity
        self.prevPos = position
        self.prevPos_time = None
        self.velocity = None
        
        # need to have uncertainty and estimated position and direction
        self.uncertainty = None
        self.ePosition = None
        
        # Kalman filter for target
        # four states. position, velocity, Cartesian coordinates
        # two measurements, position
        self.dim_x = 4
        self.dim_z = 4
        self.ekf = EKF(dim_x = self.dim_x, dim_z = self.dim_z)
        
        # transition model
        F = np.zeros((self.dim_x, self.dim_x))
        F[0,0] = 1
        F[1,1] = 1
        F[2,2] = 1
        F[3,3] = 1
        F[0,2] = vars.dt
        F[1,3] = vars.dt
        self.ekf.F = F
        
        # position std
        range_std = 7;
        self.ekf.R = np.eye(self.dim_x) * range_std * range_std
        
        # covariance of process noise
        processNoise = 0.01
        self.ekf.Q = np.eye(self.dim_x)*processNoise
        
        # uncertainty covariance
        self.ekf.P = np.eye(self.dim_x) * 2 * range_std ** 2
        
        
    def step(self):
        '''
        move target
        generate random direction periodically
        keep target within bounds
        '''
        
        # generate random direction if need be
        if(random.random() < vars.targetRand_dir):
            self.direction = random.random() * 2 * pi
            
        # move target
        newPosition = self.position + vars.targetSpeed * np.array([cos(self.direction), sin(self.direction)]) / vars.fps
        
        # check bounds, if out of bounds move target towards center
        if(np.linalg.norm(newPosition) > vars.search_radius):
            # get angular position
            self.direction = asin(self.position[1]/np.linalg.norm(self.position))
            if(self.position[0] < 0):
                self.direction = pi - self.direction
            # go towards center
            self.direction += pi
            newPosition = self.position + vars.targetSpeed * np.array([cos(self.direction), sin(self.direction)]) / vars.fps
        
        # update position
        self.position = newPosition
        
    def measure(self, measuredPosition):
        '''
        update uncertainties of position and direction
        
        INPUT:
            visualized:    Numpy array, Cartesian coordinates for the measured position of the target
        '''
        
        # get column vector
        if measuredPosition is not None:
            measuredPosition.shape = (4,1)
        self.ekf.predict_update(z = measuredPosition, HJacobian = self.HJacobian, Hx = self.hx)
        
        
        self.uncertainty = self.ekf.P
        self.ePosition = self.ekf.x
        
        
    def predict(self):
        # if no meausure value, just predict...
        self.ekf.predict()
        self.uncertainty = self.ekf.P
        self.ePosition = self.ekf.x
        
    # Stuff for EKF
    # Jacobian of measurement
    def HJacobian(self, x):
        array = np.zeros((self.dim_x, self.dim_x))
        array[0,0] = 1
        array[0,2] = vars.dt
        array[1,1] = 1
        array[1,3] = vars.dt
        array[2,2] = 1
        array[3,3] = 1
        return array
        
    # measurement
    def hx(self, x):
        return np.array([x[0]+x[2] * vars.dt, x[1] +x[3]*vars.dt, x[2],x[3]])
    
    def updateVelocity(self):
        # update the velocity
        if self.prevPos_time == None:
            self.prevPos_time = vars.time
            self.velocity = np.zeros((1,2))
            self.prevPos = self.position
        else:
            self.velocity = (self.position - self.prevPos)/(vars.time - self.prevPos_time)
            self.prevPos = self.position
            self.prevPos_time = vars.time 















    
        
        
        
        
        