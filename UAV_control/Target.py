'''
Created on Feb 19, 2018

@author: DougBrownWin
'''
from random import Random
import GlobalVariables as vars
from math import cos, sin, pi
import numpy as np

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
        
    def step(self):
        '''
        just move uav based speed and direction
        there is a probability of random direction every second
        '''
        
        # generate random direction if need be
        if(Random.random() < vars.targetRand_dir):
            self.direction = Random.random() * 2 * pi
            
        # move target
        self.position += vars.targetSpeed * np.array([cos(self.direction), sin(self.direction)])