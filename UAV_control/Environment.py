'''
Created on Feb 20, 2018

@author: dabrown
'''

import GlobalVariables as vars
from UAV import UAV
from Target import Target
import random
import math
import numpy as np

class Environement(object):
    '''
    This is the environment for the game world
    Basically this class will control the world of a UAV tracking a target
    '''


    def __init__(self):
        '''
        Constructor
        
        Will create the world based on the globalVariables
        '''
        
        # create UAV and targets
        self.uav = None
        self.targets = []
        self.populateUAV()
        self.populateTargets()
        
        
        
    def populateUAV(self):
        '''
        create a single UAV
        '''
        # randomly place UAV in search radius
        # random distance and angle
        radius = random.random() * vars.search_radius
        angle = random.random() * math.pi * 2
        position = np.array([math.cos(angle) * radius, math.sin(angle) * radius])
        
        # placing UAV at center to start
        # TODO: maybe change later
        position = np.array([0,0], dtype = np.float64)
        
        
        # get random direction
        direction = 2 * math.pi * random.random()
        
        self.uav = UAV(position=position, direction=direction)
            
    def populateTargets(self):
        '''
        create targets and populate targets list
        '''
        # create targets
        for indx in range(vars.target_num):
            # random distance and angle
            radius = random.random() * vars.search_radius
            angle = random.random() * math.pi * 2
            position = np.array([math.cos(angle) * radius, math.sin(angle) * radius])
            
            # putting Target at center
            # TODO: maybe change laters
            position = np.array([0,0], dtype = np.float64)
            
            # random direction
            direction = 2 * math.pi * random.random()
            
            # create targets
            self.targets.append(Target(position=position, direction=direction))
          
        
    def reset(self):
        '''
        Reset the environment 
        '''
        self.uav = None
        self.targets = []
        self.populateUAV()
        self.populateTargets()
        
    def checkContinue(self):
        '''
        Check if should continue
        
        OUTPUT:
            shoudlContinue: boolean, if UAV is out of bounds or not
        '''
        return np.linalg.norm(self.uav.position) < vars.search_radius
        
        
    def step(self, action):
        '''
        This makes one step in the environment
        
        INPUT:
            action: this is the action for the UAV
            
        OUPUT:
            UAV:     UAV object
            targets:    target object
            Cost: total uncertainty of target positions
        '''
        
        # move all targets
        # save the total uncertainty
        cost = 0
        for target in self.targets:
            target.step(self.uav)
            # get total cost
            cost += target.uncertainty
            
        # move UAV
        self.uav.step(action)
        
        # return that stuff!!
        return self.uav, self.targets, cost
        























        
        
        
        