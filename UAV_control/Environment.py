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
        estTargets is the estimated position of targets, an object for each target
        '''
        
        # create UAV and targets
        self.uav = None
        self.targets = []
        self.populateUAV()
        self.populateTargets()
        
        # limit on how large uncertainty for a single target can be
        self.uncertLimit = vars.uav_dfov
        
        
    def populateUAV(self):
        '''
        create a single UAV
        '''
        # randomly place UAV in search radius
        # random distance and angle
        radius = random.random() * vars.search_radius
        direction = random.random() * math.pi * 2
        position = np.array([(2*random.random() - 1) * radius, (2*random.random() - 1) * radius])
        
        self.uav = UAV(position=position, direction=direction)
            
    def populateTargets(self):
        '''
        create targets and populate targets list
        '''
        # create targets
        for indx in range(vars.target_num):
            # randomly place target within dfov of UAV
            radius = random.random() * vars.uav_dfov/2
            direction = random.random() * math.pi * 2
            position = np.array([(2*random.random() - 1) * radius, (2*random.random() - 1) * radius]) + self.uav.position
            
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
        targetsCont = True
        for target in self.targets:
            if(np.linalg.norm(target.position) > vars.search_radius):
                targetsCont = False
        test = (np.linalg.norm(self.uav.position) < vars.search_radius) and targetsCont 
        return test
        
        
    def step(self, action):
        '''
        This makes one step in the environment
        
        INPUT:
            action: this is the action for the UAV
                defined in global variables
            
        OUPUT:
            state: (row vector)
                UAV Position (x,y)
                Target e position (x,y)
                Target uncertainty (x)
            Reward: reward for run
                +1: detecting target
                +0.1: for exploring
                -1: if uncertainty > uncertaintyLimit for a single target
            terminal:
                bool to continue
        '''
        
        # move all targets
        for target in self.targets:
            # update predicitons
            if(np.linalg.norm(self.uav.position - target.position) < vars.uav_dfov/2):
                # add some noise to the measurement
                measured = np.array([target.position]).T + np.random.normal(vars.noiseMean, vars.noiseStd, (2,1))
                target.measure(measured)
                reward = 1
            else:
                target.predict()
                reward = 0.1
            
            # move target
            target.step()
            
            # three sigma of uncertainty
            uncertainty = 3*(target.uncertainty[0,0]**2 + target.uncertainty[1,1]**2)**0.5
            if(uncertainty > self.uncertLimit):
                cost = uncertainty
                reward = -1
            
        # move UAV
        self.uav.step(action)
        
        # get state info
        state = self.uav.position
        for target in self.targets:
            uncertainty = 3 * math.sqrt(target.uncertainty[0,0]**2 + target.uncertainty[0,0]**2)
            state = np.append(state, target.position)
            state = np.append(state, uncertainty)
            
        return state, reward, self.checkContinue()
        























        
        
        
        