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
        self.uncertLimit = vars.uav_dfov/2
        self.uncertLimit_termial = vars.uav_dfov*2
        
        
    def populateUAV(self):
        '''
        create a single UAV
        '''
        # randomly place UAV in search radius reduced by 50%
        # random distance and angle
#         radius = random.random() * vars.search_radius * 0.5
        direction = random.random() * math.pi * 2
#         position = np.array([(2*random.random() - 1) * radius, (2*random.random() - 1) * radius])
        position = np.array([0.0,0.0])
        
        self.uav = UAV(position=position, direction=direction)
            
    def populateTargets(self):
        '''
        create targets and populate targets list
        '''
        # create targets
        for indx in range(vars.target_num):
            # randomly place target outside of UAV FOV, so this radius is in addition to d_fov
            radius = random.random() * vars.uav_dfov/2
            # get random angle direction to place target
            angle = random.random()*2*math.pi
            direction = random.random() * math.pi * 2
            position = np.array([(radius+vars.uav_dfov/2)*math.cos(angle), 
                                 (radius+vars.uav_dfov/2)*math.sin(angle)]) + self.uav.position
            
            # create target
            target = Target(position=position, direction=direction)
            
            # add measurement of initial location
#             measured = np.array([target.position]).T
#             target.measure(measured)
            
            
            # create targets
            self.targets.append(target)
            
    def getUncert(self,target):
        # INPUT:
        #     target
        # OUTPUT:
        #     uncertainty
        
        # get Pythagorean uncertainty of position for variance
        # square root of variance and then go to 97 % confidence (3 * sigma)
        uncertainty = 3 * math.sqrt(math.sqrt(target.uncertainty[0,0]**2 + target.uncertainty[0,0]**2))
        
        return uncertainty
            
            
        
    def reset(self):
        '''
        Reset the environment 
        '''
        self.uav = None
        self.targets = []
        self.populateUAV()
        self.populateTargets()
        
    def checkTerminal(self):
        '''
        Check if should continue
        
        OUTPUT:
            shoudlContinue: boolean, if UAV is out of bounds or not
        '''
        terminal = False
        for target in self.targets:
            # if targets are outside
            if(np.linalg.norm(target.position) > vars.search_radius):
                terminal = True
            
            # if uncertainty is too large
            # pathagorean and square root of variance
            uncertainty = self.getUncert(target)
            if(uncertainty > self.uncertLimit_termial):
                terminal = True
                
        # if uav is within search radius
        test = (np.linalg.norm(self.uav.position) > vars.search_radius) or terminal 
        
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
            
#             # move target
            target.step()
            
            # three sigma of uncertainty
            # so Pathagorean and square root of variance 
            uncertainty = self.getUncert(target)
            if(uncertainty > self.uncertLimit):
                cost = uncertainty
                reward = -1
            
        # move UAV
        self.uav.step(action)
        
        # get state info
        # UAV position, direction
        # target position, direction relative to UAV, uncertainty
        uavPos = self.uav.position
        uavDir = self.uav.direction
#         state = uavPos
#         state = np.append(state, uavDir)
        state = None
        
        totalUncertainty = 0
        for target in self.targets:
            uncertainty = self.getUncert(target)
#             state = np.append(state, target.position)
            totalUncertainty += uncertainty
            
            # get vector than angle of target relative to uav position and direction
            tPos = target.ePosition
            tPos = np.array([tPos[0,0],tPos[1,0]])
            vect = tPos - uavPos
            dist = np.linalg.norm(vect)
            angle = np.angle(vect[0] + vect[1]*1j)
            angle = angle - uavDir
            while angle > math.pi:
                angle -= 2*math.pi
            while angle < -math.pi:
                angle += 2*math.pi
            
            # append distance angle and uncertainty to state
            if state is None:
                state = np.array([dist])
            else:
                state = np.append(state,dist)
            state = np.append(state,angle)
            state = np.append(state, uncertainty)
            
        # just do uncertainty for reward
        reward = -totalUncertainty+10
        reward = reward/100
        
        return state, reward, self.checkTerminal()
        























        
        
        
        