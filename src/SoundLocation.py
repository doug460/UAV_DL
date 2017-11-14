'''
Created on Nov 12, 2017

@author: DougBrownWin
'''

import numpy as np
import math

class Sound(object):
    '''
    classdocs
    
    Create a sound object t
    '''


    def __init__(self, simulation, location, speed, num_uavs):
        '''
        Constructor
        '''
        # save initial simulation, location
        # speed of sound, and if it has been detected by a specific uav
        self.simulation_initial = simulation
        self.location = np.copy(location)
        self.speed = speed
        
        # keep track of detection and phase
        self.detected_array = np.zeros(num_uavs)
        self.phase_array = np.zeros(num_uavs)
        
    
        
    def getCircle(self, simulation):
        # plot circle of sound wave
        radius = self.getRadius(simulation) 
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = self.location[0] + np.cos(t) * radius
        y = self.location[1] + np.sin(t) * radius
        
        return x, y
    
    def getRadius(self, simulation):
        # get radius for sound wave
        return self.speed * (simulation - self.simulation_initial) 
    
    def checkDetection(self, uav_indx, uav_location, simulation):
        # check if it is the first time the sound wave has been detected by a uav
        radius = self.getRadius(simulation)
        
        detected = False
    
        if(radius > np.linalg.norm(uav_location - self.location) and self.detected_array[uav_indx] == 0):
            
            self.phase_array[uav_indx] = np.linalg.norm(uav_location - self.location)/self.speed
            self.detected_array[uav_indx] = 1
            detected = True
        
        return detected
    
    def getLine(self, uav_loc):
        # get line from initial sound position to uav position
        x = np.array([self.location[0], uav_loc[0]])
        y = np.array([self.location[1], uav_loc[1]])
        return x, y
            
        
        
        
        
        
    
    
        