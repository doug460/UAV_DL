'''
Created on Sep 6, 2018

@author: dabrown
'''

import numpy as np
import random as rng
from builtins import Exception

class Env(object):
    '''
    classdocs
    '''
    
    # Keep everything positive
    # length of evn
    Size = 100
    
    # detection radius
    det_radius = 5
    
    # UAV Stuff
    # important to have speed less than det_radius
    UAV_speed   = 3
    UAV_pos     = np.array([ 0 ])
    
    # actions
    ActionsSize = 2
    Left        = np.array([ 1,0 ])
    Right       = np.array([ 0,1 ])
    
    # target Stuff
    Tar_pos     = np.array([ 0 ])
    
    # size of the state returned with a buffer
    Buffer      = 4
    StateSize   = Size + Buffer * 2 
    
    # track to see if goal is reached
    # if positive goal is reached, if negative goal is not reached
    # magnitude is inversely related to time alive
    TrackGoals  = []
    
    
    
    

    def __init__(self):
        '''
        Constructor
        '''
        self.reset()
        
        
    def reset(self):
        
        # random Positions
        Env.UAV_pos[0] = rng.random() * Env.Size
        Env.Tar_pos[0] = rng.random() * Env.Size
        
        # make sure not starting on target
        while( np.linalg.norm( Env.UAV_pos - Env.Tar_pos ) < Env.det_radius ):
            Env.UAV_pos[0] = rng.random() * Env.Size
            Env.Tar_pos[0] = rng.random() * Env.Size
        
        # keep track of iterations
        self.count = 0
        
    def getState(self):
        state = np.zeros( Env.StateSize )
        state[ Env.Tar_pos + Env.Buffer ] = 1
        state[ Env.UAV_pos + Env.Buffer ] = 1
        return state
        
        
    def step(self, action):
        '''
        Take a single step
        '''
        self.count += 1
        
        terminal = False
        
        # move UAV
        if( np.array_equal( action, Env.Left )):
            
            Env.UAV_pos[0] -= Env.UAV_speed
            
        elif( np.array_equal( action, Env.Right )):
            
            Env.UAV_pos[0] += Env.UAV_speed
            
        else:
            raise Exception('input action not recognized!')

        # if out of bounds
        if( Env.UAV_pos[0] > Env.Size or Env.UAV_pos[0] < 0 ):
            terminal = True
        
        # get reward
        reward = 0
        
        if( np.linalg.norm( Env.UAV_pos - Env.Tar_pos ) < Env.det_radius ):
            reward      = 1
            terminal    = True
            Env.TrackGoals.append( 1/self.count )
            
        elif( terminal ):
            reward = -1
            Env.TrackGoals.append( -1/self.count )
            
        
        
        return self.getState(), reward, terminal


















        