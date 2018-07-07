'''
Created on Jul 6, 2018

@author: dabrown
'''

from random import random
import numpy as np

class Env(object):
    '''
    classdocs
    '''
    
    # position of agent and target
    target = 0
    agent = 0
    
    # range over which objects can be placed
    RANGE = 20
    
    # restart range
    RANGE_LIMIT = 40
    
    # system state
    # agent pos, target pos, direction sign
    state = np.zeros((3))
    
    # terminal
    terminal = False
    
    # action constants
    LEFT = 0
    STAY = 1
    RIGHT = 2
    
    


    def __init__(self):
        '''
        Constructor
        '''
        
        
        
    def reset(self):
        '''
        reset the evn with random positions
        '''
        # random pos for agent and target 
        pos = random() * self.RANGE
        self.agent = int( pos )
        
        pos = random() * self.RANGE
        self.target = int( pos )
        
        # should program stop
        self.terminal = False
        
        
        
        
        
        
    def step(self, actionArray):
        '''
        step env one second forward
        
        INPUT
            action = { -1, 0, 1 } for { left, stay, right } respectively
        
        OUTPUT
            agent position
            target position
            sign             ( +-0 for direction to target )
            reward           ( see code )
        '''
        action = np.argmax(actionArray)
        
        # calculate reward based on best possible move
        best = np.sign( self.target - self.agent ) + 1
        dist = abs( self.target - self.agent )
        
        # if on target and stay then reward is one
        if( dist == 0 ):
            reward = 1
            self.terminal = True
            
        # if taking best action but not on target reward = 0.1
        elif( best == action ):
            reward = 0.1
            
        # if bad action
        elif( dist < self.RANGE_LIMIT ):
            reward = -0.1
            
        # if outside of range reset and penalize
        else:
            reward = -1
            self.reset()
        
        # left, right, or stay
        if( action == self.LEFT ):
            self.agent -= 1
        elif( action == self.RIGHT ):
            self.agent += 1
            
        self.state[0] = self.agent
        self.state[1] = self.target 
        self.state[2]  = best
            
        # return info on stuff
        return self.state, reward, self.terminal
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        