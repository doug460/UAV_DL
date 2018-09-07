'''
Created on May 16, 2018

@author: DougBrownWin

This is very simplistic environment to test q network
Going to pair with QTest

Basically move left,right only
move to goal
within bound stay at goal 
too far from goal is penalty
time limit determines terminal

states are distance to goal
and 0,1 for if goal is to left,right
'''

import numpy as np
import random



class Env_LR(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
        # initialize goal and agent
        self.reset()
        # actions for agent
        self.LEFT = np.array([1,0,0])
        self.RIGHT = np.array([0,1,0])
        self.STAY = np.array([0,0,1])
        
        # have time limit
        self.timeLimit = 100
        
        # goal range and penalty range
        self.goalRadius = 3
        self.penaltyRadius = 30
        
    def reset(self):
        '''
        This sets the agent an initial position and goal randomly around agent
        '''
        
        # create agent
        offset = 50
        self.agent = offset
        
        # create goal over range with an inner limit radius
        radius = 20
        closest = 0
        goal = (random.random()*2 - 1) * (radius-closest)
        self.goal = goal + closest * goal/abs(goal) + offset
        
        # keep track of time
        self.time = 0


    def step(self, action):
        '''
        move the agent in a direction
        get the reward for that action
        check terminal
        
        INPUT: 
            action: array for agent [left, stay, right]
            
        OUTPUT: 
            state:    array [agent pos, goal pos]
            reward:    1 if within goal range, -1 if too far, 0 else
            terminal:    once time limit is reached
        '''
        
        terminal = False
        
        # update time
        self.time += 1
        
        # move agent
        if(np.array_equal(action, self.RIGHT)):
            self.agent += 1
            
        elif(np.array_equal(action, self.LEFT)):
            self.agent -= 1
            
        # else assuming action is stay
        
        
        # get reward
        # if within goal 
        if(abs(self.agent - self.goal) < self.goalRadius):
            reward = 1
            terminal = True
        # if too far from goal
        elif(abs(self.agent - self.goal) > self.penaltyRadius):
            reward = -1
            terminal = True
        else:
            reward = 0


        if(self.time > self.timeLimit):
            terminal = True
            
        # get state
        state = np.array([self.agent, self.goal])
        
        # return schtuff
        return state, reward, terminal
            

        



























