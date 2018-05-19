#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import sys
import random
import numpy as np
from collections import deque
import GlobalVariables as vars
from Environment import Environement

GAME = 'oneUav' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
            # actions are [forward, left, right]
GAMMA = 0.90 # decay rate of past observations
OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 100000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_MEMORY = 25000 # number of previous transitions to remember
TOTAL_TRAIN = 1000000 # total number of training steps
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.66)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def createNetwork():  
    numIn = vars.uav_num * vars.uav_states + vars.target_num * vars.target_states
    h1Size = 50
    h2Size = 50
    outSize = ACTIONS
    # input
    input = tf.placeholder('float', [None, numIn])
    h1 = newLayer(input, numIn, h1Size, True)
    h2 = newLayer(h1, h1Size, h2Size, True)
    out = newLayer(h2, h2Size, outSize, False)
    
    return input, out

def newLayer(input, inSize, outSize, relu):
    weight = weight_variable([inSize, outSize])
    bias = bias_variable([outSize])
    layer = tf.matmul(input,weight) + bias
    if relu:
        layer = tf.nn.relu(layer)
        
    return layer


def trainNetwork(input, output, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(output, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = Environement()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    s_t, r_0, terminal = game_state.step(do_nothing)

    # saving and loading networks
    sess.run(tf.global_variables_initializer())

    # keep track of goas reached
    goals = 0

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while t < TOTAL_TRAIN:
        # choose an action epsilon greedily
        readout_t = output.eval(feed_dict={input : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        s_t1, r_t, terminal = game_state.step(a_t)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = output.eval(feed_dict = {input : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                input : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1
        
        # if terminal, restart game
        if terminal:
            game_state.reset()

        # save progress every 10000 iterations
#         if t % 10000 == 0:
#             saver.save(sess, dir + 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print my own info every 1000 iterations, basically get sample run to compare performance
        if t % 1000 == 0:
            testGame = Environement()
            do_nothing = np.zeros(ACTIONS)
            do_nothing[0] = 1
            state, reward, terminal = testGame.step(do_nothing)
            
            # save overall uncertainty
            uncert = 0
            # run for 10 seconds
            for indx in range(10 * vars.fps):
                readout_t = output.eval(feed_dict={input : [state]})[0]
                action = np.zeros([ACTIONS])   
                action_index = np.argmax(readout_t)
                action[action_index] = 1
                state, reward, terminal = testGame.step(action)
                
                # save uncertainty of state
                uncert += state[2]
                
                # if terminal break out of for loop
                if terminal:
                    break;
                
            if reward > 0:
                goals += 1
            
            # average uncertainty
            uncert = uncert / indx
            
            
                
        if t % 10000 == 0:
            print('Iteration: %7d. goals %5d of %5d' % (t, goals, 10))
            goals = 0
            
            

def playGame():
    sess = tf.InteractiveSession()
    input, output = createNetwork()
    trainNetwork(input, output, sess)

if __name__ == "__main__":
    playGame()

















