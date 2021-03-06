#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
from Env_LR import Env_LR as ENV
from collections import deque


ACTIONSnUM = 3 # number of valid actions
            # actions are [forward, left, right]
GAMMA = 0.90 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 10000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_MEMORY = 25000 # number of previous transitions to remember
TOTAL_TRAIN = 50000 # total number of training steps
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

h1Size = 15
h2Size = 15

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.66)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def createNetwork():  
    numIn = 2
#     h1Size = 100
    outSize = ACTIONSnUM
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
    a = tf.placeholder("float", [None, ACTIONSnUM])
    desired_Q = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(output, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(desired_Q - readout_action))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = ENV()
    
    goals = 0;

    # store the previous observations in replay memory
    D = deque()
    
    # get the first state by doing nothing
    do_nothing = np.zeros(ACTIONSnUM)
    do_nothing[2] = 1
    s_t, r_0, terminal = game_state.step(do_nothing)

    # saving and loading networks
    sess.run(tf.global_variables_initializer())


    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while t < TOTAL_TRAIN:
        # choose an action epsilon greedily
        Q = output.eval(feed_dict={input : [s_t]})[0]
        a_t = np.zeros([ACTIONSnUM])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONSnUM)
                a_t[random.randrange(ACTIONSnUM)] = 1
            else:
                action_index = np.argmax(Q)
                a_t[action_index] = 1
        else:
            a_t[2] = 1 # do nothing

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
            s_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s1_batch = [d[3] for d in minibatch]

            y_batch = []
            Q1 = output.eval(feed_dict = {input : s1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(Q1[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                desired_Q : y_batch,
                a : a_batch,
                input : s_batch}
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
        if t % 500 == 0:
            testGame = ENV()
            do_nothing = np.zeros(ACTIONSnUM)
            do_nothing[2] = 1
            state, reward, terminal = testGame.step(do_nothing)
             
            
            # run for 50 steps
            for indx in range(50):
                readout_t = output.eval(feed_dict={input : [state]})[0]
                action = np.zeros([ACTIONSnUM])   
                action_index = np.argmax(readout_t)
                action[action_index] = 1
                state, reward, terminal = testGame.step(action)
                
                # if terminal break out of for loop
                if terminal:
                    break;
                 
            # if reached goal
            if reward > 0:
                goals = goals + 1
                       
                       
        if t % 10000 == 0:
            print('Iteration: %7d. goals %4d losses %4d' % (t, goals,20-goals))
            goals = 0

def playGame():
    sess = tf.InteractiveSession()
    input, output = createNetwork()
    trainNetwork(input, output, sess)

if __name__ == "__main__":
    playGame()

















