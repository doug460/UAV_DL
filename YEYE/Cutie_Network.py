'''
Created on Sep 6, 2018

@author: dabrown
'''

import tensorflow           as tf
import random               as rng
import numpy                as np
import matplotlib.pyplot    as plt

class Cutie(object):
    '''
    classdocs
    '''    


    def __init__(self, env):
        '''
        Constructor
        
        import the env so as to know how big everything should be (i.e. state variables size)
        '''
        
         # initial parameters
        self.LearningRate       = 1e-4      # learning Rate
        self.BatchSize          = 32        # size of each batch
        self.FutureDiscount     = 0.90      # decay rate of future values
        self.InitialEps         = 0.5       # probability of random Action
        self.FinalEps           = 0.1       # final probability of random action
        self.TotalTrain         = 30000     # Total train iterations
        self.ReplaySize         = 2000      # Size of replay memory
        self.ReplayMem          = []        # null list for memory
        self.Explore            = 1000      # frames to explore ( anneal epsilon )
        
        # define state size and layer sizes
        self.StateSize  = env.StateSize
        self.h1Size     = 30
        self.h2Size     = 30
        self.outSize    = env.ActionsSize
        
        # populate memory
        self.popMemory(env)
        
    def train_nework(self, env):
        '''
        ** Rocky Music starts playing **
        
        just train network
        '''
        
        # ------------------ all inputs ------------------------
        s       = tf.placeholder( tf.float32,    [ None, self.StateSize ],   name='s'   )   # input State
        a       = tf.placeholder( tf.int32,      [ None, env.ActionsSize ],  name='a'   )   # input Action
        
        # define the network
        h1     = self.newLayer( s,  self.StateSize,    self.h1Size,    True )
        h2     = self.newLayer( h1, self.h1Size,       self.h2Size,    True )
        out    = self.newLayer( h2, self.h2Size,       self.outSize,   True )
        
        # define Q parameters
        Q_desired  = tf.placeholder         ( "float", [ None ], name="Q_desired" )
        Q_action   = tf.reduce_sum          ( tf.multiply( out, tf.to_float(a) ), reduction_indices=1)
        cost       = tf.reduce_mean         ( tf.square( Q_desired - Q_action )) 
        train      = tf.train.AdamOptimizer ( self.LearningRate ).minimize( cost )
        
        # initialize session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        
        # annealing stuff
        epsilon     = self.InitialEps
        deltaEps    = ( self.InitialEps - self.FinalEps ) / self.TotalTrain
        
        # get initial state
        state   = env.getState()
        
        # initialize action
        action  = np.zeros((env.ActionsSize))
        
        # track the error
        error = []
        
        # run training stuff
        for iteration in range( self.TotalTrain ):
            
            # print progress
            if( iteration % int( self.TotalTrain / 100 ) == 0 ):
                print( iteration / self.TotalTrain )
            
            # zero out action
            action = np.zeros((env.ActionsSize))
            
            # scale down epsilon
            if epsilon > self.FinalEps:
                
                epsilon -= deltaEps
                
            # get Q values
            Q = out.eval(feed_dict = {s : [ state ]})
            
            # use Q or generate random
            if rng.random() <= epsilon:
                
                indx            = rng.randrange( env.ActionsSize )
                action[indx]   = 1
                
            else:
                
                indx            = np.argmax(Q)
                action[indx]    = 1
        
            # take step
            state1, reward, terminal = env.step( action )
            
            # replace random index in memory list
            indx = rng.randrange( self.ReplaySize )
            self.ReplayMem[ indx ] = [ state, action, reward, state1, terminal]
            
            # reset if terminal
            if( terminal ):
                env.reset()
            
            # get sub batch
            batch = rng.sample( self.ReplayMem, self.BatchSize )
            
            s_batch     = np.zeros(( self.BatchSize, env.StateSize   ))
            a_batch     = np.zeros(( self.BatchSize, env.ActionsSize ))
            r_batch     = np.zeros(( self.BatchSize ))
            s1_batch    = np.zeros(( self.BatchSize, env.StateSize ))
            term_batch  = [] 
            
            
            for indx in range( len( batch ) ):
                s_batch[ indx ]  =   batch[ indx ][ 0 ] 
                a_batch[ indx ]  =   batch[ indx ][ 1 ]
                r_batch[ indx ]  =   batch[ indx ][ 2 ]
                s1_batch[ indx ] =   batch[ indx ][ 3 ]
                term_batch.append  ( batch[ indx ][ 4 ] )
                
            
            # get a batch of results 
            Q_next = out.eval( feed_dict = { s : s1_batch } )
            
            # desired Q values
            Q_desired_batch = []
            
            # loop through batch to get desired Q value
            for indx in range( len( Q_next ) ):
                terminal = term_batch[ indx ]
                
                if( terminal ):
                    Q_desired_batch.append( r_batch[ indx ])
                    
                else:
                    Q_desired_batch.append( r_batch[ indx ] + self.FutureDiscount * np.max( Q_next[ indx ] ) )
            
            # run training step 
            train.run(
                        feed_dict = {
                                    Q_desired   : Q_desired_batch,
                                    s           : s_batch,
                                    a           : a_batch
                                    }
                     )
            
            error.append( cost.eval(
                                    feed_dict = {
                                                Q_desired   : Q_desired_batch,
                                                s           : s_batch,
                                                a           : a_batch
                                                }
                                  )
                         )
        
        # just a plot of the error
        plt.figure()
        plt.plot ( error )
        plt.title('The error')
        
        # plot goals reached
        plt.figure()
        plt.plot ( env.TrackGoals )
        plt.title( 'Goals and time ')
        
        
        plt.show()
        
        
    def popMemory(self, env):
        '''
        populate the replay memory
        '''
        
        # get the initial state
        s = env.getState()
        
        # fill replay memory
        for iteration in range( self.ReplaySize ):
            
            # just a bunch of random actions
            a               = np.array([ 0, 0 ])
            randIndx        = rng.randint( 0, env.ActionsSize - 1 )
            a[ randIndx ]   = 1
            s1, r, terminal     = env.step( a )
            
            # store in memory
            self.ReplayMem.append([ s, a, r, s1, terminal ])
            
            # update state
            s = s1
            
            # reset if terminal
            if( terminal ):
                env.reset()
                s = env.getState()
        
    
    def newLayer(self, input, inSize, outSize, relu):
        '''
        input is the input layer
        inSize is the size of input layer
        outSize is the Size of this layer
        Relu is boolean
        '''
        weight = self.weight_variable([ inSize, outSize ])
        bias = self.bias_variable([outSize])
        layer = tf.matmul(input,weight) + bias
        if relu:
            layer = tf.nn.relu(layer)
            
        return layer
    
    def weight_variable(self, shape):
        '''
        weights for a layer of given shape
        '''
        initial = tf.truncated_normal(shape, stddev = 0.66)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        '''
        add bias for all nodes in layer
        '''
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
        











        
        
        
        
        