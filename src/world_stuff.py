'''
Created on Nov 13, 2017

@author: dabrown
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math
from SoundLocation import Sound
import random
import matplotlib.animation as animation

class World(object):
    '''
    classdocs
    '''


    def __init__(self):
        # world parameters
        self.world_width = 600
        self.world_height = self.world_width
        
        # sound rate, speed, and list to hold sound info
        self.sound_rate = 30
        self.sound_speed = 10
        self.sound_list = []
        
        # car parameters
        self.num_cars = 4
        self.car_speed_max = 3
        
        self.car_locs = np.random.randint(self.world_width, size=(self.num_cars,2))
        self.car_speeds = np.zeros((self.num_cars),dtype = 'int32')
        self.car_dirs = np.random.randint(4,size = (self.num_cars))
        
        
        # UAV parameters
        self.num_UAVs = 2
        self.UAV_locs = np.random.randint(self.world_width, size=(self.num_UAVs,2))
        self.UAV_speeds = np.ones((self.num_cars),dtype = 'int32')*4
        self.UAV_dirs = np.random.uniform(0,360,size = (self.num_UAVs))
        #print(UAV_dirs) # 0 = E, 1 = N, ...
        self.UAV_Fh = 100 # that's half-width of UAV Field of View
        
        #     plt.ion()
        self.fig, self.ax = plt.subplots()
        
        
        
        

    
    def dir_to_vector(self, dir):
        vec = np.zeros((2)).astype(int)
        # 0 = E, 1 = N, ...
        if dir == 0:
            vec = np.array([1,0])
        elif dir == 1:
            vec = np.array([0,1])    
        elif dir == 2:
            vec = np.array([-1,0])
        elif dir == 3:
            vec = np.array([0,-1])
        return vec
    
    def update_car_speeds(self, car_speeds,car_speed_max,car_dirs):
        car_accs = np.random.randint(-1,2,(self.num_cars))
        car_speeds += car_accs
        car_speeds[car_speeds > car_speed_max] = car_speed_max
        car_speeds[car_speeds < 0] = 0
        for n in range(0,self.num_cars):
            if car_speeds[n] == 0:
                dir_mod = int(np.random.choice(np.arange(3))-1)
                car_dirs[n] = (car_dirs[n]+dir_mod)%4 # 4 is number of directions
    #     print(car_speeds)
    
    def update_car_locs(self, car_locs):
        for n in range(0,len(car_locs)):
            car_locs[n] += self.dir_to_vector(self.car_dirs[n])*self.car_speeds[n]
            # boundary check
            if (np.max(car_locs[n]) > self.world_width) | (np.min(car_locs[n]) < 0):
                self.car_dirs[n]= (self.car_dirs[n]-2)%4 # 4 is the number of directions
                
        return car_locs
    
    def rotate_pt(self, center_pt,pt,ang):
        # this rotates a point into the coordinate system w/ origin at the UAV and 'x' axis in the direction
        # of the velocity of the UAV
        r_pt = np.zeros([2])
        o_pt = pt - center_pt
        r_pt[0] = o_pt[0]*math.cos(math.radians(ang)) + o_pt[1]*math.sin(math.radians(ang))
        r_pt[1] = o_pt[0]*math.sin(math.radians(ang))*(-1) + o_pt[1]*math.cos(math.radians(ang))
        r_pt += center_pt
        return r_pt
    
    def get_FOVs(self, UAV_locs,Fh):
        corners = np.empty([num_UAVs,4,2]) # order is UL, UR, LR, LL
        nc = np.empty([num_UAVs,4,2]) # order is UL, UR, LR, LL
        for n in range(0,num_UAVs):
            corners[n,0,:] = np.array([(UAV_locs[n,0]-Fh),(UAV_locs[n,1]+Fh)])
            corners[n,1,:] = np.array([(UAV_locs[n,0]+Fh),(UAV_locs[n,1]+Fh)])
            corners[n,2,:] = np.array([(UAV_locs[n,0]+Fh),(UAV_locs[n,1]-Fh)])
            corners[n,3,:] = np.array([(UAV_locs[n,0]-Fh),(UAV_locs[n,1]-Fh)])
            # now rotate
            ang = UAV_dirs[n]
    #         print(ang)
            for c in range(0,4):
                nc[n,c,:] = rotate_pt(UAV_locs[n,:],corners[n,c,:],ang)
    #     print(corners)
    #     print(nc)    
        return nc
    
    def calc_single_UAV_score(self, UAV_loc,UAV_dir):
        score = 0
        ncls = np.copy(car_locs) # new car locs, rotated into UAV coord. frame
        for n in range(0,num_cars):
            ncls[n,:] = rotate_pt(UAV_loc,car_locs[n,:],UAV_dir)
            if (abs(ncls[n,0]-UAV_loc[0]) < UAV_Fh) & (abs(ncls[n,1]-UAV_loc[1]) < UAV_Fh) and not car_in_FOV[n]:
                score += 1 
                car_in_FOV[n] = True
        return score
    
    def update_UAV_locs(self, UAV_locs):
        for n in range(0,len(UAV_locs)):
            x_inc = math.cos(math.radians(self.UAV_dirs[n]))
            y_inc = math.sin(math.radians(self.UAV_dirs[n]))
            UAV_locs[n] += (np.array([x_inc,y_inc])*self.UAV_speeds[n]).astype(int)
            # boundary check
            if (np.max(UAV_locs[n]) > self.world_width) | (np.min(UAV_locs[n]) < 0):
    #             dir_mod = int(np.random.choice(np.arange(1,4)))
                self.UAV_dirs[n] = self.UAV_dirs[n]+90
                
        return UAV_locs
    
    
    def plotSound(self, ax, simulation):
        # plot the sound waves propogating out
        
        for sound in self.sound_list:
            # plot circle
            x, y = sound.getCircle(simulation)
            ax.plot(x,y)
            
            # check detection
            for uav in range(self.num_UAVs):
                if(sound.checkDetection(uav, self.UAV_locs[uav], simulation)):
                    # draw line
                    x, y = sound.getLine(self.UAV_locs[uav])
                    ax.plot(x,y)
                
            
        
    
    def show_world(self, ax, simulation):
        ax.cla()
        ax.plot(self.car_locs[:,0],self.car_locs[:,1],'bs')
    #     plt.hold(True)
        ax.plot(self.UAV_locs[:,0],self.UAV_locs[:,1],'ro')
        
        self.plotSound(ax, simulation)
        
    
        ax.set_aspect('equal')
        plt.title('Car simulation')
        plt.axis([0,self.world_width,0,self.world_width])

    # show_world(ax)    
    
    def animate(self, frame):
    
        self.update_car_speeds(self.car_speeds,self.car_speed_max,self.car_dirs)
        self.car_locs = self.update_car_locs(self.car_locs)
        self.UAV_locs = self.update_UAV_locs(self.UAV_locs)
        
        total_score = 0
        car_in_FOV = np.zeros((self.num_cars),dtype = 'int32')
         
        # create sound
        if(frame%self.sound_rate == 0):
            location = self.car_locs[random.randint(0,self.num_cars-1),:]
            self.sound_list.append(Sound(frame, location, self.sound_speed, self.num_UAVs))
         
        # get rid of ones that are too large
        for sound in self.sound_list:
            if(sound.getRadius(frame) > 2 * self.world_width):
                self.sound_list.remove(sound)
             
         
         
                 
        self.show_world(self.ax, frame)
    
    
    
    
    
    
    
    
        