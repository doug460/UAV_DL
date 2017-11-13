'''
Created on Nov 12, 2017

@author: DougBrownWin
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math
from SoundLocation import Sound
import random

def dir_to_vector(dir):
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

def update_car_speeds(car_speeds,car_speed_max,car_dirs):
    car_accs = np.random.randint(-1,2,(num_cars))
    car_speeds += car_accs
    car_speeds[car_speeds > car_speed_max] = car_speed_max
    car_speeds[car_speeds < 0] = 0
    for n in range(0,num_cars):
        if car_speeds[n] == 0:
            dir_mod = int(np.random.choice(np.arange(3))-1)
            car_dirs[n] = (car_dirs[n]+dir_mod)%4 # 4 is number of directions
#     print(car_speeds)

def update_car_locs(car_locs):
    for n in range(0,len(car_locs)):
        car_locs[n] += dir_to_vector(car_dirs[n])*car_speeds[n]
        # boundary check
        if (np.max(car_locs[n]) > world_width) | (np.min(car_locs[n]) < 0):
            car_dirs[n]= (car_dirs[n]-2)%4 # 4 is the number of directions
            
    return car_locs

def rotate_pt(center_pt,pt,ang):
    # this rotates a point into the coordinate system w/ origin at the UAV and 'x' axis in the direction
    # of the velocity of the UAV
    r_pt = np.zeros([2])
    o_pt = pt - center_pt
    r_pt[0] = o_pt[0]*math.cos(math.radians(ang)) + o_pt[1]*math.sin(math.radians(ang))
    r_pt[1] = o_pt[0]*math.sin(math.radians(ang))*(-1) + o_pt[1]*math.cos(math.radians(ang))
    r_pt += center_pt
    return r_pt

def get_FOVs(UAV_locs,Fh):
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

def calc_single_UAV_score(UAV_loc,UAV_dir):
    score = 0
    ncls = np.copy(car_locs) # new car locs, rotated into UAV coord. frame
    for n in range(0,num_cars):
        ncls[n,:] = rotate_pt(UAV_loc,car_locs[n,:],UAV_dir)
        if (abs(ncls[n,0]-UAV_loc[0]) < UAV_Fh) & (abs(ncls[n,1]-UAV_loc[1]) < UAV_Fh) and not car_in_FOV[n]:
            score += 1 
            car_in_FOV[n] = True
    return score

def update_UAV_locs(UAV_locs):
    for n in range(0,len(UAV_locs)):
        x_inc = math.cos(math.radians(UAV_dirs[n]))
        y_inc = math.sin(math.radians(UAV_dirs[n]))
        UAV_locs[n] += (np.array([x_inc,y_inc])*UAV_speeds[n]).astype(int)
        # boundary check
        if (np.max(UAV_locs[n]) > world_width) | (np.min(UAV_locs[n]) < 0):
#             dir_mod = int(np.random.choice(np.arange(1,4)))
            UAV_dirs[n] = UAV_dirs[n]+90
            
    return UAV_locs


def plotSound(ax, simulation):
    # plot the sound waves propogating out
    
    for sound in sound_list:
        # plot circle
        x, y = sound.getCircle(simulation)
        ax.plot(x,y)
        
        # check detection
        for uav in range(num_UAVs):
            if(sound.checkDetection(uav, UAV_locs[uav], simulation)):
                # draw line
                x, y = sound.getLine(UAV_locs[uav])
                ax.plot(x,y)
            
        
    

def show_world(ax, simulation):
    ax.cla()
    ax.plot(car_locs[:,0],car_locs[:,1],'bs')
#     plt.hold(True)
    ax.plot(UAV_locs[:,0],UAV_locs[:,1],'ro')
    
    plotSound(ax, simulation)
    

    ax.set_aspect('equal')
    plt.title('Car simulation')
    plt.axis([0,world_width,0,world_width])
    plt.pause(0.01)
    plt.draw()    
    
# show_world(ax)    


if __name__ == '__main__':
    pass
    
    # world parameters
    world_width = 600
    world_height = world_width
    
    # sound rate, speed, and list to hold sound info
    sound_rate = 30
    sound_speed = 10
    sound_list = []
    
    # car parameters
    num_cars = 4
    car_speed_max = 3
    
    car_locs = np.random.randint(world_width, size=(num_cars,2))
    car_speeds = np.zeros((num_cars),dtype = 'int32')
    car_dirs = np.random.randint(4,size = (num_cars))
    
    
    # UAV parameters
    num_UAVs = 2
    UAV_locs = np.random.randint(world_width, size=(num_UAVs,2))
    UAV_speeds = np.ones((num_cars),dtype = 'int32')*4
    UAV_dirs = np.random.uniform(0,360,size = (num_UAVs))
    #print(UAV_dirs) # 0 = E, 1 = N, ...
    UAV_Fh = 100 # that's half-width of UAV Field of View
    
    plt.ion()
    fig, ax = plt.subplots()
    
    for n in range(0,250):
    #     ch = sys.stdin.read(1)
    #     ch = input()
    #     print(ch)
        update_car_speeds(car_speeds,car_speed_max,car_dirs)
        car_locs = update_car_locs(car_locs)
        UAV_locs = update_UAV_locs(UAV_locs)
    #     UAV_dirs += 10
        total_score = 0
        car_in_FOV = np.zeros((num_cars),dtype = 'int32')
        
        # create sound
        if(n%sound_rate == 0):
            location = car_locs[random.randint(0,num_cars-1),:]
            sound_list.append(Sound(n, location, sound_speed, num_UAVs))
        
        # get rid of ones that are too large
        for sound in sound_list:
            if(sound.getRadius(n) > 2 * world_width):
                sound_list.remove(sound)
            
        
        
        
        show_world(ax, n)















    