'''
Created on Feb 19, 2018

@author: DougBrownWin
'''

from math import sin, pi

# speed of simulation (FPS)
fps = 10
dt = 1/fps

# limiting search radius (m)
search_radius = 50

# define UAV parameters
# UAV speed (m/s)
uavSpeed = 20
# minimum turning radius of UAV (m)
uavTurn_radius = 25
# field of view radius (m)
uav_fov = 40
# number of UAVs
# the environment only creates a single UAV at a time
uav_num = 1

# define target parameters
# speed (m/s)
targetSpeed = 6
# probability every second for random direction (0->1)
targetRand_dir = 0.1
targetRand_dir = targetRand_dir / fps
# number of targets
target_num = 1


# define types of actions
A_FORWARD = 0
A_LEFT = 1
A_RIGHT = 2
ACTIONS = [A_FORWARD, A_LEFT, A_RIGHT]







####### NOT USER INPUT ###########
# UAV maximum turning angle and distance traveled during that time
# distance from https://www.quora.com/How-does-one-calculate-the-straight-line-distance-between-two-points-on-a-circle-if-the-radius-and-arc-length-are-known
# basically law of sines
uavTurn_angle = uavSpeed / (fps * uavTurn_radius)
uavTurn_distance =  uavTurn_radius * sin(uavTurn_angle) / sin((pi - uavTurn_angle)/2)
uavForward_distance = uavSpeed / fps