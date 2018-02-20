'''
Created on Feb 19, 2018

@author: DougBrownWin
'''

from math import sin, pi

# speed of simulation (fps)
fps = 30

# define UAV parameters
# uav speed (m/s)
uavSpeed = 20
# minimum turning radius of uav (m)
uavTurn_radius = 30
# field of view radius (m)
uav_fov = 50

# define target parameters
# speed (m/s)
targetSpeed = 5
# probability every second for random direction (0->1)
targetRand_dir = 0.1
targetRand_dir = targetRand_dir / fps


# define types of actions
A_FORWARD = 0
A_LEFT = 1
A_RIGHT = 2






####### NOT USER INPUT ###########
# uav maximum turning angle and distance traveled during that time
# distnace from https://www.quora.com/How-does-one-calculate-the-straight-line-distance-between-two-points-on-a-circle-if-the-radius-and-arc-length-are-known
# basically law os sines
uavTurn_angle = uavSpeed / (fps * uavTurn_radius)
uavTurn_distance =  uavTurn_radius * sin(uavTurn_angle) * 2 / (sin(pi - uavTurn_angle))
uavForward_distance = uavSpeed / fps