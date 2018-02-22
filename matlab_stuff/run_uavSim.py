'''
Created on Oct 27, 2017

@author: dabrown

this is an attempt to run the UAV simulation through eclipse
'''

import os
import matlab.engine

dir_uav = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/NMSU Research Dr. Sun/Programming/Matlab/MultiTargetTracking(MTT)_dbVersion'

if __name__ == '__main__':
    pass

    # basically just change directory for schtuff
    os.chdir(dir_uav)
    eng = matlab.engine.start_matlab()
    eng.param(nargout = 0)
    eng.run_sim(nargout = 0)
    