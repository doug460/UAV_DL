'''
Created on Oct 27, 2017

@author: dabrown
 basically just proof of concept!!
'''

import matlab.engine

import os 

dir_uav = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/NMSU Research Dr. Sun/Programming/Matlab/MultiTargetTracking(MTT)_dbVersion'
dir_test = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/NMSU Research Dr. Sun/Programming/Matlab/test'


if __name__ == '__main__':
    pass
    os.chdir(dir_test)
    eng = matlab.engine.start_matlab()
    eng.f1(nargout=0)
    eng.f2(nargout=0)