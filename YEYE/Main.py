'''
Created on Sep 6, 2018

@author: dabrown
'''

from Cutie_Network import Cutie
from Environment import Env

if __name__ == '__main__':
    pass

    # first Init the Env
    env     = Env()
    cutie   = Cutie     ( env )
    cutie.train_nework  ( env )
    
    
    