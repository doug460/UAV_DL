'''
Created on Feb 19, 2018

@author: DougBrownWin
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sayHi():
    print(string)
if __name__ == '__main__':
    pass

    array = np.zeros(500)
    print(array.shape)
    for indx in range(500):
        array[indx] = random.random()
    
    plt.hist(array)
    plt.show()