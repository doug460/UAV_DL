'''
Created on Nov 12, 2017
@author: DougBrownWin
'''


from world_stuff import World
import matplotlib.animation as animation
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass
    
    world = World()
    
    
    ani = animation.FuncAnimation(world.fig, world.animate, interval = 1, frames =250)
    
    buf = "/home/dabrown/Downloads/" + "animation" + '.mp4'
        
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps = 15, metadata=dict(artist='Me'), bitrate=1800)
    
    print('Saving Animation...')
    ani.save(buf, writer = writer)
    print('Save Complete!')
    



