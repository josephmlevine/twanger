#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:48:32 2019

@author: jlevine7
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

plt.close('all')
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure('5')
ax = plt.axes(xlim=(0, 1.1), ylim=(-1, 1))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0,1,50)
    y = solution[:,i]
    line.set_data(x, y)
    return line,





# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=300, interval=20, blit=True)

"""
save the animation as an mp4.  This requires ffmpeg or mencoder to be
installed.  The extra_args ensure that t he x264 codec is used, so that
the video can be embedded in html5.  You may need to adjust this for
your system: for more information, see
http://matplotlib.sourceforge.net/api/animation_api.html
"""
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])



plt.show()