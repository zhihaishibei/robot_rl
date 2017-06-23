#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

error = np.loadtxt("cost.txt")
print error.shape
noise = np.loadtxt("noise.txt")


distance = np.sqrt(np.square(noise[:,0]*1000)+np.square(noise[:,1]*1000))
print distance.shape
angle = noise[:,5]
print angle.shape

#plt.plot(distance, error, '.')

#plt.plot(angle, error, '.')

#distance, angle = np.meshgrid(distance, angle)

ax.scatter(distance, angle, error)
ax.view_init(elev=30., azim=-30)
plt.show()



