import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def f(abs_distance, abs_velocity, goal_velocity=0.05):
    dist_reward = (1 - abs_distance/2) ** 4
    vel_discount = np.abs(abs_velocity - goal_velocity)
    out = (dist_reward - vel_discount * 20 - 1)
    for i in range(len(out)):
        for j in range(len(out[0])):
            if abs_distance[i,j] < 0.05:
                out[i,j] = 1
            if abs_velocity[i,j] > 0.2:
                out[i,j] = -4            
    return out  

x = np.linspace(0, 2, 100)
y = np.linspace(0, 0.4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y, 0.1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('r')
plt.xlabel('d')
plt.ylabel('v')
plt.show()