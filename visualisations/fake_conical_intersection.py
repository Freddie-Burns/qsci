"""
Visualise fake conical intersection data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Coordinate grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Energy surfaces
E0 = 0
R = np.sqrt(X**2 + Y**2)
E_plus = E0 + R
E_minus = E0 - R

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, E_plus, cmap='viridis', alpha=0.8, label="Upper surface")
ax.plot_surface(X, Y, E_minus, cmap='plasma', alpha=0.8, label="Lower surface")

# Labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Energy')
ax.set_title('Conical Intersection')

plt.show()
