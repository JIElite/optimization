import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from surface import BowlShape
from optim import GradientDescent, Adagrad


surface = BowlShape(a=1, b=9)
x = np.arange(-10., 10., 0.1)
y = np.arange(-10., 10., 0.1)
X, Y = np.meshgrid(x, y)
Z = surface.compute_func(X, Y)
fig = plt.figure(figsize=(20, 5))
contour = plt.contour(X, Y, Z, colors='gray')
plt.clabel(contour, inline=1, fontsize=10) # denotes the contour level

# initialize starting point: () on the surface
optimizer1 = GradientDescent(10, 2, surface)
optimizer2 = Adagrad(10, 2, surface)
def update(i):
    if i == 0:
        plt.legend()
    x, y, _ = optimizer1.update(lr=0.105)
    x_, y_, _ = optimizer2.update(lr=2.5)
    plt.scatter(np.array(x), np.array(y), c='b', label='gradient_descent')
    plt.scatter(np.array(x_), np.array(y_), c='r', label='adagrad')
    plt.plot(np.array(optimizer1.X), np.array(optimizer1.Y), c='blue')   # plot the line path
    plt.plot(np.array(optimizer2.X), np.array(optimizer2.Y), c='red')  # plot the line path

ani = animation.FuncAnimation(fig, update, 30, repeat=False, interval=500)
# ani.save('demo_contour.gif', writer='imagemagick', fps=60)
plt.show()