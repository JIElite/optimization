import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from surface import SaddlePoint
from optim import GradientDescent, Adagrad


x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)
X, Y  = np.meshgrid(x, y)
surface = SaddlePoint()
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(X, Y, surface.compute_func(X, Y), cmap=cm.coolwarm, alpha=0.5)

# initialize starting point: (20, 100) on the surface
optimizer1 = GradientDescent(20, 100, surface)
optimizer2 = Adagrad(20, 100, surface)

def update(i):
    if i == 0:
        plt3d.legend()
    x, y, z = optimizer1.update(lr=0.03)
    x_, y_, z_ = optimizer2.update(lr=8)
    plt3d.scatter(np.array(x), np.array(y), np.array(z), c='red', label='gradient_descent')
    plt3d.scatter(np.array(x_), np.array(y_), np.array(z_), c='blue', label='adagrad')
    # plt3d.plot(optimizer1.X, optimizer1.Y, optimizer1.Z, c='red')  # plot the line path
    # plt3d.plot(optimizer2.X, optimizer2.Y, optimizer2.Z, c='blue') # plot the line path

ani = animation.FuncAnimation(plt3d.figure, update, 30, repeat=False)
# ani.save('demo_3D.gif', writer='imagemagick', fps=60)
plt.show()
