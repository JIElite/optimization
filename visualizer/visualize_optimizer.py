import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


# Define the surface
class BowlShape:
    def __init__(self):
        pass

    def compute_value(self, x, y):
        return 3*x**2 + 5*y**2

    def gradient(self, x, y):
        return 6*x, 10*y

class Valley:
    def __init__(self):
        pass
    
    def compute_value(self, x, y):
        return y**2

    def gradient(self, x, y):
        return 0, 2*y

class SaddlePoint:
    def __init__(self):
        pass

    def compute_value(self, x, y):
        return -x**2 + y**2

    def gradient(self, x, y):
        return -2*x, 2*y

class GradientDescent:
    def __init__(self, x, y):
        self.X = [x]
        self.Y = [y]

    def update(self, surface, lr=0.05):
        dx, dy = surface.gradient(self.X[-1], self.Y[-1])
        x = self.X[-1] - lr*dx
        y = self.Y[-1] - lr*dy
        z = surface.compute_value(x, y)
        self.X.append(x)
        self.Y.append(y)
        return x, y, z


# Define Optimization Algorithms
class Adagrad:
    def __init__(self, x, y):
        self.X = [x]
        self.Y = [y]
        self.gradients_x = []
        self.gradients_y = []
        self.epsilon = 1e-6

    def update(self, surface, lr=1):
        dx, dy = surface.gradient(self.X[-1], self.Y[-1])
        self.gradients_x.append(dx)
        self.gradients_y.append(dy)

        G_x = sum([dx**2 for dx in self.gradients_x]) + self.epsilon
        G_y = sum([dy**2 for dy in self.gradients_y]) + self.epsilon
        x = self.X[-1] - lr*dx / (G_x)**(1/2)
        y = self.Y[-1] - lr*dy / (G_y)**(1/2)
        z = surface.compute_value(x, y)
        self.X.append(x)
        self.Y.append(y)
        return x, y, z


x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)
a, b  = np.meshgrid(x, y)
surface = SaddlePoint()
plt3d = plt.figure().gca(projection='3d')
plt3d.plot_surface(a, b, surface.compute_value(a, b), cmap=cm.coolwarm, alpha=0.5)

# initial point is (4, 10)
optimizer1 = GradientDescent(20, 100)
optimizer2 = Adagrad(20, 100)

def update(args):
    x, y, z = optimizer1.update(surface, lr=0.03)
    x_, y_, z_ = optimizer2.update(surface, lr=8)
    plt3d.scatter(np.array([x, x_]), np.array([y, y_]), np.array([z, z_]), c=['red', 'blue'])

ani = animation.FuncAnimation(plt3d.figure, update, 30, repeat=False)
# ani.save('demo.gif', writer='imagemagick', fps=60)
plt.show()