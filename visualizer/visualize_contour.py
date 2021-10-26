import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Surface:
    def __init__(self):
        self.A = 3
        self.B = 10
    
    def compute_func(self, x, y):
        return self.A*x**2 + self.B*y**2

    def compute_grad(self, x, y):
        return self.A*2*x, self.B*2*y

    def compute_hessian(self, x, y):
        pass


class GradientDescent:
    def __init__(self, x, y, surface):
        self.X = [x]
        self.Y = [y]
        self.Z = [surface.compute_func(x, y)]
        self.surface = surface

    def update(self, lr=0.05):
        dx, dy = self.surface.compute_grad(self.X[-1], self.Y[-1])
        x_ = self.X[-1] - lr*dx
        y_ = self.Y[-1] - lr*dy
        z_ = surface.compute_func(x, y)
        self.X.append(x_)
        self.Y.append(y_)
        self.Z.append(z_)
        return x_, y_, z_


surface = Surface()
x = np.arange(-10., 10., 0.1)
y = np.arange(-10., 10., 0.1)
X, Y = np.meshgrid(x, y)
Z = surface.compute_func(X, Y)
fig = plt.figure(figsize=(20, 5))
contour = plt.contour(X, Y, Z, colors='gray')
plt.clabel(contour, inline=1, fontsize=10)

optimizer = GradientDescent(10, 2, surface)
def update(args):
    x, y, _ = optimizer.update(lr=0.09)
    plt.scatter(np.array(x), np.array(y), c='r')
    plt.plot(np.array(optimizer.X), np.array(optimizer.Y), c='r')
ani = animation.FuncAnimation(fig, update, 30, repeat=False, interval=500)
plt.show()