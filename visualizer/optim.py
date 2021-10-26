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
        z_ = self.surface.compute_func(self.X[-1], self.Y[-1])
        self.X.append(x_)
        self.Y.append(y_)
        self.Z.append(z_)
        return x_, y_, z_


class Adagrad:
    def __init__(self, x, y, surface):
        self.X = [x]
        self.Y = [y]
        self.Z = [surface.compute_func(x, y)]
        self.gradients_x = []
        self.gradients_y = []
        self.surface = surface
        self.epsilon = 1e-6

    def update(self, lr=1):
        dx, dy = self.surface.compute_grad(self.X[-1], self.Y[-1])
        self.gradients_x.append(dx)
        self.gradients_y.append(dy)

        G_x = sum([dx**2 for dx in self.gradients_x]) + self.epsilon
        G_y = sum([dy**2 for dy in self.gradients_y]) + self.epsilon
        x_ = self.X[-1] - lr*dx / (G_x)**(1/2)
        y_ = self.Y[-1] - lr*dy / (G_y)**(1/2)
        z_ = self.surface.compute_func(x_, y_)
        self.X.append(x_)
        self.Y.append(y_)
        self.Z.append(z_)
        return x_, y_, z_