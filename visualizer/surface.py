class BowlShape:
    def __init__(self, a=1, b=1):
        assert a > 0 and b > 0
        self.a = a
        self.b = b

    def compute_func(self, x, y):
        return self.a*x**2 + self.b*y**2

    def compute_grad(self, x, y):
        return self.a*2*x, self.b*2*y


class Valley:
    def __init__(self):
        pass
    
    def compute_func(self, x, y):
        return y**2

    def compute_grad(self, x, y):
        return 0, 2*y


class SaddlePoint:
    def __init__(self, a=-1, b=1):
        assert a*b < 0
        self.a = a
        self.b = b

    def compute_func(self, x, y):
        return self.a*x**2 + self.b*y**2

    def compute_grad(self, x, y):
        return self.a*2*x, self.b*2*y