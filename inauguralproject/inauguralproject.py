from types import SimpleNamespace


class InauguralprojectClass:
    def __init__(self, alpha=1/3, beta=2/3, w1A=0.8, w2A=0.3, p2=1):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A
        self.p2 = p2

    def utility_A(self, x1A, x2A):
        return x1A**self.alpha * x2A**(1-self.alpha)

    def utility_B(self, x1B, x2B):
        return x1B**self.beta * x2B**(1-self.beta)

    def set_C(self, x1A, x2A):
        return self.utility_A(x1A, x2A) >= self.utility_A(self.w1A, self.w2A) and \
            self.utility_B(1 - x1A, 1 - x2A) >= self.utility_B(1 - self.w1A, 1 - self.w2A)

    def demand_A(self, p1):
        return (self.alpha * ((p1 * self.w1A + self.p2 * self.w2A) / p1),
                 (1-self.alpha) * ((p1 * self.w1A + self.p2 * self.w2A) / self.p2))

    def demand_B(self, p1):
        return (self.beta * ((p1 * self.w1B + self.p2 * self.w2B) / p1),
                (1 - self.beta) * ((p1 * self.w1B + self.p2 * self.w2B) / self.p2))
        
    def negative_utility_A(self, p1):
        # Utility function but returns negative for optimization
        x1A, x2A = self.demand_A(p1)
        return -self.utility_A(x1A, x2A)

    def epsilons(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        eps1 = x1A - self.w1A + x1B - self.w1B
        eps2 = x2A - self.w2A + x2B - self.w2B
        return eps1, eps2
    
    def max_u_a(self,x):
        return -self.utility_A(x[0], x[1])

    def max_u_ab(self,x):
        utility_A = self.utility_A(x[0], x[1])
        utility_B = self.utility_B(1-x[0], 1-x[1])
        return -(utility_A + utility_B) 