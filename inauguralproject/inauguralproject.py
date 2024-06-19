from types import SimpleNamespace
import numpy as np
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

    def pareto_C(self, x1A, x2A):
        return self.utility_A(x1A, x2A) >= self.utility_A(self.w1A, self.w2A) and \
            self.utility_B(1 - x1A, 1 - x2A) >= self.utility_B(1 - self.w1A, 1 - self.w2A)

    def demand_A(self, p1):
        return (self.alpha * ((p1 * self.w1A + self.p2 * self.w2A) / p1),
                 (1-self.alpha) * ((p1 * self.w1A + self.p2 * self.w2A) / self.p2))

    def demand_B(self, p1):
        return (self.beta * ((p1 * self.w1B + self.p2 * self.w2B) / p1),
                (1 - self.beta) * ((p1 * self.w1B + self.p2 * self.w2B) / self.p2))
        
    def neg_utility_A(self,p1):
        x1B, x2B = self.demand_B(p1)
        x1B = np.clip(x1B, 0, 1)
        x2B = np.clip(x2B, 0, 1)
        x1A = 1 - x1B
        x2A = 1 - x2B
        return -self.utility_A(x1A, x2A)

    def epsilons(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        eps1 = x1A - self.w1A + x1B - self.w1B
        eps2 = x2A - self.w2A + x2B - self.w2B
        return eps1, eps2
    
    def max_u_a(self, x):
        return -self.utility_A(x[0], x[1])

    def max_u_ab(self, x):
        utility_A = self.utility_A(x[0], x[1])
        utility_B = self.utility_B(1-x[0], 1-x[1])
        return -(utility_A + utility_B) 
    
    def find_equilibrium(self, w1A, w2A):
        p1_range = np.linspace(0.01, 3, 100)
        market_clearing = []
        for p1 in p1_range:
            self.w1A = w1A
            self.w2A = w2A
            self.w1B = 1 - w1A
            self.w2B = 1 - w2A
            x1A, x2A = self.demand_A(p1)
            x1B, x2B = self.demand_B(p1)
            epsilon1 = abs((x1A + x1B) - 1)
            epsilon2 = abs((x2A + x2B) - 1)
            market_clearing.append(epsilon1 + epsilon2)
        min_index = np.argmin(market_clearing)
        p1_eq = p1_range[min_index]
        x1A, x2A = self.demand_A(p1_eq)
        x1B, x2B = self.demand_B(p1_eq)
        return x1A, x2A, x1B, x2B, p1_eq