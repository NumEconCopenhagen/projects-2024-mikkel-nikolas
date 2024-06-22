import numpy as np
from types import SimpleNamespace

class ProductionEconomy:
    def __init__(self, A=1.0, gamma=0.5, alpha=0.3, nu=1.0, epsilon=2.0):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon

    def firm_labor_demand(self, w, pj):
        return ((self.gamma * self.A * pj) / w) ** (1 / (1 - self.gamma))

    def firm_output(self, w, pj):
        return self.A * self.firm_labor_demand(w, pj) ** self.gamma
    
    def firm_profit(self, w, pj):
        return ((1-self.gamma) / self.gamma) * w * self.firm_labor_demand(w, pj)

    def consumer_utility(self, c1, c2, ell):
        return np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * ell ** (1 + self.epsilon) / (1 + self.epsilon)

    def consumer_budget(self, p1, p2, w, tau, T, ell):
        pi1 = self.firm_profit(w, p1)
        pi2 = self.firm_profit(w, p2)
        income = w * ell + T + pi1 + pi2
        c1 = self.alpha * income / p1
        c2 = (1 - self.alpha) * income / (p2 + tau)
        return c1, c2, income

    def consumer_optimal_labor(self, p1, p2, w, tau, T):
        income = lambda ell: w * ell + T + self.firm_profit(w, p1) + self.firm_profit(w, p2)
        utility = lambda ell: np.log((self.alpha * income(ell) / p1) ** self.alpha * 
                                     ((1 - self.alpha) * income(ell) / (p2 + tau)) ** (1 - self.alpha)) - \
                              self.nu * ell ** (1 + self.epsilon) / (1 + self.epsilon)
        return max((utility(ell), ell) for ell in np.linspace(0, 10, 1000))[1]
    
