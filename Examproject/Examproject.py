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
    
    def firm_output_p2(self, w, p2):
        return self.A * self.firm_labor_demand(w, p2) ** self.gamma

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

    def C1_star(self, p1, p2, w, tau, T):
        ell_star = self.consumer_optimal_labor(p1, p2, w, tau, T)
        c1, _, _ = self.consumer_budget(p1, p2, w, tau, T, ell_star)
        return c1

    def C2_star(self, p1, p2, w, tau, T):
        ell_star = self.consumer_optimal_labor(p1, p2, w, tau, T)
        _, c2, _ = self.consumer_budget(p1, p2, w, tau, T, ell_star)
        return c2

    def calculate_T(self, p1, p2, w, tau):
        c2_star = self.C2_star(p1, p2, w, tau, tau * 0)  # T is initially 0
        return tau * c2_star

    def SWF_function(self, p1, p2, w, tau, kappa):
        T = self.calculate_T(p1, p2, w, tau)
        ell_opt = self.consumer_optimal_labor(p1, p2, w, tau, T)
        c1, c2, _ = self.consumer_budget(p1, p2, w, tau, T, ell_opt)
        consumer_utility = self.consumer_utility(c1, c2, ell_opt)
        firm_output_p2 = self.firm_output_p2(w, p2)
        return consumer_utility - kappa * firm_output_p2


 