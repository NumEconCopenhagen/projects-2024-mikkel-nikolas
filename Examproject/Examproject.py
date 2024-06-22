import numpy as np
from types import SimpleNamespace

class ProductionEconomy:
    def __init__(self, A=1.0, gamma=0.5, alpha=0.3, nu=1.0, epsilon=2.0, kappa=0.1):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.kappa = kappa

    def firm_labor_demand(self, w, pj):
        return ((self.gamma * self.A * pj) / w) ** (1 / (1 - self.gamma))

    def firm1_labor_demand(self, w, p1):
        return ((self.gamma * self.A * p1) / w) ** (1 / (1 - self.gamma))

    def firm2_labor_demand(self, w, p2):
        return ((self.gamma * self.A * p2) / w) ** (1 / (1 - self.gamma))

    def firm_output(self, w, pj):
        return self.A * self.firm_labor_demand(w, pj) ** self.gamma

    def firm1_output_opt(self, w, p1):
        return self.A * self.firm1_labor_demand(w, p1) ** self.gamma    

    def firm2_output_opt(self, w, p2):
        return self.A * self.firm2_labor_demand(w, p2) ** self.gamma

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
        firm_output_p2 = self.firm_output(w, p2)
        return consumer_utility - kappa * firm_output_p2

class Problem:
    def __init__(self):
        pass

    def point_in_triangle(self, A, B, C, y):
        """Calculate barycentric coordinates and check if point is inside the triangle."""
        A1, A2 = A
        B1, B2 = B
        C1, C2 = C
        y1, y2 = y

        try:
            denom = (B2 - C2) * (A1 - C1) + (C1 - B1) * (A2 - C2)
            r1 = ((B2 - C2) * (y1 - C1) + (C1 - B1) * (y2 - C2)) / denom
            r2 = ((C2 - A2) * (y1 - C1) + (A1 - C1) * (y2 - C2)) / denom
            r3 = 1 - r1 - r2
            return (r1, r2, r3) if 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1 else None
        except ZeroDivisionError:
            return None  # Handles degenerate triangle case.

    def find_closest_point(self, X, y, condition):
        """Find the closest point in X to y that satisfies a condition."""
        valid_points = [pt for pt in X if condition(pt, y)]
        if not valid_points:
            return None
        distances = np.linalg.norm(np.array(valid_points) - y, axis=1)
        return valid_points[np.argmin(distances)]
    
    def find_closest_point_index(self, X, y, condition):
        """Find the closest point index in X to y that satisfies a condition."""
        valid_indices = [i for i, pt in enumerate(X) if condition(pt, y)]
        if not valid_indices:
            return None
        distances = np.linalg.norm(X[valid_indices] - y, axis=1)
        return valid_indices[np.argmin(distances)]

    def draw_triangle(self, ax, points, keys, color, label):
        """Draw a triangle given vertex keys."""
        triangle = np.array([points[key] for key in keys])
        ax.plot(*np.append(triangle, [triangle[0]], axis=0).T, linestyle='-', color=color, label=label)
    
    def approximate_value(self, r, values):
        """Approximate the function value given barycentric coordinates and vertex values."""
        if r is None:
            return None
        return r[0] * values[0] + r[1] * values[1] + r[2] * values[2]
    
    def approximate_point(self, X, F, y):
        indices = {
        'A': self.find_closest_point_index(X, y, lambda pt, y: pt[0] > y[0] and pt[1] > y[1]),
        'B': self.find_closest_point_index(X, y, lambda pt, y: pt[0] > y[0] and pt[1] < y[1]),
        'C': self.find_closest_point_index(X, y, lambda pt, y: pt[0] < y[0] and pt[1] < y[1]),
        'D': self.find_closest_point_index(X, y, lambda pt, y: pt[0] < y[0] and pt[1] > y[1])
        }

        if all(idx is not None for idx in indices.values()):
            points = {key: X[idx] for key, idx in indices.items()}
            values = {key: F[idx] for key, idx in indices.items()}

        coords_abc = self.point_in_triangle(points['A'], points['B'], points['C'], y)
        if coords_abc:
            return self.approximate_value(coords_abc, [values['A'], values['B'], values['C']])
        else:
            coords_cda = self.point_in_triangle(points['C'], points['D'], points['A'], y)
            if coords_cda:
                return self.approximate_value(coords_cda, [values['C'], values['D'], values['A']])
        return None
