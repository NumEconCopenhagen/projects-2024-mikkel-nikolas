import numpy as np
import matplotlib.pyplot as plt

class Problem:
    def __init__(self):
        pass

    def Building_block_1(self, A, B, C, y):
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

    def Building_block_2(self, X, F, y):
        """Determine and use closest points to y for barycentric calculations."""
        indices = {
            'A': self.find_closest_point_index(X, y, lambda pt: pt[0] > y[0] and pt[1] > y[1]),
            'B': self.find_closest_point_index(X, y, lambda pt: pt[0] > y[0] and pt[1] < y[1]),
            'C': self.find_closest_point_index(X, y, lambda pt: pt[0] < y[0] and pt[1] < y[1]),
            'D': self.find_closest_point_index(X, y, lambda pt: pt[0] < y[0] and pt[1] > y[1])
        }

        if all(indices.values()):
            points = {key: X[idx] for key, idx in indices.items() if idx is not None}
            values = {key: F[idx] for key, idx in indices.items() if idx is not None}
            coords_abc = self.Building_block_1(points['A'], points['B'], points['C'], y)
            if coords_abc:
                return self.approximate_value(coords_abc, [values['A'], values['B'], values['C']])
            coords_cda = self.Building_block_1(points['C'], points['D'], points['A'], y)
            if coords_cda:
                return self.approximate_value(coords_cda, [values['C'], values['D'], values['A']])
        return None

    def Algorithm(self, r, values):
        """Approximate the function value given barycentric coordinates and vertex values."""
        if r is None:
            return None
        return r[0] * values[0] + r[1] * values[1] + r[2] * values[2]

    def draw_triangle(self, ax, points, keys, color, label):
        """Draw a triangle given vertex keys."""
        triangle = np.array([points[key] for key in keys])
        ax.plot(*np.append(triangle, [triangle[0]], axis=0).T, linestyle='-', color=color, label=label)
        
        