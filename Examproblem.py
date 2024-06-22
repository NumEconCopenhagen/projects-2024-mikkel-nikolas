import numpy as np
import matplotlib.pyplot as plt

class Problem3:
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

    def draw_triangle(self, ax, points, keys, color, label):
        """Draw a triangle given vertex keys."""
        triangle = np.array([points[key] for key in keys])
        ax.plot(*np.append(triangle, [triangle[0]], axis=0).T, linestyle='-', color=color, label=label)
    def barycentric_coordinates(A, B, C, y):
        """ Calculate the barycentric coordinates of point y with respect to triangle ABC. """
        mat = np.array([
        [B[0] - C[0], A[0] - C[0]],
        [B[1] - C[1], A[1] - C[1]]
        ])
        vec = np.array([y[0] - C[0], y[1] - C[1]])
        if np.linalg.det(mat) == 0:
            return None  # Degenerate triangle
        r1_r2 = np.linalg.solve(mat, vec)
        r1, r2 = r1_r2[0], r1_r2[1]
        r3 = 1 - r1 - r2
        return r1, r2, r3