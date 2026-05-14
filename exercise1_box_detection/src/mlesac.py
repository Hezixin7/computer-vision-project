from ransac import fit_plane_from_3_points, point_to_plane_distance
import numpy as np

def mlesac_plane(points: np.ndarray, threshold: float = 0.01, gamma=0.05, max_iterations: int = 1000):
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points") # points.shape = (N, 3)

    best_normal = None
    best_d = None
    best_inlier_mask = None
    best_cost = np.inf

    num_points = points.shape[0]

    for _ in range(max_iterations):
        # # select 3 points randomly 
        # samples = np.random.choice(num_points, size=3 , replace=False)
        # p1, p2, p3 = points[samples]

        try:
            normal, d = fit_plane_from_3_points(points)
        except ValueError:
            continue

        distances = point_to_plane_distance(points, normal, d)
        # cost 
        costs = np.where(distances<threshold, distances, gamma)
        total_cost = np.sum(costs)

        # plane
        if total_cost < best_cost:
            best_normal = normal
            best_d = d
            best_inlier_mask = distances < threshold
            best_cost = total_cost
            

    if best_normal is None:
        raise RuntimeError("MLESAC failed to find a valid plane.")

    return best_normal, best_d, best_inlier_mask
