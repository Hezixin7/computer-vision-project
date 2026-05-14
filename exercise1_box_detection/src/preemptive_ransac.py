import numpy as np
from ransac import fit_plane_from_3_points, point_to_plane_distance

def preemptive_ransac_plane(points, M=100, B=100, threshold=0.01, gamma=0.05):
    # generate M models
    hypotheses = [fit_plane_from_3_points(points) for _ in range(M)]
    costs = np.zeros(M)

    # shuffle points
    shuffled = np.random.permutation(points)

    # process in block
    for i in range(0, len(points),B):
        blocks = shuffled[i:i+B]

        # update costs of every model
        for j, (normal, d) in enumerate(hypotheses):
            distances = point_to_plane_distance(blocks, normal, d)
            costs[j] += np.sum(np.minimum(distances, gamma))

        # sort model
        order = np.argsort(costs)
        hypotheses = [hypotheses[k] for k in order]
        costs = costs[order]

        # preserve top model
        keep = max(1, int(len(hypotheses)/2))
        hypotheses = hypotheses[:keep]
        costs = costs[:keep]

        if len(hypotheses) == 1:
            break

    best_normal, best_d = hypotheses[0]
    distances = point_to_plane_distance(points, best_normal, best_d)
    best_inlier_mask = distances < threshold

    return best_normal, best_d, best_inlier_mask

