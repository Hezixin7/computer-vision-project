import numpy as np

def fit_plane_from_3_points(points):
    samples = np.random.choice(points.shape[0], size=3 , replace=False)
    p1, p2, p3 = points[samples]

    v1 = p2 - p1
    v2 = p3 - p1

    # calculate normal vector and norm
    normal = np.cross(v1, v2) # 
    norm = np.linalg.norm(normal)

    # if v1 v2 share the same line
    if  norm < 1e-8:
        raise ValueError("Failed to define a plane.")

    # unit normal
    normal = normal / norm 
    
    d = np.dot(normal, p1)

    return normal, d

    

def point_to_plane_distance(points: np.ndarray, normal: np.ndarray, d: float):
    distances = np.abs(points @ normal - d) # n·x

    return distances



def ransac_plane(points: np.ndarray, threshold: float = 0.01, max_iterations: int = 1000):
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points") # points.shape = (N, 3)

    best_normal = None
    best_d = None
    best_inlier_mask = None
    best_num_inliers = 0

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
        # plane
        inlier_mask = distances < threshold # 0,1 
        num_inliers = np.sum(inlier_mask)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_normal = normal
            best_d = d
            best_inlier_mask = inlier_mask

    if best_normal is None:
        raise RuntimeError("RANSAC failed to find a valid plane.")

    return best_normal, best_d, best_inlier_mask


"""
def calculate_box_dimensions(points: np.ndarray, normal: np.ndarray, d: float):
    # calculate distance from non-floor points to the floor plane
    distances = point_to_plane_distance(points, normal, d)

    h_min = np.percentile(distances, 1) 
    h_max = np.percentile(distances, 99)
    
    height = h_max - h_min

    # project points onto floor plane 
    projected = points - np.outer(distances, normal)

    # PCA 
    center = projected.mean(axis=0)
    X = projected - center
    
    Cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(Cov)

    # sort eigenvalues in descending order 
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    v1 = eigenvectors[:, 0] # v1 represents length
    v2 = eigenvectors[:, 1] # v2 represents width 

    proj_v1 = X @ v1
    proj_v2 = X @ v2

    length = np.percentile(proj_v1, 99) - np.percentile(proj_v1, 1)
    width  = np.percentile(proj_v2, 99) - np.percentile(proj_v2, 1)

    return length, width, height 
"""  
    

    
    
    

    


