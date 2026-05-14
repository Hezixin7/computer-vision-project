from ransac import point_to_plane_distance
import numpy as np

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
