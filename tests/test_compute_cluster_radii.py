import numpy as np
from clip2class_dist.evaluate import compute_cluster_radii

def test_compute_cluster_radii():
    cluster_centers = {
        "class1": np.array([1, 1]),
        "class2": np.array([2, 2]),
    }
    embeddings = {
        "class1": np.array([[1, 1], [2, 2]]),
        "class2": np.array([[2, 2], [3, 3]]),
    }
    cluster_radii = compute_cluster_radii(cluster_centers, embeddings)
    
    assert np.isclose(cluster_radii["class1"], np.sqrt(2))
    assert np.isclose(cluster_radii["class2"], np.sqrt(2))
