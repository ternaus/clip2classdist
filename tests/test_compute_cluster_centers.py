import numpy as np
import pytest
from clip2class_dist.evaluate import compute_cluster_centers


def test_compute_cluster_centers():
    embeddings = {
        "class1": np.array([[1, 1], [1, 1]]),
        "class2": np.array([[2, 2], [2, 2]]),
    }
    cluster_centers = compute_cluster_centers(embeddings)
    
    assert np.allclose(cluster_centers["class1"], [1, 1])
    assert np.allclose(cluster_centers["class2"], [2, 2])
