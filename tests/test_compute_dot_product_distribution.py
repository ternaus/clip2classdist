import numpy as np
from clip2class_dist.evaluate import compute_dot_product_distributions

def test_compute_dot_product_distributions():
    embeddings = {
        "class1": np.array([[1, 0], [0, 1]]),
        "class2": np.array([[2, 0], [0, 2]]),
    }
    dot_product_distributions = compute_dot_product_distributions(embeddings)
    
    assert np.isclose(dot_product_distributions["class1"], 0)
    assert np.isclose(dot_product_distributions["class2"], 0)