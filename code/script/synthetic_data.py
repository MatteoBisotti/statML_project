import numpy as np

# generate synthetic data points for each cluster
# dataset is generated synthetically with a fixed random seed for reproducibility
def generate_blobs(centers, cluster_sizes, std, n_features, random_state):
    rng = np.random.default_rng(random_state)

    X = []
    y = []

    n_clusters = len(centers)

    X = []
    y = []

    for k in range(n_clusters):
        center = centers[k]
        size = cluster_sizes[k]

        points = rng.normal(size=(size, n_features)) * std + np.array(center)
        labels = np.full(size, k)

        X.append(points)
        y.append(labels)

    X = np.vstack(X) 
    y = np.hstack(y) 

    return X, y