import numpy as np

def calc_mahalanobis(x, y, n_neighbors):
    from sklearn.neighbors import DistanceMetric, NearestNeighbors
    DistanceMetric.get_metric('mahalanobis', V=np.cov(x))

    nn = NearestNeighbors(n_neighbors=n_neighbors,
                          algorithm='brute',
                          metric='mahalanobis',
                          metric_params={'V': np.cov(x)})
    return nn.fit(x).kneighbors(y)

def calc_distance(x, y):
    return np.sum((x - y)**2, 1)

def modified_calc_distance(x, y, bonus_coef=-0.5, malus_coef=0.5):
    A = (x - y)**2
    A[:, -1] = np.zeros(A.shape[0])
    dist_init = np.sum(A, axis=1)
    modified_last_col = np.zeros(A.shape[0])

    for i in len(range(A.shape[0])):
        if x[i, -1] == -1:
            pass
        elif x[i, -1] != y[i, -1]:
            modified_last_col[i] = malus_coef * dist_init[i]
        elif x[i, -1] == y[i, -1]:
            modified_last_col[i] = bonus_coef * dist_init[i]
    
    A[:, -1] = modified_last_col


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    x, y = make_classification()
    print(calc_mahalanobis(x, x[0], 2))
