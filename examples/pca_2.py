import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


def cov2corr(cov):
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)

    return corr


if __name__ == "__main__":
    xy = np.random.multivariate_normal([0, 10], [[0.2, 0.9], [0.9, 2]], 50)

    scaler = StandardScaler()
    xy_ = scaler.fit_transform(xy)

    corr = cov2corr(np.cov(xy_.T))
    evals, evects = np.linalg.eig(corr)

    # plot xy
    plt.scatter(xy_[:, 0], xy_[:, 1] )
    plt.show()

    # pca on xy
    pca = decomposition.PCA(2)
    xy_pca = pca.fit_transform(xy_)  # = np.dot(xy_, pca.components_)

    # plot xy_pca
    plt.scatter(xy_pca[:, 0], xy_pca[:, 1])
    plt.show()

    # check:  evects.T * corr * evects = evals
    assert np.all(np.abs(np.diag(np.dot(evects.T, np.dot(corr, evects))) - evals) < 1e-14)

    # variance explained
    explained_var_ratio_1 = evals[0]/np.sum(evals)
    explained_var_ratio = np.var(xy_pca, axis=0) / np.sum(np.var(xy_pca, axis=0))
    explained_var_ratio_orig = np.var(xy_, axis=0) / np.sum(np.var(xy_, axis=0))

    assert np.abs(pca.explained_variance_ratio_[0] - explained_var_ratio_1) < 1e-13
    assert np.sum(np.abs(pca.explained_variance_ratio_ - explained_var_ratio)) < 1e-15
