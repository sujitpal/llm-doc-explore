# code adapted from:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
#
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from umap import UMAP

from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM

DATA_DIR = "../data"
CHUNKS_DIR = os.path.join(DATA_DIR, "llamaindex-chunks")
CLUSTER_PREDS_FP = os.path.join(DATA_DIR, "gmm_cluster_preds.tsv")
RANDOM_STATE = 0


def extract_embeddings_from_chunk_files(chunk_dir: str) -> np.ndarray:
    embeddings, chunk_fps = [], []
    for chunk_fn in os.listdir(chunk_dir):
        chunk_fp = os.path.join(chunk_dir, chunk_fn)
        with open(chunk_fp, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        if "rel_CHILD" in chunk["metadata"].keys():
            # summary from semantic chunking, skip
            continue
        embeddings.append(np.array(chunk["metadata"]["embedding"]))
        chunk_fps.append(chunk_fp)
    X = np.array(embeddings)
    return X, chunk_fps


def plot_umap(X_red: np.ndarray):
    plt.scatter(X_red[:, 0], X_red[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of Sentence Transformer embeddings')
    _ = plt.show()


def compute_best_gmm_components(X_red: np.ndarray,
                                range: np.arange):
    n_components = range
    aic_values, bic_values = [], []
    for n_component in n_components:
        gmm = GMM(n_components=n_component,
                  covariance_type='full',
                  random_state=RANDOM_STATE)
        gmm.fit(X_red)
        aic_values.append(gmm.aic(X_red))
        bic_values.append(gmm.bic(X_red))

    plt.plot(n_components, bic_values, label='BIC')
    plt.plot(n_components, aic_values, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    _ = plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm: GMM,
             X: np.ndarray,
             n_components: int,
             label: bool = True,
             ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    ax.set_xlabel('UMAP Component #1')
    ax.set_ylabel('UMAP Component #2')
    ax.set_title("GMM clusters ({:d} components)".format(n_components))
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    _ = plt.show()


if __name__ == "__main__":
    X, chunk_fps = extract_embeddings_from_chunk_files(CHUNKS_DIR)
    print("X.shape:", X.shape)

    # scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reduce the dimensionality
    reducer = UMAP(n_components=2, random_state=RANDOM_STATE)
    X_red = reducer.fit_transform(X_scaled)
    print("X_red.shape:", X_red.shape)

    # # plot the UMAP projection
    # plot_umap(X_red)

    # find a good number of clusters
    # compute_best_gmm_components(X_red, range=np.arange(1, 101))

    # n_components = 10  # corresponding to BIC trough
    # n_components = 12  # corresponding to chapters
    n_components = 90
    gmm = GMM(n_components=n_components,
              covariance_type='full',
              random_state=RANDOM_STATE)
    # plot_gmm(gmm, X_red, n_components)

    # write out cluster predictions from GMM
    # :NOTE: gmm.predict_proba returns a (357, 90) array of floats which
    # may be used to compute soft cluster assignments. This seems useful
    # when n_components are low but at 90, cluster assignments seems to be
    # reasonaly clear cut.
    with open(CLUSTER_PREDS_FP, "w", encoding="utf-8") as fout:
        pred_clusters = gmm.predict(X_red)
        assert len(pred_clusters) == len(chunk_fps)
        for pred, chunk_fp in zip(pred_clusters, chunk_fps):
            chunk_fn = os.path.basename(chunk_fp)
            fout.write("{:s}\t{:d}\n".format(chunk_fn, pred))
