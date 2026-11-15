"""
geometric_metrics.py
====================
Layer-wise geometry metrics for the "Isotropy Cliffs" analysis.

Conventions
-----------
* ``embeddings`` always has shape ``(n_samples, n_layers, hidden_dim)``.
* ``labels`` is a 1-D numpy array with one label per sample (strings or ints).
* Layer ``l`` is the output of the l-th transformer block (0-based).
* A *transition* ``L -> L+1`` is the step between two neighbouring layers. Functions
  that return one value per transition (CKA, deltas) have ``n_layers - 1`` entries,
  and entry ``i`` describes the step ``i -> i+1``.

Metrics
-------
* :func:`isoscore` / :func:`isoscore_per_layer`    - uniformity of variance (Rudman et al., 2022)
* :func:`knn_purity` / :func:`knn_purity_per_layer` - local class structure
* :func:`cka_neighboring_layers`                    - linear CKA between layers l and l+1
* :func:`intrinsic_dim_per_layer`                   - TwoNN intrinsic dimension (optional, needs ``scikit-dimension``)
* :func:`find_cliff`                                - heuristic to locate the isotropy cliff

Plots: :func:`plot_layerwise`, :func:`plot_cliff_overview`, :func:`plot_tsne_pair`.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

C_ISO, C_KNN, C_CKA = "#2563eb", "#dc2626", "#7c3aed"


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------
# IsoScore: reference implementation, https://github.com/bcbi-edu/p_eickhoff_isoscore
# ----------------------------------------------------------------------------------------
# The reference functions below are unchanged and expect points of shape (n_dims, n_samples).
def pca_normalization(points):  # step 2
    points = np.transpose(points)
    n_samples, n_features = points.shape
    max_components = min(n_samples, n_features)
    pca = PCA(n_components=max_components)
    points = pca.fit_transform(points)
    return np.transpose(points)


def get_diag_of_cov(points):  # step 3
    n = np.shape(points)[0]
    cov = np.cov(points)
    return cov[np.diag_indices(n)]


def normalize_diagonal(cov_diag):  # step 4
    n = len(cov_diag)
    return (cov_diag * np.sqrt(n)) / np.linalg.norm(cov_diag)


def get_isotropy_defect(cov_diag_normalized):  # step 5
    n = len(cov_diag_normalized)
    iso_diag = np.eye(n)[np.diag_indices(n)]
    l2_norm = np.linalg.norm(cov_diag_normalized - iso_diag)
    normalization_constant = np.sqrt(2 * (n - np.sqrt(n)))
    return l2_norm / normalization_constant


def get_IsoScore(isotropy_defect, points):  # steps 6 and 7
    n = np.shape(points)[0]
    return ((n - (isotropy_defect**2) * (n - np.sqrt(n))) ** 2 - n) / (n * (n - 1))


def IsoScore(points):
    """Reference IsoScore. ``points`` has shape ``(n_dims, n_samples)``."""
    points_PCA = pca_normalization(points)
    cov_diag = get_diag_of_cov(points_PCA)
    cov_diag_normalized = normalize_diagonal(cov_diag)
    isotropy_defect = get_isotropy_defect(cov_diag_normalized)
    return get_IsoScore(isotropy_defect, points)


def isoscore(layer_embeddings: np.ndarray) -> float:
    """IsoScore of one layer, given as ``(n_samples, n_dims)``. Same as ``IsoScore(layer_embeddings.T)`` in the original scripts.

    PCA uses ``min(n_samples, n_dims)`` components. If ``n_samples < n_dims`` the variance vector is shorter than
    ``n_dims`` and the values are NOT comparable to runs with ``n_samples >= n_dims`` (a warning is shown).
    """
    n_samples, n_dims = layer_embeddings.shape
    if n_samples < n_dims:
        warnings.warn(
            f"{n_samples} samples < {n_dims} dimensions: IsoScore uses only {n_samples} PCA components, so values are "
            "not comparable to runs with n_samples >= n_dims.",
            stacklevel=2,
        )
    return float(IsoScore(layer_embeddings.T))


def isoscore_per_layer(embeddings: np.ndarray) -> np.ndarray:
    """IsoScore for every layer. Returns an array of shape ``(n_layers,)``."""
    return np.array([isoscore(embeddings[:, l, :]) for l in tqdm(range(embeddings.shape[1]), desc="IsoScore")])


# ----------------------------------------------------------------------------------------
# k-NN purity
# ----------------------------------------------------------------------------------------
def knn_purity(points: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Average fraction of the k nearest neighbours (Euclidean, excluding the point itself) sharing the point's label."""
    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(points)
    neighbours = nn.kneighbors(points, return_distance=False)[:, 1:]
    return float((labels[neighbours] == labels[:, None]).mean())


def knn_purity_per_layer(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> np.ndarray:
    n_layers = embeddings.shape[1]
    return np.array([knn_purity(embeddings[:, l, :], labels, k) for l in tqdm(range(n_layers), desc="k-NN purity")])


# ----------------------------------------------------------------------------------------
# CKA
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def cka_neighboring_layers(embeddings: np.ndarray, device=None) -> np.ndarray:
    """Linear CKA (Kornblith et al., 2019) between layer ``l`` and ``l+1`` for every l.

    Returns an array of shape ``(n_layers - 1,)``; entry ``i`` is CKA(layer i, layer i+1).
    Only two ``n_samples x n_samples`` Gram matrices are held in memory at a time.
    """
    device = device or default_device()
    n_layers = embeddings.shape[1]

    def centered_gram(l: int) -> torch.Tensor:
        X = torch.as_tensor(embeddings[:, l, :], dtype=torch.float32, device=device)
        X = X - X.mean(dim=0, keepdim=True)
        return X @ X.T  # already centred, so no extra H K H step is needed

    scores, K = [], centered_gram(0)
    for l in tqdm(range(n_layers - 1), desc="CKA"):
        K_next = centered_gram(l + 1)
        scores.append(float((K * K_next).sum() / (K.norm() * K_next.norm() + 1e-12)))
        K = K_next
    return np.array(scores)


# ----------------------------------------------------------------------------------------
# Intrinsic dimension (optional)
# ----------------------------------------------------------------------------------------
def intrinsic_dim_per_layer(embeddings: np.ndarray) -> np.ndarray:
    """TwoNN intrinsic dimension (Facco et al., 2017) per layer. Needs ``pip install scikit-dimension``."""
    from skdim.id import TwoNN

    n_layers = embeddings.shape[1]
    return np.array([float(np.squeeze(TwoNN().fit_transform(embeddings[:, l, :]))) for l in tqdm(range(n_layers), desc="TwoNN")])


# ----------------------------------------------------------------------------------------
# Locating the cliff
# ----------------------------------------------------------------------------------------
def find_cliff(iso: np.ndarray, knn: np.ndarray, cka: np.ndarray, ignore_last: int = 2) -> dict:
    """Heuristic: where does each signal change the most? Every value ``L`` means the transition ``L -> L+1``.

    * ``iso_drop``: largest drop in IsoScore
    * ``knn_jump``: largest increase in k-NN purity
    * ``cka_dip``:  lowest CKA between neighbouring layers

    The last ``ignore_last`` transitions are skipped because the final layers often change a lot for
    unrelated reasons (they prepare the next-token output). Always check the result visually.
    """
    iso, knn, cka = (np.asarray(a, dtype=float) for a in (iso, knn, cka))
    n_transitions = len(iso) - 1 - ignore_last
    return {
        "iso_drop": int(np.argmin(np.diff(iso)[:n_transitions])),
        "knn_jump": int(np.argmax(np.diff(knn)[:n_transitions])),
        "cka_dip": int(np.argmin(cka[:n_transitions])),
    }


def delta_log_iso(iso: np.ndarray, layer: int) -> float:
    """Change in log IsoScore across the transition ``layer -> layer+1`` (the paper's cliff steepness; negative = drop)."""
    return float(np.log(iso[layer + 1]) - np.log(iso[layer]))


# ----------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------
def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")  # save BEFORE show, otherwise the file can be blank
    plt.show()
    plt.close(fig)


def plot_layerwise(values, ylabel: str, title: str = "", color: str = C_ISO, save_path=None, ylim=None):
    """Simple line plot of one value per layer."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(values)), values, marker="o", ms=4, color=color)
    ax.set(xlabel="Layer", ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    _finish(fig, save_path)


def plot_cliff_overview(iso, knn, cka, cliff_layer=None, title: str = "", save_path=None):
    """IsoScore (left axis) and k-NN purity (right axis) on top, CKA of neighbouring layers below."""
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[2, 1])
    ax2 = ax1.twinx()
    ax1.plot(iso, color=C_ISO, lw=2, label="IsoScore")
    ax2.plot(knn, color=C_KNN, lw=2, label="k-NN purity")
    ax3.plot(range(len(cka)), cka, color=C_CKA, lw=2, label="CKA(l, l+1)")

    ax1.set_ylabel("IsoScore", color=C_ISO)
    ax2.set_ylabel("k-NN purity", color=C_KNN)
    ax2.set_ylim(0, 1.05)
    ax3.set_ylabel("CKA(l, l+1)")
    ax3.set_xlabel("Layer")
    ax3.set_ylim(0, 1.05)
    ax1.set_title(title)
    if cliff_layer is not None:
        for ax in (ax1, ax3):
            ax.axvspan(cliff_layer, cliff_layer + 1, color="grey", alpha=0.25, lw=0)
        ax1.text(cliff_layer + 0.5, ax1.get_ylim()[1], f"{cliff_layer}→{cliff_layer + 1}", ha="center", va="bottom", fontsize=9)
    for ax in (ax1, ax3):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, save_path)


def plot_tsne_pair(embeddings, labels, layers, perplexity: int = 30, pca_dims: int = 50, seed: int = 0, save_path=None):
    """t-SNE of the given layers side by side, coloured by label.

    Each layer is first reduced to ``pca_dims`` PCA components, which keeps t-SNE fast and barely changes the picture.
    Note that every t-SNE run is independent, so positions are not comparable across layers.
    """
    _, numeric = np.unique(np.asarray(labels), return_inverse=True)
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4.5), squeeze=False)
    for ax, l in zip(axes[0], layers):
        X = PCA(n_components=min(pca_dims, *embeddings[:, l, :].shape), random_state=seed).fit_transform(embeddings[:, l, :])
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)
        ax.scatter(xy[:, 0], xy[:, 1], c=numeric, cmap="viridis", s=6, alpha=0.8)
        ax.set_title(f"Layer {l}  (silhouette {silhouette_score(xy, numeric):.2f})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    _finish(fig, save_path)
