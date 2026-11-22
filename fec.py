from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of the median
        pairwise distance (e.g., 0.2 times the median) for more
        discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or 0 in X.shape:
            raise ValueError(
                "X must be a nonempty 2-D array of shape "
                "(n_samples, n_features)")
        centroid = X.mean(axis=0)
        # Single observations form their own cluster
        n_points = X.shape[0]
        if n_points == 1:
            self.w_ij = np.empty((1, 0), dtype=float)
            self.flux = np.ones(1, dtype=float)
            self.sinks = np.array([True])
            self.labels = np.array([0], dtype=int)
            return self
        # A nearby point is a stable origin for Gram distances.
        centered = X - centroid
        d = np.linalg.norm(centered, axis=1)
        shifted = X - X[np.argmin(d)]
        gram = shifted @ shifted.T
        norms_squared = np.diag(gram)
        distances_squared = (
            norms_squared[:, None] + norms_squared[None, :] - 2 * gram)
        # Cancellation can produce tiny negative squared distances.
        np.maximum(distances_squared, 0.0, out=distances_squared)
        np.fill_diagonal(distances_squared, 0.0)
        # Each unordered pair contributes exactly once to the median.
        pair_indices = np.triu_indices(n_points, k=1)
        unique_pairs = np.sqrt(distances_squared[pair_indices])
        sigma = (
            self.sigma
            if self.sigma is not None
            else 0.2 * np.median(unique_pairs))
        if sigma == 0:
            sigma = 1.0
        # Use a partial selection
        effective_k = min(self.k, n_points - 1)
        np.fill_diagonal(distances_squared, np.inf)
        neighbor_indices = np.empty((n_points, effective_k), dtype=int)
        neighbor_distances = np.empty((n_points, effective_k))
        roundoff = 64 * np.finfo(float).eps
        largest_squared_norm = norms_squared.max()
        for i in range(n_points):
            row = distances_squared[i]
            candidates = np.argpartition(
                row, effective_k - 1
            )[:effective_k]
            cutoff = row[candidates].max()
            # Candidates at cutoff that Gram split
            margin = roundoff * (
                largest_squared_norm + norms_squared[i] + cutoff)
            eligible = np.flatnonzero(row <= cutoff + margin)
            # Recheck only eligible distances without an (n, n, m)
            # temporary. A stable sort of ascending indices breaks
            # exact distance ties toward the lowest observation index.
            exact_distances = np.linalg.norm(
                X[eligible] - X[i], axis=1)
            chosen = np.argsort(
                exact_distances, kind="stable"
            )[:effective_k]
            neighbor_indices[i] = eligible[chosen]
            neighbor_distances[i] = exact_distances[chosen]
        w_ij = np.exp(-(neighbor_distances ** 2) / (sigma ** 2))
        # One observation contributes one unit of flux initially
        flux = np.ones(n_points, dtype=float)
        # Tolerance for determining downhill connections
        tau = 0.1 * sigma
        # Neighbor's centroid distance compared with its source point
        neighbor_centroid_distances = d[neighbor_indices]
        downhill_mask = neighbor_centroid_distances <= (d[:, None] + tau)
        # Initialize final outgoing flow for the T=0 case.
        edge_flux = np.zeros_like(w_ij)
        outgoing_flux = np.zeros(n_points, dtype=float)
        # Redistribute flux through the graph for T iterations.
        for iteration in range(self.T):
            if iteration == 0 or self.beta == 0:
                reinforcement = np.ones_like(flux)
            elif self.beta is None:
                reinforcement = np.log1p(flux)
            else:
                reinforcement = flux ** self.beta
            # Eligible connections using destination-node flux
            adjusted_weights = w_ij * reinforcement[neighbor_indices]
            adjusted_weights *= downhill_mask
            # Normalize each node's outgoing connections
            weight_totals = adjusted_weights.sum(axis=1, keepdims=True)
            transport_proportions = np.divide(
                adjusted_weights,
                weight_totals,
                out=np.zeros_like(adjusted_weights),
                where=weight_totals > 0)
            # Flux movement through each connection
            edge_flux = self.alpha * flux[:, None] * transport_proportions
            # Flux outgoing each observation
            outgoing_flux = edge_flux.sum(axis=1)
            # bincount accumulates contributions to each destination.
            incoming_flux = np.bincount(
                neighbor_indices.ravel(),
                weights=edge_flux.ravel(),
                minlength=n_points,)
            flux = flux - outgoing_flux + incoming_flux
        # Observations with negligible outgoing flux
        sinks = outgoing_flux < self.epsilon
        # Select closest point to centroid if no sinks found
        if not np.any(sinks):
            closest_to_centroid = np.argmin(d)
            sinks[closest_to_centroid] = True
        # Start assignments for clustering
        labels = np.full(n_points, -1, dtype=int)
        # Unique cluster label per sink
        sink_indices = np.flatnonzero(sinks)
        labels[sink_indices] = np.arange(len(sink_indices))
        # Find strongest out-connection for each observation
        strongest_connection = np.argmax(edge_flux, axis=1)
        steepest_neighbor = neighbor_indices[
            np.arange(n_points), strongest_connection]
        # Trace observations toward a sink
        for start in range(n_points):
            if labels[start] != -1:
                continue
            path = []
            visited = set()
            current = start
            while labels[current] == -1:
                # Stop if path loops to a visited observation
                if current in visited:
                    cycle_sink = path[-1]
                    labels[cycle_sink] = labels.max() + 1
                    sinks[cycle_sink] = True
                    break
                visited.add(current)
                path.append(current)
                current = int(steepest_neighbor[current])
            # Terminal cluster assigned to whole path
            if labels[current] != -1:
                cluster_label = labels[current]
            else:
                cluster_label = labels[path[-1]]
            for observation in path:
                labels[observation] = cluster_label
        # Min cluster size
        min_cluster_size = max(2, n_points // 20)
        # Count observations in clusters
        cluster_ids, cluster_sizes = np.unique(
            labels, return_counts=True)
        # ID clusters to receive smaller clusters
        large_clusters = cluster_ids[
            cluster_sizes >= min_cluster_size]
        # Threshhold size not met = use largest cluster
        if len(large_clusters) == 0:
            large_clusters = np.array([
                cluster_ids[np.argmax(cluster_sizes)]])
        # Calculate centroids of larger clusters
        large_centroids = np.array([
            X[labels == cluster_id].mean(axis=0)
            for cluster_id in large_clusters])
        # Merge small clusters into their nearest larger cluster
        for cluster_id, cluster_size in zip(cluster_ids, cluster_sizes):
            if cluster_id in large_clusters:
                continue
            small_centroid = X[labels == cluster_id].mean(axis=0)
            distances_to_large = np.linalg.norm(
                large_centroids - small_centroid,
                axis=1)
            nearest_cluster = large_clusters[np.argmin(distances_to_large)]
            labels[labels == cluster_id] = nearest_cluster

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
