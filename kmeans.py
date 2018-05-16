import numpy as np
import tqdm
import warnings

def centroid(data):
    """Find the centroid of the given data."""
    return np.mean(data)


def sse(data):
    """Calculate the SSE of the given data."""
    u = centroid(data)
    return np.sum(np.linalg.norm(data - u, 2, 1))

def find_nearest(array, value):
    """
    Get nearest entry in an array to the given value.
    Returns: Index of that value.
    Source: https://stackoverflow.com/a/2566508/1421555
    """
    idx = (np.abs(array - value)).argmin()
    return idx



class KMeansClusterer:
    """The standard k-means clustering algorithm."""

    def __init__(self, data=None, k=2, min_gain=0.01, max_iter=100,
                 max_epoch=10, verbose=True, initial_centroids=None, invariant_centroids=None):
        """Learns from data if given."""
        if data is not None:
            self.fit(data, k, min_gain, max_iter, max_epoch, verbose, initial_centroids, invariant_centroids)
        else:
            assert 0

    def fit(self, data, k=2, min_gain=0.01, max_iter=100, max_epoch=10,
            verbose=True, initial_centroids=None, invariant_centroids=None):
        """Learns from the given data.

        Args:
            data:                   The dataset with m rows each with n features
            k:                      The number of clusters
            min_gain:               Minimum gain to keep iterating
            max_iter:               Maximum number of iterations to perform
            max_epoch:              Number of random starts, to find global optimum
            verbose:                Print diagnostic message if True
            initial_centroids:      Seeder centroids, to speed up convergence.
            invariant_centroids:    Static values for centroids.

        Returns:
            self
        """
        # Developer Reference:
        # u: Centroid Location
        # c: points in cluster
        # k: cluster amount

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if len(np.unique(data)) <= k:
            print("Values are not unique enough to be clustered")
            self.data = np.matrix(data)
            self.k = k
            self.min_gain = min_gain
            self.u = None
            self.C = [None] * k
            return self

        # Validation
        if invariant_centroids is not None:
            invariant_centroids = [invariant_centroids]
            if initial_centroids is None:
                warnings.warn("Initial Centroids initialized with invariant values.")
                initial_centroids = invariant_centroids

            for c in invariant_centroids:
                if c not in initial_centroids:
                    warnings.warn("Cannot guarantee good clustering initialization if desired centroid not included in starting seeds")

        # Pre-process
        self.data = np.matrix(data)
        self.k = k
        self.min_gain = min_gain
        self.u = None   # Default in case no solution is reached.

        if self.data.shape[0] == 1:
            # Swap dimensions if 1D
            self.data = np.swapaxes(self.data,0,1)

        # Perform multiple random init for global optimum
        min_sse = np.inf

        if initial_centroids is not None:
            if max_epoch != 1:
                warnings.warn("Max Epoch set to 1 internally as additional epoch redundant with initialized centroids.")
                max_epoch = 1

#        for epoch in tqdm.tqdm(range(max_epoch)):
        for epoch in range(max_epoch):
            print("Epoch")

            # Randomly initialize k centroids
            if initial_centroids is not None:
                print("A")
                # TODO: Allow partial creation

                if len(initial_centroids) < k:
                    # Use Random values for those values not initialized.
                    indices = np.random.choice(len(data), k - len(initial_centroids), replace=False)
                    u = np.array(self.data[indices, :])

                else:
                    u = np.array([])

                for i in initial_centroids:
                    u = np.append(u,[i])

                u = u.reshape((u.shape[0], 1))

            else:
                # Use Random values
                print("B")
                indices = np.random.choice(len(data), k, replace=False)
                u = self.data[indices, :]

            # Loop
            t = 0
            old_sse = np.inf
            while True:
                # print(">")
                t += 1

                # Cluster assignment
                C = [None] * k
                for x in self.data:
                    j = np.argmin(np.linalg.norm(x - u, 2, 1))
                    C[j] = x if C[j] is None else np.vstack((C[j], x))

                #  Centroid update
                for j in range(k):
                    # print("@", C[j])
                    if C[j] is None:
                        print(C)
                        for jj in range(k):
                            if C[jj] is None:
                                C[jj] = 0
                                u[jj] = centroid(C[jj])
                        self.u = u
                        # print("Return Early")
                        return self         # Data has been fully clustered, we cannot do any more
                    u[j] = centroid(C[j])

                """
                Substitute in the invariant centroids for the cloest predictions
                This is important so that centroids can continue to pass over each other.
                A straight retention of the index would result in a 'barrier' in the clustering
                """

                if invariant_centroids is not None:
                    idx = find_nearest(u, invariant_centroids[0])
                    u[idx] = invariant_centroids[0]

                # Loop termination condition
                if t >= max_iter:
                    # print("MAxed out")
                    break
                new_sse = np.sum([sse(C[j]) for j in range(k)])
                gain = old_sse - new_sse
                if verbose:
                    line = "Epoch {:2d} Iter {:2d}: SSE={:10.4f}, GAIN={:10.4f}"
                    print(line.format(epoch, t, new_sse, gain))
                if gain < self.min_gain or t+1 >= max_iter:
                    if new_sse < min_sse:
                        min_sse, self.C, self.u = new_sse, C, u
                    # print("Gain")
                    break
                else:
                    old_sse = new_sse

            if verbose:
                print('')  # blank line between every epoch

        return self


class BisectingKMeansClusterer:
    """Bisecting k-means clustering algorithm.

    It internally uses the standard k-means algorithm with k=2.
    """

    def __init__(self, data, max_k=10, min_gain=0.1, verbose=True):
        """Learns from data if given."""
        if data is not None:
            self.fit(data, max_k, min_gain, verbose)

    def fit(self, data, max_k=10, min_gain=0.1, verbose=True):
        """Learns from given data and options.

        Args:
            data:     The dataset with m rows each with n features
            max_k:    Maximum number of clusters
            min_gain: Minimum gain to keep iterating
            verbose:  Print diagnostic message if True

        Returns:
            self
        """

        self.kmeans = KMeansClusterer()
        self.C = [data, ]
        self.k = len(self.C)
        self.u = np.reshape(
            [centroid(self.C[i]) for i in range(self.k)], (self.k, 2))

        if verbose:
            print("k={:2d}, SSE={:10.4f}, GAIN={:>10}".format(
                self.k, sse(data), '-'))

        while True:
            # pick a cluster to bisect
            sse_list = [sse(data) for data in self.C]
            old_sse = np.sum(sse_list)
            data = self.C.pop(np.argmax(sse_list))
            # bisect it
            self.kmeans.fit(data, k=2, verbose=False)
            # add bisected clusters to our list
            self.C.append(self.kmeans.C[0])
            self.C.append(self.kmeans.C[1])
            self.k += 1
            self.u = np.reshape(
                [centroid(self.C[i]) for i in range(self.k)], (self.k, 2))
            # check sse or k
            sse_list = [sse(data) for data in self.C]
            new_sse = np.sum(sse_list)
            gain = old_sse - new_sse
            if verbose:
                print("k={:2d}, SSE={:10.4f}, GAIN={:10.4f}".format(
                    self.k, new_sse, gain))
            if gain < min_gain or self.k >= max_k:
                break

        return self
