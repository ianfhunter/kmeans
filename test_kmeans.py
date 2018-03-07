#!/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np

from kmeans import BisectingKMeansClusterer, KMeansClusterer


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description='Run the k-means clustering algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--datafile', type=str, default='data.txt',
        help='The data file containing m x n matrix')
    parser.add_argument(
        '-k', '--k', type=int, default=10,
        help='Number of clusters')
    parser.add_argument(
        '-g', '--min-gain', type=float, default=0.1,
        help='Minimum gain to keep iterating')
    parser.add_argument(
        '-t', '--max-iter', type=int, default=100,
        help='Maximum number of iterations per epoch')
    parser.add_argument(
        '-e', '--epoch', type=int, default=10,
        help='Number of random starts, for global optimum')
    parser.add_argument(
        '-v', '--verbose', default=0, action='store_true',
        help='Show verbose info')
    parser.add_argument(
        '-i', '--invariant', type=int, default=None,
        help='Single invariant centroid.')
    args = parser.parse_args()

    # prepare data
    data = np.loadtxt(args.datafile)

    # initialize clusterer
    c = KMeansClusterer(
        data, k=args.k, max_iter=args.max_iter, max_epoch=args.epoch,
        verbose=args.verbose, initial_centroids=None, invariant_centroids=args.invariant)

    # the result
    plt.figure(1)

    if len(c.C[0][1]) == 1:
        dims = 1
    else:
        dims = 0    # TODO: Implement support for variable amount of dimensions. 2D only other version supported.

    # plot the clusters in different colors
    for i in range(c.k):
        if dims == 1:
            plt.plot(c.C[i][:, 0], np.zeros((c.C[i].shape[0], 1)), 'x')
        else:
            plt.plot(c.C[i][:, 0], c.C[i][:, 1], 'x')

    # plot the centroids in black squares
    if dims == 1:
        d = c.u
        np.set_printoptions(suppress=False)
        # print("Centroids Calculated: \n", d)
        plt.plot(c.u[:, 0], np.zeros((c.u.shape[0], 1)), 'ks')
    else:
        plt.plot(c.u[:, 0], c.u[:, 1], 'ks')
    plt.show()


if __name__ == '__main__':
    main()
