# -*- coding: utf-8 -*-
import logging
import argparse
import sys
import math

import numpy as np
import networkx as nx
import Pycluster
import pygame

BLACK = (0, 0, 0)
COLORS = {0: (25, 25, 12), 1: (0, 100, 0), 2: (255, 248, 4), 3: (255, 139, 4),
          4: (91, 61, 27), 5: (141, 133, 124), 6: (255, 0, 188),
          7: (57, 5, 43)}


def kMeans(X, K, maxIters = 10, centroids=None):
    """Credits: https://gist.github.com/bistaumanga/6023692 """
    if centroids is None:
        centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return C, np.array(centroids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', default=False,
                        action='store_true', help='Use for more verbose output'
                        ' DEFAULT disabled')
    parser.add_argument('input', type=argparse.FileType('rb'),
                        help='input file', default=sys.stdin)
    parser.add_argument('-k', '--kappa', type=int, default=2,
                        help='Specify the kappa parameter')
    return parser.parse_args()


def normalized_laplacian(lattice, nodelist, node_ids):
    num_nodes = len(nodelist)
    identity_matrix = np.zeros((num_nodes, num_nodes))
    np.fill_diagonal(identity_matrix, 1.0)
    degree_matrix = np.zeros(shape=(num_nodes, num_nodes))
    inv_sqrt_deg_matrix = np.zeros(shape=(num_nodes, num_nodes))
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))

    for n in nodelist:
        node_id = node_ids[n]
        neighbors = lattice.neighbors(n)
        degree_matrix[node_id, node_id] = len(neighbors)
        inv_sqrt_deg_matrix[node_id, node_id] = 1.0/math.sqrt(len(neighbors))
        for neighbor in neighbors:
            neighbor_id = node_ids[neighbor]
            adj_matrix[node_id, neighbor_id] = 1.0

    norm_lapl = identity_matrix - np.dot(inv_sqrt_deg_matrix,
                                         np.dot(adj_matrix,
                                                inv_sqrt_deg_matrix)
                                         )

    # numpy implementation
    #norm_lapl = nx.linalg.normalized_laplacian_matrix(lattice)

    return norm_lapl


def display(grid):
    width = 20
    height = 20
    margin = 5
    size = [555, 555]
    pygame.init()
    #clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)
    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((255, 255, 255))
    screen.blit(background, (0, 0))
    pygame.display.flip()

    done = False
    while not done:
        screen.blit(background, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If user clicked close
                done = True
        screen.fill((0, 0, 0))
        # Draw the grid
        for i, row in enumerate(grid):
            for j, color in enumerate(row):
                #color = (255, 255, 255)
                pygame.draw.rect(screen,
                                 color,
                                 [(margin+width)*j+margin,
                                  (margin+height)*i+margin,
                                  width,
                                  height])
        pygame.display.flip()

    pygame.quit()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.debug('Reading %s', args.input.name)
    src_nodes = np.loadtxt(args.input, dtype=int)

    lattice_width, lattice_height = src_nodes.shape
    lattice = nx.grid_2d_graph(lattice_width, lattice_height)

    # find and remove wall
    wall_nodes = map(lambda e: tuple(e),
                     np.transpose(np.nonzero(src_nodes)))
    lattice.remove_nodes_from(wall_nodes)
    assert len(lattice.nodes()) == (lattice_width * lattice_height - len(wall_nodes))

    nodelist = list(lattice.nodes())
    node_ids = {n: i for i, n in enumerate(nodelist)}
    assert len(nodelist) == len(node_ids)

    # compute normalized laplacian
    norm_lapl = normalized_laplacian(lattice, nodelist, node_ids)
    
    # compute eigenvalues and eigenvectors
    eigen_val, eigen_vec = np.linalg.eig(norm_lapl)
    # kmeans
    labels, _, _ = Pycluster.kcluster(eigen_vec[:, :args.kappa+1],
                                      args.kappa,
                                      dist='e', npass=100, initialid=None)
    # assign colors
    colors = [COLORS[i] for i in labels]
    assert len(colors) == len(labels)
    # compute grid lattice_height x lattice_width containing colors
    grid = []
    colored, non_colored = 0, 0
    its = 0
    for i in xrange(lattice_height):
        grid.append([])
        for j in xrange(lattice_width):
            node_id = node_ids.get((i, j))
            color = colors[node_id] if node_id is not None else BLACK
            grid[i].append(color)
            if color == BLACK:
                non_colored += 1
            else:
                colored += 1
    assert non_colored == len(wall_nodes)
    display(grid)

    


if __name__ == '__main__':

    main()

