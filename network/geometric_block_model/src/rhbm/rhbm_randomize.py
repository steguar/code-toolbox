"""
Copyright (C) 2020 Stefano Guarino, Enrico Mastrostefano, Davide Torre.
Licensed under GNU General Public License v3.0

This file is part of the RHBM (Random Hyperbolic Block Model) Library.
This library implements the Random Hyperbolic Block Model for generating synthetic networks with community structure in hyperbolic space.

RHBM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

RHBM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with RHBM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import os
import sys
import argparse
import igraph
from rhbm_lib import randomize, save_data
import logging
import time
from tqdm.auto import tqdm

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes messages through tqdm.write to avoid breaking progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to input graph', required=True)
    parser.add_argument('-o', '--output', type=str, default='output', help='path to output folder')
    parser.add_argument('-b', '--beta', type=float, default=2, help='the value of beta to use')
    parser.add_argument('-m', '--membership_attribute', type=str, default="community", help='the attribute storing the membership')
    parser.add_argument('-n', '--communities', type=int, default=None, help='number of communities, alternative to -m; if given, the communities are assumed to be equally sized and the vertices are assumed to be sorted by community')
    parser.add_argument('-f', '--fast', action='store_true', help='whether to skip the hidden degree adjustment')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--n_graphs', type=int, default=10, help='number of synthetic graphs to generate')
    parser.add_argument('--dump_p', action='store_true', help='whether to dump the probability matrix; with this option, n_graphs is ignored')
    args = parser.parse_args()
    ###############################################
    G_file = args.input
    output_folder = args.output
    beta = args.beta
    block_attribute = args.membership_attribute
    n = args.communities
    if n is not None:
        block_attribute = None
    fast = args.fast
    adjust_features = True
    if fast:
        adjust_features = False
    iters = args.n_runs
    n_graphs = args.n_graphs
    dump_p = args.dump_p
    ############################################### 
    # Print the title of the program
    print("\n=== Geometric Block Model: Randomization ===\n")
    # Print parsed arguments for verification
    logger.info(f"""Parsed arguments:
    \tInput graph: {G_file}
    \tMembership attribute: {block_attribute}
    \tNumber of communities: {n}
    \tbeta: {beta}
    \tAdjust hidden features: {adjust_features}
    \tNumber of runs: {iters}
    \tNumber of graphs per run: {n_graphs}
    \tDump probability matrix: {dump_p}
    \tOutput folder: {output_folder}
    """)
    ###############################################
    G = igraph.read(G_file)
    logger.info('Input graph correctly loaded:\n\t'+str(G.summary()).replace('\n','\n\t'))
    if block_attribute is None or not block_attribute in G.vertex_attributes():
        if n is None:
            logger.error('Parameter n not given but the indicated membership attributed does not exist. Exit.')
            sys.exit(1)
        N = G.vcount()
        block_sizes = [N//n for _ in range(n)]
        i = 0
        while sum(block_sizes)<N:
            block_sizes[i] += 1
            i += 1
            if i==n:
                i = 0
        if block_attribute is None:
            block_attribute = 'community'
        G.vs[block_attribute] = np.concat([[i]*c for i,c in enumerate(block_sizes)])
    degrees = G.degree()
    if min(degrees)==0:
        logger.warning('Input graph contains isolated vertices, these will be removed before proceeding')
        G.delete_vertices([i for i in range(len(degrees)) if degrees[i]==0])
    logger.info('Graph after pruning:\n\t'+str(G.summary()).replace('\n','\n\t'))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")

    G.write_graphml(os.path.join(output_folder,'input_graph.graphml'))
    with open(os.path.join(output_folder,'info.csv'), 'w') as fout:
        print('graph,communities,beta,runs,n_graphs', file=fout)
        print(f'input_graph.graphml,{block_attribute},{beta},{iters},{n_graphs}', file=fout)

    node_list = [[i,c] for i,c in enumerate(G.vs[block_attribute])]
    save_data(f'node_list.txt', output_folder, node_list, fmt='%d')
    logger.info(f'node list with membership dumped')
    
    start_time = time.time()
    randomize(G, beta, block_attribute, adjust_features, iters, output_folder, dump_p, n_graphs, logger=logger)
    end_time = time.time()
    logger.info(f"Total randomization time for {iters} runs: {end_time - start_time:.2f} seconds.")
        
