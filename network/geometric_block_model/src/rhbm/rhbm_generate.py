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
np.set_printoptions(suppress=True)
import os
import argparse
from rhbm_lib import generate_matrix, read_matrix, generate, save_data, get_block_sizes, get_node_list
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
    parser.add_argument("-N", type=int)
    args = parser.parse_args()

    print("ARGS:", args, flush=True)
    time.sleep(2)
    print("DONE", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--size', type=int, default=1000, help='network size')
    parser.add_argument('-k', '--avgk', type=int, default=10, help='average degree')
    parser.add_argument('-g', '--gamma', type=float, default=2.5, help='exponent of the degree distribution')
    parser.add_argument('-n', '--communities', type=int, default=5, help='number of communities')
    parser.add_argument('-s', '--sizes', type=str, default=None, help='file containing the community sizes')
    parser.add_argument('-d', '--delta', type=str, default=None, help='file containing the mixing matrix')
    parser.add_argument('-p', '--assortativity', type=float, default=0.5, help='parameter that controls the assortativity of the mixing matrix')
    parser.add_argument('-q', '--order_decay', type=float, default=1, help='parameter that controls how the connectivity decays with the distance from the diagonal in the the mixing matrix')
    parser.add_argument('-o', '--output', type=str, help='path to output folder')
    parser.add_argument('-b', '--beta', type=float, default=2, help='the value of beta to use')
    parser.add_argument('-f', '--fast', action='store_true', help='whether to skip the hidden features adjustment')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--n_graphs', type=int, default=10, help='number of synthetic graphs to generate')
    parser.add_argument('--dump_p', action='store_true', help='whether to dump the probability matrix; with this option, n_graphs is ignored')
    args = parser.parse_args()
    print(args, flush=True)

    ###############################################
    N = args.size
    avg_deg = args.avgk
    gamma = args.gamma
    n = args.communities
    bs_file = args.sizes
    block_sizes = get_block_sizes(bs_file,N,n,logger)
    delta_file = args.delta
    rho = args.assortativity
    q = args.order_decay
    if delta_file is None:
        delta = generate_matrix(n, rho, q)
    else:
        delta = read_matrix(delta_file)
        rho = q = None
    output_folder = args.output
    beta = args.beta
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
    \tGraph size: {N}
    \tAverage degree: {avg_deg}
    \tNumber of communities: {n}
    \tsizes: {bs_file}
    \tdelta: {delta_file}
    \trho: {rho}
    \tq: {q}
    \tgamma: {gamma}
    \tbeta: {beta}
    \tAdjust hidden features: {adjust_features}
    \tNumber of runs: {iters}
    \tNumber of graphs per run: {n_graphs}
    \tDump probability matrix: {dump_p}
    \tOutput folder: {output_folder}
    """)
    ###############################################
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")

    with open(os.path.join(output_folder,'info.csv'), 'w') as fout:
        print('N,beta,gamma,k,n,rho,q,runs,n_graphs', file=fout)
        print(f'{N},{beta},{gamma},{avg_deg},{n},{rho},{q},{iters},{n_graphs}', file=fout)
    
    node_list = get_node_list(block_sizes)
    save_data(f'node_list.txt', output_folder, node_list, fmt='%d')
    logger.info(f'node list with membership dumped')
    
    start_time = time.time()
    generate(N, beta, avg_deg, gamma, delta, block_sizes, adjust_features, iters, output_folder, dump_p, n_graphs, logger=logger)
    end_time = time.time()
    logger.info(f"Total time for {iters} runs: {end_time - start_time:.2f} seconds.")

