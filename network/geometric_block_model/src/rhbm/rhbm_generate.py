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
import sys
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

def run_rhbm_generate(
    *,
    size: int = 1000,
    avgk: int = 10,
    gamma: float = 2.5,
    communities: int = 5,
    sizes: str | None = None,
    delta: str | None = None,
    assortativity: float = 0.5,
    order_decay: float = 1.0,
    output: str = "rhbm_output",
    beta: float = 2.0,
    fast: bool = False,
    n_runs: int = 1,
    n_graphs: int = 10,
    dump_p: bool = False,
    log_to_stdout: bool = True,
):
    """
    Programmatic interface for RHBM generation.

    This function replicates exactly the CLI logic in this file, but without argparse,
    so it can be imported and called directly (e.g., from Streamlit) without subprocess.

    Parameters match CLI flags:
    - size (-N), avgk (-k), gamma (-g), communities (-n)
    - sizes (-s): path to community sizes file (optional)
    - delta (-d): path to mixing matrix file (optional)
      If delta is None -> delta matrix is generated via generate_matrix(n, rho, q)
    - assortativity (-p) and order_decay (-q) are used only when delta is None
    - output (-o): output folder path (must be writable)
    - beta (-b), fast (-f), n_runs (--n_runs), n_graphs (--n_graphs), dump_p (--dump_p)
    - log_to_stdout: if True, route logging to stdout (useful in Streamlit)
    """

    # ---- configure logger (avoid Streamlit stderr split; send to stdout) ----
    local_logger = logging.getLogger(__name__)
    local_logger.setLevel(logging.INFO)

    # prevent duplicated handlers if Streamlit reruns
    local_logger.handlers = []
    local_logger.propagate = False

    handler_stream = sys.stdout if log_to_stdout else sys.stderr
    h = logging.StreamHandler(handler_stream)
    h.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    h.setFormatter(fmt)
    local_logger.addHandler(h)

    # ---- replicate CLI logic exactly ----
    N = size
    avg_deg = avgk
    n = communities
    bs_file = sizes
    delta_file = delta
    rho = assortativity
    q = order_decay

    block_sizes = get_block_sizes(bs_file, N, local_logger)

    if delta_file is None:
        delta_matrix = generate_matrix(n, rho, q)
    else:
        delta_matrix = read_matrix(delta_file)
        rho = None
        q = None

    output_folder = output
    adjust_features = not fast
    iters = n_runs

    # ---- output folder ----
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        local_logger.info(f"Created output directory: {output_folder}")

    # info.csv
    with open(os.path.join(output_folder, 'info.csv'), 'w') as fout:
        print('N,beta,gamma,k,n,rho,q,runs,n_graphs', file=fout)
        print(f'{N},{beta},{gamma},{avg_deg},{n},{rho},{q},{iters},{n_graphs}', file=fout)

    # node_list
    node_list = get_node_list(block_sizes)
    save_data('node_list.txt', output_folder, node_list, fmt='%d')
    local_logger.info('node list with membership dumped')

    # run generator
    start_time = time.time()
    generate(
        N, beta, avg_deg, gamma,
        delta_matrix, block_sizes,
        adjust_features, iters,
        output_folder, dump_p, n_graphs,
        logger=local_logger
    )
    end_time = time.time()
    local_logger.info(f"Total time for {iters} runs: {end_time - start_time:.2f} seconds.")

    return output_folder


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-N", type=int)
    # args = parser.parse_args()
    # 
    # print("ARGS:", args, flush=True)
    # time.sleep(2)
    # print("DONE", flush=True)

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

