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
np.set_printoptions(precision=3, suppress=True, linewidth=120)
import os
from scipy.optimize import least_squares
from scipy.special import hyp2f1
from scipy.stats import pareto
import argparse
import igraph
import time
import logging
from tqdm.auto import tqdm
import psutil


###############################################################################
# UTILS
###############################################################################

def auto_workers():
    physical = psutil.cpu_count(logical=False)
    return max(1, physical - 1)


def save_data(filename, output_folder, data, **kwargs):
    # First, check if the folder is available, else create the folder
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder, exist_ok = True)
        logger.warning(f"Output_folder not found. Directory created at: {output_folder}.")
    if 'fmt' in kwargs.keys():
        save_function = np.savetxt
    elif isinstance(data, igraph.Graph): # Check if it's igraph class
        save_function = lambda path, data, **kwargs: data.write_ncol(path, weights=None, **kwargs)
    else:
        save_function = np.save
    # Make the path with os
    path = os.path.join(output_folder, filename)
    # Save it, baby!
    save_function(path, data, **kwargs)
    return


def get_node_list(block_sizes):
    k = 0
    node_list = []
    for i,j in enumerate(block_sizes):
        node_list.extend(list(zip(range(k,j+k),[i]*j)))
        k += j
    return node_list

###############################################################################
# SAMPLING FUNCTIONS
###############################################################################

def sample_angle(N):
    """
    Sample N angular coordinates uniformly in [0, 2π).
    
    Args:
        N (int): Number of angles to sample
        
    Returns:
        np.ndarray: Array of N angular coordinates
    """
    return np.random.uniform(0, 2 * np.pi, N)


def sample_pareto(N, alpha, loc=0, scale=1, mean=None):
    """
    Sample N values from a Pareto distribution.
    
    Args:
        N (int): Number of samples
        alpha (float): Shape parameter (alpha > 1)
        loc (float): Location parameter (default: 0)
        scale (float): Scale parameter (default: 1)
        mean (float): If provided, rescale samples to have this mean
        
    Returns:
        np.ndarray: Array of N samples from Pareto distribution
    """
    b = alpha - 1
    r = pareto.rvs(b, loc=loc, scale=scale, size=N)
    
    if mean:
        r = (r / r.mean() * mean).round()
    
    return r


def get_R(N):
    """
    Compute the radius of the hyperbolic disk for N nodes.
    
    Args:
        N (int): Number of nodes
        
    Returns:
        float: Radius R = N / (2π)
    """
    return N / (2 * np.pi)


###############################################################################
# HYPERBOLIC GEOMETRY FUNCTIONS
###############################################################################

def get_2f1(x, beta):
    """
    Compute the Gaussian hypergeometric function 2F1(1, 1/β, 1+1/β, -x^β).
    
    Args:
        x (float or np.ndarray): Input value(s)
        beta (float): Shape parameter
        
    Returns:
        float or np.ndarray: Result of hypergeometric function
    """
    return hyp2f1(1, 1/beta, 1 + 1/beta, -(x**beta))


def I_beta(beta):
    """
    Compute the normalization constant I(β) = π / (β * sin(π/β)).
    
    Args:
        beta (float): Shape parameter (β > 0)
        
    Returns:
        float: Normalization constant (0 if β = ∞)
    """
    if beta == np.inf:
        return 0
    return np.pi / (beta * np.sin(np.pi / beta))


def delta_theta(theta_i, theta_j):
    """
    Compute the angular distance between two angles on a circle.
    The distance is the minimum arc length in [0, π].
    
    Args:
        theta_i (float or np.ndarray): First angle(s)
        theta_j (float or np.ndarray): Second angle(s)
        
    Returns:
        float or np.ndarray: Angular distance in [0, π]
    """
    return np.pi - np.abs(np.pi - np.abs(theta_i - theta_j))


###############################################################################
# UTILITY FUNCTIONS FOR OPTIMIZATION
###############################################################################

def safe_exp(x):
    """
    Compute exponential with clipping to avoid overflow/underflow.
    
    Args:
        x (float or np.ndarray): Input value(s)
        
    Returns:
        float or np.ndarray: exp(x) with x clipped to [-20, 20]
    """
    return np.exp(np.clip(x, -20, 20))


###############################################################################
# PROBABILITY AND CONNECTIVITY FUNCTIONS
###############################################################################

def get_p(theta, fitness, community_array, K, beta=2):
    """
    Compute the probability matrix for the RHBM.
    
    The connection probability between nodes i and j is:
    p_ij = 1 / (1 + (Δθ_ij / (β·sin(π/β) · f_i · f_j · K_IJ))^β)
    
    where:
    - Δθ_ij is the angular distance between nodes i and j
    - f_i, f_j are the fitness values of nodes i and j
    - K_IJ is the block connectivity between communities I and J
    - β is the clustering parameter
    
    Args:
        theta (np.ndarray): Angular coordinates of nodes (shape: N)
        fitness (np.ndarray): Fitness values of nodes (shape: N)
        community_array (np.ndarray): Community membership (shape: N)
        K (np.ndarray): Block connectivity matrix (shape: n×n)
        beta (float): Clustering parameter (default: 2)
        
    Returns:
        np.ndarray: Connection probability matrix (shape: N×N)
    """
    N = len(theta)
    
    # Compute angular distance matrix
    p = delta_theta(theta.reshape(N, 1), theta.reshape(1, N))
    p = p / (beta * np.sin(np.pi / beta))
    
    # Divide by block connectivity
    p = p / K[community_array][:, community_array]
    
    # Divide by fitness products
    p = p / (fitness.reshape((N, 1)) * fitness.reshape((1, N)))
    
    # Apply connection probability formula
    p = 1 / (1 + np.power(p, beta))
    
    # No self-loops
    np.fill_diagonal(p, 0)
    
    return p

def get_block_sizes(block_sizes, N, n, logger):
    if block_sizes is None:
        block_sizes = [N//n for _ in range(n)]
    else:
        with open(block_sizes) as fin:
            block_sizes = [int(x.strip()) for x in fin.read().split()]
            if sum(block_sizes)<N:
                logger.info(f"Warning: the sum of the block sizes does not match with the given N; exceeding nodes will be distributed equally")
    i = 0
    while sum(block_sizes)<N:
        block_sizes[i] += 1
        i += 1
        if i==n:
            i = 0
    return block_sizes

def get_K(p, community_array, n_coms):
    """
    Compute the expected number of edges between communities.
    
    Args:
        p (np.ndarray): Connection probability matrix (shape: N×N)
        community_array (np.ndarray): Community membership (shape: N)
        n_coms (int): Number of communities
        
    Returns:
        np.ndarray: Expected edge count matrix between communities (shape: n×n)
    """
    n = n_coms
    K = np.asarray([
        [p[community_array == i, :][:, community_array == j].sum() 
         for j in range(n)] 
        for i in range(n)
    ])
    return K


def extract_edge_list(p):
    """
    Sample edges from probability matrix to create an edge list.
    
    Args:
        p (np.ndarray): Connection probability matrix (shape: N×N)
        
    Returns:
        np.ndarray: Edge list (shape: E×2), where E is the number of edges
    """
    # Get upper triangular indices (excluding diagonal)
    idx = np.triu_indices_from(p, 1)
    flat_p = p[idx]
    
    # Sample edges based on probabilities
    edges = np.where(np.random.random(flat_p.shape) < flat_p)[0]
    
    return np.transpose((idx[0][edges], idx[1][edges]))


###############################################################################
# OPTIMIZATION FUNCTIONS
###############################################################################


def compute_residuals(x, theta, community_array, c, beta=2):
    """
    Objective function for hidden feature optimization.
    
    This function computes the normalized residuals between expected and target values
    for both node degrees and inter-community edge counts.
    
    Args:
        x (np.ndarray): Optimization variables [log(fitness), log(F)]
        theta (np.ndarray): Angular coordinates
        community_array (np.ndarray): Community membership
        c (np.ndarray): Target values [degrees, K_upper_triangular]
        beta (float): Clustering parameter
        
    Returns:
        np.ndarray: Residuals (target - expected)
    """
    N = len(theta)
    L = (c[:N]).sum() / 2  # Total number of edges
    
    # Extract fitness and block fitness from log-space
    fitness = safe_exp(x[:N])
    F = safe_exp(x[N:])
    
    # Reconstruct full symmetric matrix F from upper triangular
    t = len(F)
    n = int((np.sqrt(8*t + 1) - 1) / 2)
    uids = np.triu_indices(n)
    
    F_full = np.zeros((n, n))
    F_full[uids] = F
    F_full = F_full + F_full.T - np.diag(F_full.diagonal())
    
    # Compute probability matrix
    p = get_p(theta, fitness, community_array, L * F_full, beta)
    
    # Expected degrees and inter-community edges
    expected_degrees = p.sum(axis=0)
    expected_K = get_K(p, community_array, n)
   
    # Normalized residuals to have errors with equal magnitude
    target_degrees = c[:N]
    target_K_flat = c[N:]
    degree_scale = np.mean(target_degrees)
    K_scale = np.mean(target_K_flat)
    residuals_deg = (expected_degrees - target_degrees) / degree_scale
    residuals_K = (expected_K[uids] - target_K_flat) / K_scale
    
    return np.concatenate([residuals_deg, residuals_K])


def compute_jacobian(x, theta, community_array, c, beta=2):
    """
    Jacobian of the objective function for optimization.
    
    Computes the derivatives of expected degrees and K with respect to
    fitness values and block fitness values.
    
    Args:
        x (np.ndarray): Optimization variables [log(fitness), log(F)]
        theta (np.ndarray): Angular coordinates
        community_array (np.ndarray): Community membership
        c (np.ndarray): Target values [degrees, K_upper_triangular]
        beta (float): Clustering parameter
        
    Returns:
        np.ndarray: Jacobian matrix
    """
    N = len(theta)
    L = (c[:N]).sum() / 2
    
    # Extract fitness and block fitness from log-space
    fitness = safe_exp(x[:N])
    F = safe_exp(x[N:])
    
    # Reconstruct full symmetric matrix F
    t = len(F)
    n = int((np.sqrt(8*t + 1) - 1) / 2)
    uids = np.triu_indices(n)
    
    F_full = np.zeros((n, n))
    F_full[uids] = F
    F_full = F_full + F_full.T - np.diag(F_full.diagonal())
    
    # Compute probability and variance matrices
    p = get_p(theta, fitness, community_array, L * F_full, beta)
    q = p * (1 - p)  # Variance: p(1-p)
    
    # Derivative of deg_i with respect to F_IJ
    B = np.zeros((N, t))
    uids_array = np.column_stack(uids)
    
    for i, (F_IJ, (I, J)) in enumerate(zip(F, uids_array)):
        mask_I = community_array == I
        mask_J = community_array == J
        # B[mask_I, i] = beta / F_IJ * q[mask_I, :][:, mask_J].sum(axis=1)
        # if J!=I:
        #     B[mask_J, i] = beta / F_IJ * q[mask_J, :][:, mask_I].sum(axis=1)
        B[mask_I, i] = beta * q[mask_I, :][:, mask_J].sum(axis=1)
        if J!=I:
            B[mask_J, i] = beta * q[mask_J, :][:, mask_I].sum(axis=1)
    
    # Derivative of K_IJ with respect to f_i
    C = np.zeros((t, N))
    
    for i, (I, J) in enumerate(uids_array):
        mask_I = community_array == I
        mask_J = community_array == J
        # C[i, mask_I] = beta / fitness[mask_I] * q[mask_I, :][:, mask_J].sum(axis=1)
        # if J!=I:
        #     C[i, mask_J] = beta / fitness[mask_J] * q[mask_J, :][:, mask_I].sum(axis=1)
        C[i, mask_I] = beta * q[mask_I, :][:, mask_J].sum(axis=1)
        if J!=I:
            C[i, mask_J] = beta * q[mask_J, :][:, mask_I].sum(axis=1)
    
    # Derivative of K_IJ with respect to F_QR
    K = get_K(q, community_array, n)
    # D = np.diag(beta / F * K[uids])
    D = np.diag(beta * K[uids])
    
    # Derivative of deg_i with respect to f_j
    np.fill_diagonal(q, q.sum(axis=0))
    # A = beta / fitness * q
    A = beta * q
    
    # Assemble Jacobian
    J = np.vstack((
        np.hstack((A, B)),
        np.hstack((C, D))
    ))
    
    # # Apply chain rule for log-space variables
    # # d/d(log f) = f * d/df
    # for i in range(N):
    #     J[:, i] = J[:, i] * fitness[i]
    # 
    # for j in range(len(uids_array)):
    #     J[:, N + j] = J[:, N + j] * F[j]
    
    # Jacobian normalization
    target_degrees = c[:N]
    target_K_flat = c[N:]
    degree_scale = np.mean(target_degrees)
    K_scale = np.mean(target_K_flat)
    J[:N, :] /= degree_scale
    J[N:, :] /= K_scale
    
    return J


def adjust_hidden_features(target_degrees, target_K, theta, init_node_fitness, 
                                    init_block_fitness, beta, community_array,
                                    err=5, max_iter=300, method='auto', logger=None):
    """
    Adjust hidden features (fitness and block connectivity) to match target statistics.
    
    This function solves a system of equations to find fitness values and block
    connectivity that produce the desired degree sequence and inter-community edge counts.
    
    Args:
        target_degrees (np.ndarray): Target degree sequence (shape: N)
        target_K (np.ndarray): Target inter-community edge counts (shape: n×n)
        theta (np.ndarray): Angular coordinates (shape: N)
        init_node_fitness (np.ndarray): Initial fitness values (shape: N)
        init_block_fitness (np.ndarray): Initial block connectivity (shape: n×n)
        beta (float): Clustering parameter
        community_array (np.ndarray): Community membership (shape: N)
        err (float): Maximum acceptable error (default: 5)
        max_iter (int): Maximum iterations (default: 300)
        method (str): Method to use -- 'auto', 'trust-region', 'lm', or 'weighted' (default: 'auto')

    Returns:
        tuple: (node_fitness_opt, block_fitness_opt, converged)
            - node_fitness_opt (np.ndarray): Adjusted fitness values
            - block_fitness_opt (np.ndarray): Adjusted block connectivity
            - converged (bool): Whether optimization converged
    """
    
    N = len(init_node_fitness)
    n = init_block_fitness.shape[0]
    
    target_degrees = np.asarray(target_degrees)
    target_K = np.asarray(target_K)
    c = np.concatenate([target_degrees, target_K[np.triu_indices(n)]])
    community_array = np.asarray(community_array)
    L = target_degrees.sum() / 2

    # Initial vector in log-space: [log(fitness), log(K_upper_triangular)]
    _m = init_block_fitness[np.triu_indices(n)]
    x0 = np.concatenate([
        np.log(init_node_fitness), 
        np.log(_m, out=-N*np.ones_like(_m, dtype=np.float64), where=(_m!=0))
    ])
    
    # Try different methods, in order
    methods_to_try = []
    if method == 'auto':
        methods_to_try = [
            ('trf', {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'max_nfev': max_iter * 100}),
            ('lm', {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'max_nfev': max_iter * 100}),
            ('dogbox', {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'max_nfev': max_iter * 100})
        ]
    elif method == 'trust-region':
        methods_to_try = [('trf', {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'max_nfev': max_iter * 100})]
    elif method == 'lm':
        methods_to_try = [('lm', {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8, 'max_nfev': max_iter * 100})]
    
    best_result = None
    best_error = float('inf')
    
    for method_name, options in methods_to_try:
        logger.info(f"Trying method: {method_name}")
        try:
            result = least_squares(
                compute_residuals, 
                x0, 
                jac=compute_jacobian,
                method=method_name,
                args=(theta,community_array,c,beta),
                # workers=auto_workers,
                **options
            )
            
            # Compute actual (non normalized) error
            sol = result.x
            node_fitness_test = safe_exp(sol[:N])
            block_fitness_test_flat = safe_exp(sol[N:])
            
            uids = np.triu_indices(n)
            block_fitness_test = np.zeros((n, n))
            block_fitness_test[uids] = block_fitness_test_flat
            block_fitness_test = block_fitness_test + block_fitness_test.T - np.diag(block_fitness_test.diagonal())
            
            p_test = get_p(theta, node_fitness_test, community_array, L * block_fitness_test, beta)
            degs_test = p_test.sum(axis=0)
            K_test = get_K(p_test, community_array, n)
            
            max_err_degs = np.max(np.abs(degs_test - target_degrees))
            max_err_K = np.max(np.abs(K_test - target_K))
            total_error = max_err_degs + max_err_K
            
            logger.info(f"Method {method_name}: deg_error={max_err_degs:.4f}, K_error={max_err_K:.4f}")
            
            if total_error < best_error:
                best_error = total_error
                best_result = result
                
            if max_err_degs <= err and max_err_K <= err:
                logger.info(f"✓ Converged with {method_name}!")
                break
                
        except Exception as e:
            logger.warning(f"✗ Method {method_name} failed: {e}")
            continue
    
    if best_result is None:
        logger.warning("All methods failed!")
        return init_node_fitness, init_block_fitness, False
    
    # Extract solution from log-space
    sol = best_result.x
    node_fitness_opt = safe_exp(sol[:N])
    block_fitness_opt_flat = safe_exp(sol[N:])
    
    # Reconstruct symmetric block fitness matrix
    uids = np.triu_indices(n)
    block_fitness_opt = np.zeros((n, n))
    block_fitness_opt[uids] = block_fitness_opt_flat
    block_fitness_opt = block_fitness_opt + block_fitness_opt.T - np.diag(block_fitness_opt.diagonal())
    
    # Verify convergence
    p_final = get_p(theta, node_fitness_opt, community_array, L * block_fitness_opt, beta)
    degs_final = p_final.sum(axis=0)
    K_final = get_K(p_final, community_array, n)
    
    max_err_degs = np.max(np.abs(degs_final - target_degrees))
    max_err_K = np.max(np.abs(K_final - target_K))
    
    logger.info(f"Final errors: deg={max_err_degs:.4f}, K={max_err_K:.4f}")
    logger.info(f"Final K:\n\t"+str(K_final).replace('\n', '\n\t'))
    
    converged = max_err_degs <= err and max_err_K <= err
    
    return node_fitness_opt, block_fitness_opt, converged


###############################################################################
# MIXING MATRIX GENERATION
###############################################################################

def read_matrix(delta_file):
    with open(delta_file) as fin:
        delta = np.array([[float(x.strip()) for x in line.split()] for line in fin])
    if not np.isclose(delta,delta.T).all():
        delta = (delta+delta.T)/2
    delta = 2*delta/delta.sum()
    return delta


def generate_matrix(n, p, q):
    """
    Generate a mixing matrix with controllable assortativity.
    
    Args:
        n (int): Number of communities
        p (float): Assortativity parameter in [-1, 1]
            p = -1: maximally disassortative (anti-diagonal)
            p = 0: neutral (uniform)
            p = 1: maximally assortative (diagonal)
        q (float): Decay parameter in (0, 1]
            Controls how connectivity decays away from the diagonal
            q = 1: uniform off-diagonal elements
            q < 1: exponential decay
            
    Returns:
        np.ndarray: Normalized mixing matrix (shape: n×n)
    """
    # Create off-diagonal matrix with decay
    H = np.zeros((n, n))
    for i in range(1, n):
        H += np.diag(np.ones(n - i) * (q**i), i)
        H += np.diag(np.ones(n - i) * (q**i), -i)
    H = 2 * H / H.sum()
    
    # Create identity matrix (diagonal)
    I = 2 / n * np.eye(n)
    
    # Combine diagonal and off-diagonal parts
    K = (p + 1) / 2 * I + (1 - p) / 2 * H
    
    return K


###############################################################################
# GRAPH GENERATION
###############################################################################

def generate(N, beta, avg_deg, alpha, delta, block_sizes, 
             adjust_features=True, iters=1, output_folder='', 
             dump_p=False, n_graphs=10, logger=None, verbose=False):
    """
    Generate synthetic networks using the RHBM model.
    
    Args:
        N (int): Total number of nodes
        beta (float): Clustering parameter
        avg_deg (float): Target average degree
        alpha (float): Exponent for Pareto degree distribution
        delta (np.ndarray): Mixing matrix (shape: n×n)
        block_sizes (list): List of block sizes (length: n)
        adjust_features (bool): Whether to adjust hidden features (default: True)
        iters (int): Number of independent realizations (default: 1)
        output_folder (str): Path to output directory
        dump_p (bool): Whether to dump probability matrix (default: False)
        n_graphs (int): Number of graphs per iteration (default: 10)
    """
    # Logger initialization
    if logger is None:
        # set information level logger to warning
        logger = logging.getLogger('randomize')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        if verbose:
            # Use tqdm for info level logging
            logging.basicConfig(level=logging.INFO, format=formatter, emitter=lambda st: tqdm.write(st))
        else:
            logging.basicConfig(level=logging.WARNING, format=formatter)
    
    # Initialization
    L = N * avg_deg / 2  # Total number of edges
    n = len(delta)
    community_array = np.concat([[i] * c for i, c in enumerate(block_sizes)])
    target_K = delta * L
    K = target_K
    block_fitness = delta
    logger.info(f"Target K matrix:\n\t"+str(K).replace('\n', '\n\t'))

    i = 0
    while i<iters:
        # Sample random angles
        theta = sample_angle(N)
        
        # Sample fitness from Pareto distribution, normalize per block
        node_fitness = sample_pareto(N, alpha)
        for I in range(n):
            mask = community_array == I
            node_fitness[mask] = node_fitness[mask] / node_fitness[mask].sum()
        
        # Compute target degrees
        target_degrees = np.zeros(N)
        for I in range(n):
            mask = community_array == I
            target_degrees[mask] = node_fitness[mask] * target_K[I].sum()
        
        # Adjust features if requested
        while adjust_features:
            start_time_adjust = time.time()
            logger.info(f'Adjusting hidden degrees, iter {i}...')
            node_fitness, block_fitness, success = adjust_hidden_features(
                target_degrees, target_K, theta, node_fitness, 
                block_fitness, beta, community_array, logger=logger
            )
            end_time_adjust = time.time()
            if success:
                K = L * block_fitness
                logger.info(f'Success! Took {end_time_adjust - start_time_adjust:.2f} seconds.\n')
                break
            else:
                logger.warning(f'Failure! Took {end_time_adjust - start_time_adjust:.2f} seconds.\n')
        p = get_p(theta, node_fitness, community_array, K, beta)
        if dump_p:
            save_data(f'probability_matrix_{i}', output_folder, p)
            logger.info(f'randomized matrix {i} dumped')
        else:
            K_exp = get_K(p, community_array, n)
            save_data(f'mixing_matrix_{i}', output_folder, K_exp)
            logger.info(f'expected mixing matrix {i} dumped')
            degs = p.sum(axis=0)
            save_data(f'degrees_{i}', output_folder, degs)
            logger.info(f'expected degrees {i} dumped')
            for j in tqdm(range(n_graphs), desc=f"Generating graphs for run {i}", leave=False):
                edge_list = extract_edge_list(p)
                save_data(f'edge_list_{i}_{j}.txt', output_folder, edge_list, fmt='%d')
        i += 1
    logger.info(f"Generation complete. Saved {iters} mixing matrices and degree sequences in '{output_folder}'.")


###############################################################################
# GRAPH RANDOMIZATION
###############################################################################

def get_blocks(G, attribute):
    """
    Extract block structure from a graph based on a vertex attribute.
    
    Args:
        G (igraph.Graph): Input graph
        attribute (str): Name of the vertex attribute indicating community
        
    Returns:
        tuple: (community_array, K, community_names)
            - community_array (np.ndarray): Community membership
            - K (np.ndarray): Adjacency matrix of block graph
            - community_names (list): Names of communities
            
    Raises:
        ValueError: If attribute doesn't exist in graph
    """
    if attribute not in G.vs.attributes():
        raise ValueError(f"Attribute {attribute} not found in graph. Aborting")
    
    cl = igraph.VertexClustering.FromAttribute(G, attribute)
    H = cl.cluster_graph(
        combine_vertices={attribute: 'first'}, 
        combine_edges=False
    )
    
    community_names = H.vs[attribute]
    community_array = np.array(cl.membership)
    K = np.array(H.get_adjacency())
    
    return community_array, K, community_names


def randomize(G, beta, block_attribute='community', adjust_features=True, iters=1, output_folder=None,
              dump_p=False, n_graphs=10, logger=None, verbose=False):
    """
    Randomize a graph while preserving degree sequence and block structure.
    
    Args:
        G (igraph.Graph): Input graph to randomize
        beta (float): Clustering parameter
        block_attribute (str): Vertex attribute for community (default: 'community')
        adjust_features (bool): Whether to adjust hidden features (default: True)
        iters (int): Number of randomizations (default: 1)
        output_folder (str): Path to output directory
        dump_p (bool): Whether to dump probability matrix (default: False)
        n_graphs (int): Number of graphs per iteration (default: 10)
        eps (float): Minimum degree for nodes with degree 0 (default: 1)
        
    Returns:
        np.ndarray: Probability matrix if iters=1, None otherwise
    """
    # Logger initialization
    if logger is None:
        # set information level logger to warning
        logger = logging.getLogger('randomize')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        if verbose:
            # Use tqdm for info level logging
            logging.basicConfig(level=logging.INFO, format=formatter, emitter=lambda st: tqdm.write(st))
        else:
            logging.basicConfig(level=logging.WARNING, format=formatter)

    # Initialization
    N = G.vcount()
    degrees = G.degree()
    if min(degrees)==0:
        logger.warning('Input graph contains isolated vertices, these will be removed before proceeding')
        G.delete_vertices([i for i in range(N) if degrees[i]==0])
        N = G.vcount()
        logger.info(f'the graph now has {N} vertices')
    
    L = G.ecount()
    community_array, K, com_names = get_blocks(G, block_attribute)
    logger.info(f"Input K matrix:\n\t"+str(K).replace('\n', '\n\t'))
    
    # Set target statistics
    target_K = K
    block_fitness = target_K / L
    target_degrees = np.array(G.degree())
   
    # Check consistency
    n = len(com_names)
    for I in range(n):
        mask = community_array == I
        assert target_degrees[mask].sum() == target_K[I].sum()

    # Initialize node fitness
    node_fitness = np.zeros(N)
    n = len(com_names)
    for I in range(n):
        mask = community_array == I
        node_fitness[mask] = target_degrees[mask] / target_degrees[mask].sum()
    
    # Generate randomized graphs
    i = 0
    while i<iters:
        theta = sample_angle(N)
        # Adjust features if requested
        while adjust_features:
            start_time_adjust = time.time()
            logger.info(f'Adjusting hidden degrees, iter {i}...')
            node_fitness, block_fitness, success = adjust_hidden_features(
                target_degrees, target_K, theta, node_fitness, 
                block_fitness, beta, community_array, logger=logger
            )
            end_time_adjust = time.time()
            if success:
                logger.info(f'Success! Took {end_time_adjust - start_time_adjust:.2f} seconds.\n')
                break
            else:
                logger.warning(f'Failure! Took {end_time_adjust - start_time_adjust:.2f} seconds.\n')
        p = get_p(theta, node_fitness, community_array, L * block_fitness, beta)
        if dump_p:
            save_data(f'randomized_probability_matrix_{i}', output_folder, p)
            logger.info(f'randomized matrix {i} dumped')
        else:
            K_exp = get_K(p, community_array, n)
            save_data(f'mixing_matrix_{i}', output_folder, K_exp)
            logger.info(f'expected mixing matrix {i} dumped')
            degs = p.sum(axis=0)
            save_data(f'degrees_{i}', output_folder, degs)
            logger.info(f'expected degrees {i} dumped')
            for j in tqdm(range(n_graphs), desc=f"Generating graphs for run {i}", leave=False):
                edge_list = extract_edge_list(p)
                save_data(f'edge_list_{i}_{j}.txt', output_folder, edge_list, fmt='%d')
        i += 1
    logger.info(f"Randomization complete. Saved {iters} mixing matrices and degree sequences in '{output_folder}'.")


###############################################################################
# MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate synthetic networks using RHBM'
    )
    parser.add_argument('-N', '--size', type=int, default=1000, 
                       help='Network size')
    parser.add_argument('-k', '--avgk', type=int, default=10, 
                       help='Average degree')
    parser.add_argument('-g', '--gamma', type=float, default=2.5, 
                       help='Exponent of degree distribution')
    parser.add_argument('-n', '--communities', type=int, default=5, 
                       help='Number of communities')
    parser.add_argument('-p', '--assortativity', type=float, default=0.5, 
                       help='Assortativity parameter for mixing matrix')
    parser.add_argument('-q', '--order_decay', type=float, default=1, 
                       help='Decay parameter for mixing matrix')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Path to output folder')
    parser.add_argument('-b', '--beta', type=float, default=2, 
                       help='Clustering parameter')
    parser.add_argument('-f', '--fast', action='store_true', 
                       help='Skip hidden degree adjustment')
    parser.add_argument('--n_runs', type=int, default=1, 
                       help='Number of runs')
    parser.add_argument('--n_graphs', type=int, default=10, 
                       help='Number of synthetic graphs to generate')
    parser.add_argument('--dump_p', action='store_true', 
                       help='Dump probability matrix instead of edge lists')
    
    args = parser.parse_args()
    
    # Extract parameters
    N = args.size
    avg_deg = args.avgk
    gamma = args.gamma
    n = args.communities
    rho = args.assortativity
    q = args.order_decay
    output_folder = args.output
    beta = args.beta
    adjust_degrees = not args.fast
    iters = args.n_runs
    n_graphs = args.n_graphs
    dump_p = args.dump_p
    
    # Create block sizes (approximately equal)
    block_sizes = [N // n for _ in range(n)]
    i = 0
    while sum(block_sizes) < N:
        block_sizes[i] += 1
        i += 1
        if i == n:
            i = 0
    
    # Generate mixing matrix
    delta = generate_matrix(n, rho, q)
    
    # Save metadata
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'info.csv'), 'w') as fout:
        print('N,beta,gamma,k,n,rho,q,runs', file=fout)
        print(f'{N},{beta},{gamma},{avg_deg},{n},{rho},{q},{iters}', file=fout)
    
    # Generate networks
    generate(N, beta, avg_deg, gamma, delta, block_sizes, 
            adjust_degrees, iters, output_folder, dump_p, n_graphs)
