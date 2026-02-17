#
#
#    Copyright (C) 2020 Stefano Guarino, Enrico Mastrostefano, Davide Torre 
#
#    This file is part of RHBM.
#
#    RHBM is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RHBM is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with USN.  If not, see <http://www.gnu.org/licenses/>.
#
#

import numpy as np
import os
import argparse
from scipy.spatial.distance import pdist,cdist 
from rhbm_lib import get_K, get_blocks, extract_edge_list


def delta_theta(theta_i,theta_j):
    return np.pi-np.abs(np.pi-np.abs(theta_i-theta_j))

def delta_theta_from_vectors(v_i,v_j=None):
    if v_j is None:
        return np.arccos(1 - cdist(v_i, v_i, 'cosine'))
    else:
        return np.arccos(1 - cdist(v_i, v_j, 'cosine'))

def generate_S1(N, beta, R, mu, kappas, thetas):
    """
    compute the probability matrix for the S^1 model as p_ij = 1/(1 + ((R*Dtheta) / (mu*k_i*k_j))^beta)
    """
    kappas = np.asarray(kappas) 
    thetas = np.asarray(thetas)
    p = (R*delta_theta(thetas.reshape(N,1),thetas.reshape(1,N)))/(mu*(kappas.reshape((N,1))*kappas.reshape((1,N))))
    p = 1/(1+np.power(p,beta))
    np.fill_diagonal(p,0)
    return p

def generate_SD(N, beta, R, mu, D, kappas, coordinates):
    """
    compute the probability matrix for the S^D model as p_ij = 1/(1 + ((R*Dtheta) / (mu*k_i*k_j)^(1/D) )^beta)
    """
    kappas = np.asarray(kappas) 
    p = R*delta_theta_from_vectors(coordinates)/np.power(mu*(kappas.reshape((N,1))*kappas.reshape((1,N))),1/D)
    p = 1/(1+np.power(p,beta))
    np.fill_diagonal(p,0)
    return p

def read_dmercator(inf_coord_file):
    kappas = []
    coordinates = []
    with open(inf_coord_file, 'r') as fin:
        while True:
            line = fin.readline()
            if line.startswith('# Parameters'):
                break
        N = int(fin.readline().rsplit(' ',1)[-1])
        beta = float(fin.readline().rsplit(' ',1)[-1])
        mu = float(fin.readline().rsplit(' ',1)[-1])
        R = float(fin.readline().rsplit(' ',1)[-1])
        line = fin.readline()
        while line.startswith('#'):
            line = fin.readline()
        vals = line.split()
        D = len(vals)-3
        if D == 1:
            while True:
                i,k,t,_ = vals
                kappas.append(float(k))
                coordinates.append(float(t))
                line = fin.readline()
                if not line or line.startswith('#'):
                    break
                vals = line.split()
        else:
            D = D-1
            while True:
                kappas.append(float(vals[1]))
                coordinates.append([float(t) for t in vals[3:]])
                line = fin.readline()
                if not line or line.startswith('#'):
                    break
                vals = line.split()
    assert N == len(kappas)
    return N,D,beta,R,mu,kappas,coordinates

def generate_dmercator(inf_coord_file):
    N,D,beta,R,mu,kappas,coordinates = read_dmercator(inf_coord_file)
    print(f'input parameters: N={N}, D={D}, beta={beta}, R={R}')
    if D==1:
        return generate_S1(N, beta, R, mu, kappas, coordinates)
    else:
        return generate_SD(N, beta, R, mu, D, kappas, coordinates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to input file with inferred coords')
    parser.add_argument('-o', '--output', type=str, help='path to output basefile')
    parser.add_argument('-n', '--n_coms', type=int, help='number of communities')
    parser.add_argument('-g', '--n_graphs', type=int, help='number of graphs')
    args = parser.parse_args()
    inf_coord_file = args.input
    output_base = args.output
    n = args.n_coms
    n_graphs = args.n_graphs
    ######
    N,D,beta,R,mu,input_kappas,_ = read_dmercator(inf_coord_file)
    block_sizes = [N//n for _ in range(n)]
    i = 0
    while sum(block_sizes)<N:
        block_sizes[i] += 1
        i += 1
        if i==n:
            i = 0
    community_array = np.concat([[i]*c for i,c in enumerate(block_sizes)])
    p = generate_dmercator(inf_coord_file)
    # np.save(os.path.join(output_folder,'probability_matrix'),p)
    K_exp = get_K(p,community_array,len(block_sizes))
    np.save(output_base+f'_dmercator_mixing_matrix', K_exp)
    print(f'expected mixing matrix dumped')
    exp_degs = p.sum(axis=0)
    np.save(output_base+'_dmercator_degrees', exp_degs)
    print(f'expected degrees dumped')
    for j in range(n_graphs):
        edge_list = extract_edge_list(p)
        np.savetxt(output_base+f'_dmercator_edge_list_{j}.txt', edge_list, fmt='%d')
        print(f'edge_list {j} dumped', end='\r')


