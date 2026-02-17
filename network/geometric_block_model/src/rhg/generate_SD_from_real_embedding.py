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
from rhbm_lib import get_K, get_blocks, extract_edge_list, InvalidFileExtensionError
import igraph

def print_np_matrix(A, d_int = False):
    rows, cols = A.shape
    for ir in range(rows):
        if d_int:
            print(np.array([int(d) for d in A[ir, :]])) 
        else:
            print(np.array([d for d in A[ir, :]])) 
    return

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
    names = []
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
                names.append(i)
                kappas.append(float(k))
                coordinates.append(float(t))
                line = fin.readline()
                if not line or line.startswith('#'):
                    break
                vals = line.split()
        else:
            D = D-1
            while True:
                names.append(vals[0])
                kappas.append(float(vals[1]))
                coordinates.append([float(t) for t in vals[3:]])
                line = fin.readline()
                if not line or line.startswith('#'):
                    break
                vals = line.split()
    assert N == len(kappas)
    return N,D,beta,R,mu,kappas,coordinates,names

def generate_dmercator(inf_coord_file):
    N,D,beta,R,mu,kappas,coordinates,names = read_dmercator(inf_coord_file)
    print(f'input parameters: N={N}, D={D}, beta={beta}, R={R}')
    if D==1:
        return generate_S1(N, beta, R, mu, kappas, coordinates)
    else:
        return generate_SD(N, beta, R, mu, D, kappas, coordinates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to the graph in format .graphml')
    parser.add_argument('-e', '--embedding', type=str, help='path to the inf_coord file')
    parser.add_argument('-o', '--output', type=str, help='path to output basefile')
    parser.add_argument('-c', '--com_att', type=str, help='name of the vertex attribute to be used to determine the communities; defaults to "community"', default='community')
    parser.add_argument('-g', '--n_graphs', type=int, help='number of graphs')
    args = parser.parse_args()
    input_graph_file = args.input
    inf_coord_file = args.embedding
    com_att = args.com_att
    output_base = args.output
    n_graphs = args.n_graphs
    # Sanity checks
    ## Input graph
    if not os.path.exists(input_graph_file):
                raise FileNotFoundError(f"File {input_graph_file} not found.")

    if not input_graph_file.endswith('.graphml'):
        # Get the actual extension for better error reporting
        actual_ext = os.path.splitext(input_graph_file)[1] if '.' in input_graph_file else None
        raise InvalidFileExtensionError(input_graph_file, '.graphml', actual_ext)

    ## Embedding
    if not os.path.exists(inf_coord_file):
        raise FileNotFoundError(f"File {inf_coord_file} not found.")

    if not inf_coord_file.endswith('.inf_coord'):
        # Get the actual extension for better error reporting
        actual_ext = os.path.splitext(inf_coord_file)[1] if '.' in inf_coord_file else None
        raise InvalidFileExtensionError(inf_coord_file, '.inf_coord', actual_ext)

    ######
    N,D,beta,R,mu,input_kappas,_,names = read_dmercator(inf_coord_file)
    in_graph = igraph.read(input_graph_file)
    ## Sanity check: is it the same graph? At least count the number of vertices...
    if N != in_graph.vcount():
        raise RuntimeError(f"The graphs and the embedding have different number of vertices. Are they the same?")
    
    community_array,K_in,communities = get_blocks(in_graph, com_att)
    ## dmercator orders alfabetically nodes name. We have to make sure that this order is respected
    dmercator_order = np.argsort(in_graph.vs['name']) if 'name' in in_graph.vs.attributes() else np.arange(N)
    community_array_input_graph_order = community_array.copy()
    community_array = community_array[dmercator_order]
    
    p = generate_dmercator(inf_coord_file)
    K_exp = get_K(p,community_array,len(communities))
    np.save(output_base+f'_dmercator_mixing_matrix', K_exp)
    print(f'expected mixing matrix dumped')
    exp_degs = p.sum(axis=0)
    exp_degs = exp_degs[dmercator_order]
    np.save(output_base+'_dmercator_degrees', exp_degs)
    print(f'expected degrees dumped')
    for j in range(n_graphs):
        edge_list = extract_edge_list(p)
        np.savetxt(output_base+f'_dmercator_unsorted_edge_list_{j}.txt', edge_list, fmt='%d')
        print(f'edge_list {j} dumped', end='\r')
    #### Other stats
    average_degree_dmercator = np.mean(exp_degs)
    expected_edges_dmercator = np.sum(p) / 2
    average_degree_input_graph = 2 * in_graph.ecount() / in_graph.vcount()
    K_miss_extra = np.abs(K_exp - K_in)
    faulty_edges = np.sum(K_miss_extra) / 2
    print("-------- EMBEDDING STATITICS --------")
    print("metric\t| embedding\t| input graph\t| error\t\t|")
    print(f"<deg>\t| {average_degree_dmercator:.2f}\t\t| {average_degree_input_graph:.2f}\t\t| {(average_degree_dmercator - average_degree_input_graph) / average_degree_input_graph *100:.2f} %\t|")
    print("----------------------------------------------------------")
    print("K input:")
    print_np_matrix(K_in, d_int=True)
    print("K output:")
    print_np_matrix(K_exp, d_int = True)
    print("Missing/abbundant edges:")
    print_np_matrix(K_miss_extra, d_int = False)
    print("Faulty edges among comunities:")
    print(faulty_edges)
    print("Faulty edges ratio:")
    print(f"{faulty_edges / expected_edges_dmercator * 100:.2f}%")
    print("Total error:")
    print(f"{faulty_edges / (2 * K_in.sum()) * 100:.2f}%")
