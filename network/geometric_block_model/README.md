### WARNING: The code is under continuous development and some features may not yet be documented or conversely may have become obsolete. Please contact us if you encounter any problems!

Implementation of the model described in the paper:  
[**Random Hyperbolic Graphs with Arbitrary Mesoscale Structures**](https://arxiv.org/abs/2506.02686)

# GEOMETRIC BLOCK MODEL
This repository contains the code used in [1](#RHBM) to:
- generate a synthetic graph using the Random Hyperbolic Block Model (RHBM);
- use the Random Hyperbolic Block Model (RHBM) to randomize any input graph $G^*$; 
- use D-Mercator [2](#dmercator) to embed in $\mathbb{S}^D$ any graph $G^*$, and generate randomized versions of $G^*$ from the embedding;
- measure the degree-sequence, clustering coefficient and block-mixing structure of some graph $G$ and possibly compare them with those of another graph $G^*$.

## Generating graphs with RHBM

The RHBM requires the following parameters:
- $N$: the number of nodes in the network;
- $k$: the average degree of the network;
- $n$: the number of blocks (i.e. communities)
- $\beta$: the inverse temperature of the model;
- $\gamma$: the power-law exponent of the degree distribution (this is assumed to be Pareto-distributed);
- $N_1,\ldots,N_n$: the number of nodes in each of the $n$ blocks;
- $\Delta$: the block-mixing matrix, which must satisfy $\sum_{IJ} \Delta_{IJ}=2$, where $\Delta_{IJ}$ is the ratio of all edges in the network that connect blocks $I$ and $J$, twice that number if $I=J$.

### Generating graphs
To generate RHBM graphs, run:
```sh
python3 src/rhbm/rhbm_generate.py -N N -k k -n n -b beta -g gamma --block_sizes block_sizes_csv --delta delta_csv -o outputfolder --n_graphs ngraphs --dump_p
```

The block sizes $N_1,\ldots,N_n$ are passed using a csv file with no header formatted as follows:
```
N1,N2,...,Nn
```

The $\Delta$ matrix is passed using a csv file with no header formatted as follows:
```
D11,D12,...,D1n
D21,D22,...,D2n
...
Dn1,Dn2,...,Dnn
```

The last 3 parameters are used as folows:
- `-o` specifies the path where all output files must be saved
- `--n_graphs` specifies the number of synthetic graphs to generate
- `--dump_p` specifies that the probability matrix must be dumped

The script dumps the following files in the `outputfolder':
- `degrees.npy`: the expected degree sequence, as a pickled numpy array
- `mixing_matrix.npy`: the expected mixing matrix, as a pickled numpy array
- `edge_list_i.txt`: for $i=1,\ldots,\text{ngraphs}$ the edge list for the $i$th sampled graph
- possibly, `probability_matrix.npy` the $N\times N$ matrix of edge probabilities, as a pickled numpy array 

### Generating a mixing matrix
We defined a model to generate mixing matrices (see [1](#RHBM)) parametrized by two parameters:
- $\rho\in[−1, 1]$ interpolates between disassortative and assortative mixing
- $q\in(0, 1]$ controls the decay of connectivity away from the diagonal, making it possible to obtain ordered community structures

To generate a mixing matrix with this model, run:
```sh
python3 src/rhbm/generate_matrix.py -n n -p rho -q q -o outputfolder
```
The matrix will be dumped to `outputfolder/delta.csv` in the format described above.

### Running a full experiment
To run experiments that involve the creation of the matrix and the generation of the graph, run:
```sh
script/run_rhbm.py N k n beta gamma rho q
```

All files will be created under the `output/job_XXXX` folder, where `XXXX` is a pseudo-random experiment id. The folder will contain also a `info.csv` file reporting all parameters used in that run.

## Randomizing graphs with D-Mercator

In [1](#RHBM) we show what happens to block/attribute-based mixing when you use D-Mercator to randomize a graph.
The randomization works by first embedding the graph in $\mathbb{S}^D$ for some $D\geq 1$ and then generating one or more new graphs from the embedding.

### Randomizing a graph generated with RHBM
To randomize a graph generated with RHBM, enter the `script` folder and run:
```sh
script/run_dmercator.sh job_XXXX graph_id n_runs n_graphs
```
where:
- `job_XXXX` is the folder that contains the RHBM graph;
- `graph_id` is the id of the RHBM graph that must be used, meaning that the script will use the file `edge_list_{graph_id}.txt`;
- `n_runs` is the number of embeddings to generate;
- `n_graphs` is the number of synthetic graphs to generate from each embedding.

The embeddings are saved in `output/job_XXXX`. 

**NB**: in this case, all blocks are assumed to be equally sized!

### Randomizing a real graph
First, make sure that the graph is stored in both the NCOL and GRAPHML formats in two files named `graph.ncol` and `graph.graphml`, respectively. These must be stored in the same subfolder of the `graphs` folder.

To randomize the graph, enter the `script` folder and run:
```sh
script/run_dmercator_real.sh subfolder n_runs n_graphs com_att
```
where:
- `subfolder` is the subfolder of `graphs` that contains the NCOL and GRAPHML files;
- `n_runs` is the number of embeddings to generate;
- `n_graphs` is the number of synthetic graphs to generate from each embedding;
- `com_att` is the name of the node attribute that identifies the blocks in the GRAPHML file.

The embeddings are saved in `output/subfolder`. 

**NB**: in this case, all blocks are assumed to be equally sized!
