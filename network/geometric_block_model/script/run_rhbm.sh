#!/bin/bash
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
#    along with RHBM.  If not, see <http://www.gnu.org/licenses/>.
#
#

ROOT=$(pwd)
OUTPUT_BASE="$ROOT/output"
mkdir -p "$OUTPUT_BASE"

N=$1
k=$2
n=$3
beta=$4
gamma=$5
rho=$6
q=$7
iters=1
n_graphs=1

# Create unique output folder
OUTPUT_DIR=$(mktemp -d "$OUTPUT_BASE/job_XXXX")

echo "N,beta,gamma,k,n,rho,q,runs,n_graphs" > "$OUTPUT_DIR/info.csv"
echo "$N,$beta,$gamma,$k,$n,$rho,$q,$iters,$n_graphs" >> "$OUTPUT_DIR/info.csv"

python3 src/rhbm/generate_matrix.py -n "$n" -p "$rho" -q "$q" -o "$OUTPUT_DIR"
python3 src/rhbm/rhbm_generate.py -N "$N" -k "$k" -n "$n" -b "$beta" -g "$gamma" -o "$OUTPUT_DIR" --n_runs "$iters" --n_graphs "$n_graphs" --delta "$OUTPUT_DIR/delta.csv" > "$OUTPUT_DIR/rhbm_log.txt"
