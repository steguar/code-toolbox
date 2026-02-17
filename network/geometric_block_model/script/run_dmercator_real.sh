#!/bin/bash

ROOT=$(pwd)
INPUT_DIR="$ROOT/graphs/$1"
OUTPUT_BASE="$ROOT/output"
OUTPUT_DIR="$OUTPUT_BASE/$1"
mkdir -p $OUTPUT_DIR
log="${OUTPUT_DIR}/dmercator_log.txt"
nruns="$2"
ngraphs="$3"
com_att="$4"
edge_list="${INPUT_DIR}/graph.ncol"
# python3 get_GC.py $edge_list
for d in 1 2 3 4 5; do
    for ((i=0;i<$nruns;i++)); do
        embedding="${OUTPUT_DIR}/embedding_dim_${d}_${i}"
        echo "running d-mercator for graph $1 and dimension $d" >> "$log"
	cd $ROOT/src/d-mercator/ 
        ./mercator -d "$d" -o "$embedding" "$edge_list" >> "$log"
        cd $ROOT
	python3 src/rhg/generate_SD_from_real_embedding.py -i "$INPUT_DIR" -e "$OUTPUT_DIR" -o "$embedding" -c "$com_att" -g "$ngraphs" >> "$log"
    done
done

