#!/bin/bash

ROOT=$(pwd)
OUTPUT_BASE="$ROOT/output"
OUTPUT_DIR="$OUTPUT_BASE/$1"
mkdir -p $OUTPUT_DIR
n=$(sed -n '2p' "$OUTPUT_DIR/info.csv" | cut -f5 -d',')
log="${OUTPUT_DIR}/dmercator_log.txt"
j="${2:-0}"
nruns="$3"
ngraphs="$4"
edge_list="${OUTPUT_DIR}/edge_list_${j}.txt"
# python3 get_GC.py $edge_list
for d in 1 2 3 4 5; do
    for ((i=0;i<$nruns;i++)); do
        embedding="${OUTPUT_DIR}/embedding_dim_${d}_${i}"
        echo "running d-mercator for graph $j and dimension $d" >> "$log"
        cd $ROOT/src/d-mercator/ 
        ./mercator -d "$d" -o "$embedding" "$edge_list" >> "$log"
        cd $ROOT 
        python3 src/rhg/generate_SD_from_rhbm_embedding.py -i "${embedding}.inf_coord" -o "$embedding" -n "$n" -g "$ngraphs" >> "$log"
    done
done

