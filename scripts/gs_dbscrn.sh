#!/bin/bash

set -e

function grid_search () {
  dataset=$1
  k_candidates=$2

  for k in ${k_candidates}; do
    ./DBSCRN_clustering \
      --input_file "datasets/points/${dataset}" \
      --ground_truth_file "datasets/ground_truth/${dataset}" \
      --algorithm DBSCRN \
      --k ${k} \
      --minkowski_power 2 \
      --TI_optimized true
done;
}

DATASETS="cluto-t7-10k.tsv complex9.tsv dim512.tsv example.tsv letter.tsv"

grid_search "example.tsv" "3 4 5 6"
grid_search "complex9.tsv" "5 10 15 20"
