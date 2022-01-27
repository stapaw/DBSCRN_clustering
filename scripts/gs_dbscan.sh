#!/bin/bash

set -e

function grid_search () {
  dataset=$1
  eps_candidates=$2
  min_pts_candidates=$3

  for eps in ${eps_candidates}; do
    for min_pts in ${min_pts_candidates}; do
      ./DBSCRN_clustering \
        --input_file "datasets/points/${dataset}" \
        --ground_truth_file "datasets/ground_truth/${dataset}" \
        --algorithm DBSCAN \
        --eps ${eps} \
        --minPts ${min_pts} \
        --minkowski_power 2
    done;
  done;
}

DATASETS="cluto-t7-10k.tsv complex9.tsv dim512.tsv example.tsv letter.tsv"

grid_search "example.tsv" "1 2 3" "4 5 6"
grid_search "complex9.tsv" "1 2 3" "5 10 15"
