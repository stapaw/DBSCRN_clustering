#!/bin/bash

function grid_search () {
  dataset=$1
  eps=$2
  min_pts=$3

  python Python/run.py \
    -a dbscan \
    -d "datasets/points/${dataset}" \
    -o grid_search \
    --eps ${eps} \
    --min_samples ${min_pts} \
    --plot \
    --cache
}

dataset="complex9.tsv"
grid_search $dataset 10 15

dataset="dim512.tsv"
grid_search $dataset 10 15

dataset="cluto-t7-10k.tsv"
grid_search $dataset 10 15

dataset="letter.tsv"
grid_search $dataset 10 15

dataset="example.tsv"
grid_search $dataset 10 15
