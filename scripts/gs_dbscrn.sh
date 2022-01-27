#!/bin/bash

function grid_search () {
  dataset=$1
  k=$2

  python Python/run.py \
    -a dbscrn \
    --ti \
    -d "datasets/points/${dataset}" \
    -o grid_search \
    -k ${k} \
    --plot \
    --cache
}

dataset="complex9.tsv"
grid_search $dataset 20
grid_search $dataset 21
grid_search $dataset 22
grid_search $dataset 23
grid_search $dataset 24
grid_search $dataset 25
grid_search $dataset 26
grid_search $dataset 27
grid_search $dataset 28
grid_search $dataset 29
grid_search $dataset 30

dataset="dim512.tsv"
grid_search $dataset 50 55 60 65 70

dataset="cluto-t7-10k.tsv"
grid_search $dataset 20
grid_search $dataset 21
grid_search $dataset 22
grid_search $dataset 23
grid_search $dataset 24
grid_search $dataset 25
grid_search $dataset 26
grid_search $dataset 27
grid_search $dataset 28
grid_search $dataset 29
grid_search $dataset 30

dataset="letter.tsv"
grid_search $dataset 350 400 450 500 550 600 650 700 750 800

dataset="example.tsv"
grid_search $dataset 3
