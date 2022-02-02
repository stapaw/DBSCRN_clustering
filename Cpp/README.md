Build from sources using CMakeLists.txt file (preferably in CLion IDE):

1) Install jsoncpp and boost_program_options libraries

2) Update paths to libraries in CMakeLists.txt

3) Build from sources using cmake:
https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html#seealso

All commands used to generate cpp output files in the results folder:

```
./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/example.tsv \
--ground_truth_file ../datasets/ground_truth/example.tsv \
--min_pts 4 \
--eps 2

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/example.tsv \
--ground_truth_file ../datasets/ground_truth/example.tsv \
--min_pts 4 \
--eps 2 \
--ti true

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/complex9.tsv \
--ground_truth_file ../datasets/ground_truth/complex9.tsv \
--min_pts 27 \
--eps 23

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/complex9.tsv \
--ground_truth_file ../datasets/ground_truth/complex9.tsv \
--min_pts 27 \
--eps 23 \
--ti true

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/cluto-t7-10k.tsv \
--ground_truth_file ../datasets/ground_truth/cluto-t7-10k.tsv \
--min_pts 35 \
--eps 15

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/cluto-t7-10k.tsv \
--ground_truth_file ../datasets/ground_truth/cluto-t7-10k.tsv \
--min_pts 35 \
--eps 15 \
--ti true

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/dim512.tsv \
--ground_truth_file ../datasets/ground_truth/dim512.tsv \
--min_pts 13 \
--eps 60

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/dim512.tsv \
--ground_truth_file ../datasets/ground_truth/dim512.tsv \
--min_pts 13 \
--eps 60 \
--ti true

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/letter.tsv \
--ground_truth_file ../datasets/ground_truth/letter.tsv \
--min_pts 39 \
--eps 3.6 \
--skip_silhouette true

./Clustering \
--algorithm DBSCAN \
--input_file ../datasets/points/letter.tsv \
--ground_truth_file ../datasets/ground_truth/letter.tsv \
--min_pts 39 \
--eps 3.6 \
--ti true \
--skip_silhouette true


./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/example.tsv \
--ground_truth_file ../datasets/ground_truth/example.tsv \
--k 3

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/example.tsv \
--ground_truth_file ../datasets/ground_truth/example.tsv \
--k 3 \
--ti true


./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/complex9.tsv \
--ground_truth_file ../datasets/ground_truth/complex9.tsv \
--k 27

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/complex9.tsv \
--ground_truth_file ../datasets/ground_truth/complex9.tsv \
--k 27 \
--ti true

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/cluto-t7-10k.tsv \
--ground_truth_file ../datasets/ground_truth/cluto-t7-10k.tsv \
--k 35

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/cluto-t7-10k.tsv \
--ground_truth_file ../datasets/ground_truth/cluto-t7-10k.tsv \
--k 35 \
--ti true

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/dim512.tsv \
--ground_truth_file ../datasets/ground_truth/dim512.tsv \
--k 13

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/dim512.tsv \
--ground_truth_file ../datasets/ground_truth/dim512.tsv \
--k 13 \
--ti true

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/letter.tsv \
--ground_truth_file ../datasets/ground_truth/letter.tsv \
--k 39 \
--skip_silhouette true

./Clustering \
--algorithm DBSCANRN \
--input_file ../datasets/points/letter.tsv \
--ground_truth_file ../datasets/ground_truth/letter.tsv \
--k 39 \
--ti true \
--skip_silhouette true

```