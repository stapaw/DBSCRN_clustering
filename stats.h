//
// Created by stanislaw on 10.01.2022.
//

#ifndef CLUSTERING_STATS_H
#define CLUSTERING_STATS_H

#include <vector>

struct stats {
    int point_number;
    int dimensions;
    int cluster_number;
    int TP;
    int TN;
    int number_of_pairs;
    double rand;
    double purity;
    double silhouette;
    double davies_bouldin;
    int noise_points;
    int border_points;
    int core_points;
    int non_core_points;
    int avg_dist_calculation;
};

double calculate_davies_bouldin(stats stats);

double calculate_silhouette(int i);

stats calculate_ground_truth_stats(stats stats, int point_number, const std::vector<int> &vector);
#endif //CLUSTERING_STATS_H
