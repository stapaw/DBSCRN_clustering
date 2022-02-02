//
// Created by stanislaw on 08.01.2022.
//

#ifndef CLUSTERING_SETTINGS_H
#define CLUSTERING_SETTINGS_H
struct settings {
    int minkowski_distance_order = 2;
    double eps = 2;
    int minPts = 4;
    int k = 3;
};
extern settings settings;

static const char *const clock_phases[] = {
        "1_read_input_file",
        "2_sort_by_ref_point_distances",
        "3_eps_neighborhood/rnn_calculation",
        "4_clustering",
        "5_stats_calculation", "total_runtime"};

static const char *const K_PARAM_NAME = "k";
static const char *const EPS_PARAM_NAME = "eps";
static const char *const MIN_PTS_PARAM_NAME = "min_pts";
static const char *const ALGORITHM_PARAM_NAME = "algorithm";
static const char *const INPUT_FILE_PARAM_NAME = "input_file";
static const char *const MINKOWSKI_PARAM_NAME = "minkowski_power";
static const char *const TI_OPTIMIZED_PARAM_NAME = "ti";
static const char *const LABELS_FILE_PARAM_NAME = "ground_truth_file";
static const char *const SKIP_SILHOUETTE_PARAM_NAME = "skip_silhouette";
#endif //CLUSTERING_SETTINGS_H
