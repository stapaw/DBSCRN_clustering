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

static const char *const K_PARAM_NAME = "k";
static const char *const EPS_PARAM_NAME = "eps";
static const char *const MIN_PTS_PARAM_NAME = "minPts";
static const char *const ALGORITHM_PARAM_NAME = "algorithm";
static const char *const INPUT_FILE_PARAM_NAME = "input_file";
static const char *const MINKOWSKI_PARAM_NAME = "minkowski_power";
static const char *const TI_OPTIMIZED_PARAM_NAME = "TI_optimized";
static const char *const LABELS_FILE_PARAM_NAME = "ground_truth_file";
#endif //CLUSTERING_SETTINGS_H
