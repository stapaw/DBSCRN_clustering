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
#endif //CLUSTERING_SETTINGS_H
