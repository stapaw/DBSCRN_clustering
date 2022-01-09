//
// Created by stanislaw on 08.01.2022.
//

#ifndef CLUSTERING_POINT_H
#define CLUSTERING_POINT_H

enum point_type {
    core, non_core, border, noise
};

struct point {
    int id{};
    int distanceCalculationNumber = 0;
    double max_eps;
    double min_eps;
    point_type type;
    std::vector<double> dimensions;
    std::vector<int> knn;
    std::vector<int> rnn;
    std::vector<int> eps_neighborhood;
};
extern std::vector<point> points;
extern int clusters[100000];
extern bool visited[100000];

#endif //CLUSTERING_POINT_H
