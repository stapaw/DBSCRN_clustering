//
// Created by stanislaw on 08.01.2022.
//

#ifndef CLUSTERING_POINT_H
#define CLUSTERING_POINT_H

enum point_type {
    noise, border, core
};

struct point {
    int id{};
    int distanceCalculationNumber = 0;
    double max_eps;
    double min_eps;
    point_type type = noise;
    std::vector<double> dimensions;
    std::vector<int> knn;
    std::vector<int> rnn;
    std::vector<int> eps_neighborhood;
};
extern std::vector<point> points;
extern int clusters[100000];
extern double reference_values[10000];

#endif //CLUSTERING_POINT_H
