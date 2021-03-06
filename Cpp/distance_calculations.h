//
// Created by stanislaw on 08.01.2022.
//

#ifndef CLUSTERING_DISTANCE_CALCULATIONS_H
#define CLUSTERING_DISTANCE_CALCULATIONS_H

#include "point.h"

extern double big_number;


struct distance_x {
    int id{};
    double dist{};
};

struct dist_comparator {
    inline bool operator()(const distance_x &struct1, const distance_x &struct2) {
        return (struct1.dist < struct2.dist);
    }
};

void calculate_eps_neighborhood(double eps);

void calculate_eps_neighborhood_optimized(const double &eps, std::vector<struct distance_x> distances);

void
calculate_eps_neighborhood_optimized(std::vector<struct distance_x> reference_distances, int distance_idx, int point_id,
                                     double eps);

void calculate_knn(int k);

void calculate_knn_optimized(int k, std::vector<struct distance_x> distances);

double calculate_distance(const point &point, const struct point &other);

std::vector<distance_x> sort_by_ref_point(const point &reference_point);

#endif //CLUSTERING_DISTANCE_CALCULATIONS_H
