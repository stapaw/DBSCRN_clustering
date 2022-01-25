//
// Created by stanislaw on 08.01.2022.
//

#include <boost/program_options.hpp>
# include "csv.h"
#include <functional>
#include <json/writer.h>
#include <json/value.h>
#include <time.h>
#include <fstream>
#include <queue>
#include<cmath>
#include <iostream>
#include "distance_calculations.h"
#include "point.h"
#include "settings.h"



double calculate_distance(const point &point, const struct point &other) {
    int dimension = point.dimensions.size();
    double dist = 0;
    for (int i = 0; i < dimension; i++) {
        double diff = point.dimensions.at(i) - other.dimensions.at(i);
        dist += pow(diff, settings.minkowski_distance_order);
    }
    return pow(dist, ((double)1/settings.minkowski_distance_order)); // sqrt is much faster
}

std::vector<distance_x> calculate_distances_for_knn(point p, int distance_number) {
    std::vector<distance_x> distances;
    for (int j = 0; j < distance_number; j++) {
        point other = points.at(j);
        double dist = calculate_distance(p, other);
        distance_x distance;
        distance.id = j;
        distance.dist = dist;
        distances.push_back(distance);
    }
    return distances;
}

void calculate_knn(int k) {
    int size = points.size();
    for (int i = 0; i < size; i++) {
        std::vector<distance_x> distances = calculate_distances_for_knn(points.at(i), size);
        sort(distances.begin(), distances.end(), dist_comparator());

        points.at(i).distanceCalculationNumber += size;
        for (int j = 1; j <= k; j++) {
            if (points.at(i).id != distances.at(j).id) {
                points.at(i).knn.push_back(distances.at(j).id);

                points.at(distances.at(j).id).rnn.push_back(i);
            }
        }
    }
}

void calculate_eps_neighborhood(double eps) {
    int size = points.size();
    for (int i = 0; i < size; i++) {
        std::vector<distance_x> distances = calculate_distances_for_knn(points.at(i), points.size());

        sort(distances.begin(), distances.end(), dist_comparator());
        points.at(i).distanceCalculationNumber += size;
        int j = 0;
        while ((j < size) && (distances.at(j).dist <= eps)) {
            points.at(i).eps_neighborhood.push_back(distances.at(j).id);
            j++;
        }
    }
}

void calculate_knn_optimized(std::vector<distance_x> distances, int distance_idx, int point_id, int k) {
    int max_index = distances.size() - 1;
    std::vector<distance_x> k_dist;
    int p_idx = distance_idx;
    std::priority_queue<distance_x, std::vector<distance_x>, dist_comparator> queue(k_dist.begin(), k_dist.end());
    int up = 1;
    int down = 1;
    for (int i = 0; i < k; i++) {
        int up_idx = p_idx - up;
        int down_idx = p_idx + down;
        double up_value, down_value;

        if (up_idx < 0) up_value = big_number;
        else up_value = distances.at(p_idx).dist - distances.at(up_idx).dist;

        if (down_idx > max_index) down_value = big_number;
        else down_value = distances.at(down_idx).dist - distances.at(p_idx).dist;

        point other;
        if (up_value < down_value) {
            other = points.at(distances.at(up_idx).id);
            up++;
        } else {
            other = points.at(distances.at(down_idx).id);
            down++;
        }
        distance_x distance;
        distance.id = other.id;
        distance.dist = calculate_distance(points.at(point_id), other);
        queue.push(distance);
    }
    points.at(point_id).distanceCalculationNumber += k;
    double radius = queue.top().dist;
    points.at(point_id).max_eps = radius;

    while (1) {
        int up_idx = p_idx - up;
        int down_idx = p_idx + down;
        double up_value, down_value;

        if (up_idx < 0) up_value = big_number;
        else up_value = distances.at(p_idx).dist - distances.at(up_idx).dist;

        if (down_idx > max_index) down_value = big_number;
        else down_value = distances.at(down_idx).dist - distances.at(p_idx).dist;

        point other;
        if (up_value < down_value) {
            if (up_value < radius) {
                other = points.at(distances.at(up_idx).id);
                up++;
            } else break;
        } else {
            if (down_value < radius) {
                other = points.at(distances.at(down_idx).id);
                down++;
            } else break;
        }
        distance_x distance;
        distance.id = other.id;
        distance.dist = calculate_distance(points.at(point_id), other);
        points.at(point_id).distanceCalculationNumber++;

        queue.push(distance);
        queue.pop();
        radius = queue.top().dist;
    }
    points.at(point_id).min_eps = radius;

    for (int j = 0; j < k; j++) {
        if (point_id != queue.top().id) {
            points.at(point_id).knn.push_back(queue.top().id);

            points.at(queue.top().id).rnn.push_back(point_id);
        }
        queue.pop();
    }
}

void calculate_eps_neighborhood_optimized(const double &eps, std::vector<distance_x> distances) {
    int size = distances.size();
    for (int i = 0; i < size; i++) {
        calculate_eps_neighborhood_optimized(distances, i, distances.at(i).id, eps);
    }
}

void
calculate_eps_neighborhood_optimized(std::vector<struct distance_x> reference_distances, int distance_idx, int point_id,
                                     double eps) {

    int j = distance_idx;
    points.at(point_id).distanceCalculationNumber +=1;
    while ((j >= 0) && ((reference_distances.at(distance_idx).dist - reference_distances.at(j).dist) <= eps)) {
        double dist = calculate_distance(points.at(point_id), points.at(reference_distances.at(j).id));
        points.at(point_id).distanceCalculationNumber +=1;
        if (dist <= eps) {
            points.at(point_id).eps_neighborhood.push_back(reference_distances.at(j).id);
        }
        j--;
    }

    j = distance_idx + 1;
    int max_index = reference_distances.size() - 1;
    while ((j <= max_index) && ((reference_distances.at(j).dist - reference_distances.at(distance_idx).dist) <= eps)) {
        double dist = calculate_distance(points.at(point_id), points.at(reference_distances.at(j).id));
        points.at(point_id).distanceCalculationNumber +=1;
        if (dist <= eps) {
            points.at(point_id).eps_neighborhood.push_back(reference_distances.at(j).id);
        }
        j++;
    }
}

void calculate_knn_optimized(int k, std::vector<distance_x> distances) {
    int size = distances.size();
    for (int i = 0; i < size; i++) {
        calculate_knn_optimized(distances, i, distances.at(i).id, k);
    }
}

std::vector<distance_x> sort_by_ref_point(const point &reference_point) {
    std::vector<distance_x> distances = calculate_distances_for_knn(reference_point, points.size());
    sort(distances.begin(), distances.end(), dist_comparator());
    return distances;
}
