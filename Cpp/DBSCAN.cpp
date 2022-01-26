//
// Created by stanislaw on 08.01.2022.
//

#include <boost/program_options.hpp>
#include "point.h"
#include <queue>


int DBSCAN(const int &minPts) {
    int size = points.size();
    int cluster_number = 1;
    for (int i = 0; i < size; i++) {
        std::queue<int> seeds;
        bool extended_flag = false;

        if (clusters[i] == 0) {
            if (points.at(i).eps_neighborhood.size() >= minPts) {
                clusters[i] = cluster_number;
                points.at(i).type = core;
                for (int n: points.at(i).eps_neighborhood) {
                    seeds.push(n);
                    extended_flag = true;
                }
            }
        }
        while (!seeds.empty()) {
            if (clusters[seeds.front()] == 0) {
                clusters[seeds.front()] = cluster_number;
                if (points.at(seeds.front()).eps_neighborhood.size() >= minPts) {
                    points.at(seeds.front()).type = core;
                    for (int n: points.at(seeds.front()).eps_neighborhood) {
                        seeds.push(n);
                    }
                }
            }
            seeds.pop();
        }
        if (extended_flag)cluster_number++;
    }
    return cluster_number - 1;
}