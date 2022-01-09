//
// Created by stanislaw on 08.01.2022.
//

#include <boost/program_options.hpp>
#include "DBSCAN.h"
#include "settings.h"
#include "point.h"
#include "distance_calculations.h"
# include "csv.h"
#include <functional>
#include <json/writer.h>
#include <json/value.h>
#include <time.h>
#include <fstream>
#include <queue>
#include<cmath>
#include <iostream>
#include "DBSCRN.h"

int DBSCRN(int k) {
    std::vector<int> S_non_core;
    std::vector<int> S_core;
    int cluster_number = 1;
    for (int i = 0; i < points.size(); i++) {
        if (points.at(i).rnn.size() < k) S_non_core.push_back(i);
        else {
            S_core.push_back(i);
            points.at(i).type = core;
            int cluster_to_expand;
            if (clusters[i] != 0) cluster_to_expand = clusters[i];
            else {
                cluster_to_expand = cluster_number;
                cluster_number++;
            }
            DBSCRN_expand_cluster(i, k, cluster_to_expand);
        }
    }
    for (int non_core_point : S_non_core) {
        points.at(non_core_point).type = non_core;
        if (clusters[non_core_point] == 0) {
            clusters[non_core_point] = get_cluster_of_nearest_core_point(non_core_point, S_core);
        }
    }
    return cluster_number - 1;
}

int get_cluster_of_nearest_core_point(int point, const std::vector<int> &vector) {
//    TODO: calculate distances;
    return -1;
}

void DBSCRN_expand_cluster(int i, int k, int cluster_number) {
    std::vector<int> S_tmp;

    clusters[i] = cluster_number;
    S_tmp.push_back(i);
    visited[i] = true;
    for (int j = 0; j < S_tmp.size(); j++) {
        int y_k = S_tmp.at(j);
        for (int y_j: points.at(y_k).rnn) {
            //TODO: change for math pi value
            if (points.at(y_j).rnn.size() > (2 * k / 3.14)) {
                for (int p:points.at(y_j).rnn) {
                    if (!visited[p]) {
                        S_tmp.push_back(p);
                        visited[p] = true;
                    }
                }
            }
            if (clusters[y_j] == 0) {
                clusters[y_j] = cluster_number;
            }
        }
    }
}