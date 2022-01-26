//
// Created by stanislaw on 08.01.2022.
//
#include "DBSCRN.h"
#include "point.h"
#include<cmath>
#include <queue>


int DBSCRN(int k) {
    int size = points.size();
    int cluster_number = 1;
    std::vector<int> S_non_core;
    for (int i = 0; i < size; i++) {
        std::queue<int> seeds;
        bool extended_flag = false;

        if (clusters[i] == 0) {
            if (points.at(i).rnn.size() >= k) {
                clusters[i] = cluster_number;
                points.at(i).type = core;
                for (int n: points.at(i).rnn) {
                    seeds.push(n);
                    extended_flag = true;
                }
            }
            else{
                S_non_core.push_back(i);
            }
        }
        while (!seeds.empty()) {
            if (clusters[seeds.front()] == 0) {
                if (points.at(seeds.front()).rnn.size() >= k) {
                    clusters[seeds.front()] = cluster_number;
                    points.at(seeds.front()).type = core;
                    for (int n: points.at(seeds.front()).rnn) {
                        seeds.push(n);
                    }
                }
            }
            seeds.pop();
        }
        if (extended_flag)cluster_number++;
    }
    for (int non_core_point : S_non_core) {
        clusters[non_core_point] = get_cluster_of_nearest_core_point(non_core_point);
        }
    return cluster_number - 1;
}

int get_cluster_of_nearest_core_point(int point_id) {
//  currently implemented base method modification - points not having core point in neighbours are treated as outliers;
    for(int i=points.at(point_id).knn.size()-1; i>=0; i--){
        if(points.at(points.at(point_id).knn.at(i)).type == core){
            points.at(point_id).type = border;
            return clusters[points.at(points.at(point_id).knn.at(i)).id];
        }
    }
    // it is a noise/outlier point
    points.at(point_id).type = noise;
    return 0;
}