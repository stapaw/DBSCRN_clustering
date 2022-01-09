//
// Created by stanislaw on 08.01.2022.
//

#ifndef CLUSTERING_DBSCRN_H
#define CLUSTERING_DBSCRN_H

#include <vector>

int get_cluster_of_nearest_core_point(int point, const std::vector<int> &vector);

void DBSCRN_expand_cluster(int i, int k, int cluster_number);

int DBSCRN(int k);
#endif //CLUSTERING_DBSCRN_H
