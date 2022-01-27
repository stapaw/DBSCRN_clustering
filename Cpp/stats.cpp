//
// Created by stanislaw on 10.01.2022.
//

#include <utility>
#include <boost/program_options.hpp>
#include "output.h"
#include "point.h"
#include "distance_calculations.h"
#include <json/value.h>
#include "stats.h"



int get_number_of_pairs(int point_number);

double calculate_davies_bouldin(stats stats) {
    map<int, point> cluster_centroids;
    map<int, double> cluster_distances;
    map<int, int> cluster_cardinalities;

    for (int i = 0; i <= stats.cluster_number; i++) {
        point p{};
        for (int j = 0; j < stats.dimensions; j++) {
            p.dimensions.push_back(0.0);
        }
        cluster_centroids[i] = p;
        cluster_cardinalities[i] = 0;
    }

    for (int i = 0; i < stats.point_number; i++) {
        point p = points.at(i);
        for (int j = 0; j < stats.dimensions; j++) {
            cluster_centroids[clusters[i]].dimensions[j] += p.dimensions.at(j);
        }
        cluster_cardinalities[clusters[i]]++;
    }

    for (auto &imap: cluster_centroids) {
        for (int j = 0; j < stats.dimensions; j++) {
            imap.second.dimensions[j] /= cluster_cardinalities[imap.first];
        }
    }
    cluster_distances[0] = 0;
    for (int i = 0; i < stats.point_number; i++) {
        point p = points.at(i);
        cluster_distances[clusters[i]] += calculate_distance(p, cluster_centroids[clusters[i]]);
    }

    for (auto &imap: cluster_distances) {
        imap.second /= cluster_cardinalities[imap.first];
    }
    double DS = 0;
    for (int i = 0; i <= stats.cluster_number; i++) {
        if(cluster_cardinalities[i] != 0) {
            double R_i = 0;
            for (int j = 0; j <= stats.cluster_number; j++) {
                if (cluster_cardinalities[j] != 0) {
                    if (i != j) {
                        double s_i = cluster_distances[i];
                        double s_j = cluster_distances[j];
                        double dist = calculate_distance(cluster_centroids[i], cluster_centroids[j]);
                        double R_ij = (s_i + s_j) / dist;
                        R_i = max(R_i, R_ij);
                    }
                }
            }
            DS += R_i;
        }
    }
    int cluster_number = cluster_cardinalities[0] > 0 ? stats.cluster_number +1:stats.cluster_number;
    return DS/cluster_number;
}

double calculate_silhouette(int point_number, const stats& stats) {

    double global_s_i = 0;
    for (int i = 0; i < point_number; i++) {
        int outlier_cluster_counter = stats.cluster_number;
        map<int, double> cluster_distances;
        map<int, double> cluster_cardinalities;
        for (int j = 0; j < point_number; j++){
//            outliers as separate single point clusters:
            if(clusters[j] == 0){
                outlier_cluster_counter++;
                cluster_distances[outlier_cluster_counter] = calculate_distance(points.at(i), points.at(j));
                cluster_cardinalities[outlier_cluster_counter] += 1;
            }
            else{
                if (i != j) {
                    cluster_distances[clusters[j]] += calculate_distance(points.at(i), points.at(j));
                    cluster_cardinalities[clusters[j]] += 1;
                }
            }
        }

        double a_i = clusters[i] == 0 ? 0:cluster_distances[clusters[i]] / cluster_cardinalities[clusters[i]];
        double b_i = big_number;
        for (auto const &imap: cluster_distances) {
            if (imap.first != clusters[i]) {
                b_i = min(b_i, clusters[i] == 0 ? 1:imap.second / cluster_cardinalities[imap.first]);
            }
        }

        double s_i = (b_i - a_i) / max(b_i, a_i);
        global_s_i = (global_s_i * i + s_i) / (i + 1);
    }
    return global_s_i;
}

stats calculate_ground_truth_stats(stats stats, int point_number, const vector<int> &labels) {
    map<pair<int, int>, int> clustering;
    map<int, int> purity_values;
    vector<pair<int, int>> keys;
    vector<int> vints;

    for (int i = 0; i < point_number; i++) {
        clustering[make_pair(labels.at(i), clusters[i])] += 1;
    }

    for (auto const &imap: clustering) {
        keys.push_back(imap.first);
        vints.push_back(imap.second);
    }

    int size = keys.size();

    int TP = 0;
    for (int i = 0; i < size; i++) {
        if (vints.at(i) > 1) {
            TP += get_number_of_pairs(vints.at(i));
        }
        purity_values[keys.at(i).first] = max(purity_values[keys.at(i).first], vints.at(i));
    }

    int TN = 0;
    for (int i = 0; i < size; i++) {
        for (int j = i+1; j < size; j++) {
            if ((keys.at(i).first != keys.at(j).first) && (keys.at(i).second != keys.at(j).second)) {
                TN += vints.at(i) * vints.at(j);
            }
        }
    }

    int purity = 0;
    for (auto const &imap: clustering) {
        if (imap.second == purity_values[imap.first.first])purity += imap.second;
    }

    stats.TP = TP;
    stats.TN = TN;
    stats.number_of_pairs = get_number_of_pairs(point_number);
    stats.rand = (double) (TP + TN) / stats.number_of_pairs;
    stats.purity = (double) purity / point_number;
    return stats;
}

int get_number_of_pairs(int point_number) { return (point_number * (point_number - 1)) / 2; }