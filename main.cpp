#include <iostream>
#include<cmath>
#include <queue>
#include <fstream>
#include <time.h>
#include <json/value.h>
#include <json/writer.h>
#include <functional>


# include "csv.h"
#include "distance_calculations.h"
#include "point.h"
#include "settings.h"
#include "DBSCAN.h"
#include "DBSCRN.h"
#include "output.h"
#include <boost/program_options.hpp>
#include <utility>

namespace po = boost::program_options;
using namespace std;

double big_number = 9999999;
vector<point> points;
struct settings settings;
int clusters[100000] = {0};
bool visited[100000] = {false};
double reference_values[10000] = {big_number};

struct stats {
    int point_number;
    int dimensions;
    int cluster_number;
    int TP;
    int TN;
    int number_of_pairs;
    double rand;
    double purity;
    double silhouette;
    double davies_bouldin;
};

stats calculate_ground_truth_stats(stats stats, int point_number, const vector<int> &vector);

int get_number_of_pairs(int point_number);

double calculate_silhouette(int i);

double calculate_davies_bouldin(stats stats);

int main(int argc, char *argv[]) {
    clock_t start_time, input_read_time, sort_by_reference_point_time, rnn_neighbour_time, clustering_time, stats_calculation_time, output_write_time;
    const string clock_phases[] = {"read input file", "sort by reference point distances", "rnn calculation",
                                   "clustering", "stats calculation", "write output files", "total"};

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("in_file", po::value<string>()->default_value("../datasets/example.txt"),
             "input filename")
            ("ground_truth_file", po::value<string>()->default_value("../datasets/ground_truth/example.txt"),
             "ground truth (cluster labels) filename")
            ("alg", po::value<string>()->default_value("DBSCAN"), "algorithm name (DBSCAN|DBCSRN)")
            ("k", po::value<int>()->default_value(3), "number of nearest neighbors")
            ("eps", po::value<double>()->default_value(2), "eps parameter for DBSCAN")
            ("minPts", po::value<int>()->default_value(4), "minPts parameter for DBSCAN")
            ("minkowski_order", po::value<int>()->default_value(2), "Minkowski distance order")
            ("optimized", po::value<bool>()->default_value(true), "run optimized version");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    start_time = clock();
    ifstream InputFile(vm["in_file"].as<string>());

    int point_number, dimensions;
    InputFile >> point_number >> dimensions;

    for (int i = 0; i < point_number; i++) {
        point p;
        p.id = i;
        for (int j = 0; j < dimensions; j++) {
            double v;
            InputFile >> v;
            p.dimensions.push_back(v);
            if (v < reference_values[j]) {
                reference_values[j] = v;
            }
        }
        points.push_back(p);
    }
    settings.minkowski_distance_order = vm["minkowski_order"].as<int>();
    settings.eps = vm["eps"].as<double>();
    settings.minPts = vm["minPts"].as<int>();
    settings.k = vm["k"].as<int>();

    input_read_time = clock();
    int cluster_number;
    if (vm["alg"].as<string>() == "DBSCAN") {
        if (vm["optimized"].as<bool>()) {
            point reference_point;
            for (int i = 0; i < dimensions; i++) {
                reference_point.dimensions.push_back(reference_values[i]);
            }
            calculate_eps_neighborhood_optimized(vm["eps"].as<double>(), reference_point);
        } else {
            calculate_eps_neighborhood(vm["eps"].as<double>());
        }
        cluster_number = DBSCAN(vm["minPts"].as<int>());
    } else {
        int k = vm["k"].as<int>();
        if (vm["optimized"].as<bool>()) {
            point reference_point;
            for (int i = 0; i < dimensions; i++) {
                reference_point.dimensions.push_back(reference_values[i]);
            }
//        vector<distance_x> distances = calculate_distances_for_knn(reference_point, points.size());
//        sort(distances.begin(), distances.end(), dist_comparator());
//        TODO: not real value - update code
            sort_by_reference_point_time = clock();
            calculate_knn_optimized(k, reference_point);
        } else {
            sort_by_reference_point_time = clock();
            calculate_knn(k);
        }
        rnn_neighbour_time = clock();


        cluster_number = DBSCRN(k);
    }
    for (int i = 0; i < point_number; i++) {
        point p = points.at(i);
        cout << endl;
        cout << p.id << " " << p.dimensions.at(0) << " " << p.dimensions.at(1) << endl;
    }
    clustering_time = clock();
    stats_calculation_time = clock();

    string filename_suffix = get_filename_suffix(vm, point_number, dimensions);
    write_to_out_file(point_number, "OUT" + filename_suffix);
    write_to_debug_file(point_number, "DEBUG" + filename_suffix);

    output_write_time = clock();
    double time_diffs[] = {(double) (input_read_time - start_time) / CLOCKS_PER_SEC,
                           (double) (sort_by_reference_point_time - input_read_time) / CLOCKS_PER_SEC,
                           (double) (rnn_neighbour_time - sort_by_reference_point_time) / CLOCKS_PER_SEC,
                           (double) (clustering_time - rnn_neighbour_time) / CLOCKS_PER_SEC,
                           (double) (stats_calculation_time - clustering_time) / CLOCKS_PER_SEC,
                           (double) (output_write_time - stats_calculation_time) / CLOCKS_PER_SEC,
                           (double) (clock() - start_time) / CLOCKS_PER_SEC
    };
    int number_of_phases = sizeof(clock_phases) / sizeof(*clock_phases);
    for (int i = 0; i < number_of_phases; i++)cout << clock_phases[i] << ": " << time_diffs[i] << endl;

    map<std::string, double> values{
            {"#_of_points",           point_number},
            {"#_of_point_dimensions", dimensions},
            {"#_clusters",            cluster_number},
//            {"#_core_points",         core_points_number},
//            {"#_non_core_points",     non_core_points_number}
    };

    stats stats{};
    vector<int> ground_truth;
    ifstream LabelsFile(vm["ground_truth_file"].as<string>());
    int label;
    while (LabelsFile >> label)ground_truth.push_back(label);

    stats = calculate_ground_truth_stats(stats, point_number, ground_truth);
    stats.silhouette = calculate_silhouette(point_number);
    stats.davies_bouldin = calculate_davies_bouldin(stats);
    stats.point_number = point_number;
    stats.dimensions = dimensions;
    stats.cluster_number = cluster_number;
    cout << stats.number_of_pairs << " " << "TP: " << stats.TP << " TN: " << stats.TN << " purity: " << stats.purity
         << endl;

    write_to_stats_file(clock_phases, time_diffs, number_of_phases, vm, values, "STAT" + filename_suffix);
}

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

    for (int i = 0; i < stats.point_number; i++) {
        point p = points.at(i);
        cluster_distances[clusters[i]] += calculate_distance(p, cluster_centroids[clusters[i]]);
    }

    for (auto &imap: cluster_distances) {
        imap.second /= cluster_cardinalities[clusters[imap.first]];
    }
    double DS = 0;
    for (int i = 0; i <= stats.cluster_number; i++) {
        double R_i = 0;
        for (int j = 0; j <= stats.cluster_number; j++) {
            if(i != j){
                double s_i = cluster_distances[i];
                double s_j = cluster_distances[j];
                double dist = calculate_distance(cluster_centroids[i], cluster_centroids[j]);
                double R_ij = (s_i + s_j)/dist;
                R_i = max(R_i, R_ij);
            }
        }
        DS += R_i;
    }
//TODO: zero cluster as outlier? - fix wrong division numbers
    return DS/stats.cluster_number;
}

double calculate_silhouette(int point_number) {
    map<int, double> cluster_distances;
    map<int, double> cluster_cardinalities;
    double global_s_i = 0;

    for (int i = 0; i < point_number; i++) {
        for (int j = 0; j < point_number; j++)
            if (i != j) {
                cluster_distances[clusters[j]] += calculate_distance(points.at(i), points.at(j));
                cluster_cardinalities[clusters[j]] += 1;
            }
        double a_i = cluster_distances[clusters[i]] / cluster_cardinalities[clusters[i]];
        double b_i = big_number;
        for (auto const &imap: cluster_distances) {
            if (imap.first != clusters[i]) {
                b_i = min(b_i, imap.second / cluster_cardinalities[imap.first]);
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
    std::vector<pair<int, int>> keys;
    std::vector<int> vints;

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
        for (int j = 1; j < size; j++) {
            if (keys.at(i) != keys.at(j)) {
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

