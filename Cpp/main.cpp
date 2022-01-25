#include <iostream>
#include <fstream>
#include <ctime>
#include <functional>
#include "settings.h"
#include <vector>
#include "distance_calculations.h"
#include "point.h"
#include "DBSCAN.h"
#include "DBSCRN.h"
#include "output.h"
#include "stats.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

double big_number = 9999999;
vector<point> points;
struct settings settings;

double get_time_in_sec(clock_t from, clock_t to);

clock_t save_checkpoint_time(clock_t from, clock_t to, stats &stats);

int clusters[100000] = {0};
bool visited[100000] = {false};
double reference_values[10000] = {big_number};

int main(int argc, char *argv[]) {
    clock_t start_time, last_checkpoint_time;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            (INPUT_FILE_PARAM_NAME, po::value<string>()->default_value("../datasets/points/example.tsv"),
             "input file path")
            (LABELS_FILE_PARAM_NAME, po::value<string>()->default_value("../datasets/ground_truth/example.tsv"),
             "ground truth (cluster labels) file path")
            (ALGORITHM_PARAM_NAME, po::value<string>()->default_value("DBSCRN"), "algorithm name (DBSCAN|DBCSRN)")
            (K_PARAM_NAME, po::value<int>()->default_value(3), "number of nearest neighbors for DBSCRN")
            (EPS_PARAM_NAME, po::value<double>()->default_value(2), "eps parameter for DBSCAN")
            (MIN_PTS_PARAM_NAME, po::value<int>()->default_value(4), "minPts parameter for DBSCAN")
            (MINKOWSKI_PARAM_NAME, po::value<int>()->default_value(2), "Minkowski distance power")
            (TI_OPTIMIZED_PARAM_NAME, po::value<bool>()->default_value(true),
             "If true, TI optimized calculations are enabled (true|false)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    stats stats{};
    start_time = clock();
    ifstream InputFile(vm[INPUT_FILE_PARAM_NAME].as<string>());

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
    settings.minkowski_distance_order = vm[MINKOWSKI_PARAM_NAME].as<int>();
    settings.eps = vm[EPS_PARAM_NAME].as<double>();
    settings.minPts = vm[MIN_PTS_PARAM_NAME].as<int>();
    settings.k = vm[K_PARAM_NAME].as<int>();

    last_checkpoint_time = save_checkpoint_time(start_time, clock(), stats);

    int cluster_number;
    if (vm[ALGORITHM_PARAM_NAME].as<string>() == "DBSCAN") {
        if (vm[TI_OPTIMIZED_PARAM_NAME].as<bool>()) {
            point reference_point{};
            for (int i = 0; i < dimensions; i++) {
                reference_point.dimensions.push_back(reference_values[i]);
            }
            std::vector<distance_x> distances = sort_by_ref_point(reference_point);
            last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, clock(), stats);

            calculate_eps_neighborhood_optimized(vm[EPS_PARAM_NAME].as<double>(), distances);
        } else {
            calculate_eps_neighborhood(vm[EPS_PARAM_NAME].as<double>());
        }
        last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, clock(), stats);

        cluster_number = DBSCAN(vm[MIN_PTS_PARAM_NAME].as<int>());
    } else {
        int k = vm[K_PARAM_NAME].as<int>();
        if (vm[TI_OPTIMIZED_PARAM_NAME].as<bool>()) {
            point reference_point{};
            for (int i = 0; i < dimensions; i++) {
                reference_point.dimensions.push_back(reference_values[i]);
            }
            std::vector<distance_x> distances = sort_by_ref_point(reference_point);
            last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, clock(), stats);

            calculate_knn_optimized(k, distances);
        } else {
            last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, last_checkpoint_time, stats);

            calculate_knn(k);
        }
        last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, clock(), stats);

        cluster_number = DBSCRN(k);
    }
    last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, clock(), stats);

    string filename_suffix = get_filename_suffix(vm, point_number, dimensions);
    write_to_out_file(point_number, "OUT" + filename_suffix);
    write_to_debug_file(point_number, "DEBUG" + filename_suffix);


    vector<int> ground_truth;
    ifstream LabelsFile(vm[LABELS_FILE_PARAM_NAME].as<string>());
    int label;
    while (LabelsFile >> label)ground_truth.push_back(label);

    stats.point_number = point_number;
    stats.dimensions = dimensions;
    stats.cluster_number = cluster_number;


    stats = calculate_ground_truth_stats(stats, point_number, ground_truth);
    stats.silhouette = calculate_silhouette(point_number);
    stats.davies_bouldin = calculate_davies_bouldin(stats);

    double avg_distance_calculation_number = 0;
    int point_types[4] = {0};
    for (int i = 0; i < point_number; i++) {
        point p = points.at(i);
        avg_distance_calculation_number += p.distanceCalculationNumber;
        if (p.type == noise && clusters[i] != 0 && clusters[i] != -1) p.type = border;
        point_types[p.type]++;
    }

    stats.avg_dist_calculation = (double) avg_distance_calculation_number / point_number;
    stats.core_points = point_types[core];
    stats.non_core_points = point_types[non_core];
    stats.border_points = point_types[border];
    stats.noise_points = point_types[noise];

    last_checkpoint_time = save_checkpoint_time(last_checkpoint_time, clock(), stats);
    stats.time_diffs.push_back(get_time_in_sec(start_time, last_checkpoint_time)); // total

    write_to_stats_file(stats, vm, "STAT" + filename_suffix);
}

clock_t save_checkpoint_time(clock_t from, clock_t to, stats &stats) {
    double time_from_last_checkpoint = get_time_in_sec(from, to);
    stats.time_diffs.push_back(time_from_last_checkpoint);
    int index = stats.time_diffs.size()-1;
    cout << clock_phases[index] << " runtime: " << stats.time_diffs.at(index) << endl;
    return to;
}

double get_time_in_sec(clock_t from, clock_t to) {
    return (double) (to - from) / CLOCKS_PER_SEC;
}


