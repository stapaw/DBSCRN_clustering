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

namespace po = boost::program_options;
using namespace std;

double big_number = 9999999;
vector<point> points;
struct settings settings;
int clusters[100000] ={0};
bool visited[100000] = {false};
double reference_values[10000] = {big_number};


int main(int argc, char *argv[]) {
    clock_t start_time, input_read_time, sort_by_reference_point_time, rnn_neighbour_time, clustering_time, stats_calculation_time, output_write_time;
    const string clock_phases[] = {"read input file", "sort by reference point distances", "rnn calculation",
                                   "clustering", "stats calculation", "write output files", "total"};

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("in_file", po::value<string>()->default_value("../datasets/example.txt"),
             "input filename")
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

    write_to_stats_file(clock_phases, time_diffs, number_of_phases, vm, values, "STAT" + filename_suffix);
}

