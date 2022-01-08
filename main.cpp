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
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

double big_number = 9999999;
vector<point> points;
int clusters[100000] = {0};
bool visited[100000] = {false};
double reference_values[10000] = {big_number};
const string SEPARATOR = ",";

void DBSCRN_expand_cluster(int i, int k, int cluster_number);

int get_cluster_of_nearest_core_point(int point, const vector<int> &vector);

Json::Value
prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, po::variables_map vm,
                   map<string, double> point_number);

void write_to_out_file(int point_number);

void write_to_debug_file(int point_number);

void
write_to_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, po::variables_map vm,
                    map<string, double> point_number);

int DBSCAN(const int &minPts);


int DBSCRN(int k) {
    vector<int> S_non_core;
    vector<int> S_core;
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

int get_cluster_of_nearest_core_point(int point, const vector<int> &vector) {
//    TODO: calculate distances;
    return -1;
}

void DBSCRN_expand_cluster(int i, int k, int cluster_number) {
    vector<int> S_tmp;

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

    write_to_out_file(point_number);

    write_to_debug_file(point_number);

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

    write_to_stats_file(clock_phases, time_diffs, number_of_phases, vm, values);
}

int DBSCAN(const int &minPts) {
    int size = points.size();
    int cluster_number = 1;
    for (int i = 0; i < size; i++) {
        queue<int> seeds;
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
                    points.at(i).type = core;
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

void
write_to_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, po::variables_map vm,
                    map<string, double> values) {
    ofstream StatsFile("../stats_file");
    Json::Value stats = prepare_stats_file(clock_phases, time_diffs, number_of_phases, std::move(vm), values);
    StatsFile << stats;
    StatsFile.close();
}

void write_to_debug_file(int point_number) {
    ofstream DebugFile("../debug_file");

    DebugFile << "id" << SEPARATOR << "max_eps" << SEPARATOR << "min_eps" << SEPARATOR << "|rnn|" << SEPARATOR;
    DebugFile << "[knn]" << SEPARATOR << "[rnn]" << endl;

    for (int i = 0; i < point_number; i++) {
        DebugFile << i << SEPARATOR
                  << points.at(i).max_eps << SEPARATOR
                  << points.at(i).min_eps << SEPARATOR
                  << points.at(i).rnn.size() << SEPARATOR;
        DebugFile << "[ ";
        for (int j: points.at(i).knn) {
            DebugFile << j << " ";
        }
        DebugFile << "]" << SEPARATOR << "[ ";
        for (int j: points.at(i).rnn) {
            DebugFile << j << " ";;
        }
        DebugFile << "]" << endl;
    }
    DebugFile.close();
}

void write_to_out_file(int point_number) {
    int dimensions = points.at(0).dimensions.size();
    ofstream OutFile("../out_file");
//    header
    OutFile << "id" << SEPARATOR;
    for (int j = 0; j < dimensions; j++) OutFile << "d" << j << SEPARATOR;
    OutFile << "distance_calculations" << SEPARATOR << "is_core" << SEPARATOR << "cluster_id" << endl;

    for (int i = 0; i < point_number; i++) {
        OutFile << i << SEPARATOR;
        for (int j = 0; j < dimensions; j++) {
            OutFile << points.at(i).dimensions.at(j) << SEPARATOR;
        }
        OutFile << points.at(i).distanceCalculationNumber << SEPARATOR
                << (points.at(i).type == core) << SEPARATOR
                << clusters[i] << endl;
    }
    OutFile.close();
}

Json::Value
prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, po::variables_map vm,
                   map<string, double> values) {
    Json::Value stats;
    Json::Value reference_point_values(Json::arrayValue); // TODO
    reference_point_values.append(Json::Value(4.2));
    reference_point_values.append(Json::Value(4));


    stats["main"]["input_filename"] = vm["in_file"].as<string>();
    stats["main"]["#_of_point_dimensions"] = values["#_of_point_dimensions"];
    stats["main"]["#_of_points"] = values["#_of_points"];

    stats["parameters"]["algorithm"] = vm["alg"].as<string>();
    stats["parameters"]["k"] = vm["k"].as<int>();
    stats["parameters"]["Eps"] = vm["eps"].as<double>();
    stats["parameters"]["minPts"] = vm["minPts"].as<int>();

    stats["main"]["#_clusters"] = values["#_clusters"];
    stats["main"]["#_noise_points"] = 0; // TODO
    stats["main"]["#_border_points"] = 0; // TODO
    stats["main"]["#_core_points"] = values["#_core_points"];
    stats["main"]["#non_core_points"] = values["#non_core_points"];
    stats["clustering_stats"]["avg_#_of_distance_calculation"] = 0; // TODO

    stats["clustering_stats"]["silhouette_coefficient"] = 0; // TODO
    stats["clustering_stats"]["davies_bouldin"] = 0; // TODO
//time stats
    for (int i = 0; i < number_of_phases; i++)
        stats["time_stats"][clock_phases[i]] = time_diffs[i];


    // if real cluster known
    stats["clustering_stats"]["TP"] = 0; // TODO
    stats["clustering_stats"]["TN"] = 0; // TODO
    stats["clustering_stats"]["#_of_pairs"] = 0; // TODO ???
    stats["clustering_stats"]["RAND"] = 0; // TODO
    stats["clustering_stats"]["Purity"] = 0; // TODO

    cout << stats << endl;
    return stats;
}
