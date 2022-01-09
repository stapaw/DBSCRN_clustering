//
// Created by stanislaw on 09.01.2022.
//

#include <boost/program_options.hpp>
#include "DBSCRN.h"
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
#include "output.h"

const string SEPARATOR = ",";

vector<string> split(const string &s, char by);

Json::Value
prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, boost::program_options::variables_map vm,
                   std::map<string, double> point_number);

string get_filename_suffix(const boost::program_options::variables_map &vm, int point_number, int dimensions) {
    string filename_suffix = "_";
    filename_suffix += vm["optimized"].as<bool>() ? "Opt-" : "";
    filename_suffix += vm["alg"].as<string>() + "_";
    std::vector<string> splitted = split(vm["in_file"].as<string>(), '.');
    splitted = split(splitted.at(splitted.size() - 2), '/');
    filename_suffix += splitted.at(splitted.size() - 1);
    filename_suffix += "_D" + to_string(dimensions) + "_R" + to_string(point_number);
    if (vm["alg"].as<string>() == "DBSCAN") {
        filename_suffix += "_m" + to_string(vm["minPts"].as<int>()) + "_e" + to_string(vm["eps"].as<double>());
    } else {
        filename_suffix += "_k" + vm["k"].as<string>();
    }
    filename_suffix += "_rMin.csv";
    return filename_suffix;
}

std::vector<string> split(const string &ss, char by) {
    std::stringstream input(ss);
    string s;
    std::vector<string> splitted;
    while (getline(input, s, by)) {
        splitted.push_back(s);
    }
    return splitted;
}

void
write_to_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, boost::program_options::variables_map vm,
                    std::map<string, double> values, const string& filename) {
    std::ofstream StatsFile(filename);
    Json::Value stats = prepare_stats_file(clock_phases, time_diffs, number_of_phases, std::move(vm), values);
    StatsFile << stats;
    StatsFile.close();
}

void write_to_debug_file(int point_number, const string& filename) {
    std::ofstream DebugFile(filename);

    DebugFile << "id" << SEPARATOR << "max_eps" << SEPARATOR << "min_eps" << SEPARATOR << "|rnn|" << SEPARATOR;
    DebugFile << "[knn]" << SEPARATOR << "[rnn]" << std::endl;

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
        DebugFile << "]" << std::endl;
    }
    DebugFile.close();
}

void write_to_out_file(int point_number, const string& filename) {
    int dimensions = points.at(0).dimensions.size();
    std::ofstream OutFile(filename);
//    header
    OutFile << "id" << SEPARATOR;
    for (int j = 0; j < dimensions; j++) OutFile << "d" << j << SEPARATOR;
    OutFile << "distance_calculations" << SEPARATOR << "is_core" << SEPARATOR << "cluster_id" << std::endl;

    for (int i = 0; i < point_number; i++) {
        OutFile << i << SEPARATOR;
        for (int j = 0; j < dimensions; j++) {
            OutFile << points.at(i).dimensions.at(j) << SEPARATOR;
        }
        OutFile << points.at(i).distanceCalculationNumber << SEPARATOR
                << (points.at(i).type == core) << SEPARATOR
                << clusters[i] << std::endl;
    }
    OutFile.close();
}

Json::Value
prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, boost::program_options::variables_map vm,
                   std::map<string, double> values) {
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

    std::cout << stats << std::endl;
    return stats;
}