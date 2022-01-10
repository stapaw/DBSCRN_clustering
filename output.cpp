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
prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, stats stats);

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
write_to_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, stats stats,
                    const boost::program_options::variables_map& vm, const string &filename) {
    std::ofstream StatsFile(filename);
    Json::Value output;
    Json::Value reference_point_values(Json::arrayValue); // TODO
    reference_point_values.append(Json::Value(4.2));
    reference_point_values.append(Json::Value(4));


    output["main"]["input_filename"] = vm["in_file"].as<string>();
    output["main"]["#_of_point_dimensions"] = stats.dimensions;
    output["main"]["#_of_points"] = stats.point_number;

    output["parameters"]["algorithm"] = vm["alg"].as<string>();
    output["parameters"]["k"] = vm["k"].as<int>();
    output["parameters"]["Eps"] = vm["eps"].as<double>();
    output["parameters"]["minPts"] = vm["minPts"].as<int>();

    output["main"]["#_clusters"] = stats.cluster_number;
    output["main"]["#_noise_points"] = stats.noise_points;
    output["main"]["#_border_points"] = stats.border_points;
    output["main"]["#_core_points"] = stats.core_points;
    output["main"]["#non_core_points"] = stats.non_core_points;
    output["clustering_stats"]["avg_#_of_distance_calculation"] = stats.avg_dist_calculation;

    output["clustering_stats"]["silhouette_coefficient"] = stats.silhouette;
    output["clustering_stats"]["davies_bouldin"] = stats.davies_bouldin;
    //time stats
    for (int i = 0; i < number_of_phases; i++)
        output["time_stats"][clock_phases[i]] = time_diffs[i];


    // if real cluster known
    output["clustering_stats"]["TP"] = stats.TP;
    output["clustering_stats"]["TN"] = stats.TN;
    output["clustering_stats"]["#_of_pairs"] = stats.number_of_pairs;
    output["clustering_stats"]["RAND"] = stats.rand;
    output["clustering_stats"]["Purity"] = stats.purity;

    StatsFile << output;
    StatsFile.close();

    cout << output;
}

void write_to_debug_file(int point_number, const string &filename) {
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

void write_to_out_file(int point_number, const string &filename) {
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