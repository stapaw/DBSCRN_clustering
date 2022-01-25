//
// Created by stanislaw on 09.01.2022.
//

#include <boost/program_options.hpp>
#include "DBSCRN.h"
#include "settings.h"
#include "point.h"
#include <functional>
#include <json/writer.h>
#include <json/value.h>
#include <fstream>
#include <iostream>
#include "output.h"

const string SEPARATOR = ",";

vector<string> split(const string &s, char by);

Json::Value
prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, stats stats);

string get_filename_suffix(const boost::program_options::variables_map &vm, int point_number, int dimensions) {
    string filename_suffix = "_";
    filename_suffix += vm[TI_OPTIMIZED_PARAM_NAME].as<bool>() ? "Opt-" : "";
    filename_suffix += vm[ALGORITHM_PARAM_NAME].as<string>() + "_";
    std::vector<string> splitted = split(vm[INPUT_FILE_PARAM_NAME].as<string>(), '.');
    splitted = split(splitted.at(splitted.size() - 2), '/');
    filename_suffix += splitted.at(splitted.size() - 1);
    filename_suffix += "_D" + to_string(dimensions) + "_R" + to_string(point_number);
    if (vm[ALGORITHM_PARAM_NAME].as<string>() == "DBSCAN") {
        filename_suffix += "_minPts" + to_string(vm[MIN_PTS_PARAM_NAME].as<int>()) + "_e" + to_string(vm[EPS_PARAM_NAME].as<double>());
    } else {
        filename_suffix += "_k" + to_string(vm[K_PARAM_NAME].as<int>());
    }
    filename_suffix += "_minkowski" + to_string(vm[MINKOWSKI_PARAM_NAME].as<int>());
    filename_suffix += "_refMin.csv";
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
write_to_stats_file(stats stats, const boost::program_options::variables_map &vm, const string &filename) {
    std::ofstream StatsFile(filename);
    Json::Value output;
    Json::Value reference_point_values(Json::arrayValue);
    for(int i=0; i<stats.dimensions; i++){
        reference_point_values.append(Json::Value(reference_values[i]));
    }

    output[STATS_MAIN][INPUT_FILE_PARAM_NAME] = vm[INPUT_FILE_PARAM_NAME].as<string>();
    output[STATS_MAIN]["#_dimensions"] = stats.dimensions;
    output[STATS_MAIN]["#_points"] = stats.point_number;
    output[STATS_MAIN][ALGORITHM_PARAM_NAME] = vm[ALGORITHM_PARAM_NAME].as<string>();

    if (output[STATS_MAIN][ALGORITHM_PARAM_NAME] == "DBSCAN") {
        output[STATS_PARAMETERS][EPS_PARAM_NAME] = vm[EPS_PARAM_NAME].as<double>();
        output[STATS_PARAMETERS][MIN_PTS_PARAM_NAME] = vm[MIN_PTS_PARAM_NAME].as<int>();
    } else {
        output[STATS_PARAMETERS][K_PARAM_NAME] = vm[K_PARAM_NAME].as<int>();
    }
    output[STATS_PARAMETERS][MINKOWSKI_PARAM_NAME] = vm[MINKOWSKI_PARAM_NAME].as<int>();
    output[STATS_PARAMETERS][TI_OPTIMIZED_PARAM_NAME] = vm[TI_OPTIMIZED_PARAM_NAME].as<bool>();
    output[STATS_PARAMETERS]["reference_point_values"] = reference_point_values;


    output[STATS_CLUSTERING_STATS]["#_clusters"] = stats.cluster_number;
    output[STATS_CLUSTERING_STATS]["#_core_points"] = stats.core_points;

    if (output[STATS_MAIN][ALGORITHM_PARAM_NAME] == "DBSCAN") {
        output[STATS_CLUSTERING_STATS]["#_border_points"] = stats.border_points;
    } else {
        output[STATS_CLUSTERING_STATS]["#non_core_points"] = stats.non_core_points;

    }
    output[STATS_CLUSTERING_STATS]["#_noise_points"] = stats.noise_points;
    output[STATS_CLUSTERING_STATS]["avg_#_of_distance_calculation"] = stats.avg_dist_calculation;

    //time stats
    int number_of_phases = stats.time_diffs.size();
    for (int i = 0; i < number_of_phases; i++)
        output[STATS_CLUSTERING_TIME][clock_phases[i]] = stats.time_diffs.at(i);

    output[STATS_CLUSTERING_METRICS]["silhouette_coefficient"] = stats.silhouette;
    output[STATS_CLUSTERING_METRICS]["davies_bouldin"] = stats.davies_bouldin;
    // if real cluster known
    output[STATS_CLUSTERING_METRICS]["TP"] = stats.TP;
    output[STATS_CLUSTERING_METRICS]["TN"] = stats.TN;
    output[STATS_CLUSTERING_METRICS]["#_of_pairs"] = stats.number_of_pairs;
    output[STATS_CLUSTERING_METRICS]["RAND"] = stats.rand;
    output[STATS_CLUSTERING_METRICS]["Purity"] = stats.purity;

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