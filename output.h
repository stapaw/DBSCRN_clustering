//
// Created by stanislaw on 09.01.2022.
//

#ifndef CLUSTERING_OUTPUT_H
#define CLUSTERING_OUTPUT_H

static const char *const STATS_MAIN = "#main";
static const char *const STATS_PARAMETERS = "#parameters";
static const char *const STATS_CLUSTERING_STATS = "clustering_stats";
static const char *const STATS_CLUSTERING_METRICS = "clustering_metrics";
static const char *const STATS_CLUSTERING_TIME = "clustering_time";


#include <boost/program_options/variables_map.hpp>
#include <json/json.h>
#include<iostream>
#include "stats.h"

using namespace std;

string get_filename_suffix(const boost::program_options::variables_map &vm, int point_number, int dimensions);

void
write_to_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, stats stats, const boost::program_options::variables_map& vm,
                    const string &filename);

void write_to_debug_file(int point_number, const string& filename);

void write_to_out_file(int point_number, const string& filename);



#endif //CLUSTERING_OUTPUT_H
