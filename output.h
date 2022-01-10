//
// Created by stanislaw on 09.01.2022.
//

#ifndef CLUSTERING_OUTPUT_H
#define CLUSTERING_OUTPUT_H

#include <boost/program_options/variables_map.hpp>
#include <json/json.h>
#include<iostream>
using namespace std;

string get_filename_suffix(const boost::program_options::variables_map &vm, int point_number, int dimensions);

void
write_to_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases, boost::program_options::variables_map vm,
                    std::map<string, double> point_number, const string& filename);

void write_to_debug_file(int point_number, const string& filename);

void write_to_out_file(int point_number, const string& filename);



#endif //CLUSTERING_OUTPUT_H