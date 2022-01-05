#include <iostream>
#include<cmath>
#include <queue>
#include <fstream>
#include <time.h>
#include <json/json.h>
#include <json/value.h>
#include <json/writer.h>
#include <functional>


# include "csv.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;



struct point {
    int id{};
    bool isCore = true;
    int distanceCalculationNumber = 0;
    double max_eps;
    double min_eps;
    vector<double> dimensions;
    vector<int> knn;
    vector<int> rnn;
};


struct distance_x {
    int id{};
    double dist{};
};

struct find_id : std::unary_function<distance_x, bool> {
    int id;

    find_id(int id) : id(id) {}

    bool operator()(distance_x const &m) const {
        return m.id == id;
    }
};

struct dist_comparator {
    inline bool operator()(const distance_x &struct1, const distance_x &struct2) {
        return (struct1.dist < struct2.dist);
    }
};


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

void calculate_knn_optimized(int k, point reference_point);

double big_number = 9999999;
vector<point> points;
int clusters[100000] = {0};
bool visited[100000] = {0};
double reference_values[10000] = {big_number};
const string SEPARATOR = ",";

double calculate_distance(const point &point, const struct point &other) {
    int dimension = point.dimensions.size();
    double dist = 0;
    for (int i = 0; i < dimension; i++) {
        double diff = point.dimensions.at(i) - other.dimensions.at(i);
        dist += diff * diff;
    }
    return sqrt(dist);
}

vector<distance_x> calculate_distances_for_knn(point p, int distance_number) {
    vector<distance_x> distances;
    for (int j = 0; j < distance_number; j++) {
        point other = points.at(j);
        double dist = calculate_distance(p, other);
        distance_x distance;
        distance.id = j;
        distance.dist = dist;
        distances.push_back(distance);
    }
    return distances;
}

template<typename T>
void print_queue(T q) { // NB: pass by value so the print uses a copy
    while (!q.empty()) {
        cout << q.top().id << ' ' << q.top().dist;
        q.pop();
    }
    cout << '\n';
}

void calculate_knn(int k) {
    for (int i = 0; i < points.size(); i++) {
        vector<distance_x> distances = calculate_distances_for_knn(points.at(i), points.size());

        sort(distances.begin(), distances.end(), dist_comparator());
//        cout << i << ":" << endl;
//        for (distance_x j: distances) cout << j.id << ' ' << j.dist << endl;
        for (int j = 1; j <= k; j++) {
            if (points.at(i).id != distances.at(j).id) {
                points.at(i).knn.push_back(distances.at(j).id);

                points.at(distances.at(j).id).rnn.push_back(i);
            }
        }
    }
}

void calculate_knn_optimized(vector<distance_x> distances, int distance_idx, int point_id, int k) {
    int max_index = distances.size() - 1;
    //TODO: empty queue
    vector<distance_x> k_dist;
    int p_idx = distance_idx;
    priority_queue<distance_x, vector<distance_x>, dist_comparator> queue(k_dist.begin(), k_dist.end());
    int up = 1;
    int down = 1;
    for (int i = 0; i < k; i++) {
        int up_idx = p_idx - up;
        int down_idx = p_idx + down;
        double up_value, down_value;

        if (up_idx < 0) up_value = big_number;
        else up_value = distances.at(p_idx).dist - distances.at(up_idx).dist;

        if (down_idx > max_index) down_value = big_number;
        else down_value = distances.at(down_idx).dist - distances.at(p_idx).dist;

        point other;
        if (up_value < down_value) {
            other = points.at(distances.at(up_idx).id);
            up++;
        } else {
            other = points.at(distances.at(down_idx).id);
            down++;
        }
        distance_x distance;
        distance.id = other.id;
        distance.dist = calculate_distance(points.at(point_id), other);
        queue.push(distance);
    }
    points.at(point_id).distanceCalculationNumber += k;
    double radius = queue.top().dist;
    points.at(point_id).max_eps = radius;

    while (1) {
        int up_idx = p_idx - up;
        int down_idx = p_idx + down;
        double up_value, down_value;

        if (up_idx < 0) up_value = big_number;
        else up_value = distances.at(p_idx).dist - distances.at(up_idx).dist;

        if (down_idx > max_index) down_value = big_number;
        else down_value = distances.at(down_idx).dist - distances.at(p_idx).dist;

        point other;
        if (up_value < down_value) {
            if (up_value < radius) {
                other =  points.at(distances.at(up_idx).id);
                up++;
            } else break;
        } else {
            if (down_value < radius) {
                other =  points.at(distances.at(down_idx).id);
                down++;
            } else break;
        }
        distance_x distance;
        distance.id = other.id;
        distance.dist = calculate_distance(points.at(point_id), other);
        points.at(point_id).distanceCalculationNumber++;

        queue.push(distance);
        queue.pop();
        radius = queue.top().dist;
    }
    points.at(point_id).min_eps = radius;

    for (int j = 0; j < k; j++) {
        if (point_id != queue.top().id) {
            points.at(point_id).knn.push_back(queue.top().id);

            points.at(queue.top().id).rnn.push_back(point_id);
        }
        queue.pop();
    }
}

auto DBSCRN(int k) {
    vector<int> S_non_core;
    vector<int> S_core;
    int cluster_number = 1;
    for (int i = 0; i < points.size(); i++) {
        if (points.at(i).rnn.size() < k) S_non_core.push_back(i);
        else {
            S_core.push_back(i);
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
        points.at(non_core_point).isCore = false;
        if (clusters[non_core_point] == 0) {
            clusters[non_core_point] = get_cluster_of_nearest_core_point(non_core_point, S_core);
        }
    }
    struct result {
        int cluster_number;
        int core_points_number;
        int non_core_points_number;
    };
    return result{cluster_number - 1, (int) S_core.size(), (int) S_non_core.size()};
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
//        cout << "y_k: " << letters[y_k] << endl;
        for (int y_j: points.at(y_k).rnn) {
            //TODO: change for math pi value
            if (points.at(y_j).rnn.size() > (2 * k / 3.14)) {
//                cout << "   y_j: " << letters[y_j] <<endl;
                for (int p:points.at(y_j).rnn) {
//                    TODO: add visited;
                    if (!visited[p]) {
                        S_tmp.push_back(p);
                        visited[p] = true;
//                       cout <<"    added to S_tmp: " << letters[p] << endl;
                    }
                }
            }
            if (clusters[y_j] == 0) {
                clusters[y_j] = cluster_number;
//                cout << "assigned " <<letters[y_j] << " -> " << cluster_number  << endl;
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

    int k = vm["k"].as<int>();
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

    for (int i = 0; i < point_number; i++) {
        point p = points.at(i);
        cout << endl;
        cout << p.id << " " << p.dimensions.at(0) << " " << p.dimensions.at(1) << endl;
//        cout << "knn: ";
//        for(int j : p.knn){
//            cout << letters[j] << " ";
//        }
//        cout << endl << "rnn: ";
//        for(int j : p.rnn){
//            cout << letters[j] << " ";
//        }
    }
    auto[cluster_number, core_points_number, non_core_points_number] = DBSCRN(k);

    for (int i = 0; i < point_number; i++) {
        cout << i << " " << clusters[i] << endl;
        point p = points.at(i);
        cout << "rnn size: " << p.rnn.size() << endl;
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
            {"#_core_points",         core_points_number},
            {"#_non_core_points",     non_core_points_number}};

    write_to_stats_file(clock_phases, time_diffs, number_of_phases, vm, values);
}

void calculate_knn_optimized(int k, point reference_point) {
    vector<distance_x> distances = calculate_distances_for_knn(reference_point, points.size());
    sort(distances.begin(), distances.end(), dist_comparator());
    int size = distances.size();
    for (int i = 0; i < size; i++) {
        calculate_knn_optimized(distances, i, distances.at(i).id, k);
    }
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
    ofstream OutFile("../out_file");
    for (int i = 0; i < point_number; i++) {
        OutFile << i << SEPARATOR;
        int dimensions = points.at(0).dimensions.size();
        for (int j = 0; j < dimensions; j++) {
            OutFile << points.at(i).dimensions.at(j) << SEPARATOR;
        }
        OutFile << points.at(i).distanceCalculationNumber << SEPARATOR
                << points.at(i).isCore << SEPARATOR
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
