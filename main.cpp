#include <iostream>
#import<cmath>
#include <queue>
#include <fstream>
#include <time.h>
#include <json/json.h>
#include <json/value.h>
#include <json/writer.h>


# include "csv.h"

using namespace std;

struct point {
    int id{};
    double x{};
    double y{};
    bool isCore = true;
    int distanceCalculationNumber = 0;
    vector<int> knn;
    vector<int> rnn;
};


struct distance_x {
    int id{};
    double dist{};
};

struct dist_comparator {
    inline bool operator()(const distance_x &struct1, const distance_x &struct2) {
        return (struct1.dist < struct2.dist);
    }
};


void DBSCRN_expand_cluster(int i, int k, int i1);

int get_cluster_of_nearest_core_point(int point, const vector<int> &vector);

Json::Value prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases);

vector<point> points;
int clusters[100000] = {0};
const string SEPARATOR = ",";
//string letters = "ABCDEFGHIJKL";

double calculate_distance(const point &point, const struct point &other) {
    double dx = point.x - other.x;
    double dy = point.y - other.y;
    double dist = dx * dx + dy * dy;
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


void calculate_knn(int k) {
    for (int i = 0; i < points.size(); i++) {
        vector<distance_x> distances = calculate_distances_for_knn(points.at(i), points.size());
        sort(distances.begin(), distances.end(), dist_comparator());
//        cout << i << ":" << endl;
//        for (distance_x j: distances) cout << j.id << ' ' << j.dist << endl;
        for (int j = 0; j < k; j++) {
            points.at(i).knn.push_back(distances.at(j).id);

            points.at(distances.at(j).id).rnn.push_back(i);
        }
    }
}

void calculate_knn_optimized(point p, int k) {
    point r;
    r.x = 0;
    r.y = 0;
    vector<distance_x> distances = calculate_distances_for_knn(r, points.size());
    sort(distances.begin(), distances.end(), dist_comparator());
    vector<distance_x> k_dist = calculate_distances_for_knn(p, k);
    double radius = max_element(k_dist.begin(), k_dist.end(), dist_comparator())->dist;
//    p_idx = find(distances.begin(), distances.end(), )
//    while()
    // calculate distances within border
    // sort and return
}

void DBSCRN(int k) {
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
        clusters[non_core_point] = get_cluster_of_nearest_core_point(non_core_point, S_core);
    }
}

int get_cluster_of_nearest_core_point(int point, const vector<int> &vector) {
//    TODO: calculate distances;
    return -1;
}

void DBSCRN_expand_cluster(int i, int k, int cluster_number) {
    vector<int> S_tmp;

    clusters[i] = cluster_number;
    S_tmp.push_back(i);
    for (int j = 0; j < S_tmp.size(); j++) {
        int y_k = S_tmp.at(j);
//        cout << "y_k: " << letters[y_k] << endl;
        for (int y_j: points.at(y_k).rnn) {
            //TODO: change for math pi value
            if (points.at(y_j).rnn.size() > 2 * k / 3.14) {
//                cout << "   y_j: " << letters[y_j] <<endl;
                for (int p:points.at(y_j).rnn) {
                    if (std::find(S_tmp.begin(), S_tmp.end(), p) == S_tmp.end()) {
                        S_tmp.push_back(p);
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

int main() {
    clock_t start_time, input_read_time, sort_by_reference_point_time, rnn_neighbour_time, clustering_time, stats_calculation_time, output_write_time;
    const string clock_phases[] = {"read input file", "sort by reference point distances", "rnn calculation", "clustering", "stats calculation", "write output files", "total"};
    start_time = clock();

    int k, point_number, dimensions;
    cin >> k >> point_number >> dimensions;
//  TODO: update for more than 2 dimensions
    for (int i = 0; i < point_number; i++) {
        point p;
        p.id = i;
        cin >> p.x >> p.y;
        points.push_back(p);
    }
    input_read_time = clock();

    point r;
    r.x = 4.2;
    r.y = 4;
    vector<distance_x> distances = calculate_distances_for_knn(r, points.size());
    sort(distances.begin(), distances.end(), dist_comparator());

    sort_by_reference_point_time = clock();

    calculate_knn(k);

    rnn_neighbour_time = clock();

//    for(int i=0; i<point_number; i++){
//        point p = points.at(i);
//        cout << endl;
//        cout << p.id << " " << p.x << " " << p.y << endl;
////        cout << "knn: ";
//        for(int j : p.knn){
////            cout << letters[j] << " ";
//        }
////        cout << endl << "rnn: ";
//        for(int j : p.rnn){
////            cout << letters[j] << " ";
//        }
//    }
    DBSCRN(k);

    clustering_time = clock();
    stats_calculation_time = clock();


    ofstream OutFile("../out_file");
    ofstream StatsFile("../stats_file");
    ofstream DebugFile("../debug_file");

    for (int i = 0; i < point_number; i++) {
        OutFile << i << SEPARATOR
                << points.at(i).x << SEPARATOR << points.at(i).y << SEPARATOR
                << points.at(i).distanceCalculationNumber << SEPARATOR
                << points.at(i).isCore << SEPARATOR
                << clusters[i] << endl;
    }
    OutFile.close();


    output_write_time = clock();
    double time_diffs[] = {(double) (input_read_time - start_time) / CLOCKS_PER_SEC,
                           (double) (sort_by_reference_point_time - input_read_time) / CLOCKS_PER_SEC,
                           (double) (rnn_neighbour_time - sort_by_reference_point_time) / CLOCKS_PER_SEC,
                           (double) (clustering_time - rnn_neighbour_time) / CLOCKS_PER_SEC,
                           (double) (stats_calculation_time - clustering_time) / CLOCKS_PER_SEC,
                           (double) (output_write_time - stats_calculation_time) / CLOCKS_PER_SEC,
                           (double) (clock() - start_time) / CLOCKS_PER_SEC
    };
    int number_of_phases = sizeof(clock_phases)/sizeof(*clock_phases);
    for(int i=0; i<number_of_phases; i++)cout << clock_phases[i] << ": " << time_diffs[i] << endl;


    Json::Value stats = prepare_stats_file(clock_phases, time_diffs, number_of_phases);
    StatsFile << stats;
    StatsFile.close();


    DebugFile.close();

    //    unsigned int n;
//    io::CSVReader<n> in("../ram.csv");
//    in.read_header(io::ignore_extra_column, "a", "b", "c");
//    std::string vendor; int size; double speed;
//    while(in.read_row(vendor, size, speed)){
//        // do stuff with the data
//    }

}

Json::Value prepare_stats_file(const string *clock_phases, const double *time_diffs, int number_of_phases) {
    Json::Value stats;
    Json::Value reference_point_values(Json::arrayValue); // TODO
    reference_point_values.append(Json::Value(4.2));
    reference_point_values.append(Json::Value(4));


    stats["main"]["input_filename"] = "input_file"; // TODO
    stats["main"]["#_of_point_dimensions"] = 2; // TODO
    stats["main"]["#_of_points"] = 12; // TODO
    stats["parameters"]["algorithm"] = "parameter"; // TODO
    stats["parameters"]["k"] = "parameter"; // TODO
    stats["parameters"]["Eps"] = "parameter"; // TODO
    stats["parameters"]["minPts"] = "parameter"; // TODO
    stats["main"]["#_clusters"] = 2; // TODO
    stats["main"]["#_noise_points"] = 0; // TODO
    stats["main"]["#_core_points"] = 0; // TODO
    stats["main"]["#_border_points"] = 0; // TODO
    stats["clustering_stats"]["avg_#_of_distance_calculation"] = 0; // TODO

    stats["clustering_stats"]["silhouette_coefficient"] = 0; // TODO
    stats["clustering_stats"]["davies_bouldin"] = 0; // TODO
//time stats
    for(int i=0; i<number_of_phases; i++)
         stats["time_stats"][clock_phases[i]]  = time_diffs[i];


    // if real cluster known
    stats["clustering_stats"]["TP"] = 0; // TODO
    stats["clustering_stats"]["TN"] = 0; // TODO
    stats["clustering_stats"]["#_of_pairs"] = 0; // TODO ???
    stats["clustering_stats"]["RAND"] = 0; // TODO
    stats["clustering_stats"]["Purity"] = 0; // TODO

    cout << stats << endl;
    return stats;
}
