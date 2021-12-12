#include <iostream>
#import<cmath>
#include <queue>
# include "csv.h"
using namespace std;

struct point {
    int id{};
    double x{};
    double y{};
    vector<int> knn;
    vector<int> rnn;
};


struct distance_x {
    int id{};
    double dist{};
};

struct dist_comparator
{
    inline bool operator() (const distance_x& struct1, const distance_x& struct2)
    {
        return (struct1.dist < struct2.dist);
    }
};


void DBSCRN_expand_cluster(int i, int k, int i1);

int get_cluster_of_nearest_core_point(int point, const vector<int>& vector);

vector<point> points;
int clusters[100000] = {0};
bool S_tmp_visited[100000] = {false};
string letters = "ABCDEFGHIJKL";

double calculate_distance(const point& point, const struct point& other) {
    double dx = point.x - other.x;
    double dy = point.y - other.y;
    double dist = dx * dx + dy * dy;
    return sqrt(dist);
}

vector<distance_x> calculate_distances_for_knn(int i, int k){
    vector<distance_x> distances;
    point p = points.at(i);
    for(int j=0; j<points.size(); j++){
        if(j!=i){
            point other = points.at(j);
            double dist = calculate_distance(p, other);
            distance_x distance;
            distance.id = j;
            distance.dist = dist;
            distances.push_back(distance);
        }
    }
    return distances;
}



void calculate_knn(int k){
    for(int i=0; i<points.size(); i++){
        vector<distance_x> distances = calculate_distances_for_knn(i, k);
        sort(distances.begin(), distances.end(), dist_comparator());
//        cout << i << ":" << endl;
//        for (distance_x j: distances) cout << j.id << ' ' << j.dist << endl;
        for(int j=0; j<k; j++){
            points.at(i).knn.push_back(distances.at(j).id);

            points.at(distances.at(j).id).rnn.push_back(i);
        }
    }
}

void DBSCRN(int k){
    vector<int> S_non_core;
    vector<int> S_core;
    int cluster_number = 1;
    for(int i=0; i<points.size(); i++){
        if(points.at(i).rnn.size() < k) S_non_core.push_back(i);
        else{
            S_core.push_back(i);
            int cluster_to_expand;
            if(clusters[i] != 0) cluster_to_expand=clusters[i];
            else{
                cluster_to_expand = cluster_number;
                cluster_number++;
            }
            DBSCRN_expand_cluster(i, k, cluster_to_expand);
        }
    }
    for(int non_core_point : S_non_core){
        clusters[non_core_point] = get_cluster_of_nearest_core_point(non_core_point, S_core);
    }
}

int get_cluster_of_nearest_core_point(int point, const vector<int>& vector) {
//    TODO: calculate distances;
    return -1;
}

void DBSCRN_expand_cluster(int i, int k, int cluster_number) {
    vector<int> S_tmp;

    clusters[i] = cluster_number;
    S_tmp.push_back(i);
    S_tmp_visited[i] = true;
    for(int j=0; j<S_tmp.size(); j++){
        cout << "Processing" << S_tmp.at(j) << endl;
        int y_k = S_tmp.at(j);
        for(int y_j: points.at(y_k).rnn){
            //TODO: change for math pi value
            if(points.at(y_j).rnn.size() > 2*k/3.14){
                cout << "point_id: " << y_j <<endl;
               for(int p:points.at(y_j).rnn){
                   if(!S_tmp_visited[p]){
                       S_tmp_visited[p] = true;
                       S_tmp.push_back(p);
                       cout <<"added" << p << endl;
                   }
               }
            }
            cout << "processed: " << y_j << " vis: " << S_tmp_visited[y_j] << " clust: " << clusters[y_j] << endl;
            if((!S_tmp_visited[y_j]) && (clusters[y_j] == 0)){
                clusters[y_j] = cluster_number;
                cout << "assigned " <<y_j << " -> " << cluster_number  << endl;
            }
        }
    }
//    Cleaning S_tmp_visited
    for(int point_id:S_tmp)S_tmp_visited[point_id] = false;
}

int main(){
    int k, point_number, dimensions;
    cin >> k >> point_number >> dimensions;
//  TODO: update for more than 2 dimensions
    for(int i=0; i<point_number; i++){
        point p;
        p.id = i;
        cin >> p.x >> p.y;
        points.push_back(p);
    }

    calculate_knn(k);
    for(int i=0; i<point_number; i++){
        point p = points.at(i);
        cout << endl;
        cout << letters[p.id] << " " << p.x << " " << p.y << endl;
        cout << "knn: ";
        for(int j : p.knn){
            cout << letters[j] << " ";
        }
        cout << endl << "rnn: ";
        for(int j : p.rnn){
            cout << letters[j] << " ";
        }
    }
    DBSCRN(k);
    for(int i=0; i<point_number; i++){
        cout << i << " " << clusters[i] << endl;
    }
//    unsigned int n;
//    io::CSVReader<n> in("../ram.csv");
//    in.read_header(io::ignore_extra_column, "a", "b", "c");
//    std::string vendor; int size; double speed;
//    while(in.read_row(vendor, size, speed)){
//        // do stuff with the data
//    }

}