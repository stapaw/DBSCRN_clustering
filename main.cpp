#include <iostream>
#import<cmath>
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


vector<point> points;

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

    calculate_knn(3);
    for(int i=0; i<point_number; i++){
        point p = points.at(i);
        cout << endl;
        cout << p.id << " " << p.x << " " << p.y << endl;
        cout << "knn: ";
        for(int j : p.knn){
            cout << j << " ";
        }
        cout << endl << "rnn: ";
        for(int j : p.rnn){
            cout << j << " ";
        }
    }
//    unsigned int n;
//    io::CSVReader<n> in("../ram.csv");
//    in.read_header(io::ignore_extra_column, "a", "b", "c");
//    std::string vendor; int size; double speed;
//    while(in.read_row(vendor, size, speed)){
//        // do stuff with the data
//    }

}