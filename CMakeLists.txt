cmake_minimum_required(VERSION 3.15)
project(DBSCRN_clustering)

add_executable(DBSCRN_clustering Cpp/main.cpp Cpp/DBSCAN.cpp Cpp/DBSCRN.cpp Cpp/distance_calculations.cpp Cpp/stats.cpp Cpp/output.cpp Cpp/csv.h Cpp/point.h Cpp/settings.h)

INCLUDE_DIRECTORIES( /home/linuxbrew/.linuxbrew/Cellar/jsoncpp/1.9.4_1/include )
LINK_DIRECTORIES(  /home/linuxbrew/.linuxbrew/Cellar/jsoncpp/1.9.4_1/lib )

file(GLOB LIBRARIES "/home/linuxbrew/.linuxbrew/Cellar/jsoncpp/1.9.4_1/lib/*.so")
message("LIBRARIES = ${LIBRARIES}")


TARGET_LINK_LIBRARIES(DBSCRN_clustering ${LIBRARIES})
TARGET_LINK_LIBRARIES(DBSCRN_clustering boost_program_options)

