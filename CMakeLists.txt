cmake_minimum_required(VERSION 3.15)
project(Clustering)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(Clustering Cpp/main.cpp Cpp/DBSCAN.cpp Cpp/DBSCRN.cpp Cpp/distance_calculations.cpp Cpp/stats.cpp Cpp/output.cpp Cpp/csv.h Cpp/point.h Cpp/settings.h)

INCLUDE_DIRECTORIES( /home/linuxbrew/.linuxbrew/Cellar/jsoncpp/1.9.4_1/include )
LINK_DIRECTORIES(  /home/linuxbrew/.linuxbrew/Cellar/jsoncpp/1.9.4_1/lib )

file(GLOB LIBRARIES "/home/linuxbrew/.linuxbrew/Cellar/jsoncpp/1.9.4_1/lib/*.so")
message("LIBRARIES = ${LIBRARIES}")


TARGET_LINK_LIBRARIES(Clustering ${LIBRARIES})
TARGET_LINK_LIBRARIES(Clustering boost_program_options)

