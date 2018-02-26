cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)

file(GLOB SOURCES
    *.h
    *.cpp
)
SET(CMAKE_CXX_FLAGS "-std=c++11")

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )

SET(test FALSE)
if (TRUE)
    add_executable (pcd_read ${SOURCES} main.cpp)
    target_link_libraries (pcd_read ${PCL_LIBRARIES})
    target_link_libraries( pcd_read ${OpenCV_LIBS} )
endif()

if (FALSE)
    add_executable (track_example trackbar.cpp)
    target_link_libraries (track_example ${OpenCV_LIBS})
endif()