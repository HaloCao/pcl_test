cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcl_test)

set(DCMAKE_BUILD_TYPE = Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# find_package(Boost 1.45.0 COMPONENTS filesystem regex)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED)
find_package(yaml-cpp REQUIRED)

include(${VTK_USE_FILE})

include_directories(${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})

link_directories(${PCL_LIBRARY_DIRS} )

add_definitions(${PCL_DEFINITIONS})

add_executable (scp src/scp.cc)

target_link_libraries (scp
# ${Boost_LIBRARIES}
${VTK_LIBRARIES}
${PCL_LIBRARIES}
yaml-cpp
)
