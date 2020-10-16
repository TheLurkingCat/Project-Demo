cmake_minimum_required(VERSION 3.2)

project(Demo CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

find_library(Darknet_LIBRARIES
    PATHS ${CMAKE_CURRENT_LIST_DIR}
    NAMES darknet
    REQUIRED
)

add_executable(demo ${CMAKE_CURRENT_LIST_DIR}/demo.cpp ${CMAKE_CURRENT_LIST_DIR}/darknet.hpp ${CMAKE_CURRENT_LIST_DIR}/marker.hpp)
target_link_libraries(demo PRIVATE opencv_highgui opencv_imgcodecs opencv_imgproc opencv_core)
target_link_libraries(demo PRIVATE ${Darknet_LIBRARIES})

set_target_properties(demo
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
