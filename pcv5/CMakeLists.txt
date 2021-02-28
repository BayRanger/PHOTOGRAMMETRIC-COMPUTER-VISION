cmake_minimum_required(VERSION 3.1)

project( pcv5 LANGUAGES CXX )

find_package( OpenCV REQUIRED )




add_library(code 
    Pcv5.cpp
    Pcv5.h
    Helper.cpp
    Helper.h
)

set_target_properties(code PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(code 
    PUBLIC
        opencv_core opencv_highgui opencv_imgproc opencv_calib3d
)




add_executable(main
    main.cpp 
)

target_link_libraries(main
    PRIVATE
        code
)

add_executable(unit_test 
    unit_test.cpp 
)

target_link_libraries(unit_test 
    PRIVATE
        code
)


set_target_properties(main PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
set_target_properties(unit_test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

