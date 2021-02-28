cmake_minimum_required(VERSION 3.1)

project( pcv1 LANGUAGES CXX )

find_package( OpenCV REQUIRED )



add_library(code 
    Pcv1.cpp
    Pcv1.h
)

set_target_properties(code PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(code 
    PUBLIC
        ${OpenCV_LIBS}
)




add_executable(main 
    main.cpp 
)

set_target_properties(main PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(main 
    PRIVATE
        code 
)



add_executable(unit_test 
    unit_test.cpp 
)

set_target_properties(unit_test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(unit_test 
    PRIVATE
        code
)

