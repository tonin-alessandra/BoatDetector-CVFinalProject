cmake_minimum_required(VERSION 3.0.0)
project(Tonin_FinalProject VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})