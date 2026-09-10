cmake_minimum_required(VERSION 3.18)
project(plates_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(RFDETR_ROOT "$ENV{HOME}/work/libs/RF-DETR-ONNX/cpp")
set(ONNXRUNTIME_ROOT_DIR "$ENV{HOME}/work/libs/onnxruntime-osx-arm64-1.22.0")

add_executable(plates
    src/main.cpp
    src/ocr_engine.cpp
    src/plate_tracker.cpp
    src/plate_formatter.cpp
    src/annotator.cpp
)

# Use target_include_directories and add both ONNX root and sub-include paths
target_include_directories(plates PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${RFDETR_ROOT}/include
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime
    ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime/core/session
    include
)

target_link_directories(plates PRIVATE
    ${RFDETR_ROOT}/build/src
    ${ONNXRUNTIME_ROOT_DIR}/lib
)

target_link_libraries(plates PRIVATE
    rfdetr_onnx
    onnxruntime
    ${OpenCV_LIBS}
)