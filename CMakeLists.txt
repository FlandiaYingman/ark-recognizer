cmake_minimum_required(VERSION 3.1.3)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.23.318.tar.gz"
        SHA1 "0475be76dbf51385790bbe96be57279e2ff0861d"
)

project(ark_recognizer_cpp)
set(CMAKE_CXX_STANDARD 17)

# https://hunter.readthedocs.io/en/latest/packages/pkg/nlohmann_json.html
hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)
# https://hunter.readthedocs.io/en/latest/packages/pkg/OpenCV.html
hunter_add_package(OpenCV)
find_package(OpenCV REQUIRED)

add_executable(recognizer src/ark_recognizer_cpp/recognizer.cpp src/ark_recognizer_cpp/item.h src/ark_recognizer_cpp/item.cpp src/ark_recognizer_cpp/recognize_icon.cpp src/ark_recognizer_cpp/recognize_icon.h src/ark_recognizer_cpp/recognize_type.cpp src/ark_recognizer_cpp/recognize_type.h src/ark_recognizer_cpp/recognize_number.cpp src/ark_recognizer_cpp/recognize_number.h src/ark_recognizer_cpp/utils.h src/ark_recognizer_cpp/utils.cpp)
target_link_libraries(recognizer PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(recognizer PRIVATE ${OpenCV_LIBS})