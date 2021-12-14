cmake_minimum_required(VERSION 3.1.3)

if(NOT DEFINED WASM)
        set(WASM OFF)
endif()

if(WASM)
        include(FetchContent)
        FetchContent_Declare(opencv 
                GIT_REPOSITORY https://github.com/opencv/opencv.git
                GIT_TAG 4.5.4
                GIT_SHALLOW ON
        )
        FetchContent_Declare(nlohmann_json 
                GIT_REPOSITORY https://github.com/nlohmann/json.git
                GIT_TAG v3.10.4
                GIT_SHALLOW ON
        )

        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/libs/opencv)
        set(nlohmann_json_DIR ${CMAKE_SOURCE_DIR}/libs/nlohmann)
else()
        include("cmake/HunterGate.cmake")
        HunterGate(
                URL "https://github.com/cpp-pm/hunter/archive/v0.23.318.tar.gz"
                SHA1 "0475be76dbf51385790bbe96be57279e2ff0861d"
        )
endif()


project(ark_recognizer_cpp)
set(CMAKE_CXX_STANDARD 17)

if(NOT WASM)
        # https://hunter.readthedocs.io/en/latest/packages/pkg/nlohmann_json.html
        hunter_add_package(nlohmann_json)
        find_package(nlohmann_json CONFIG REQUIRED)

        # https://hunter.readthedocs.io/en/latest/packages/pkg/OpenCV.html
        hunter_add_package(OpenCV)
        find_package(OpenCV REQUIRED)
endif(NOT WASM)


set(SOURCES
        src/ark_recognizer_cpp/recognizer.cpp
        src/ark_recognizer_cpp/item.h
        src/ark_recognizer_cpp/item.cpp
        src/ark_recognizer_cpp/recognize_icon.cpp
        src/ark_recognizer_cpp/recognize_icon.h
        src/ark_recognizer_cpp/recognize_type.cpp
        src/ark_recognizer_cpp/recognize_type.h
        src/ark_recognizer_cpp/recognize_number.cpp
        src/ark_recognizer_cpp/recognize_number.h
        src/ark_recognizer_cpp/utils.h
        src/ark_recognizer_cpp/utils.cpp
        )

add_executable(recognizer ${SOURCES})

if(WASM)
        if(NOT EXISTS ${OpenCV_DIR}) 
                FetchContent_Populate(opencv)

                set(opencv_BUILD_JS ${opencv_SOURCE_DIR}/platforms/js/build_js.py)
        
                file(READ ${opencv_BUILD_JS} filedata)
                string(REPLACE "-DWITH_JPEG=OFF" "-DWITH_JPEG=OFF" filedata "${filedata}")
                string(REPLACE "-DWITH_PNG=OFF" "-DWITH_PNG=ON" filedata "${filedata}")
                string(REPLACE "-DBUILD_opencv_imgcodecs=OFF" "-DBUILD_opencv_imgcodecs=ON" filedata "${filedata}")
                file(WRITE ${opencv_BUILD_JS} "${filedata}")
        
                include(FindPython3)
                find_package (Python3 COMPONENTS Interpreter)
        
                execute_process(COMMAND emcmake ${Python3_EXECUTABLE} ${opencv_BUILD_JS} opencv_build)

                execute_process(COMMAND ${CMAKE_COMMAND} --build opencv_build)
                execute_process(COMMAND ${CMAKE_COMMAND} --install opencv_build --prefix ${OpenCV_DIR})
        endif()

        if(NOT EXISTS ${nlohmann_json_DIR})
                FetchContent_Populate(nlohmann_json)
                file(COPY ${nlohmann_json_SOURCE_DIR}/single_include/ DESTINATION ${nlohmann_json_DIR}/include)      
        endif()
        


        target_include_directories(recognizer PRIVATE ${OpenCV_DIR}/include/opencv4/)
        target_include_directories(recognizer PRIVATE ${nlohmann_json_DIR}/include)

        # Link to opencv.js precompiled libraries
        file(GLOB OpenCV_LIBS ${OpenCV_DIR}/lib/*.a)
        target_link_libraries(recognizer ${OpenCV_LIBS})

        file(GLOB OpenCV_3RD_LIBS ${OpenCV_DIR}/lib/opencv4/3rdparty/*.a)
        target_link_libraries(recognizer ${OpenCV_3RD_LIBS})

        target_link_options(recognizer PRIVATE "-s EXPORTED_FUNCTIONS='[_recognize, _load_item, _load_hash]'")
else()
        target_link_libraries(recognizer PUBLIC nlohmann_json::nlohmann_json)
        target_link_libraries(recognizer PRIVATE ${OpenCV_LIBS})
endif()