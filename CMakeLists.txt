cmake_minimum_required(VERSION 3.14)
project(simpleNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Wymuś jednolite użycie /MDd (MultiThreadedDebugDLL) — dynamiczny runtime
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL" CACHE STRING "" FORCE)

# Źródła Twojej biblioteki
add_library(simpleNN
    src/Network.cpp
    src/Layer.cpp
    src/Neuron.cpp
    src/utils.cpp
    src/Trainer.cpp
)

target_include_directories(simpleNN PUBLIC include)

# Główna aplikacja (plik wykonywalny)
add_executable(simpleNNN src/main.cpp)
target_link_libraries(simpleNNN simpleNN)

# GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE 
)

# Ustaw GTest na użycie tej samej biblioteki runtime (dynamicznej)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Testy
enable_testing()

add_executable(tests
    tests/test_iris.cpp
    tests/test_xor.cpp
)

target_link_libraries(tests
    simpleNN
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)
