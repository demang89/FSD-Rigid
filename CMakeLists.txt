CMAKE_MINIMUM_REQUIRED(VERSION 3.12 FATAL_ERROR)

project(PSERigid  LANGUAGES C CXX CUDA)
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
#set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

find_package(HOOMD 5.0.1 REQUIRED)
message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location.
set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Installation prefix" FORCE)

#set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})
#message(STATUS "Install plugin to: " ${PYTHON_MODULE_BASE_DIR})

# Enable compiler warnings on gcc and clang.
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result")
endif()

# add subdirectories
add_subdirectory(source)
