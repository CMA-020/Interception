cmake_minimum_required(VERSION 3.22)
project(KeyLaunch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ====================================================================
# GPU OPTION
# ====================================================================
option(NO_GPU_E "Exclude GPU-dependent code" OFF)

if(NO_GPU_E)
    set(USE_GPU OFF)
    add_definitions(-DNO_GPU)
else()
    set(USE_GPU ON)
    add_definitions(-DUSE_GPU)
endif()

# ====================================================================
# ARCH DETECTION
# ====================================================================
execute_process(COMMAND gcc -dumpmachine OUTPUT_VARIABLE TARGET_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)

if(USE_GPU)
    if(TARGET_ARCH MATCHES "aarch64")
        add_definitions(-DPLATFORM_TEGRA)
        set(CUDA_VER "12.2")
        set(NVDS_VERSION "7.0")
    elseif(TARGET_ARCH MATCHES "x86_64")
        set(CUDA_VER "12.9")
        set(NVDS_VERSION "7.1")
    endif()
    set(LIB_INSTALL_DIR "/opt/nvidia/deepstream/deepstream-${NVDS_VERSION}/lib")
endif()

# ====================================================================
# ROS2 SDK MODE
# ====================================================================
set(ROS2_DISTRO humble)
set(CMAKE_PREFIX_PATH "/opt/ros/${ROS2_DISTRO}")

find_package(rclcpp REQUIRED)
find_package(rcutils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(vs_engine_interfaces REQUIRED)

# ====================================================================
# vs_engine
# ====================================================================
set(VS_ENGINE_PROJECT_ROOT "/home/paar-pi5/Visual-Streaming-Engine_Sentinel-1")
set(VS_ENGINE_LIBRARY_DIR "${VS_ENGINE_PROJECT_ROOT}/build/vs_engine")

file(GLOB_RECURSE ALL_HEADERS
    "${VS_ENGINE_PROJECT_ROOT}/src/*.h"
    "${VS_ENGINE_PROJECT_ROOT}/src/*.hpp"
    "${VS_ENGINE_PROJECT_ROOT}/src/*.tpp"
)

if(NO_GPU_E)
    list(FILTER ALL_HEADERS EXCLUDE REGEX "cuda_dependent")
endif()

set(VS_ENGINE_INCLUDE_DIRS "")
foreach(HEADER_FILE ${ALL_HEADERS})
    get_filename_component(HEADER_DIR ${HEADER_FILE} DIRECTORY)
    list(APPEND VS_ENGINE_INCLUDE_DIRS ${HEADER_DIR})
endforeach()
list(REMOVE_DUPLICATES VS_ENGINE_INCLUDE_DIRS)

# ====================================================================
# GST + OpenCV
# ====================================================================
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-video-1.0
    gstreamer-rtsp-server-1.0
    gstreamer-app-1.0
    cairo
)

find_package(OpenCV REQUIRED)
pkg_check_modules(X11 REQUIRED x11)

# SeeCube dependency
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)
# ====================================================================
# EXECUTABLE
# ====================================================================
add_executable(${PROJECT_NAME}
    main.cpp
)

# ====================================================================
# INCLUDE
# ====================================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${VS_ENGINE_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${X11_INCLUDE_DIRS}
    ${LIBUSB_INCLUDE_DIRS}
)
if(USE_GPU)
    target_include_directories(${PROJECT_NAME} PRIVATE
        "/usr/local/cuda-${CUDA_VER}/include"
        "${LIB_INSTALL_DIR}/../sources/includes"
    )
endif()

# ====================================================================
# LINK DIRS
# ====================================================================
target_link_directories(${PROJECT_NAME} PRIVATE
    ${VS_ENGINE_LIBRARY_DIR}
    ${LIBUSB_LIBRARY_DIRS}
)

if(USE_GPU)
    target_link_directories(${PROJECT_NAME} PRIVATE
        "/usr/local/cuda-${CUDA_VER}/lib64"
        ${LIB_INSTALL_DIR}
    )
endif()

# ====================================================================
# LINK LIBRARIES
# ====================================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    vs_engine

    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    ${X11_LIBRARIES}

    rclcpp::rclcpp
    rcutils::rcutils
    cv_bridge::cv_bridge
    image_transport::image_transport
    std_msgs::std_msgs__rosidl_typesupport_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    vs_engine_interfaces::vs_engine_interfaces__rosidl_typesupport_cpp

    yaml-cpp
    pthread
    m
    udev

    -Wl,--no-as-needed
    ${LIBUSB_LIBRARIES}
    -Wl,--as-needed
)

if(USE_GPU)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        cudart cuda
        nvinfer nvinfer_plugin nvonnxparser
        nvdsgst_meta nvds_meta nvdsgst_helper
    )
endif()

# ====================================================================
# RPATH
# ====================================================================
set(MY_RPATH "${VS_ENGINE_LIBRARY_DIR}" "/opt/ros/${ROS2_DISTRO}/lib")

if(USE_GPU)
    list(APPEND MY_RPATH "${LIB_INSTALL_DIR}" "/usr/local/cuda-${CUDA_VER}/lib64")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${MY_RPATH}"
    BUILD_WITH_INSTALL_RPATH ON
)