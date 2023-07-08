cmake_minimum_required(VERSION 3.26)
project(Spacevec CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#find_package(doctest)
find_package(Vulkan REQUIRED)

add_custom_command(
  OUTPUT vertex.spv
  COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/assets/vertex.glsl -o vertex.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/assets/vertex.glsl
  VERBATIM)

add_custom_command(
  OUTPUT fragment.spv
  COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/assets/fragment.glsl -o fragment.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/assets/fragment.glsl
  VERBATIM)

add_executable(${PROJECT_NAME} src/main.cpp src/renderer.cpp vertex.spv fragment.spv)

target_link_libraries(${PROJECT_NAME} PUBLIC -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi)
