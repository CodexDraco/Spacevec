cmake_minimum_required(VERSION 3.25)
project(Spacevec CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#find_package(doctest)
find_package(Vulkan REQUIRED)
find_package(fmt REQUIRED)
find_package(glfw3 3.3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  fetch_vk_bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG v1.3.268
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)

add_custom_command(
  OUTPUT vertex.spv
  COMMAND Vulkan::glslc ${CMAKE_CURRENT_SOURCE_DIR}/assets/vertex.glsl -o vertex.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/assets/vertex.glsl
  VERBATIM)

add_custom_command(
  OUTPUT fragment.spv
  COMMAND Vulkan::glslc ${CMAKE_CURRENT_SOURCE_DIR}/assets/fragment.glsl -o fragment.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/assets/fragment.glsl
  VERBATIM)

add_executable(${PROJECT_NAME} src/main.cpp vertex.spv fragment.spv)

target_link_libraries(${PROJECT_NAME} PUBLIC glfw Vulkan::Vulkan fmt::fmt vk-bootstrap::vk-bootstrap)
