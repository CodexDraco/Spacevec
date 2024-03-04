# Spacevec 
Vulkan based vector rendering engine written in C++.

This software is experimental.

## Requirements ##
* GLFW3
* Vk-Bootstrap
* Vulkan headers
* {fmt}
* CMake

If you use Nix, the provided shell.nix should install all dependencies.

## Building ##
``` bash
cmake -S . -B build -G Ninja
cd build
cmake --build .
```

## Running
``` bash
cd build
./Spacevec
```

