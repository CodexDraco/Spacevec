#include "renderer.h"
#include <cstdio>
#include <fmt/core.h>

int main() {
    spacevec::Renderer renderer;

    spacevec::init_result result;
    if(!(result = renderer.init())) {
        fmt::print("Error initializing Vulkan renderer: {}", result.message);
        return 1;
    }
    if(!(result = renderer.main_loop())) {
        fmt::print(stderr, "Error rendering images: {}", result.message);
        return 1;
    }
    renderer.destroy();

    return 0;
}
