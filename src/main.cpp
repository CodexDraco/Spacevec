#include "renderer.h"

#include <exception>
#include <fmt/core.h>
#include <span>

void parse_arguments(const std::span<const char *> &args) {
  for (auto const &arg : args) {
    fmt::print(stderr, "{}\n", arg);
  }
}

int main(int argc, const char *argv[]) {
  if (argc > 1) {
    std::span args(argv + 1, argc - 1);
    parse_arguments(args);
  }

  try {
    auto renderer = engine::renderer();
    renderer.init();

    fmt::print(stderr, "Initialization completed.\n");

    renderer.main_loop();
  } catch (const std::exception &e) {
    fmt::print(stderr, "Error initializing rendering engine: {}\n", e.what());
  }
}
