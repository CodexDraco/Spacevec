#include "renderer.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <span>

void parse_arguments(const auto &args) {
  for (auto const &arg : args) {
    std::cout << arg << '\n';
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

    std::cerr << "Initialization completed.\n";

    renderer.main_loop();
  } catch (const std::exception &e) {
    std::cerr << "Error initializing rendering engine: " << e.what() << '\n';
  }
}
