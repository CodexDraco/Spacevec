{ pkgs ? import <nixpkgs> {} }:
pkgs.llvmPackages_16.libcxxStdenv.mkDerivation {
  name = "env";
  nativeBuildInputs = with pkgs; [
    cmake
    clang-tools_16
    ninja
    vulkan-headers
    vulkan-loader
    vulkan-validation-layers
    shaderc
  ];
  buildInputs = with pkgs; [
    glfw
    fmt
  ];
}
