{ pkgs ? import <nixpkgs> {} }:
let new-vk-bootstrap = pkgs.vk-bootstrap.overrideAttrs (old: {
  version = "1.3.268";
  src = pkgs.fetchFromGitHub {
        owner = "charles-lunarg";
        repo = "vk-bootstrap";
        rev = "v1.3.268";
        hash = "sha256-Vp3dHLqf19xXWVaTYcLS/ZZme1RI9UqENOqIzsFIRms=";
  };
  postPatch = ''
    # Upstream uses cmake FetchContent to resolve glfw and catch2
    # needed for examples and tests
    sed -iE 's=add_subdirectory(ext)==g' CMakeLists.txt
    sed -iE 's=add_subdirectory(tests)==g' CMakeLists.txt
  '';
});
in
pkgs.stdenv.mkDerivation {
  name = "spacevec-shell";
  nativeBuildInputs = with pkgs; [
    cmake
    clang-tools_16
    ninja
    shaderc
  ];
  buildInputs = with pkgs; [
    glfw
    fmt
    glm
    new-vk-bootstrap
    vulkan-headers
    vulkan-validation-layers
  ];
}
