# `gpu-memperf`

## Building The Project

This project requires the NVIDIA toolchain. It is recommended to have a
relatively up-to-date version of this. Ubuntu instructions can be found on
[this page](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). On
Arch, it can be installed like so:

```sh
pacman -S cuda
```

Then, just run `make` (or `make all`) to build the binary.

## Code Style

Please format all `.cc`, `.cu`, `.hh`, and `.cuh` files using `clang-format`
before committing. A `.clang-format` file is provided to maintain consistency in
the project. All source and include files can be formatted like so:

```sh
find . -type f \( -name "*.cc" -o -name "*.cu" -o -name "*.cuh" -o -name "*.hh" \) -exec clang-format -i {} +
```
