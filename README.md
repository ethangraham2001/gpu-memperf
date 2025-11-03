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

## Usage

The idea is to run every benchmark as follows:

```
./gpu-memperf <benchname> <list-of-arguments>
```

For example, for the pchase_cpu benchmark for estimating CPU cache size, we
could call it like:

```sh
./gpu-memperf pchase_cpu --num_iters=10000 --multiplier=2
```

To run on the student cluster using the NVidia RTX 5060 Ti GPU, with benchmark default arguments, we could call it like:
```sh
srun -A dphpc --gpus 5060ti:1 ./gpu-memperf pchase_cpu
```

The full set of options, and how they should be parsed, are defined by the
individual benchmarks.

https://github.com/ethangraham2001/gpu-memperf/blob/b4daf09a9ae45d63b5c8d4a8d58ad4c8fd50c40b/include/PchaseCPUBenchmark.hh#L33-L37

## Quickstart

To get started quickly, we provide a google colab notebook that sets up the benchmarks with GPU. Copy the notebook and choose the benchmarks you want to run.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RcWO9uD51dTs9U8HxkvwoWA2Mi_OcHut?usp=sharing)

## Opinionated Style Guide

Please use `camelCase` instead of `snake_case`.

All comments should use a `/* */` block (not `//`) - they should begin with a
capital and end with a full stop. For documentation, use `/** */`. Try not to
leave whitespace at the ends of lines.

Please format all `.cc`, `.cu`, `.hh`, and `.cuh` files using `clang-format`
before committing. A `.clang-format` file is provided to maintain consistency in
the project. All source and include files can be formatted like so:

```sh
find . -type f \( -name "*.cc" -o -name "*.cu" -o -name "*.cuh" -o -name "*.hh" \) -exec clang-format -i {} +
```

## Commit Messages

Try and make commit messages descriptive, as it makes it easier to look through
the diff later on. I.e., avoid:

```
git commit -m "fix bug"

 README.md                     |  32 +++++++++++++++++++++++++++++++-
 include/ArgParser.hh          |  68 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 include/Benchmark.hh          |  67 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 include/PchaseCPUBenchmark.hh | 120 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 include/Util.hh               |  20 ++++++++++++++++++++
 src/main.cc                   |  24 ++++++++++++++++++++++--
 6 files changed, 328 insertions(+), 3 deletions(-)
```

The kernel has a nice, albeit opinionated, style of commit messages. It looks
something like so:

```
path/to/modified: short description, less than 65 columns

A longer description if required. This can be a single paragraph, or
multiple. Each line in this section should be no longer than 72
columns long.

If the commit fixes a bug from another commit, it can sometimes be
helpful to add the commit hash, e.g.,

fixes: 7f8d73ca214a

If you wanna be really fancy you can sign your commits off with
`git commit -s`.

Signed-off-by: John Doe <jd@example.com>
```
