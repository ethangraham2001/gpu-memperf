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
