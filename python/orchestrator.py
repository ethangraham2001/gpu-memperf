"""
orchestrator.py - do-it-all benchmark orchestration for gpu-memperf
"""

from abc import abstractmethod, ABC
from pathlib import Path
from enum import Enum
import subprocess
import argparse
import shutil
import re

from plot_global_to_shared import plot_global_to_shared
from plot_shared_to_register import (
    plot_shared_memory_error_bars,
    plot_shared_memory_multiple_threads,
)

# Path to the gpu-memperf compiled binary.
gpu_memperf_bin = "./gpu-memperf"
verbose = False


def run_command_manual_check(cmd: list[str]) -> tuple[str, bool]:
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        if not result.stdout.strip():
            return "", False
        else:
            return result.stdout, False
    else:
        return f"stdout: {result.stdout}\nstderr: {result.stderr}", True


def warn(s: str):
    print(f"[WARNING]: {s}")


def info(s: str):
    if verbose:
        print(s)


class Benchmark(ABC):
    @abstractmethod
    def get_args(self) -> list[str]:
        pass

    @abstractmethod
    def plot(self, path_to_results: Path, plot_dir: Path):
        pass

    def run(self) -> tuple[str, bool]:
        subprocess_cmd = [gpu_memperf_bin] + self.get_args()
        return run_command_manual_check(subprocess_cmd)


class GlobalToSharedBenchmark(Benchmark):
    def __init__(
        self, flops_per_elem: list[int], threads_per_block: int, num_blocks: int
    ):
        self.name = "global_to_shared"
        self.flops_per_elem = flops_per_elem
        self.threads_per_block = threads_per_block
        self.num_blocks = num_blocks

    @classmethod
    def default(cls):
        return cls(
            flops_per_elem=[2**i for i in range(8)],
            threads_per_block=1024,
            num_blocks=108,
        )

    def get_args(self) -> list[str]:
        fmt_flops_per_elem = ",".join([str(flops) for flops in self.flops_per_elem])
        return [
            self.name,
            f"--flops_per_elem={fmt_flops_per_elem}",
            f"--threads_per_block={self.threads_per_block}",
            f"--num_blocks={self.num_blocks}",
        ]

    def plot(self, path_to_results: Path, plot_dir: Path):
        result_csv = path_to_results.joinpath("result.csv")
        plot_global_to_shared(result_csv, plot_dir.joinpath("global_to_shared_bw.png"))


class RandomAccessBenchmark(Benchmark):
    def __init__(
        self,
        num_warps: list[int],
        num_accesses: int,
        working_set: int,
        mode: str,
        data_type: str,
    ):
        self.name = "random_access"
        self.num_warps = num_warps
        self.num_accesses = num_accesses
        self.working_set = working_set
        self.mode = mode
        self.data_type = data_type

    @classmethod
    def default_l1(cls):
        return cls(
            mode="l1",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=8 * 1024,
            data_type="f32",
        )

    def get_args(self) -> list[str]:
        fmt_num_warps = ",".join(str(w) for w in self.num_warps)
        return [
            self.name,
            f"--num_warps={fmt_num_warps}",
            f"--num_accesses={self.num_accesses}",
            f"--working_set={self.working_set}",
            f"--mode={self.mode}",
            f"--data_type={self.data_type}",
        ]

    def plot(self, path_to_results: Path, plot_dir: Path):
        warn("TODO plotting for random access benchmark")


class StridedAccessBenchmark(Benchmark):
    def __init__(
        self,
        mode: str,
        stride: list[int],
        iters: int,
        threads_per_block: int,
        blocks: int,
    ):
        self.name = "strided_access"
        self.mode = mode
        self.stride = stride
        self.iters = iters
        self.threads_per_block = threads_per_block
        self.blocks = blocks

    @classmethod
    def default_l1(cls):
        return cls(
            mode="L1",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            threads_per_block=1024,
            blocks=0,
        )

    def get_args(self) -> list[str]:
        fmt_stride = ",".join(str(s) for s in self.stride)
        return [
            self.name,
            f"--mode={self.mode}",
            f"--stride={fmt_stride}",
            f"--iters={self.iters}",
            f"--threads_per_block={self.threads_per_block}",
            f"--blocks={self.blocks}",
        ]

    def plot(self, path_to_results: Path, plot_dir: Path):
        warn("TODO plotting for strided benchmark")


class SharedToRegisterBenchmark(Benchmark):
    def __init__(
        self,
        sizes: list[int],
        threads: list[int],
        strides: list[int],
        num_iters: int,
        reps: int,
    ):
        self.name = "shared_to_register"
        self.sizes = sizes
        self.threads = threads
        self.strides = strides
        self.num_iters = num_iters
        self.reps = reps

    @classmethod
    def default(cls):
        return cls(
            sizes=[4096 * (2**i) for i in range(4)],
            threads=[32 * (2**i) for i in range(4)],
            strides=[2**i for i in range(6)],
            num_iters=int(1e5),
            reps=3,
        )

    def get_args(self) -> list[str]:
        fmt_sizes = ",".join(str(size) for size in self.sizes)
        fmt_threads = ",".join(str(t) for t in self.threads)
        fmt_strides = ",".join(str(stride) for stride in self.strides)
        return [
            self.name,
            f"--sizes={fmt_sizes}",
            f"--threads={fmt_threads}",
            f"--strides={fmt_strides}",
            f"--num_iters={self.num_iters}",
            f"--reps={self.reps}",
        ]

    def plot(self, path_to_results: Path, plot_dir: Path):
        result_csv = path_to_results.joinpath("result.csv")
        plot_shared_memory_error_bars(
            result_csv, plot_dir.joinpath("shared_to_regs_error_bars.png")
        )
        plot_shared_memory_multiple_threads(
            result_csv, plot_dir.joinpath("shared_to_regs_multiple_threads.png")
        )


def get_git_info():
    """
    get_git_info - retrieve git information on the current environment

    Returns:
        - The commit hash of HEAD
        - The active git branch
        - True if the active repository is dirty
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return commit, branch, dirty
    except:
        return "unknown", "unknown", False


class Program(Enum):
    GlobalToShared = "global_to_shared"
    RandomAccessL1 = "random_l1"
    StridedAccessL1 = "strided_l1"
    SharedToRegisters = "shared_to_registers"


class Orchestrator:
    def __init__(self, out_dir: str, programs: list[Program]):
        self.programs = programs if len(programs) > 0 else list(Program)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir()
        self._write_bench_info()

    def run_all(self):
        benches = map(lambda p: self._prog_to_bench(p), self.programs)
        for prog, bench in zip(self.programs, benches):
            out, failed = bench.run()
            info(out)
            if failed:
                warn(f"WARNING: program {prog.value} failed")
                continue

            result_path = self._parse_bench_output_dir(out)
            if result_path is None:
                raise Exception("could not find an output file")

            new_result_path = self.out_dir.joinpath(prog.value)
            shutil.move(result_path, new_result_path)
            bench.plot(new_result_path, self.out_dir)

    @staticmethod
    def _prog_to_bench(prog: Program) -> Benchmark:
        match prog:
            case Program.GlobalToShared:
                return GlobalToSharedBenchmark.default()
            case Program.RandomAccessL1:
                return RandomAccessBenchmark.default_l1()
            case Program.StridedAccessL1:
                return StridedAccessBenchmark.default_l1()
            case Program.SharedToRegisters:
                return SharedToRegisterBenchmark.default()

    def _write_bench_info(self):
        info_path = self.out_dir.joinpath("info")
        info_path.touch()
        with info_path.open("w") as info:
            commit, branch, dirty = get_git_info()
            info.write(f"Git commit: {commit}\n")
            info.write(f"Git branch: {branch}\n")
            info.write(
                f"Git diff:   {'uncomitted changes deteced' if dirty else 'clean'}\n"
            )

    def _parse_bench_output_dir(self, bench_result: str) -> Path | None:
        for line in bench_result.split("\n"):
            if "wrote results to" not in line:
                continue

            results_path = re.search(r'"(.*?)"', line)
            if results_path is None:
                return None
            return Path(results_path.group(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpu-memperf benchmark orchestrator")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--program",
        type=Program,
        action="append",
        default=[],
        help="Run this benchmark when enabled. If none are specified, all enabled by default",
    )
    parser.add_argument("--out", type=str, default="orchestrator_out")
    args = parser.parse_args()
    verbose = args.verbose

    orchestrator = Orchestrator(programs=args.program, out_dir=args.out)
    orchestrator.run_all()
