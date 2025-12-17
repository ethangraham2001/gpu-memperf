"""
orchestrator.py - do-it-all benchmark orchestration for gpu-memperf
"""

from abc import abstractmethod, ABC
import pandas as pd
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
from plot_random_access import plot_all
from plot_strided_access import plot_strided_access_bandwidth

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
        num_blocks: list[int] | int | None = None,
        reps: int = 1,
    ):
        self.name = "random_access"
        self.num_warps = num_warps
        self.num_accesses = num_accesses
        self.working_set = working_set
        self.mode = mode
        self.data_type = data_type

        if num_blocks is None:
            self.num_blocks = [1]
        elif isinstance(num_blocks, int):
            self.num_blocks = [num_blocks]
        else:
            self.num_blocks = num_blocks
        self.reps = reps

    @classmethod
    def default_l1(cls):
        return cls(
            mode="l1",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=8 * 1024,
            data_type="f32",
            num_blocks=[1, 36, 72, 108],
            reps=3,
        )

    @classmethod
    def default_l2(cls):
        return cls(
            mode="l2",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=8 * 1024 * 1024,
            data_type="f32",
            num_blocks=[1, 36, 72, 108],
            reps=3,
        )

    @classmethod
    def default_dram(cls):
        return cls(
            mode="dram",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=64 * 1024 * 1024,
            data_type="f32",
            num_blocks=[1, 36, 72, 108],
            reps=3,
        )

    def get_args(self, blocks: int, reps_override: int = 1) -> list[str]:
        fmt_num_warps = ",".join(str(w) for w in self.num_warps)
        args = [
            self.name,
            f"--num_warps={fmt_num_warps}",
            f"--num_accesses={self.num_accesses}",
            f"--working_set={self.working_set}",
            f"--mode={self.mode}",
            f"--data_type={self.data_type}",
            f"--reps={reps_override}",
            f"--num_blocks={blocks}",
        ]
        return args

    def run(self) -> tuple[str, bool]:
        """
        Run the benchmark for random access.
        For each mode (l1, l2, dram), we run 3 measurements per block count (1, 36, 72, 108)
        to gather sufficient data for error bars plotting.
        
        :param self: Description
        :return: Description
        :rtype: tuple[str, bool]
        """
        merged_df = pd.DataFrame()
        last_out = ""
        last_failed = False
        
        final_result_path = None

        for idx, blocks in enumerate(self.num_blocks):
            for r in range(self.reps):
                print(f"Measuring random access for {self.mode} - {blocks} blocks - {r + 1}/{self.reps}", flush=True)
                
                # Pass reps=1 to binary so it runs once
                cmd = [gpu_memperf_bin] + self.get_args(blocks, reps_override=1)
                out, failed = run_command_manual_check(cmd)
                last_out = out
                last_failed = failed
                
                if failed:
                    return out, failed
                
                # Parse output dir from this run
                res_path = None
                for line in out.split("\n"):
                    if "wrote results to" in line:
                        match = re.search(r'"(.*?)"', line)
                        if match:
                            res_path = Path(match.group(1))
                            break
                
                if res_path and (res_path / "result.csv").exists():
                    df = pd.read_csv(res_path / "result.csv")
                    df["num_blocks"] = blocks
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
                    final_result_path = res_path
                else:
                    warn(f"Could not find results for block count {blocks}")

            print("")

        if final_result_path:
            merged_df.to_csv(final_result_path / "result.csv", index=False)
            
        return last_out, last_failed

    def plot(self, path_to_results: Path, plot_dir: Path):
        plot_all(path_to_results.parent, plot_dir)


class StridedAccessBenchmark(Benchmark):
    def __init__(
        self,
        mode: str,
        stride: list[int],
        working_sets : list[int],
        iters: int,
        threads_per_block: int,
        blocks: int,
        reps: int,
    ):
        self.name = "strided_access"
        self.mode = mode
        self.stride = stride
        self.working_sets = working_sets
        self.iters = iters
        self.threads_per_block = threads_per_block
        self.blocks = blocks
        self.reps = reps

    @classmethod
    def default_l1(cls):
        return cls(
            mode="L1",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            working_sets = [1024 * x for x in (100,)],
            threads_per_block=1024,
            blocks=0,
            reps=3,
        )

    @classmethod
    def default_l2(cls):
        return cls(
            mode="L2",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            working_sets = [(1024**2) * x for x in (25,)],
            threads_per_block=1024,
            blocks=0,
            reps=3,
        )

    @classmethod
    def default_dram(cls):
        return cls(
            mode="DRAM",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            working_sets = [(1024**3) * x for x in (4,)],
            threads_per_block=1024,
            blocks=0,
            reps=3,
        )

    def get_args(self) -> list[str]:
        fmt_stride = ",".join(str(s) for s in self.stride)
        fmt_working_sets = ",".join(str(ws) for ws in self.working_sets)
        return [
            self.name,
            f"--mode={self.mode}",
            f"--stride={fmt_stride}",
            f"--iters={self.iters}",
            f"--working_set={fmt_working_sets}",
            f"--threads_per_block={self.threads_per_block}",
            f"--blocks={self.blocks}",
            f"--reps={self.reps}",
        ]

    def plot(self, path_to_results: Path, plot_dir: Path):
        result_csv = path_to_results.joinpath("result.csv")
        outfile = plot_dir.joinpath(f"strided_access_{self.mode.lower()}_bandwidth.png")
        plot_strided_access_bandwidth(result_csv, outfile, mode=self.mode)


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
    RandomAccessL1 = "random_access_l1"
    RandomAccessL2 = "random_access_l2"
    RandomAccessDRAM = "random_access_dram"
    StridedAccessL1 = "strided_l1"
    StridedAccessL2 = "strided_l2"
    StridedAccessDRAM = "strided_dram"
    SharedToRegisters = "shared_to_registers"


class Orchestrator:
    def __init__(self, out_dir: str, programs: list[Program]):
        self.programs = programs if len(programs) > 0 else list(Program)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self._write_bench_info()

    def run_all(self):
        benches = map(lambda p: self._prog_to_bench(p), self.programs)
        for prog, bench in zip(self.programs, benches):
            out, failed = bench.run()
            info(out)
            if failed:
                warn(f"program {prog.value} failed")
                continue

            result_path = self._parse_bench_output_dir(out)
            if result_path is None:
                raise Exception("could not find an output file")

            new_result_path = self.out_dir.joinpath(prog.value)
            # Remove destination if it exists to allow overwriting
            if new_result_path.exists():
                shutil.rmtree(new_result_path)
            shutil.move(result_path, new_result_path)
            bench.plot(new_result_path, self.out_dir)

    @staticmethod
    def _prog_to_bench(prog: Program) -> Benchmark:
        match prog:
            case Program.GlobalToShared:
                return GlobalToSharedBenchmark.default()
            case Program.RandomAccessL1:
                return RandomAccessBenchmark.default_l1()
            case Program.RandomAccessL2:
                return RandomAccessBenchmark.default_l2()
            case Program.RandomAccessDRAM:
                return RandomAccessBenchmark.default_dram()
            case Program.StridedAccessL1:
                return StridedAccessBenchmark.default_l1()
            case Program.StridedAccessL2:
                return StridedAccessBenchmark.default_l2()
            case Program.StridedAccessDRAM:
                return StridedAccessBenchmark.default_dram()
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
