"""
orchestrator.py - do-it-all benchmark orchestration for gpu-memperf
"""

from __future__ import annotations

from abc import abstractmethod, ABC
import pandas as pd
from pathlib import Path
from typing import Any
from enum import Enum
import subprocess
import argparse
import shutil
import json
import re

from plot_global_to_shared import plot_global_to_shared
from plot_shared_to_register import (
    plot_shared_memory_error_bars,
    plot_shared_memory_all_strides,
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

    @abstractmethod
    def bench_name(self) -> str:
        pass

    def run(self) -> tuple[str, bool]:
        subprocess_cmd = [gpu_memperf_bin] + self.get_args()
        return run_command_manual_check(subprocess_cmd)

    @staticmethod
    def parse_results_path(stdout: str) -> Path | None:
        for line in stdout.split("\n"):
            if "wrote results to" not in line:
                continue

            results_path = re.search(r'"(.*?)"', line)
            if results_path is None:
                return None
            return Path(results_path.group(1))

    @abstractmethod
    def get_csv(self, path_to_results: Path) -> pd.DataFrame:
        pass


class GlobalToSharedBenchmark(Benchmark):
    def __init__(
        self,
        flops_per_elem: list[int],
        threads_per_block: int,
        num_blocks: int,
        reps: int = 1,
    ):
        self.name = "global_to_shared"
        self.flops_per_elem = flops_per_elem
        self.threads_per_block = threads_per_block
        self.num_blocks = num_blocks
        self.reps = reps
        self.result_file = "result.csv"

    @classmethod
    def default(cls, reps: int = 1):
        return cls(
            flops_per_elem=[2**i for i in range(8)],
            threads_per_block=1024,
            num_blocks=108,
            reps=reps,
        )

    def get_args(self) -> list[str]:
        fmt_flops_per_elem = ",".join([str(flops) for flops in self.flops_per_elem])
        return [
            self.name,
            f"--flops_per_elem={fmt_flops_per_elem}",
            f"--threads_per_block={self.threads_per_block}",
            f"--num_blocks={self.num_blocks}",
        ]

    def run(self) -> tuple[str, bool]:
        merged_df = pd.DataFrame()
        last_out = ""
        last_failed = False
        final_result_path = None

        for r in range(self.reps):
            print(f"Measuring global_to_shared - {r + 1}/{self.reps}", flush=True)
            cmd = [gpu_memperf_bin] + self.get_args()
            out, failed = run_command_manual_check(cmd)
            last_out = out
            last_failed = failed

            if failed:
                return out, failed

            # Parse output dir
            res_path = None
            for line in out.split("\n"):
                if "wrote results to" in line:
                    match = re.search(r'"(.*?)"', line)
                    if match:
                        res_path = Path(match.group(1))
                        break

            if res_path and (res_path / self.result_file).exists():
                df = pd.read_csv(res_path / self.result_file)
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                final_result_path = res_path
            else:
                warn(f"Could not find results for global_to_shared rep {r}")

        if final_result_path:
            merged_df.to_csv(final_result_path / self.result_file, index=False)

        return last_out, last_failed

    def plot(self, path_to_results: Path, plot_dir: Path):
        result_csv = path_to_results.joinpath(self.result_file)
        plot_global_to_shared(result_csv, plot_dir.joinpath("global_to_shared_bw.png"))

    def bench_name(self):
        return self.name

    def get_csv(self, path_to_results: Path) -> pd.DataFrame:
        result_csv = path_to_results.joinpath(self.result_file)
        return pd.read_csv(result_csv)


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
    def default_l1(cls, reps: int = 1):
        return cls(
            mode="l1",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=8 * 1024,
            data_type="f32",
            num_blocks=[1, 36, 72, 108],
            reps=reps,
        )

    @classmethod
    def default_l2(cls, reps: int = 1):
        return cls(
            mode="l2",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=8 * 1024 * 1024,
            data_type="f32",
            num_blocks=[1, 36, 72, 108],
            reps=reps,
        )

    @classmethod
    def default_dram(cls, reps: int = 1):
        return cls(
            mode="dram",
            num_warps=[2**i for i in range(6)],
            num_accesses=int(1e7),
            working_set=2 * 1024 * 1024 * 1024,
            data_type="f32",
            num_blocks=[1, 36, 72, 108],
            reps=reps,
        )

    def get_args(self):
        fmt_num_warps = ",".join(str(w) for w in self.num_warps)
        args = [
            self.name,
            f"--num_warps={fmt_num_warps}",
            f"--num_accesses={self.num_accesses}",
            f"--working_set={self.working_set}",
            f"--mode={self.mode}",
            f"--data_type={self.data_type}",
            f"--reps={self.reps}",
            f"--num_blocks={self.num_blocks}",
        ]
        return args

    def run(self) -> tuple[str, bool]:
        """
        Run the benchmark for random access.
        For each mode (l1, l2, dram), we run multiple measurements per block count (1, 36, 72, 108)
        to gather sufficient data for error bars plotting.
        """
        merged_df = pd.DataFrame()
        last_out = ""
        last_failed = False

        final_result_path = None

        for idx, blocks in enumerate(self.num_blocks):
            for r in range(self.reps):
                print(
                    f"Measuring random access for {self.mode} - {blocks} blocks - {r + 1}/{self.reps}",
                    flush=True,
                )

                # Pass reps=1 to binary so it runs once
                cmd = [gpu_memperf_bin] + self.get_args()
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

    def get_csv(self, path_to_results: Path) -> pd.DataFrame:
        return pd.read_csv(path_to_results.joinpath("result.csv"))

    def bench_name(self):
        return self.name


class StridedAccessBenchmark(Benchmark):
    def __init__(
        self,
        mode: str,
        stride: list[int],
        working_sets: list[int],
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
    def default_l1(cls, reps: int = 3):
        return cls(
            mode="L1",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            working_sets=[1024 * x for x in (100,)],
            threads_per_block=1024,
            blocks=0,
            reps=reps,
        )

    @classmethod
    def default_unit_stride_l1(cls, reps: int = 3):
        ret = StridedAccessBenchmark.default_l1(reps)
        ret.stride = [1]
        return ret

    @classmethod
    def default_l2(cls, reps: int = 3):
        return cls(
            mode="L2",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            working_sets=[(1024**2) * x for x in (25,)],
            threads_per_block=1024,
            blocks=0,
            reps=reps,
        )

    @classmethod
    def default_unit_stride_l2(cls, reps: int = 3):
        ret = StridedAccessBenchmark.default_l2(reps)
        ret.stride = [1]
        return ret

    @classmethod
    def default_dram(cls, reps: int = 3):
        return cls(
            mode="DRAM",
            stride=[2**i for i in range(6)],
            iters=int(1e6),
            working_sets=[(1024**3) * x for x in (4,)],
            threads_per_block=1024,
            blocks=0,
            reps=reps,
        )

    @classmethod
    def default_unit_stride_dram(cls, reps: int = 3):
        ret = StridedAccessBenchmark.default_dram(reps)
        ret.stride = [1]
        return ret

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
            f"--data_type=f64",
        ]

    def plot(self, path_to_results: Path, plot_dir: Path):
        result_csv = path_to_results.joinpath("result.csv")
        outfile = plot_dir.joinpath(f"strided_access_{self.mode.lower()}_bandwidth.png")
        plot_strided_access_bandwidth(result_csv, outfile, mode=self.mode)

    def bench_name(self):
        return self.name

    def get_csv(self, path_to_results: Path) -> pd.DataFrame:
        return pd.read_csv(path_to_results.joinpath("result.csv"))


class PChaseGPUBenchmark(Benchmark):
    def __init__(
        self,
        num_experiments: int,
        multiplier: int,
        iters: int,
        start_bytes: int,
        cache_line_size: int,
    ):
        size_of_double = 8
        self.num_experiments = num_experiments
        self.multiplier = multiplier
        self.iters = iters
        self.start_bytes = start_bytes
        # It feels more intuitive to express this in terms of the size of the
        # cache line on the given arch than as a stride.
        self.stride = cache_line_size // size_of_double

        self.latency_csv = "latency.csv"

    @classmethod
    def default(cls):
        return cls(
            num_experiments=12,
            multiplier=2,
            iters=1000000,
            start_bytes=1 << 16,
            cache_line_size=128,
        )

    def get_args(self) -> list[str]:
        return [
            self.bench_name(),
            f"--num_experiments={self.num_experiments}",
            f"--multiplier={self.multiplier}",
            f"--num_iters={self.iters}"
            f"--stride={self.stride}"
            f"--thresh_coarse={1.2}",  # It isn't clear what a sensible value is yet.
        ]

    def bench_name(self) -> str:
        return "pchase_gpu"

    def plot(self, path_to_results: Path, plot_dir: Path):
        result_csv = path_to_results.joinpath(self.latency_csv)
        warn(f"cannot plot results from {result_csv}: not implemented")

    def get_csv(self, path_to_results: Path):
        result_csv = path_to_results.joinpath(self.latency_csv)
        return pd.read_csv(result_csv)

    @staticmethod
    def approximate_cache_sizes(df: pd.DataFrame) -> dict[str, dict[str, str]]:
        """
        _approximate_cache_sizes - given the latency measurements, estimate the
                                   sizes of the L1/L2 caches.

        EG: I'm not 100% sure how this is working as it's very mathy. I'll try
        and comment it out so that it's at least clear what it is doing,
        conceptually. It works pretty well for samples collected on RTX5060Ti.
        """
        df = df.sort_values(by="bytes").reset_index(drop=True)
        # Smooth the resulting curve aggressively to flatten spikes using a
        # median filter. Window=10 is chosen heuristically.
        df["latency_smooth"] = (
            df["avg_access_latency"]
            .rolling(window=10, center=True)
            .median()
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

        # Detect potential boundaries using the gradient method - we are looking
        # for locations where the latency spikes rapidly.
        df["grad"] = df["latency_smooth"].diff().fillna(0)
        threshold = df["grad"].max() * 0.05
        jump_indices = df.index[df["grad"] > threshold].tolist()

        # Group consecutive jumps. A single physical jump may span tens of data
        # points in the noisy data. We group them, taking the start of the jump
        # as the boundary candidate.
        boundaries = []
        if jump_indices:
            group = [jump_indices[0]]
            for idx in jump_indices[1:]:
                if idx - group[-1] <= 10:  # If points are close, same jump
                    group.append(idx)
                else:
                    boundaries.append(group[0])
                    group = [idx]
            boundaries.append(group[0])

        # Filter out for true plateaus, which is defined as the region between
        # two boundaries.
        valid_levels = []

        # Add start(0) and end(len) to boundaries to define all regions.
        region_points = [0] + boundaries + [len(df)]

        for i in range(len(region_points) - 1):
            start_idx = region_points[i]
            end_idx = region_points[i + 1]
            safe_end_idx = min(end_idx, len(df) - 1)

            end_bytes = df["bytes"].iloc[safe_end_idx]
            median_latency = df["avg_access_latency"].iloc[start_idx:end_idx].median()

            # Store the END of this region as the capacity limit
            # (Exception: The last region is DRAM/Global, which has no limit in
            # this view).
            is_last = i == len(region_points) - 2

            valid_levels.append(
                {"capacity": end_bytes, "latency": median_latency, "is_last": is_last}
            )

        out = {}
        for i, level in enumerate(valid_levels):
            if level["is_last"]:
                name = "DRAM"
                size_str = "N/A"
            else:
                name = f"L{i + 1}"
                size_str = f"{level["capacity"]}"
            out[name] = {"capacity": size_str, "latency": level["latency"]}

        return out


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
    def default(cls, reps: int = 3):
        return cls(
            sizes=[4096 * (2**i) for i in range(4)],
            threads=[32 * (2**i) for i in range(4)],
            strides=[2**i for i in range(6)],
            num_iters=int(1e5),
            reps=reps,
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
        plot_shared_memory_all_strides(
            result_csv, plot_dir.joinpath("shared_to_regs_all_strides.png")
        )

    def bench_name(self):
        return self.name


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
    PChaseGPU = "pchase_gpu"


class Mode(ABC):
    @staticmethod
    @abstractmethod
    def init_subparser(parser: argparse.ArgumentParser):
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def init_from_args(args: argparse.Namespace) -> "Mode":
        pass

    @abstractmethod
    def run(self):
        pass


class Orchestrator(Mode):
    def __init__(self, out_dir: str, programs: list[Program], reps: int | None = None):
        self.programs = programs if len(programs) > 0 else list(Program)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.reps = reps
        self._write_bench_info()

    @staticmethod
    def init_from_args(args):
        return Orchestrator(out_dir=args.out, programs=args.program, reps=args.reps)

    def run(self):
        benches = map(lambda p: self._prog_to_bench(p, self.reps), self.programs)
        for prog, bench in zip(self.programs, benches):
            out, failed = bench.run()
            info(out)
            if failed:
                warn(f"program {prog.value} failed")
                continue

            result_path = Benchmark.parse_results_path(out)
            if result_path is None:
                raise Exception("could not find an output file")

            new_result_path = self.out_dir.joinpath(prog.value)
            # Remove destination if it exists to allow overwriting
            if new_result_path.exists():
                shutil.rmtree(new_result_path)
            shutil.move(result_path, new_result_path)
            bench.plot(new_result_path, self.out_dir)

    @staticmethod
    def name():
        return "orchestrator"

    @staticmethod
    def init_subparser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output"
        )
        parser.add_argument(
            "--program",
            type=Program,
            action="append",
            default=[],
            help="Run this benchmark when enabled. If none are specified, all enabled by default",
        )
        parser.add_argument("--out", type=str, default="orchestrator_out")
        parser.add_argument(
            "--reps",
            type=int,
            default=None,
            help="Number of repetitions for benchmarks",
        )

    @staticmethod
    def _prog_to_bench(prog: Program, reps: int | None) -> Benchmark:
        # If global reps is specified, use it.
        kwargs = {}
        if reps is not None:
            kwargs["reps"] = reps

        match prog:
            case Program.GlobalToShared:
                return GlobalToSharedBenchmark.default(**kwargs)
            case Program.RandomAccessL1:
                return RandomAccessBenchmark.default_l1(**kwargs)
            case Program.RandomAccessL2:
                return RandomAccessBenchmark.default_l2(**kwargs)
            case Program.RandomAccessDRAM:
                return RandomAccessBenchmark.default_dram(**kwargs)
            case Program.StridedAccessL1:
                return StridedAccessBenchmark.default_l1(**kwargs)
            case Program.StridedAccessL2:
                return StridedAccessBenchmark.default_l2(**kwargs)
            case Program.StridedAccessDRAM:
                return StridedAccessBenchmark.default_dram(**kwargs)
            case Program.SharedToRegisters:
                return SharedToRegisterBenchmark.default(**kwargs)
            case Program.PChaseGPU:
                return PChaseGPUBenchmark.default()

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


class CacheSizeFinder(Mode):
    """
    CacheSizeFinder - approximate the detectable sizes of L1 / L2 cache
    """

    def __init__(self):
        self.pchase_bench = PChaseGPUBenchmark.default()

    @staticmethod
    def init_from_args(args):
        return CacheSizeFinder()

    def run(self):
        df = self._run_and_load_csv()
        info(df.to_string())
        result = PChaseGPUBenchmark.approximate_cache_sizes(df)
        print(json.dumps(result))

    def _run_and_load_csv(self) -> pd.DataFrame:
        stdout, failed = self.pchase_bench.run()
        if failed:
            raise RuntimeError(f"failed to run {self.pchase_bench.bench_name()}")
        results_path = Benchmark.parse_results_path(stdout)
        if results_path is None:
            raise RuntimeError("failed to parse results path")
        return self.pchase_bench.get_csv(results_path)

    @staticmethod
    def name():
        return "cache-sizes"

    @staticmethod
    def init_subparser(parser: argparse.ArgumentParser):
        # No-op for now.
        pass


class PropertiesFinder(Mode):
    """
    PropertiesFinder - find GPU memory properties and write them as JSON
    """

    def __init__(self, out_file: Path, reps: int = 3):
        self.out_file = out_file
        self.reps = reps

    @staticmethod
    def init_from_args(args):
        return PropertiesFinder(out_file=Path(args.out))

    def _strided_bandwidth(self) -> dict[str, Any]:
        l1_bw = self._measure_bw(StridedAccessBenchmark.default_unit_stride_l1())
        l2_bw = self._measure_bw(StridedAccessBenchmark.default_unit_stride_l2())
        dram_bw = self._measure_bw(StridedAccessBenchmark.default_unit_stride_dram())
        return {
            "l1_cache": l1_bw,
            "l2_cache": l2_bw,
            "dram": dram_bw,
        }

    def _system_info(self) -> dict[str, Any]:
        commit, _, _ = get_git_info()
        return {
            "commit": commit,
        }

    def _cache_sizes(self) -> dict[str, Any]:
        bench = PChaseGPUBenchmark.default()
        stdout, failed = bench.run()
        if failed:
            print(stdout)
            raise RuntimeError(f"{bench.bench_name()} failed")

        path = Benchmark.parse_results_path(stdout)
        if path is None:
            raise RuntimeError("unable to parse output path")

        df = bench.get_csv(path)
        return PChaseGPUBenchmark.approximate_cache_sizes(df)

    @staticmethod
    def name():
        return "properties"

    @staticmethod
    def init_subparser(parser: argparse.ArgumentParser):
        parser.add_argument("--out", type=str, default="properties.json")

    def _measure_bw(self, bench: Benchmark) -> dict[str, float]:
        dfs = []
        for rep in range(self.reps):
            stdout, failed = bench.run()
            if failed:
                warn(f"{bench.bench_name()} bandwidth rep#{rep}: measurement failed")
                continue

            results_path = Benchmark.parse_results_path(stdout)
            if results_path is None:
                warn(
                    f"{bench.bench_name()} bandwidth rep#{rep}: failed to read results"
                )
                continue

            dfs.append(bench.get_csv(results_path))

        if len(dfs) == 0:
            return {}

        merged_df = pd.concat(dfs, ignore_index=True)
        bw_col = merged_df["bandwidth"]

        return {
            "max": float(bw_col.max()),
            "min": float(bw_col.min()),
            "mean": float(bw_col.mean()),
            "median": float(bw_col.median()),
        }

    def run(self):
        data = {
            "system_info": self._system_info(),
            "linear_bandwidth": self._strided_bandwidth(),
            "estimated_cache_properties": self._cache_sizes(),
        }
        with open(self.out_file, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpu-memperf benchmark orchestrator")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    mode_registry = {cls.name(): cls for cls in Mode.__subclasses__()}

    for name, mode in mode_registry.items():
        sp = subparsers.add_parser(name)
        mode.init_subparser(sp)

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    if args.mode in mode_registry:
        mode = mode_registry[args.mode]
        instance: Mode = mode.init_from_args(args)
        instance.run()

    else:
        parser.print_help()
