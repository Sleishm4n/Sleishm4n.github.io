#!/usr/bin/env python3
"""
Converts Criterion benchmark outputs into mentats/data/kernels.json for the website.

Scans `target/criterion/**/new/estimates.json` (or `base/estimates.json`), extracts
median nanoseconds and confidence intervals, auto-detects system hardware, and formats
the results for the site's microbenchmark charts.
"""

import argparse
import datetime
import json
import os
import platform
import re
import sys
from pathlib import Path


def get_hardware_info():
    """Detects CPU and OS info for benchmarking metadata."""
    cpu_info = platform.processor() or platform.machine()
    # On Windows, try querying the environment variable or registry if processor string is sparse
    if sys.platform == "win32":
        cpu_name = os.environ.get("PROCESSOR_IDENTIFIER", cpu_info)
        # Often friendly name is available in registry, fall back to cpu_name
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        import subprocess
        try:
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            cpu_name = cpu_info
    else:  # Linux
        cpu_name = cpu_info
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return {
        "cpu": cpu_name,
        "os": f"{platform.system()} {platform.release()}",
        "rustc": get_rustc_version(),
        "threads": 1,
    }


def get_rustc_version():
    import subprocess
    try:
        out = subprocess.check_output(["rustc", "--version"]).decode().strip()
        return out.split(" ")[1]
    except Exception:
        return "stable"


def estimate_flops(group: str, label: str):
    """Calculates floating point ops for common kernels if determinable from label."""
    # Matmul NxN: 2 * N^3 FLOPs
    if "matmul" in group.lower() or "matmul" in label.lower():
        match = re.search(r"(\d+)(?:x|×)?(?:\1)?", label)
        if match:
            n = int(match.group(1))
            return 2 * (n ** 3)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Criterion estimates.json into kernels.json")
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent.parent / "mentats" / "mentats" / "target" / "criterion",
        help="Path to target/criterion directory (default: ../mentats/target/criterion)",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "mentats" / "data" / "kernels.json",
        help="Path to output kernels.json (default: mentats/data/kernels.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    crit_dir = args.criterion_dir
    out_file = args.out_file

    if not crit_dir.exists():
        print(f"Error: Criterion directory not found at {crit_dir}", file=sys.stderr)
        print("Please run `cargo bench` in mentats first.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning for Criterion benchmarks in: {crit_dir}")

    benches = []

    # Criterion structures directories as:
    # <group>/<benchmark_id>/new/estimates.json OR <benchmark_id>/new/estimates.json
    for est_path in crit_dir.glob("**/new/estimates.json"):
        # Rel path: e.g. "matmul/128/new/estimates.json"
        rel_parts = est_path.relative_to(crit_dir).parts
        if len(rel_parts) < 3:
            continue

        # Ignore reports / summary directories
        if "report" in rel_parts:
            continue

        # Extract group and label
        if len(rel_parts) == 3:  # benchmark_name/new/estimates.json
            group = "general"
            bench_id = rel_parts[0]
            label = bench_id
        else:  # group/bench_name/new/estimates.json (or deeper)
            group = rel_parts[0]
            bench_id = "/".join(rel_parts[:-2])
            label = rel_parts[-3]

        try:
            with open(est_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # point_estimate in estimates.json is in nanoseconds
            median_ns = data.get("median", {}).get("point_estimate")
            if median_ns is None:
                median_ns = data.get("mean", {}).get("point_estimate")

            if median_ns is not None:
                flops = estimate_flops(group, label)
                benches.append({
                    "id": bench_id,
                    "group": group,
                    "label": label,
                    "median_ns": round(median_ns, 2),
                    "flops": flops,
                })
                print(f"  [+] {bench_id}: {median_ns / 1e6:.3f} ms" + (f" ({flops / median_ns:.2f} GFLOP/s)" if flops else ""))
        except Exception as e:
            print(f"Warning: could not parse {est_path}: {e}", file=sys.stderr)

    if not benches:
        print("No benchmark estimates found.", file=sys.stderr)
        sys.exit(1)

    # Sort benches by group then id
    benches.sort(key=lambda b: (b["group"], b["id"]))

    result = {
        "placeholder": False,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hardware": get_hardware_info(),
        "benches": benches,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSuccessfully generated {out_file} with {len(benches)} benchmarks!")


if __name__ == "__main__":
    main()
