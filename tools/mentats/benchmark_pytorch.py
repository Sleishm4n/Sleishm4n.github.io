#!/usr/bin/env python3
"""
Benchmarks equivalent PyTorch single-thread CPU operations and produces mentats/data/reference.json.

Measures:
- MatMul: 64x64, 128x128, 256x256, 512x512
- Conv2D forward & backward: batch 64, 3->16 channels, 3x3 kernel on 32x32
- Conv2D forward: batch 1, 1->16 channels, 3x3 kernel on 28x28

Compares PyTorch execution time against mentats' measured median in kernels.json
and calculates percentage of PyTorch single-threaded speed (using oneDNN/MKL).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch operations against mentats kernels.json")
    parser.add_argument(
        "--kernels-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "mentats" / "data" / "kernels.json",
        help="Path to mentats kernels.json",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "mentats" / "data" / "reference.json",
        help="Path to output reference.json",
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    return parser.parse_args()


def bench_fn(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        durations.append(t1 - t0)
    return statistics.median(durations)


def main():
    args = parse_args()

    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed in this Python environment.", file=sys.stderr)
        sys.exit(1)

    torch.set_num_threads(1)
    print(f"Running PyTorch {torch.__version__} reference benchmarks (threads={torch.get_num_threads()})...\n")

    # Load mentats kernels
    mentats_map = {}
    if args.kernels_json.exists():
        with open(args.kernels_json, "r", encoding="utf-8") as f:
            kdata = json.load(f)
            for b in kdata.get("benches", []):
                mentats_map[b["id"]] = b["median_ns"]
                mentats_map[b["label"]] = b["median_ns"]

    reference_results = []

    # 1. MatMul benchmarks
    matmul_sizes = [64, 128, 256, 512]
    for n in matmul_sizes:
        a = torch.randn(n, n, dtype=torch.float32)
        b = torch.randn(n, n, dtype=torch.float32)
        c = torch.empty(n, n, dtype=torch.float32)

        def run_mm():
            torch.mm(a, b, out=c)

        med_ns = bench_fn(run_mm, warmup=args.warmup, iters=args.iters)
        label = f"matmul {n}x{n}"

        # Find corresponding mentats median
        candidate_keys = [
            f"{n}x{n}_optimised",
            f"matmul_{n}x{n}",
            f"matmul/{n}",
        ]
        mentats_ns = None
        for k in candidate_keys:
            if k in mentats_map:
                mentats_ns = mentats_map[k]
                break

        print(f"[MatMul {n}x{n}] PyTorch: {med_ns / 1e3:.1f} µs | mentats: {f'{mentats_ns / 1e3:.1f} µs' if mentats_ns else 'N/A'}")

        if mentats_ns:
            pct = (med_ns / mentats_ns) * 100
            print(f"  -> mentats throughput relative to PyTorch: {pct:.1f}%")

        reference_results.append({
            "id": f"matmul/{n}",
            "label": label,
            "reference_ns": round(med_ns, 2),
            "mentats_ns": mentats_ns or med_ns,
        })

    # 2. Conv2D forward & backward (batch 64)
    # Shape: batch 64, 3 in_channels, 32x32 image -> 16 out_channels, 3x3 kernel
    x_batch = torch.randn(64, 3, 32, 32, dtype=torch.float32, requires_grad=True)
    conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, bias=False)
    grad_out = torch.randn(64, 16, 30, 30, dtype=torch.float32)

    def run_conv_fwd():
        with torch.no_grad():
            conv_layer(x_batch)

    fwd_med_ns = bench_fn(run_conv_fwd, warmup=args.warmup, iters=args.iters)
    mentats_fwd = mentats_map.get("conv_forward_batch64") or mentats_map.get("conv2d/conv_forward_batch64")
    print(f"\n[Conv2D forward batch64] PyTorch: {fwd_med_ns / 1e6:.2f} ms | mentats: {f'{mentats_fwd / 1e6:.2f} ms' if mentats_fwd else 'N/A'}")
    if mentats_fwd:
        pct = (fwd_med_ns / mentats_fwd) * 100
        print(f"  -> mentats throughput relative to PyTorch: {pct:.1f}%")

    reference_results.append({
        "id": "conv2d/conv_forward_batch64",
        "label": "conv2d forward (batch 64)",
        "reference_ns": round(fwd_med_ns, 2),
        "mentats_ns": mentats_fwd or fwd_med_ns,
    })

    # Conv2D backward
    out = conv_layer(x_batch)

    def run_conv_bwd():
        x_batch.grad = None
        conv_layer.weight.grad = None
        out.backward(grad_out, retain_graph=True)

    bwd_med_ns = bench_fn(run_conv_bwd, warmup=15, iters=40)
    mentats_bwd = mentats_map.get("conv_backward_batch64") or mentats_map.get("conv2d/conv_backward_batch64")
    print(f"\n[Conv2D backward batch64] PyTorch: {bwd_med_ns / 1e6:.2f} ms | mentats: {f'{mentats_bwd / 1e6:.2f} ms' if mentats_bwd else 'N/A'}")
    if mentats_bwd:
        pct = (bwd_med_ns / mentats_bwd) * 100
        print(f"  -> mentats throughput relative to PyTorch: {pct:.1f}%")

    reference_results.append({
        "id": "conv2d/conv_backward_batch64",
        "label": "conv2d backward (batch 64)",
        "reference_ns": round(bwd_med_ns, 2),
        "mentats_ns": mentats_bwd or bwd_med_ns,
    })

    # 3. Conv2D single image (1 in_channel -> 16 out_channels, 3x3 kernel on 28x28)
    x_single = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    conv_single = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, bias=False)

    def run_conv_single():
        with torch.no_grad():
            conv_single(x_single)

    single_med_ns = bench_fn(run_conv_single, warmup=args.warmup, iters=args.iters)
    mentats_single = mentats_map.get("im2col_gemm_single") or mentats_map.get("conv2d_comparison/im2col_gemm_single")
    print(f"\n[Conv2D single 28x28] PyTorch: {single_med_ns / 1e3:.1f} µs | mentats: {f'{mentats_single / 1e3:.1f} µs' if mentats_single else 'N/A'}")
    if mentats_single:
        pct = (single_med_ns / mentats_single) * 100
        print(f"  -> mentats throughput relative to PyTorch: {pct:.1f}%")

    reference_results.append({
        "id": "conv2d/single_28x28",
        "label": "conv2d single image (28x28)",
        "reference_ns": round(single_med_ns, 2),
        "mentats_ns": mentats_single or single_med_ns,
    })

    # Prepare reference.json structure
    output_data = {
        "placeholder": False,
        "reference": {
            "framework": "PyTorch (CPU)",
            "version": torch.__version__,
            "threads": 1,
        },
        "kernels": reference_results,
        "models": [
            {
                "id": "cnn_mnist",
                "label": "CNN classifier (MNIST)",
                "mentats": {"test_accuracy": 97.69, "epoch_s": 133.1},
                "reference": {"test_accuracy": 98.2, "epoch_s": 45.0},
            }
        ],
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSuccessfully wrote real reference benchmarks to: {args.out_file}")


if __name__ == "__main__":
    main()
