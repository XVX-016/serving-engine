#!/usr/bin/env python3
"""
scripts/compare.py

Compares two tensor dump directories (e.g. data/reference vs data/dump_cpu).
Prints max abs diff, mean abs diff, and argmax agreement for each tensor.
Exits with non-zero status code if max abs diff exceeds --max-abs threshold.
"""

import os
import sys
import argparse
import numpy as np

def compare_tensors(arr_a, arr_b, tensor_name):
    if arr_a.shape != arr_b.shape:
        return {
            "error": f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}",
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "argmax_match": 0.0
        }

    # Cast to float64 for difference calculation to avoid precision loss
    diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))

    argmax_match = None
    if "logits" in tensor_name or arr_a.ndim >= 2:
        # Check argmax across the last axis
        argmax_a = np.argmax(arr_a, axis=-1)
        argmax_b = np.argmax(arr_b, axis=-1)
        match_count = np.sum(argmax_a == argmax_b)
        total_count = argmax_a.size
        argmax_match = float(match_count / total_count * 100.0)

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "argmax_match": argmax_match
    }

def find_npy_files(directory):
    npy_dict = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".npy"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, directory)
                npy_dict[rel_path] = full_path
    return npy_dict

def main():
    parser = argparse.ArgumentParser(description="Compare two tensor dump directories.")
    parser.add_argument("dir_a", help="First directory (reference)")
    parser.add_argument("dir_b", help="Second directory (target to test)")
    parser.add_argument("--max-abs", type=float, default=1e-3, help="Max absolute difference threshold (default: 1e-3)")
    parser.add_argument("--mean-abs", type=float, default=None, help="Mean absolute difference threshold")
    parser.add_argument("--quiet", action="store_true", help="Only report failures")
    args = parser.parse_args()

    if not os.path.isdir(args.dir_a):
        print(f"Error: Directory {args.dir_a} does not exist.", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.dir_b):
        print(f"Error: Directory {args.dir_b} does not exist.", file=sys.stderr)
        sys.exit(2)

    files_a = find_npy_files(args.dir_a)
    files_b = find_npy_files(args.dir_b)

    common_keys = sorted(list(set(files_a.keys()) & set(files_b.keys())))
    only_a = sorted(list(set(files_a.keys()) - set(files_b.keys())))
    only_b = sorted(list(set(files_b.keys()) - set(files_a.keys())))

    if not common_keys:
        print(f"Error: No matching .npy files found between {args.dir_a} and {args.dir_b}", file=sys.stderr)
        sys.exit(2)

    print(f"=== Comparing {len(common_keys)} tensor files ===")
    print(f"Threshold --max-abs: {args.max_abs:g}")
    print(f"{'Tensor File':<45} | {'Max Abs Diff':<14} | {'Mean Abs Diff':<14} | {'Argmax Agree':<12} | {'Status'}")
    print("-" * 100)

    any_failure = False

    for key in common_keys:
        arr_a = np.load(files_a[key])
        arr_b = np.load(files_b[key])

        res = compare_tensors(arr_a, arr_b, key)

        if "error" in res:
            print(f"{key:<45} | {res['error']}")
            any_failure = True
            continue

        max_abs = res["max_abs"]
        mean_abs = res["mean_abs"]
        argmax_str = f"{res['argmax_match']:.2f}%" if res["argmax_match"] is not None else "N/A"

        passed = max_abs <= args.max_abs
        if args.mean_abs is not None and mean_abs > args.mean_abs:
            passed = False

        status_str = "PASS" if passed else "FAIL"
        if not passed:
            any_failure = True

        if not args.quiet or not passed:
            print(f"{key:<45} | {max_abs:<14.6e} | {mean_abs:<14.6e} | {argmax_str:<12} | {status_str}")

    if only_a:
        print(f"\nWarning: {len(only_a)} files only in {args.dir_a}: {only_a[:5]}...")
    if only_b:
        print(f"\nWarning: {len(only_b)} files only in {args.dir_b}: {only_b[:5]}...")

    print("-" * 100)
    if any_failure:
        print(f"OVERALL VERDICT: FAIL (Tolerances exceeded)", file=sys.stderr)
        sys.exit(1)
    else:
        print("OVERALL VERDICT: PASS (All tensors within tolerance)")
        sys.exit(0)

if __name__ == "__main__":
    main()

