#!/usr/bin/env python3
"""
Collects and aggregates training metrics from mentats outputs into mentats/data/models.json.

Scans ../mentats/outputs/**/metrics.json, extracts model records, and updates
the models.json file consumed by the GitHub Pages site.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate mentats training metrics into models.json"
    )
    parser.add_argument(
        "--mentats-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent.parent
        / "mentats/mentats",
        help="Path to the mentats repository (default: ../mentats)",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "mentats"
        / "data"
        / "models.json",
        help="Path to output models.json (default: mentats/data/models.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mentats_dir = args.mentats_dir
    out_file = args.out_file

    if not mentats_dir.exists():
        print(f"Error: mentats directory not found at {mentats_dir}", file=sys.stderr)
        sys.exit(1)

    outputs_dir = mentats_dir / "outputs"
    print(f"Scanning for metrics in: {outputs_dir}")

    # Load existing models.json if present
    existing_data = {}
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read existing {out_file}: {e}")

    models_map = {}
    for m in existing_data.get("models", []):
        if "id" in m:
            models_map[m["id"]] = m

    found_count = 0
    if outputs_dir.exists():
        for metrics_file in outputs_dir.glob("**/metrics.json"):
            print(f"Found metrics: {metrics_file.relative_to(mentats_dir)}")
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                model_id = model_data.get("id")
                if model_id:
                    models_map[model_id] = model_data
                    found_count += 1
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}", file=sys.stderr)

    merged_models = list(models_map.values())
    result = {
        "placeholder": len(merged_models) == 0,
        "source": f"Collected {found_count} metrics files from {mentats_dir.name}/outputs",
        "models": merged_models,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"Successfully updated {out_file} with {len(merged_models)} models ({found_count} new/updated)."
    )


if __name__ == "__main__":
    main()
