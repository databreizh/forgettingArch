"""
Data aggregation module for Collaborative Forgetting experimental results.
Traverses experiment directories to consolidate individual JSON summaries 
into a unified CSV report for statistical analysis and plotting.
"""

import argparse
import json
import os
import csv
from typing import List, Dict, Any


def _find_summary_files(input_dir: str, suffix: str) -> List[str]:
    """
    Recursively discovers all summary files within the specified directory.
    
    Args:
        input_dir: Root directory containing algorithm execution traces.
        suffix: Filename suffix to identify valid summary files (e.g., '_summary').
        
    Returns:
        A list of absolute paths to discovered JSON summary files.
    """
    files = []
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            if name.endswith(suffix + ".json"):
                files.append(os.path.join(root, name))
    return files


def main():
    """
    Main entry point for the aggregation pipeline.
    Parses command-line arguments and flattens structured JSON data into tabular format.
    """
    parser = argparse.ArgumentParser(
        description="Consolidate experimental JSON summaries into a single CSV report."
    )
    parser.add_argument(
        "input_dir",
        help="Root directory containing the generated *_summary.json files",
    )
    parser.add_argument(
        "--output",
        default="experiments/summary_all.csv",
        help="Path to the output CSV file (default: experiments/summary_all.csv)",
    )
    parser.add_argument(
        "--suffix",
        default="_summary",
        help="Suffix used to identify JSON summary files (default: _summary)",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_csv = args.output
    suffix = args.suffix

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Specified input directory does not exist: {input_dir}")
        return

    # Discovery phase
    json_files = _find_summary_files(input_dir, suffix)
    print(f"[DEBUG] Found {len(json_files)} summary files under {input_dir}")

    if not json_files:
        print("[INFO] No summaries discovered. Exiting.")
        return

    # Processing phase: Flattening nested metadata for tabular representation
    rows: List[Dict[str, Any]] = []
    all_keys = set()

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten 'metadata' dictionary by adding a 'meta_' prefix to its keys.
        # This prevents namespace collisions with top-level experiment metrics.
        metadata = data.pop("metadata", {}) or {}
        flat_meta = {f"meta_{k}": v for k, v in metadata.items()}

        row = {
            **data,
            **flat_meta,
            "source_file": path, # Traceability: link row back to source file
        }

        rows.append(row)
        all_keys.update(row.keys())

    # Ensure all columns are present (even if empty for some rows) 
    # to maintain CSV structural integrity.
    fieldnames = sorted(list(all_keys))

    # Output phase: Persistence to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] Consolidated {len(rows)} summaries into: {output_csv}")


if __name__ == "__main__":
    main()