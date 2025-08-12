#!/usr/bin/env python3
"""Summarize NH-ARP experiment logs.

This is a lightweight placeholder that writes empty summary files
matching the expected schema. It allows replication scripts to run
without requiring the full analysis tooling.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

SCHEMA = [
    "problem",
    "n",
    "extra",
    "alpha",
    "mu",
    "p_anneal",
    "steps",
    "seed",
    "success",
    "final_metric",
    "gap_pct",
    "var",
    "kurt",
    "stable_step",
    "wall_clock_step_s",
    "wall_clock_total_s",
    "restarts_used",
    "phi_gamma",
    "phi_t0",
    "phi_kurt",
    "phi_subgaussian",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize log files into CSV and JSON outputs.")
    parser.add_argument("--inputs", required=True, help="Directory containing log files")
    parser.add_argument("--out", required=True, help="Path to summary JSON output")
    parser.add_argument("--csv", required=True, help="Path to summary CSV output")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # Write empty CSV with header
    with open(args.csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=SCHEMA)
        writer.writeheader()

    # Write empty JSON array
    with open(args.out, "w") as f_json:
        json.dump([], f_json)


if __name__ == "__main__":
    main()
