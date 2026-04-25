"""
Verify combined dataset includes real Hugging Face sources.

Usage:
    python scripts/verify_dataset_usage.py --dataset data/combined_sft.json

By default this fails if fallback-only sources are present. Use --allow_fallback
only for fully offline testing.
"""

import argparse
import json
import os
import sys
from collections import Counter


REQUIRED_REAL_SOURCES = ("glaive", "helpful_instructions", "xlam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dataset source composition")
    parser.add_argument("--dataset", default="data/combined_sft.json")
    parser.add_argument("--min_total", type=int, default=1000)
    parser.add_argument("--allow_fallback", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset file not found: {args.dataset}")
        return 1

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERROR] Dataset must be a JSON list of samples.")
        return 1

    total = len(data)
    counts = Counter(sample.get("source", "unknown") for sample in data)

    print("=" * 60)
    print("Dataset Source Verification")
    print("=" * 60)
    print(f"File: {args.dataset}")
    print(f"Total samples: {total}")
    print("Source counts:")
    for source, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {source}: {count}")

    errors = []

    if total < args.min_total:
        errors.append(
            f"Total samples ({total}) is below --min_total ({args.min_total})."
        )

    missing_real = [s for s in REQUIRED_REAL_SOURCES if counts.get(s, 0) == 0]
    if missing_real:
        errors.append(
            "Missing required real sources: " + ", ".join(missing_real)
        )

    fallback_sources = [s for s in counts if "fallback" in s]
    if fallback_sources and not args.allow_fallback:
        errors.append(
            "Fallback sources found (network/load issue): "
            + ", ".join(sorted(fallback_sources))
            + ". Rebuild without fallback for final submission."
        )

    if errors:
        print("\n[FAIL]")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n[PASS] Dataset includes required Hugging Face sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

