from __future__ import annotations

import argparse
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=10)
    parser.add_argument("--cadence-days", type=float, default=5)
    parser.add_argument("--locations", type=int, default=10)
    parser.add_argument("--input-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    snapshots_per_location = math.floor((args.years * 365.25) / args.cadence_days)
    samples_per_location = max(0, snapshots_per_location - args.input_len - args.horizon + 1)
    total_samples = samples_per_location * args.locations

    print(f"snapshots/location: ~{snapshots_per_location:,}")
    print(f"samples/location:   ~{samples_per_location:,}")
    print(f"locations:          {args.locations:,}")
    print(f"total samples:      ~{total_samples:,}")
    print()
    print("For ~30,000 samples at these settings, needed locations:")
    if samples_per_location > 0:
        print(f"~{math.ceil(30000 / samples_per_location)} locations")
    else:
        print("not enough snapshots per location")


if __name__ == '__main__':
    main()
