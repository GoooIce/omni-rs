#!/usr/bin/env python3
"""Point this checkout's Cargo output at the primary checkout's registered target directories."""
import argparse
import sys
from pathlib import Path

from target_layout import apply_links, check_links, load_manifests, primary_checkout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read-only check of the live layout")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    primary = primary_checkout(checkout)
    manifests = load_manifests(checkout)
    if args.check:
        failures = check_links(checkout, primary, manifests)
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print(f"Shared Cargo layout checked: {len(manifests)} manifests.")
        return 0
    try:
        ensured = apply_links(checkout, primary, manifests)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    for link, dest in ensured:
        print(f"{link.relative_to(checkout)} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
