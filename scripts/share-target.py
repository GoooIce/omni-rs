#!/usr/bin/env python3
"""Point this checkout's Cargo output at the primary checkout's registered target directories."""
import sys
from pathlib import Path

from target_layout import apply_links, load_manifests, primary_checkout


def main():
    checkout = Path(__file__).resolve().parents[1]
    try:
        ensured = apply_links(checkout, primary_checkout(checkout), load_manifests(checkout))
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    for link, dest in ensured:
        print(f"{link.relative_to(checkout)} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
