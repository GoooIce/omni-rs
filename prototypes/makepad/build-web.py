#!/usr/bin/env python3
"""Retired browser build entry point; inference is native on Apple Silicon."""
import sys


def main():
    print("WASM/WebGPU builds are no longer supported. Use Apple Silicon macOS and "
          "python3 prototypes/makepad/build-native.py.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
