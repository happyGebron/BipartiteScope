"""Command-line entry point; commands are added with their service capabilities."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="bilcs", description="BiLCS bipartite analysis engine")
    parser.add_argument("--version", action="version", version="BiLCS 0.1.0-alpha")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
