"""Command-line entry point; commands are added with their service capabilities."""

from __future__ import annotations

import argparse

from .io import load_csv_graph


def main() -> None:
    parser = argparse.ArgumentParser(prog="bilcs", description="BiLCS bipartite analysis engine")
    parser.add_argument("--version", action="version", version="BiLCS 0.1.0-alpha")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate", help="validate generic U-V edge and U-feature CSV files")
    validate.add_argument("--edges", required=True)
    validate.add_argument("--features", required=True)
    validate.add_argument("--delimiter", default=",")
    args = parser.parse_args()
    if args.command == "validate":
        graph = load_csv_graph(args.edges, args.features, delimiter=args.delimiter)
        print(f"valid: {len(graph.u_ids)} U entities, {len(graph.v_ids)} V entities, {graph.incidence.nnz} interactions")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
