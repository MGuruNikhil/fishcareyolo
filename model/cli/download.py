"""
CLI for downloading the dataset from Roboflow.

Usage:
    ROBOFLOW_API_KEY=your_key uv run mina-download
"""

import argparse

from mina.dataset import download_and_organize


def main():
    parser = argparse.ArgumentParser(
        description="Download fish disease dataset from Roboflow"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key. Can also be set via ROBOFLOW_API_KEY env var.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="mina-orfdd",
        help="Roboflow workspace name",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="mina-u7bag",
        help="Roboflow project name",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="Dataset version number",
    )

    args = parser.parse_args()

    download_and_organize(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
    )


if __name__ == "__main__":
    main()
