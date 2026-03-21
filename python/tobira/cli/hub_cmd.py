"""tobira hub-push / hub-pull - HuggingFace Hub model management."""

from __future__ import annotations

import argparse
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register ``hub-push`` and ``hub-pull`` subcommands.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    _register_push(subparsers)
    _register_pull(subparsers)


def _register_push(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "hub-push",
        help="Upload a model to HuggingFace Hub",
        description="Upload a trained model directory to HuggingFace Hub "
        "with an auto-generated model card.",
    )
    parser.add_argument(
        "model_dir",
        help="Path to the local model directory to upload",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace Hub repository ID (e.g. velocitylabo/tobira-spam-bert-ja)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private repository",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        metavar="PATH",
        help="Path to a JSON file with evaluation metrics",
    )
    parser.add_argument(
        "--license",
        default="apache-2.0",
        dest="license_name",
        help="License for the model card (default: apache-2.0)",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language code for the model card (default: ja)",
    )
    parser.add_argument(
        "--base-model",
        default="tohoku-nlp/bert-base-japanese-v3",
        help="Base model used for fine-tuning",
    )
    parser.set_defaults(func=_run_push)


def _register_pull(subparsers: "argparse._SubParsersAction[Any]") -> None:
    parser = subparsers.add_parser(
        "hub-pull",
        help="Download a model from HuggingFace Hub",
        description="Download a model from HuggingFace Hub to a local directory.",
    )
    parser.add_argument(
        "repo_id",
        help="HuggingFace Hub repository ID (e.g. velocitylabo/tobira-spam-bert-ja)",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory to save the downloaded model",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision (branch, tag, or commit hash) to download",
    )
    parser.set_defaults(func=_run_pull)


def _run_push(args: argparse.Namespace) -> int:
    """Execute the ``hub-push`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from tobira.hub import load_metrics, push_to_hub

    metrics = None
    if args.metrics is not None:
        try:
            metrics = load_metrics(args.metrics)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

    try:
        url = push_to_hub(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            metrics=metrics,
            license_name=args.license_name,
            language=args.language,
            base_model=args.base_model,
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Model uploaded: {url}")
    return 0


def _run_pull(args: argparse.Namespace) -> int:
    """Execute the ``hub-pull`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from tobira.hub import pull_from_hub

    try:
        local_path = pull_from_hub(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=args.token,
            revision=args.revision,
        )
    except ImportError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Model downloaded to: {local_path}")
    return 0
