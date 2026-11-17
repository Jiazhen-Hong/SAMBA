"""Evaluate a released SAMBA checkpoint with NeuralBench."""

from __future__ import annotations

import argparse

from neuralbench import evaluate_model

from samba_eeg import SambaModel
from samba_eeg.integrations import SambaNeuralBenchBackbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--task", default="all")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--name", default="samba")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute matching NeuralBench jobs instead of reusing cached failures/results",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download selected datasets first (use --no-download for manually installed data)",
    )
    parser.add_argument(
        "--prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare NeuralBench caches before evaluation",
    )
    args = parser.parse_args()

    model = SambaNeuralBenchBackbone(SambaModel.from_pretrained(args.checkpoint))
    scores = evaluate_model(
        model,
        "eeg",
        args.task,
        dataset=args.dataset,
        name=args.name,
        downstream_wrapper="linear_probe_mean",
        download=args.download,
        prepare=args.prepare,
        debug=args.debug,
        force=args.force,
    )
    print(scores)


if __name__ == "__main__":
    main()
