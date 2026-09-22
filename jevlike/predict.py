"""Score one context against command-line options, or one state/questions request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .data import ChoiceExample
from .interaction import answer_request, parse_request
from .model import load_checkpoint, select_device
from .train import move


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--context")
    parser.add_argument("--option", action="append")
    parser.add_argument(
        "--request",
        help="path to a state/questions JSON file, or - to read it from stdin",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.request and (args.context or args.option):
        parser.error("pass either --request or --context with --option, not both")
    if not args.request and not args.context:
        parser.error("pass --request, or --context with --option at least twice")
    if not args.request and len(args.option or []) < 2:
        parser.error("pass --option at least twice")
    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    if args.request:
        try:
            source = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot read {args.request}: {error}")
        try:
            request = parse_request(json.loads(source))
        except (json.JSONDecodeError, ValueError) as error:
            parser.error(f"bad request: {error}")
        print(json.dumps(answer_request(request, model, collator, device), indent=2))
        return
    batch = move(collator([
        ChoiceExample(args.context, tuple(args.option), 0)
    ]), device)
    model.eval()
    with torch.no_grad():
        probabilities = model(batch).softmax(-1)[0, :len(args.option)].cpu().tolist()
    print(json.dumps([
        {"option": option, "probability": probability}
        for option, probability in zip(args.option, probabilities)
    ], indent=2))


if __name__ == "__main__":
    main()
