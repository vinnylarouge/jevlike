"""JSONL loading, byte tokenisation and small public data builders."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChoiceExample:
    context: str
    options: tuple[str, ...]
    label: int


def validate(payload: dict) -> ChoiceExample:
    context, options, label = payload.get("context"), payload.get("options"), payload.get("label")
    if not isinstance(context, str) or not isinstance(options, list):
        raise ValueError("each row needs string context and list options")
    if len(options) < 2 or any(not isinstance(option, str) or not option for option in options):
        raise ValueError("options must contain at least two non-empty strings")
    if not isinstance(label, int) or not 0 <= label < len(options):
        raise ValueError("label must be an option index")
    return ChoiceExample(context, tuple(options), label)


class JsonlDataset(Dataset[ChoiceExample]):
    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as handle:
            self.examples = [validate(json.loads(line)) for line in handle if line.strip()]
        if not self.examples:
            raise ValueError(f"no examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _bytes(text: str, length: int) -> list[int]:
    return [byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]


class ByteCollator:
    def __init__(self, context_tokens: int, option_tokens: int) -> None:
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples: list[ChoiceExample]):
        contexts = [_bytes(item.context, self.context_tokens) for item in examples]
        option_rows = [[_bytes(option, self.option_tokens) for option in item.options]
                       for item in examples]
        return _tensor_batch(examples, contexts, option_rows, 0)


class HuggingFaceCollator:
    def __init__(self, tokenizer, context_tokens: int, option_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples: list[ChoiceExample]):
        contexts = self.tokenizer(
            [item.context for item in examples], truncation=True,
            max_length=self.context_tokens, add_special_tokens=True,
        )["input_ids"]
        flat = [option for item in examples for option in item.options]
        encoded = self.tokenizer(
            flat, truncation=True, max_length=self.option_tokens, add_special_tokens=True,
        )["input_ids"]
        rows, offset = [], 0
        for item in examples:
            rows.append(encoded[offset:offset + len(item.options)])
            offset += len(item.options)
        return _tensor_batch(examples, contexts, rows, self.tokenizer.pad_token_id)


def _tensor_batch(examples, contexts, option_rows, pad_id):
    batch, max_context = len(examples), max(map(len, contexts))
    max_options = max(len(row) for row in option_rows)
    max_option_tokens = max(len(tokens) for row in option_rows for tokens in row)
    context_ids = torch.full((batch, max_context), pad_id, dtype=torch.long)
    option_ids = torch.full(
        (batch, max_options, max_option_tokens), pad_id, dtype=torch.long
    )
    option_mask = torch.zeros((batch, max_options), dtype=torch.bool)
    for row, tokens in enumerate(contexts):
        context_ids[row, :len(tokens)] = torch.tensor(tokens)
    for row, options in enumerate(option_rows):
        option_mask[row, :len(options)] = True
        for column, tokens in enumerate(options):
            option_ids[row, column, :len(tokens)] = torch.tensor(tokens)
    return {
        "context_ids": context_ids, "context_mask": context_ids.ne(pad_id),
        "option_ids": option_ids, "option_token_mask": option_ids.ne(pad_id),
        "option_mask": option_mask,
        "labels": torch.tensor([item.label for item in examples], dtype=torch.long),
    }


COLOURS = ("amber", "azure", "bronze", "coral", "crimson", "gold", "green", "indigo")
ANIMALS = ("badger", "crane", "dolphin", "falcon", "gecko", "heron", "ibis", "jaguar")


def synthetic_example(seed: int) -> ChoiceExample:
    rng = random.Random(seed)
    target = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
    count = rng.randint(2, 8)
    options = {target}
    while len(options) < count:
        options.add(f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}")
    # Stabilise set iteration order before the seeded shuffle.
    options = sorted(options)
    rng.shuffle(options)
    notes = " ".join(rng.choice(("north", "south", "east", "west")) for _ in range(8))
    context = f"Choose the exact badge {target}. Notes: {notes}. Badge: {target}."
    return ChoiceExample(context, tuple(options), options.index(target))


def write_synthetic(output: Path, sizes: dict[str, int], seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    offset = 0
    for split, size in sizes.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                item = synthetic_example(seed + offset + index * 104729)
                handle.write(json.dumps({
                    "context": item.context, "options": item.options, "label": item.label,
                }) + "\n")
        offset += size * 104729


def _stable(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _title(text: str) -> str:
    return unquote(text).replace("_", " ")


def build_wikispeedia(root: Path, output: Path, max_options: int = 64) -> None:
    graph_dir = root / "wikispeedia_paths-and-graph"
    outgoing: dict[str, list[str]] = {}
    for line in (graph_dir / "links.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            source, target = line.split("\t")
            outgoing.setdefault(source, []).append(target)
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    counts = {name: 0 for name in handles}
    try:
        lines = (graph_dir / "paths_finished.tsv").read_text(encoding="utf-8").splitlines()
        for row, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            path = []
            for node in fields[3].split(";"):
                if node == "<":
                    if len(path) > 1:
                        path.pop()
                else:
                    path.append(node)
            if len(path) < 2:
                continue
            step = _stable(f"{fields[0]}:{fields[1]}:{row}") % (len(path) - 1)
            current, click, target = path[step], path[step + 1], path[-1]
            candidates = list(dict.fromkeys(outgoing.get(current, ())))
            if len(candidates) < 2 or click not in candidates:
                continue
            rng = random.Random(_stable(f"{row}:{target}:menu"))
            others = [item for item in candidates if item != click]
            rng.shuffle(others)
            menu = [click] + others[:max_options - 1]
            rng.shuffle(menu)
            article = root / "plaintext_articles" / f"{current}.txt"
            body = " ".join(article.read_text(encoding="utf-8", errors="replace").split())
            payload = {
                "context": f"Target article: {_title(target)}\nCurrent article: {_title(current)}\n{body[:2048]}",
                "options": [_title(item) for item in menu], "label": menu.index(click),
            }
            bucket = _stable(target + ":split") % 10
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps(counts, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic"))
    synthetic.add_argument("--train", type=int, default=2000)
    synthetic.add_argument("--validation", type=int, default=400)
    synthetic.add_argument("--test", type=int, default=400)
    synthetic.add_argument("--seed", type=int, default=17)
    wiki = commands.add_parser("wikispeedia")
    wiki.add_argument("--root", type=Path, required=True)
    wiki.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        write_synthetic(args.output, {
            "train": args.train, "validation": args.validation, "test": args.test,
        }, args.seed)
    else:
        build_wikispeedia(args.root, args.output)


if __name__ == "__main__":
    main()
