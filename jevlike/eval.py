"""Evaluate accuracy, calibration, shuffled-context and blank-context controls."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler

from .data import ChoiceExample, JsonlDataset
from .model import load_checkpoint, select_device
from .train import move


def _global_shuffle_sources(size: int) -> list[int]:
    """Return one deterministic derangement of row indices for the full dataset."""
    if size < 2:
        return list(range(size))
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(size, generator=generator)
    sources = torch.empty_like(order)
    sources[order] = order.roll(1)
    return sources.tolist()


@torch.no_grad()
def metrics(model, loader, device, shuffle_context=False, blank_context=False):
    if shuffle_context and blank_context:
        raise ValueError("choose either shuffled_context or blank_context")
    context_sources = None
    if shuffle_context:
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise ValueError(
                "global shuffled-context evaluation requires a sequential loader "
                "that includes every example"
            )
        context_sources = _global_shuffle_sources(len(loader.dataset))

    model.eval()
    top1 = top3 = total = 0
    confidences, predictions, labels = [], [], []
    offset = 0
    for host_batch in loader:
        batch = move(host_batch, device)
        batch_size = batch["labels"].numel()
        if context_sources is not None:
            targets = [
                loader.dataset[index]
                for index in range(offset, offset + batch_size)
            ]
            paired = [
                ChoiceExample(
                    loader.dataset[source_index].context,
                    target.options,
                    target.label,
                )
                for source_index, target in zip(
                    context_sources[offset:offset + batch_size], targets
                )
            ]
            contexts = loader.collate_fn(paired)
            batch["context_ids"] = contexts["context_ids"].to(device)
            batch["context_mask"] = contexts["context_mask"].to(device)
        elif blank_context:
            tokenizer = getattr(loader.collate_fn, "tokenizer", None)
            pad_id = getattr(tokenizer, "pad_token_id", 0)
            if pad_id is None:
                pad_id = 0
            batch["context_ids"] = torch.full(
                (batch_size, 1), pad_id, dtype=batch["context_ids"].dtype,
                device=device,
            )
            batch["context_mask"] = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=device,
            )

        logits = model(batch).cpu()
        offset += batch_size
        probabilities = logits.softmax(-1)
        ranking = logits.topk(min(3, logits.shape[1]), dim=-1).indices
        batch_labels = batch["labels"].cpu()
        top1 += int(ranking[:, 0].eq(batch_labels).sum())
        top3 += int(ranking.eq(batch_labels[:, None]).any(1).sum())
        total += batch_labels.numel()
        confidence, prediction = probabilities.max(-1)
        confidences.append(confidence)
        predictions.append(prediction)
        labels.append(batch["labels"].cpu())
    if context_sources is not None and offset != len(context_sources):
        raise ValueError("global shuffled-context evaluation did not cover the dataset")
    confidence, prediction, labels = map(
        torch.cat, (confidences, predictions, labels)
    )
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            gap = prediction[selected].eq(labels[selected]).float().mean()
            gap -= confidence[selected].mean()
            ece += float(selected.float().mean() * gap.abs())
    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "ece": ece, "examples": labels.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        JsonlDataset(args.data), batch_size=args.batch_size, collate_fn=collator,
    )
    print(json.dumps({
        "model": metrics(model, loader, device),
        "shuffled_context": metrics(model, loader, device, True),
        "blank_context": metrics(model, loader, device, blank_context=True),
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
