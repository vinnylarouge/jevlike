"""One-pass option scorers with tiny or frozen pretrained text encoders."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class AttentionHead(nn.Module):
    """Turn context tokens and option vectors into one score per option."""

    def __init__(self, input_width: int, rank: int) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(input_width)
        self.option_norm = nn.LayerNorm(input_width)
        self.query = nn.Linear(input_width, rank, bias=False)
        self.key = nn.Linear(input_width, rank, bias=False)
        self.value = nn.Linear(input_width, rank, bias=False)
        self.rank = rank

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor,
        options: torch.Tensor, option_mask: torch.Tensor,
        shuffle_context: bool = False,
    ) -> torch.Tensor:
        context = self.context_norm(context.float())
        options = self.option_norm(options.float())
        if shuffle_context and context.shape[0] > 1:
            context = context.roll(1, dims=0)
            context_mask = context_mask.roll(1, dims=0)
        query = self.query(options)
        key = self.key(context)
        value = self.value(context)
        scores = torch.einsum("bnr,blr->bnl", query, key) / math.sqrt(self.rank)
        scores = scores.masked_fill(
            ~context_mask[:, None, :], torch.finfo(scores.dtype).min
        )
        attended = torch.einsum("bnl,blr->bnr", scores.softmax(-1), value)
        logits = (query * attended).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


class TinyScorer(nn.Module):
    def __init__(self, width: int, rank: int, context_tokens: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Embedding(context_tokens, width)
        self.head = AttentionHead(width, rank)

    def forward(self, batch: dict[str, torch.Tensor], shuffle_context: bool = False):
        context_ids = batch["context_ids"]
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        context = self.embedding(context_ids) + self.position(positions)
        option_tokens = self.embedding(batch["option_ids"])
        weights = batch["option_token_mask"].unsqueeze(-1)
        options = (option_tokens * weights).sum(2) / weights.sum(2).clamp_min(1)
        return self.head(
            context, batch["context_mask"], options, batch["option_mask"],
            shuffle_context,
        )


class FrozenTransformerScorer(nn.Module):
    def __init__(self, model_name: str, rank: int) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False).eval()
        self.head = AttentionHead(self.encoder.config.hidden_size, rank)

    def forward(self, batch: dict[str, torch.Tensor], shuffle_context: bool = False):
        self.encoder.eval()
        with torch.no_grad():
            context = self.encoder(
                input_ids=batch["context_ids"],
                attention_mask=batch["context_mask"],
            ).last_hidden_state
            shape = batch["option_ids"].shape
            flat_ids = batch["option_ids"].reshape(-1, shape[-1])
            flat_mask = batch["option_token_mask"].reshape(-1, shape[-1])
            hidden = self.encoder(
                input_ids=flat_ids, attention_mask=flat_mask,
            ).last_hidden_state
            pooled = (hidden * flat_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / flat_mask.sum(1, keepdim=True).clamp_min(1)
            options = pooled.reshape(shape[0], shape[1], -1)
        return self.head(
            context, batch["context_mask"], options, batch["option_mask"],
            shuffle_context,
        )


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_system(config: dict, device: torch.device):
    from .data import ByteCollator, HuggingFaceCollator

    if config["encoder"] == "tiny":
        model = TinyScorer(
            config["width"], config["rank"], config["context_tokens"]
        )
        collator = ByteCollator(config["context_tokens"], config["option_tokens"])
    else:
        from transformers import AutoTokenizer

        name = config["hf_model"]
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FrozenTransformerScorer(name, config["rank"])
        collator = HuggingFaceCollator(
            tokenizer, config["context_tokens"], config["option_tokens"]
        )
    return model.to(device), collator


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, collator = make_system(payload["config"], device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected or any(not name.startswith("encoder.") for name in missing):
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model, collator, payload["config"]
