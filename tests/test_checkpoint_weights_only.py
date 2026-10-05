"""load_checkpoint must use weights_only=True (issue #3)."""

from pathlib import Path

import torch

from jevlike.data import ByteCollator, synthetic_example
from jevlike.model import TinyScorer, load_checkpoint, make_system, trainable_state


def test_load_checkpoint_round_trip_with_weights_only(tmp_path: Path):
    torch.manual_seed(0)
    config = {
        "encoder": "tiny",
        "rank": 16,
        "context_tokens": 64,
        "option_tokens": 16,
        "width": 32,
    }
    device = torch.device("cpu")
    model, collator = make_system(config, device)
    batch = ByteCollator(64, 16)([synthetic_example(i) for i in range(4)])
    path = tmp_path / "ckpt.pt"
    torch.save({"config": config, "state_dict": trainable_state(model)}, path)

    loaded, _collator, loaded_config = load_checkpoint(path, device)
    assert loaded_config["encoder"] == "tiny"
    assert isinstance(loaded, TinyScorer)
    out = loaded(batch)
    assert out.shape[0] == 4


def test_load_checkpoint_uses_weights_only(monkeypatch, tmp_path: Path):
    config = {
        "encoder": "tiny",
        "rank": 8,
        "context_tokens": 32,
        "option_tokens": 8,
        "width": 16,
    }
    device = torch.device("cpu")
    model, _ = make_system(config, device)
    path = tmp_path / "ckpt.pt"
    torch.save({"config": config, "state_dict": trainable_state(model)}, path)

    seen = {}

    real_load = torch.load

    def wrapped(*args, **kwargs):
        seen["weights_only"] = kwargs.get("weights_only", "MISSING")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", wrapped)
    load_checkpoint(path, device)
    assert seen["weights_only"] is True
