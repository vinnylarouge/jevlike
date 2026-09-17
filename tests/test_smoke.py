import torch
from torch.nn import functional as F

from jevlike.data import ByteCollator, synthetic_example
from jevlike.model import OptionCache, TinyScorer


def test_tiny_scorer_learns_and_normalises():
    torch.manual_seed(5)
    examples = [synthetic_example(1000 + index) for index in range(32)]
    batch = ByteCollator(128, 24)(examples)
    model = TinyScorer(width=32, rank=32, context_tokens=128)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = float(F.cross_entropy(model(batch), batch["labels"]).detach())
    for _ in range(30):
        loss = F.cross_entropy(model(batch), batch["labels"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    logits = model(batch)
    final = float(F.cross_entropy(logits, batch["labels"]).detach())
    assert final < initial * 0.6
    assert torch.allclose(logits.softmax(-1).sum(-1), torch.ones(len(examples)), atol=1e-6)


def test_option_cache_encodes_identical_options_once():
    calls = []

    def encode(ids, mask):
        calls.append(ids.shape)
        return ids.float().mean(-1, keepdim=True)

    cache = OptionCache(capacity=2)
    ids = torch.tensor([[[1, 2, 0], [3, 4, 5]]] * 4)
    mask = ids.ne(0)
    first = cache.lookup(ids, mask, encode)
    second = cache.lookup(ids.clone(), mask.clone(), encode)
    assert len(calls) == 1 and torch.equal(first, second)
    # a different option list, order or padding is a miss
    cache.lookup(ids.flip(1), mask.flip(1), encode)
    cache.lookup(ids, mask.logical_and(ids.ne(5)), encode)
    assert len(calls) == 3 and cache.hits == 1 and cache.misses == 3
    # bounded: the oldest entry is evicted, so the first batch encodes again
    cache.lookup(ids, mask, encode)
    assert len(calls) == 4
    assert OptionCache(capacity=0).lookup(ids, mask, encode) is not None and len(calls) == 5
