import torch
from torch.utils.data import DataLoader

from jevlike.data import ByteCollator, ChoiceExample
from jevlike.eval import metrics


class RecordingScorer:
    def __init__(self):
        self.context_ids = []
        self.context_masks = []

    def eval(self):
        return self

    def __call__(self, batch):
        self.context_ids.append(batch["context_ids"].cpu().clone())
        self.context_masks.append(batch["context_mask"].cpu().clone())
        return torch.zeros((batch["labels"].numel(), 2))


def _loader():
    examples = [
        ChoiceExample(chr(ord("A") + index), ("yes", "no"), 0)
        for index in range(7)
    ]
    return DataLoader(
        examples, batch_size=2, collate_fn=ByteCollator(8, 8),
    )


def test_shuffled_context_pairs_across_the_full_dataset():
    loader = _loader()
    scorer = RecordingScorer()

    metrics(scorer, loader, torch.device("cpu"), shuffle_context=True)

    observed = torch.cat(scorer.context_ids)[:, 0].tolist()
    original = [ord("A") + index + 1 for index in range(len(loader.dataset))]
    source_indices = [token_id - ord("A") - 1 for token_id in observed]

    assert sorted(observed) == sorted(original)
    assert all(source != target for target, source in enumerate(source_indices))
    assert any(
        source // loader.batch_size != target // loader.batch_size
        for target, source in enumerate(source_indices)
    )


def test_blank_context_baseline_attends_only_to_the_padding_token():
    loader = _loader()
    scorer = RecordingScorer()

    metrics(scorer, loader, torch.device("cpu"), blank_context=True)

    for ids, mask in zip(scorer.context_ids, scorer.context_masks):
        assert ids.shape[1] == 1
        assert torch.equal(ids, torch.zeros_like(ids))
        assert torch.equal(mask, torch.ones_like(mask))
