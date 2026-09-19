import json
import os
import subprocess
import sys

import pytest

from jevlike.data import validate


@pytest.mark.parametrize("seed", [17, 42])
def test_synthetic_cli_is_reproducible_across_hash_seeds(tmp_path, seed):
    sizes = {"train": 32, "validation": 8, "test": 8}
    outputs = []
    for hash_seed in ("1", "2", "3"):
        output = tmp_path / hash_seed
        subprocess.run(
            [
                sys.executable, "-m", "jevlike.data", "synthetic",
                "--output", str(output), "--seed", str(seed),
                "--train", str(sizes["train"]),
                "--validation", str(sizes["validation"]),
                "--test", str(sizes["test"]),
            ],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            timeout=60,
        )
        outputs.append({
            split: (output / f"{split}.jsonl").read_bytes() for split in sizes
        })

    for split, size in sizes.items():
        assert outputs[0][split] == outputs[1][split] == outputs[2][split], split
        rows = outputs[0][split].splitlines()
        assert len(rows) == size
        for row in rows:
            item = validate(json.loads(row))
            assert len(item.options) == len(set(item.options))
            assert item.context.endswith(f"Badge: {item.options[item.label]}.")
