# Jevlike

Train a small model that chooses among a changing list of text options.

A Jev-like model takes a piece of text and a list of `N` text options. It returns one probability for each option. It does this in one pass instead of writing an answer word by word. [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe's commercial model for this kind of task. TypeSafe has not published its design. This repository is an independent starter model with the same input and output shape.

## Demo

The same option-attention head can score controller buttons from image patches. [This ten-second film](docs/jevre-demo-10s-bgm.mp4) joins two selected five-second windows: live `deadly_corridor` combat on the seven Doom buttons, then a chess controller walking to and playing moves with five keys. The diagram shows the tensors used for each decision. The Doom window came from the supplied joint checkpoint, which averaged 0.60 kills and -97.50 reward across its ten recorded episodes. The chess window came from the stronger chess-only checkpoint, which scored 4 wins, 46 draws and 0 losses in 50 sampled games against a random mover, but 0 wins, 2 draws and 48 losses against Stockfish level 0. The windows were selected for activity and are not typical-play or competence claims.

<video src="docs/jevre-demo-10s-bgm.mp4" controls width="960"></video>

Install the game extras and record a fresh 640 by 480 Doom trace from the released joint checkpoint:

```sh
uv pip install -e '.[games]'
python examples/doom/play.py examples/checkpoints/joint-imitation.pt --episodes 10 --game-seconds 35.3 --device cpu --capture-resolution 640x480 --output runs/doom.mp4 --trace runs/doom-trace.json
```

Render the trace in the same visual layout. This writes a silent film because the author-owned soundtrack source is not part of the repository.

```sh
(cd examples/film && npm install && npx playwright install chromium)
examples/film/make-film.sh runs/doom-trace.json runs/doom-film.mp4 10
```

The release includes the [Doom example](examples/doom/README.md), the [chess example](examples/chess/README.md), the single-game checkpoints and the shared 12-option checkpoint. Both games import the visual scorer from `jevlike.vision`; there is no second model copy in either example.

## Architecture

Each option becomes a query vector, which is a short list of numbers representing its text. The query assigns attention weights to the context tokens. Those weights make one context vector for that option. A shared dot product turns each option and context pair into one score. A softmax, which converts scores into probabilities that sum to one, runs across the options.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img src="docs/architecture.svg" alt="Each option queries the context, receives an attended context vector, and becomes one probability.">
</picture>

The default encoder learns byte embeddings from scratch. An encoder is the part that turns text into vectors. The optional Hugging Face path uses a frozen pretrained encoder, whose existing weights stay fixed while the small scorer learns.

## Data format

Use one JSON object per line:

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

`label` is the zero-based index of the correct option. Each row may have a different number of options, with a minimum of two.

## Quickstart

Run these commands from the repository root. They create local synthetic data, train on it, evaluate the saved model and score one new menu.

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'

jevlike-data synthetic --output data/synthetic
jevlike-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/synthetic.pt
jevlike-eval runs/synthetic.pt data/synthetic/test.jsonl
jevlike-predict runs/synthetic.pt \
  --context "Choose the exact badge amber badger. Badge: amber badger." \
  --option "azure crane" \
  --option "amber badger" \
  --option "gold heron"
```

The evaluation prints top-1 accuracy, which is the fraction of correct first choices. Top-3 accuracy is the fraction with the right answer among the three highest scores. Expected calibration error compares confidence with observed accuracy. The command also prints a shuffled-context control, which pairs each menu with the wrong context. A useful model should beat that control.

## Use your own data

1. Export train, validation and test JSONL files in the format above.
2. Keep all options that the model will see at prediction time in each row.
3. Split related records together. For example, keep all records for one customer or one target page in one split. This prevents near-duplicates from leaking into the test set.
4. Run `jevlike-train` with your train and validation files.
5. Run `jevlike-eval` once on the held-out test file. Held-out means the file was never used for training or model selection.

The default byte encoder truncates context to 192 bytes and each option to 32 bytes. Raise `--context-tokens` or `--option-tokens` when your text needs more room. Training supports CPU, Apple MPS for a Mac GPU, and CUDA for an NVIDIA GPU through `--device`.

## Use a frozen pretrained encoder

Install the optional dependency and name any compatible encoder from Hugging Face:

```sh
uv pip install -e '.[transformers]'
jevlike-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/qwen-head.pt \
  --encoder hf \
  --hf-model Qwen/Qwen2.5-0.5B \
  --rank 256 \
  --batch-size 8
```

Pooled option vectors are cached across batches (`OptionCache`, 64 entries, keyed on the exact option tokens), so a closed option list that repeats on every row is encoded once instead of once per batch. Pass `option_cache=0` to `FrozenTransformerScorer` to switch it off.

The checkpoint stores the trained scorer head and the encoder name. It does not copy the frozen encoder weights. Loading the checkpoint therefore needs access to the same Hugging Face model.

`--rank` sets the width of the small scorer head. A wider head has more trainable weights and uses more memory.

## Wikispeedia example

[`scripts/get_wikispeedia.sh`](scripts/get_wikispeedia.sh) downloads the public SNAP archives and builds next-click JSONL files. The data stay outside this repository.

```sh
scripts/get_wikispeedia.sh
jevlike-train data/wikispeedia/jsonl/train.jsonl \
  --validation data/wikispeedia/jsonl/validation.jsonl \
  --output runs/wikispeedia.pt
```

Cite Robert West and Jure Leskovec, *Human Wayfinding in Information Networks*, WWW 2012. Review the source data terms on the [SNAP dataset page](https://snap.stanford.edu/data/wikispeedia.html).

## What to expect

In the experiments that led to this starter, the one-pass scorer reached about 98% accuracy on synthetic menus. On target-disjoint Wikispeedia next-click data, a frozen Qwen2.5-0.5B encoder plus the scorer reached 26%, against about 8% for shuffled and random-encoder controls. A small model trained from scratch on 40,000 clicks reached 29%. At eight options, one pass was about 100 times faster than a small decoder forced to write 400 tokens.

These numbers describe local experiments, not this quickstart run. We did not show equal quality with Jev or reproduce TypeSafe's private training method.

## Limitations

- This is a research starter, not a copy of Jev.
- Accuracy depends on data quality, split quality and the encoder.
- The byte encoder is cheap but weak on language meaning.
- The pretrained path may download a large model and needs more memory.
- One-pass scoring requires the complete option list before prediction.
- The speed comparison used a small local decoder rather than a large commercial model.

## Licence

Code is released under the [MIT License](LICENSE). Downloaded datasets and pretrained models keep their own terms.
