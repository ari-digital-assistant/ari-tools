# wakeword

Corpus assembly, training and evaluation for Ari's wake-word models.

See `ari-android/docs/superpowers/specs/2026-09-10-wake-word-retrain-design.md`
for why this exists and what the ship gate is.

## Two projects, two virtualenvs — on purpose

```
wakeword/
├── generate/   Piper synthesis of wake-phrase positives (torch)
└── train/      microWakeWord training + evaluation (TensorFlow)
```

They resolve together perfectly well. They are kept apart anyway, because a
combined resolution **downgrades `audiomentations` from 0.43.1 to 0.33.0** —
`piper-sample-generator` pins it hard, `microwakeword` does not pin it at all,
so the older wins.

Augmentation is the heart of this pipeline, so running the trainer's
augmentation against a library version it was never developed against is the one
silent downgrade this project cannot afford — whatever the eventual diagnosis
turns out to be.

## Python

Both pin 3.13. TensorFlow publishes cp310–cp313 wheels only, and Arch ships
3.14 — so these carry their own interpreter rather than using the system one.
`uv` fetches it; nothing is installed system-wide.

```bash
cd train    && uv sync    # or: cd generate && uv sync
uv run python -c "import microwakeword"
```

## Corpus data is never committed

Recorded positives are household voice audio, and the eval set's value depends
on staying held out rather than published. `corpus/`, `eval/` and `models/` are
gitignored at the repo root — keep it that way.

## Running it

```bash
# 1. Clean voices (GPU, ~12 clips/s)
cd generate
uv run ./generate_speech.py --model ../models/en_US-libritts_r-medium.pt \
    --count 10000 --output-dir ../corpus/positives-raw

# 2. Resample to 16 kHz and spread across a real room's dynamic range (CPU)
cd ../train
uv run ./prepare_clips.py --input ../corpus/positives-raw \
    --output ../corpus/positives --variants 5
uv run pytest
```

Step 1's output is **not** training data. Measured over 40 clips from 40
different speakers it varies by just 4.6 dB — raw generator output is uniformly
loud, and training on it unaltered would describe a world that does not exist.
Step 2 takes that to **38 dB**, the range real speech occupies across a room:

| corpus | spread | dB range | stdev/mean |
|---|---|---|---|
| raw generator | 1.7x | 4.6 | 13% |
| prepared | 80x | 38.1 | 103% |

### Negatives

The same two scripts, different text. `negatives/near_misses.txt` is derived
from the 2026-09-10 capture review rather than from intuition — spoken numbers,
"all right", and "okay Addy", which is a near-miss on `hey_ari` and `ok_ari` at
once.

```bash
cd generate
uv run ./generate_speech.py --phrases-file ../negatives/near_misses.txt \
    --model ../models/en_US-libritts_r-medium.pt --count 3000 \
    --output-dir ../corpus/near-misses-raw
```

**Maltese and French negatives cannot be produced this way.** They were the
largest trigger class in the review (5 of 18), but LibriTTS-R is English — feed
it Maltese and you get an English speaker mangling foreign words, which is not
the sound that fired the model. Those have to be real recordings.

### Negatives from real speech

```bash
cd train
uv run ./fetch_negatives.py --languages mt,fr --max-clips 3000 \
    --output ../corpus/negatives
```

Fetched 2026-09-11: **7.25 hours** of real non-English speech — 3.74 h Maltese,
3.51 h French, 6,000 clips, 34 seconds. CC0.

This is the one gap in the incumbent's training we can actually name. Its
negative set was CHiME, FMA and AudioSet: ambient noise and music, and not one
word of non-English speech. The capture review found Maltese and French were
5 of 18 real false accepts — the largest single class.

`--audioset-clips N` adds ambient/non-speech on top. Common Voice is read
speech: one person, close-mic'd, no room. It is not a substitute for a
television, so use both. AudioSet streams as parquet now rather than the tars
upstream's notebook expects, and `datasets` will not decode audio without
`torchcodec` — which would pull torch into the train venv, the exact pairing the
two-project split avoids. The fetcher takes the raw bytes and decodes them with
soundfile instead.

### Eval material is never augmented

```bash
uv run ./prepare_clips.py --input ~/captures/rejected \
    --output ../corpus/eval-negatives --variants 1 --no-augment
```

Held-out material is the only honest measurement in the project. Augmenting it
would score models against audio no microphone ever produced, and would flatter
whichever model happened to suit the augmentation.

Useful sanity check from the first real export: 16 captured false-accepts span
**18 dB** of natural variation. Training augmentation spans 38 dB — comfortably
wider than reality, without being absurd.

### Before training: prove the exam is not in the lessons

```bash
cd train
uv run ./check_leakage.py --train ../corpus/positives --eval ../corpus/eval-positives
```

Exits 1 if anything leaked, so it gates a pipeline rather than printing advice.
Three routes, because no single check finds all of them:

1. **identical** — same PCM both sides. The only one a hash catches.
2. **shared source** — `0-3.wav` in training and `0-7.wav` in eval are different
   augmentations of one original. Different bytes, same utterance underneath.
   This is how a corpus built by *this* pipeline will actually leak.
3. **same utterance, different capture path** — the one that would have got us.
   `WakeCaptureStore`'s firehose and `UtteranceCaptureStore` both record an
   accepted turn through different windows, 2 s against 4.2 s of the same
   moment. Different lengths, different directories, neither aware of the other.
   Only the epoch-ms in the filename connects them.

Verified against the real thing: run over the 2026-09-10 export it finds
**exactly 22 pairs**, which is the number that export actually contains.

Both steps write a `manifest.json` recording every setting used. The settings
are the corpus: without them, two models trained weeks apart cannot be compared
because nobody can say what differed.

## Two upstream gotchas, both load-bearing

**`piper_train` is missing from every published artifact.** `piper-sample-generator`
imports `piper_train.vits.commons` in `__main__`, but its `packages.find` includes
only `piper_sample_generator*`, so neither the PyPI wheel nor a git install ships
it. (The `except ImportError` guard around that import retries the identical
import, so it can only ever fail twice.) `generate/upstream/` is a shallow clone
of the pinned tag that exists purely to supply it:

```bash
cd generate
git clone --depth 1 --branch v3.2.0 https://github.com/rhasspy/piper-sample-generator.git upstream
PYTHONPATH=upstream uv run python -m piper_sample_generator --help
```

**The `.pt.json` config lives in the repo, not the release.** The GitHub release
carries `en_US-libritts_r-medium.pt` alone; the generator also wants
`<model>.pt.json` beside it, which only exists in the repo's `models/`.

## Generator output is 22.05 kHz — microWakeWord wants 16 kHz

Verified on a six-sample smoke run: 22050 Hz, mono, 16-bit. Everything fed to
training has to be resampled to 16 kHz first.

That smoke run also demonstrated the bug this whole project exists to fix. Six
samples, six different speakers and speeds — and peaks of 32767, 30899, 27790,
with RMS clustered in a band barely 20% wide. **Raw generator output is
uniformly loud**, so step 2 is not a nicety to add later.

A correction worth recording: this README previously claimed that uniformity was
what broke `hey_ari`. It was not. The trainer that produced it volume-augmented
at p=0.8 across -45 dB — wider than this pipeline does. See the retraction in
the retrain design section 2.3. Wide dynamic range is still correct practice;
it is just not a fix for a diagnosed fault.

## The incumbent's baseline — the number to beat

`baseline/hey_ari-2026-09-11.json`, measured by `train/eval_model.py` against 20
real wake utterances and 16 real captured false-accepts, using a port of the
app's own detection rule (window mean over `probability_cutoff`, plus the 4 s
refractory period).

| operating point | recall | hard negatives firing |
|---|---|---|
| HIGH  (0.95 / 5)   | 100% | 15/16 |
| MEDIUM (0.985 / 10) | 95% | 14/16 |
| LOW   (0.99 / 14)  | 90% | 10/16 |

Sweeping the cutoff at window 10 is the result that matters:

| cutoff | 0.50 | 0.90 | 0.95 | 0.97 | 0.985 | 0.99 | 0.999 |
|---|---|---|---|---|---|---|---|
| recall | 100% | 100% | 100% | 100% | 95% | 95% | 90% |
| negatives firing | 15/16 | 15/16 | 15/16 | 14/16 | 14/16 | 12/16 | 9/16 |

**Nothing changes between 0.5 and 0.95.** The model emits near-1.0 for genuine
wakes and for "spaghetti" alike, so no threshold anywhere in the range separates
them — and at 0.999, the most extreme the rule allows, nine of sixteen still
fire while recall has already started falling. This is design section 2.1 as a
measurement rather than an argument: threshold tuning is exhausted, and only
retraining can help.

Two honest caveats:

- Recall is measured on **one speaker** — the developer, near-field, giving
  commands. It is an upper bound. The whole point of the recorder in the Android
  app is to make this number mean something for anyone else.
- FA/hour here is measured on **known-hard negatives**, clips selected precisely
  because they already fooled a model. It is a worst case, not a user's
  experience. An ambient corpus is still needed for the real rate.

A useful validation fell out of this: all 16 negatives genuinely fired on the
device, and the harness reproduces 14-15 of them firing at the app's default.
The port of the detection rule is faithful.
