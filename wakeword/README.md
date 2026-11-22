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
on staying held out rather than published. `corpus/`, `eval/`, `models/`,
`captures/`, `features/`, `trained/` and any stray `*.wav` are gitignored at the
repo root — keep it that way.

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

## Three upstream gotchas, all load-bearing

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

**`microwakeword.audio` and `.layers` are missing from the installed package
too.** `pyproject.toml` installs `microwakeword` from git, but its packaging
drops those two subpackages, and `make_features.py` and `train_model.py` both
need them — so `train/upstream/` is a clone of the *same pinned commit*, put on
`sys.path` ahead of the installed copy. Clone it at that commit and no other; a
newer one changes the spectrogram path the committed feature banks were built
with.

```bash
cd train
git clone https://github.com/OHF-Voice/micro-wake-word.git upstream
git -C upstream checkout 4665173cd35f1cff9a61e06fc427f124766c488e
```

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

## Measuring a model — and the threshold it is judged by

`train/eval_model.py` scores a model with a port of the app's own detection rule
(window mean over `probability_cutoff`, plus the 4 s refractory period) and
writes the result into `baseline/`. Positives are **marked takes** — the
speaker's own tap — so the denominator is every take they meant to make rather
than every take something happened to detect.

The measurement that settled the project is 92 marked takes from four speakers
and twenty minutes of household ambient: a kitchen with four people talking, and
a living room. None of it had ever been heard by any model under test.

The operating points it scores at are **parsed out of the engine's
`ari-voice/src/wake_model.rs`**, so "at the app's default" always means what the phone is
actually running. That used to be a copied constant and it went stale twice —
once reporting a model uniformly better than its predecessor while the phone
said otherwise. A wrong ladder here is invisible: every row still prints, just
about the wrong app. If the parse fails it raises rather than guessing.

### The threshold was the fault, not the model

The April model (`hey_ari.tflite`) shipped on a ladder of **0.95 / 0.985 / 0.99**
that arrived in a commit called "Minor UI tweaks", was inherited from upstream,
and had never once been measured against a recording. On those 92 takes it hears
the loudest voice in the house 13 times out of 16 and the quietest **18 out of
46**. Every week of "she can't wake it" was that gap: the model could always hear
her, and the threshold was throwing the answer away.

Over the twenty minutes of ambient the highest mean across a window of 10 is
**0.349**. Quoting each cutoff by its headroom above that floor gives
**0.45 / 0.55 / 0.65** at +0.10 / +0.20 / +0.30 — 72% / 69% / 69% recall across
all four speakers, and the quietest speaker at **29 of 46** instead of 18. That
is the ladder the app ships today, in `WakeWordRegistry`.

A cutoff means nothing on its own. It is only ever a number relative to what the
room is already producing, so a ladder belongs to the model it was measured on
and travels with it — which is why operating points live per-model rather than
per-app.

### RUN 2: shipped and reverted the same day

The 2026-09-13 retrain (`baseline/hey_ari-run2-seed3.json`) went out on
2026-09-14 and came back within hours. Its strictest setting, 0.60, sat below an
ordinary kitchen's ambient peak of 0.623 — firing on conversation by
construction rather than by bad luck. Held to **equal headroom** above the floor
it manages 64 / 53 / 46% where the April model manages 74 / 71 / 70%: it
separates the phrase from speech worse, which is how it contrived to miss takes
and false-fire at the same time.

`WakeWordSensitivityTest` in the app now fails any `hey_ari` cutoff that doesn't
clear measured ambient by 0.10. The floor rises as more rooms are recorded, and
the cutoffs move with it.

> **Retraction.** An earlier version of this section presented a threshold sweep
> on one speaker's audio and concluded that "threshold tuning is exhausted, and
> only retraining can help." That was wrong, and expensively so — a week went
> into retraining before anyone checked the number the model was being judged
> by. The sweep was measured near-field on the developer's own voice, which is
> exactly the speaker the old ladder served well. A measurement that only covers
> the person who set the thing up cannot see the fault.

### What is still missing

Twenty minutes cannot price a rate of one false wake an hour. Longer ambient,
from more rooms and more households, is the measurement that would have caught
all of the above and has still not been made.

`ok_ari` and `hey_jarvis` keep the ladder they shipped on. Neither has a
household recording to justify moving it, and moving it on `hey_ari`'s numbers
would be a guess dressed as a fix.
