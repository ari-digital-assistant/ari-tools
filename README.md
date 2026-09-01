# ari-tools

Operational scripts for the [Ari digital assistant](https://github.com/ari-digital-assistant/ari).

This is the place for things that aren't part of the engine, the Android app,
or the skill registry — but are needed to keep the project running. Build
pipelines, registry maintenance, that sort of thing.

## Layout

```
ari-tools/
└── scripts/              — operational scripts, run by CI or by hand
    ├── publish_manifest.py    Publish on-device model manifests to floating releases (called by three workflows)
    ├── stt_bench.py           Replay captured utterances through STT models and compare transcripts
    └── measure_wake_cpu.sh    Measure what always-on wake-word listening costs in CPU
```

### Diagnosing a speech-to-text problem

Start with `scripts/stt_bench.py`, not with the Android pipeline. It replays
the app's debug captures through any model and flags where the bench and the
device disagree — agreement means the model is at fault and no amount of
Kotlin will help.

```bash
pip install sherpa-onnx
scripts/stt_bench.py --model kroko=./kroko-2025-08-06 captures/*.wav
```

This is not a hypothetical shortcut. Nemotron 0.6B int8 shipped as the "high
accuracy" option and was quietly mangling transcripts — "how's the weather"
arriving as "how's the weat" — while a fortnight of fixes went into audio
plumbing that had never been at fault. Twenty minutes on the bench, comparing
it against the 71 MB Kroko on the same WAVs, settled it and retired the model.
