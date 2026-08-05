#!/usr/bin/env python3
"""
Replay captured utterances through STT models and compare the transcripts.

This exists because a fortnight of fixes went into `SpeechRecognizer.kt` for a
problem that was never in `SpeechRecognizer.kt`. Transcripts were arriving
truncated ("how's the weather" as "how's the weat", "turn on the kitchen table
light" as "ton"), and every plausible cause in the Android pipeline got a fix:
decoder warm-up, pre-roll slicing, 100 ms re-batching, a parallel second-opinion
stream, custom endpointing, a silero VAD veto, a stream flush. None of it moved
the needle, because the audio was reaching sherpa intact all along and the model
was mangling it.

Twenty minutes of replaying the same WAVs offline settled it:

    audio                     nemotron 0.6b int8 (663 MB)  kroko (71 MB)
    "how's the weather"       "how's the weat"             "how's the weather"
    "turn on the kitchen..."  "ton"                        "turn on the kitchen table"
    "turn off the bedroom..." "tedroom lights"             "turn off the bedroom lights"

Ten times the size, an order of magnitude worse — int8 quantisation damage to a
0.6B streaming transducer. Nemotron was retired on the strength of that table.

So: before changing pipeline code to chase a transcription problem, run the
audio through the model here and find out whether the pipeline is even involved.

Enable Debug -> keep everything in the app, speak to it, then pull the captures:

    adb shell 'run-as dev.heyari.ari tar -c -C \\
        /data/data/dev.heyari.ari/files utterance-captures' > captures.tar
    tar -xf captures.tar

Each capture is a 16 kHz mono WAV plus a `.txt` sidecar recording what the
device actually produced — which is what makes a mismatch between device and
bench worth investigating rather than guessing at.

Models are directories of ONNX files. Pull one off a device with:

    adb shell 'run-as dev.heyari.ari tar -c -C \\
        /data/data/dev.heyari.ari/files/models kroko-2025-08-06' > m.tar

Usage:
  stt_bench.py --model kroko=./kroko-2025-08-06 captures/*.wav

  # Two models, same audio — the comparison that retired Nemotron:
  stt_bench.py --model kroko=./kroko --model nemotron=./nemotron captures/*.wav

  # Does finalising the stream recover the tail? (It does not, on NeMo models:
  # sherpa's NeMo IsReady() needs a full 121-frame window, so a partial
  # trailing chunk is never decoded and no amount of padding helps.)
  stt_bench.py --model kroko=./kroko --tails captures/*.wav

Requires `pip install sherpa-onnx`. Pin the same version the app ships (see
`app/libs/sherpa-onnx-*.aar`) or the comparison is against different code.
"""

from __future__ import annotations

import argparse
import array
import glob
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 16000
# 100 ms — SpeechRecognizer.BATCH_TARGET_SAMPLES. Feeding a different chunk size
# would not change the transcript (sherpa buffers into fixed feature frames
# regardless) but matching the app keeps the replay honest.
BATCH = 1600


def load_wav(path: str) -> list[float]:
    with wave.open(path) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            raise SystemExit(
                f"{path}: expected 16 kHz mono, got "
                f"{w.getframerate()} Hz / {w.getnchannels()}ch"
            )
        pcm = array.array("h")
        pcm.frombytes(w.readframes(w.getnframes()))
    return [s / 32768.0 for s in pcm]


def sidecar(wav_path: str) -> str | None:
    """What the device produced for this clip, from the `.txt` written beside it."""
    txt = Path(wav_path).with_suffix(".txt")
    if not txt.is_file():
        return None
    for line in txt.read_text(errors="replace").splitlines():
        if line.startswith("transcript:"):
            return line.split(":", 1)[1].strip()
    return None


def build(model_dir: Path):
    import sherpa_onnx

    def one_of(*names: str) -> str:
        for n in names:
            if (model_dir / n).is_file():
                return str(model_dir / n)
        raise SystemExit(f"{model_dir}: none of {names} found")

    # int8 and float exports use different filenames; accept either so a
    # directory pulled straight off a device works untouched.
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=one_of("tokens.txt"),
        encoder=one_of("encoder.onnx", "encoder.int8.onnx"),
        decoder=one_of("decoder.onnx", "decoder.int8.onnx"),
        joiner=one_of("joiner.onnx", "joiner.int8.onnx"),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        dither=0.0,
        # The app does its own endpointing; sherpa's would cut clips short here
        # and we are measuring the decode, not the endpoint.
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
        provider="cpu",
    )


def transcribe(rec, samples: list[float], *, finalise: bool, pad_s: float = 0.0) -> str:
    stream = rec.create_stream()
    for i in range(0, len(samples), BATCH):
        stream.accept_waveform(SAMPLE_RATE, samples[i : i + BATCH])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    if pad_s:
        stream.accept_waveform(SAMPLE_RATE, [0.0] * int(SAMPLE_RATE * pad_s))
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    if finalise:
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
    return rec.get_result(stream).strip()


def parse_model(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--model wants name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    d = Path(path).expanduser()
    if not d.is_dir():
        raise SystemExit(f"--model {name}: {d} is not a directory")
    return name, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="model to test; repeat to compare models on the same audio",
    )
    ap.add_argument(
        "--tails",
        action="store_true",
        help="also report no-finalise and silence-padded variants, to test "
        "whether the tail of the utterance is being dropped",
    )
    ap.add_argument("wavs", nargs="+", help="16 kHz mono WAV files")
    args = ap.parse_args()

    # Shells that do not expand globs, and --tails runs, both benefit from this.
    wavs: list[str] = []
    for w in args.wavs:
        wavs.extend(sorted(glob.glob(w)) if any(c in w for c in "*?[") else [w])
    if not wavs:
        raise SystemExit("no WAV files matched")

    models = [parse_model(m) for m in args.model]
    audio = {w: load_wav(w) for w in wavs}
    width = max(len(Path(w).stem) for w in wavs)

    for name, model_dir in models:
        print(f"\n=== {name}  ({model_dir}) ===")
        rec = build(model_dir)
        for w in wavs:
            got = transcribe(rec, audio[w], finalise=True)
            print(f"  {Path(w).stem:<{width}}  {got!r}")
            device = sidecar(w)
            # A disagreement here means the pipeline IS involved, and the
            # difference is the thing to chase. Agreement means it is the model.
            if device is not None and device != got:
                print(f"  {'':<{width}}  device said: {device!r}  <-- differs")
            if args.tails:
                bare = transcribe(rec, audio[w], finalise=False)
                if bare != got:
                    print(f"  {'':<{width}}  no finalise: {bare!r}")
                for pad in (0.5, 1.5):
                    padded = transcribe(rec, audio[w], finalise=True, pad_s=pad)
                    if padded != got:
                        print(f"  {'':<{width}}  +{pad}s silence: {padded!r}")
        del rec
    return 0


if __name__ == "__main__":
    sys.exit(main())
