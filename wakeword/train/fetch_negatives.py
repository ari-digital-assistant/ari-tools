#!/usr/bin/env python3
"""
Fetch real-speech negatives, with the languages that actually wake Ari.

The incumbent's negative set was CHiME, FMA and AudioSet — ambient noise and
music, and **not one word of non-English speech**. The 2026-09-10 capture review
found Maltese and French were 5 of 18 real false accepts, the single largest
class. That is a named, measurable gap rather than a hunch, and it is the most
defensible reason to retrain at all.

Mozilla Common Voice fills it. Maltese has 8.75 validated hours from 225
speakers; French has 1,102 hours. It is CC0 — public domain, no attribution
required — which also keeps the shipped model's provenance clean, unlike the
AudioSet/FMA mixture upstream's notebook warns is "non-commercial personal use
only".

Common Voice is read speech: one person, close-mic'd, no room. It is therefore
NOT a substitute for ambient audio, and `--audioset-parts` exists because the
television and the kitchen still have to be in the corpus somewhere. Use both.

Everything lands as 16 kHz mono WAV, which is what the trainer consumes.

Usage:
  ./fetch_negatives.py --output ../corpus/negatives
  ./fetch_negatives.py --languages mt,fr,en --max-clips 4000 --audioset-parts 2
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_RATE = 16000
CV_REPO = "fsicoli/common_voice_22_0"
CV_URL = "https://huggingface.co/datasets/{repo}/resolve/main/audio/{lang}/{split}/{lang}_{split}_{shard}.tar"
AUDIOSET_REPO = "agkphysics/AudioSet"

# Validated splits only. `other` is unvalidated and `invalidated` was rejected
# by contributors — for negatives the audio would arguably still do, but there
# is no shortage here and no reason to train on clips people flagged as wrong.
CV_SPLITS = ("train", "dev", "test")


def stream_tar(url: str):
    """Yield (name, bytes) from a remote tar, reading only as far as the caller goes.

    Genuinely streamed (`r|*`, no seeking), which matters more than it looks:
    the caller stops at --max-clips, and stopping here stops the download. A
    French Common Voice train shard is 1.5 GB; buffering one to pull 3,000 clips
    off the front of it would be 1.5 GB of transfer and RAM for no reason.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "ari-wakeword"})
    with urllib.request.urlopen(request) as response:
        with tarfile.open(fileobj=response, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                payload = archive.extractfile(member)
                if payload is not None:
                    yield member.name, payload.read()


def to_wav_16k(raw: bytes) -> np.ndarray | None:
    """Decode arbitrary audio bytes to float32 mono 16 kHz, or None if unreadable."""
    try:
        audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return None
    if rate != TARGET_RATE:
        audio = soxr.resample(audio, rate, TARGET_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def harvest(url: str, out_dir: Path, prefix: str, limit: int, suffixes: tuple[str, ...]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        for name, raw in stream_tar(url):
            if written >= limit:
                break
            if not name.lower().endswith(suffixes):
                continue
            audio = to_wav_16k(raw)
            if audio is None:
                continue
            stem = Path(name).stem
            sf.write(out_dir / f"{prefix}-{stem}.wav", audio, TARGET_RATE, subtype="PCM_16")
            written += 1
    except urllib.error.HTTPError as exc:
        print(f"  ! {url.rsplit('/', 1)[-1]}: HTTP {exc.code}", file=sys.stderr)
    return written


def harvest_audioset(out_dir: Path, limit: int) -> int:
    """Stream AudioSet, decoding the bytes ourselves.

    `datasets` will not decode audio without `torchcodec`, which pulls torch
    into this virtualenv — the exact heavyweight pairing the two-project split
    exists to avoid. `Audio(decode=False)` hands back raw FLAC bytes instead,
    and soundfile is already here.

    Clips are 10 s at 48 kHz stereo; to_wav_16k downmixes and resamples.
    """
    from datasets import Audio, load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(AUDIOSET_REPO, split="train", streaming=True).cast_column(
        "audio", Audio(decode=False)
    )
    written = 0
    for row in stream:
        if written >= limit:
            break
        payload = row["audio"].get("bytes")
        if not payload:
            continue
        audio = to_wav_16k(payload)
        if audio is None:
            continue
        sf.write(out_dir / f"as-{row['video_id']}.wav", audio, TARGET_RATE, subtype="PCM_16")
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--languages",
        default="mt,fr",
        help="Common Voice codes. Default is the two the capture review found: "
        "Maltese (spoken in the household) and French (media playback).",
    )
    ap.add_argument("--max-clips", type=int, default=3000, help="Per language")
    ap.add_argument(
        "--audioset-clips",
        type=int,
        default=0,
        help="Ambient/non-speech clips from AudioSet. Common Voice has no "
        "television in it; this is where that comes from.",
    )
    args = ap.parse_args()

    started = time.monotonic()
    counts: dict[str, int] = {}

    for lang in [l.strip() for l in args.languages.split(",") if l.strip()]:
        out_dir = args.output / f"common-voice-{lang}"
        total = 0
        for split in CV_SPLITS:
            if total >= args.max_clips:
                break
            url = CV_URL.format(repo=CV_REPO, lang=lang, split=split, shard=0)
            print(f"common-voice/{lang}/{split} ...", flush=True)
            total += harvest(
                url, out_dir, f"cv-{lang}", args.max_clips - total, (".mp3", ".wav", ".flac")
            )
        counts[f"common-voice-{lang}"] = total
        print(f"  {total} clips -> {out_dir}")

    if args.audioset_clips:
        print(f"audioset (streaming {args.audioset_clips} clips) ...", flush=True)
        got = harvest_audioset(args.output / "audioset", args.audioset_clips)
        counts["audioset"] = got
        print(f"  {got} clips -> {args.output / 'audioset'}")

    if not counts:
        sys.exit("Nothing fetched")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "sources": {
            "common_voice": {"repo": CV_REPO, "splits": list(CV_SPLITS), "licence": "CC0"},
            "audioset": {"repo": "agkphysics/AudioSet", "licence": "see dataset card"},
        },
        "clips_by_source": counts,
        "sample_rate_hz": TARGET_RATE,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }, indent=2))

    print(f"\n{sum(counts.values())} clips total in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
