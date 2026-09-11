#!/usr/bin/env python3
"""
Train a wake-word model on the prepared feature banks.

Writes the YAML upstream's trainer expects, then runs it. Kept as a script
rather than a notebook so a run is reproducible and reviewable: the sampling
weights and class weights below ARE the model, and a number changed by hand in
a cell is a number nobody can account for six weeks later.

Read the ship gate in the retrain design (section 5) before believing any
output of this. A model is not better because training finished; it is better
when eval_model.py says it beats the incumbent on BOTH recall and false
accepts, on audio neither model has seen.

Sampling weights are deliberately lopsided towards negatives. The incumbent's
failure in the field is false accepts on ordinary speech — 14 of 16 known-hard
clips at the shipped default — and non-English speech was the largest observed
class with no representation at all in its training. Weighting Maltese and
French heavily is the one change here aimed squarely at a measured failure.

Usage:
  ./train_model.py --features ../features --out ../models/hey_ari-v2
  ./train_model.py --features ../features --out ... --steps 20000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# Wake phrase length ceiling the streaming model will accept.
CLIP_DURATION_MS = 1500

# (name, sampling_weight, penalty_weight, truth, truncation_strategy)
#
# speech_mt and speech_fr carry the heaviest weights because they are the gap
# we can actually name: the incumbent saw no non-English speech whatsoever, and
# Maltese and French were 5 of 18 real false accepts in the capture review.
FEATURE_SETS = [
    ("positive", 2.0, 1.0, True, "truncate_start"),
    ("near_miss", 6.0, 1.0, False, "random"),
    ("speech_mt", 10.0, 1.0, False, "random"),
    ("speech_fr", 8.0, 1.0, False, "random"),
    ("ambient", 5.0, 1.0, False, "random"),
]

# Architecture from upstream's notebook. Not tuned here: changing the model and
# the data in the same run means learning nothing from either.
MIXEDNET_ARGS = [
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
]


def build_config(features: Path, out: Path, steps: int, negative_weight: int) -> dict:
    entries = []
    for name, sampling, penalty, truth, truncation in FEATURE_SETS:
        directory = features / name
        if not directory.is_dir():
            print(f"  (no {name} features — skipping)")
            continue
        entries.append({
            "features_dir": str(directory),
            "sampling_weight": sampling,
            "penalty_weight": penalty,
            "truth": truth,
            "truncation_strategy": truncation,
            "type": "mmap",
        })
    if not any(e["truth"] for e in entries):
        sys.exit("No positive feature set — nothing to learn.")
    if not any(not e["truth"] for e in entries):
        sys.exit("No negative feature sets — the model would fire at everything.")

    return {
        "window_step_ms": 10,
        "train_dir": str(out),
        "features": entries,
        "training_steps": [steps],
        "positive_class_weight": [1],
        # Heavily asymmetric: a false wake is what users actually notice and
        # complain about, and what this retrain exists to reduce.
        "negative_class_weight": [negative_weight],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": CLIP_DURATION_MS,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Where the model lands")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--negative-class-weight", type=int, default=20)
    ap.add_argument("--config-only", action="store_true", help="Write the yaml and stop")
    args = ap.parse_args()

    if not args.features.is_dir():
        sys.exit(f"No features at {args.features} — run make_features.py first")

    args.out.mkdir(parents=True, exist_ok=True)
    config = build_config(args.features, args.out, args.steps, args.negative_class_weight)
    config_path = args.out / "training_parameters.yaml"
    config_path.write_text(yaml.dump(config, sort_keys=False))
    print(f"config -> {config_path}")
    for entry in config["features"]:
        kind = "positive" if entry["truth"] else "negative"
        print(f"  {Path(entry['features_dir']).name:12} {kind:8} weight {entry['sampling_weight']}")
    if args.config_only:
        return 0

    command = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train", "1",
        "--restore_checkpoint", "1",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        # The quantized streaming model is the one that ships, so it is the only
        # one worth the time to test.
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        *MIXEDNET_ARGS,
    ]
    print("\n" + " ".join(command[:6]) + " ...\n", flush=True)
    env_path = str(HERE / "upstream")
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = env_path + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
