"""
Fine-tune FunctionGemma 270M on Modal.

Usage:
    modal run functiongemma/modal_train.py

Output is saved to a Modal Volume and downloaded locally at the end.
Requires:
    - `modal` CLI installed and authenticated (`pip install modal && modal setup`)
    - Modal secret named "huggingface" with key HF_TOKEN
"""

import modal
import subprocess
import sys

app = modal.App("ari-functiongemma")

# Persistent volume for output artifacts across runs.
volume = modal.Volume.from_name("ari-functiongemma-output", create_if_missing=True)

# Image with everything pre-installed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential", "curl")
    .pip_install(
        "torch",
        "transformers==4.57.1",
        "trl==0.25.1",
        "datasets",
        "huggingface_hub",
        "gguf",
        "pyyaml",
        "sentencepiece",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        "/root/.cargo/bin/rustc --version",
    )
    # Pre-download the base model into the image cache. This is a gated
    # model so it needs the HF token during build. The model files get
    # baked into the image layer — subsequent runs load from disk, not
    # the network.
    .run_commands(
        "python3 -c '"
        "from huggingface_hub import login; import os; login(token=os.environ[\"HF_TOKEN\"]); "
        "from transformers import AutoProcessor, AutoModelForCausalLM; "
        "AutoProcessor.from_pretrained(\"google/functiongemma-270m-it\"); "
        "AutoModelForCausalLM.from_pretrained(\"google/functiongemma-270m-it\")'",
        secrets=[modal.Secret.from_name("huggingface")],
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

ARI_ENGINE_REPO = "https://github.com/ari-digital-assistant/ari-engine.git"
ARI_SKILLS_REPO = "https://github.com/ari-digital-assistant/ari-skills.git"
ARI_TOOLS_REPO = "https://github.com/ari-digital-assistant/ari-tools.git"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
# Conversion is part of the model artifact. Pin it so a quant sweep compares
# quantisation levels rather than accidentally comparing different converter
# revisions. r127 cloned `master` without recording its commit, so it cannot be
# reproduced byte-for-byte; all recovery artifacts use this one revision.
LLAMA_CPP_COMMIT = "0cea36222fe9bac5ebfc45716c9eef11f37046c4"
SWEEP_QUANTS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

OUTPUT_DIR = "/output"
WORK_DIR = "/work"


def _prepare_llama_cpp():
    """Check out and build the pinned GGUF converter/quantizer."""
    import os
    import shutil

    checkout = f"{WORK_DIR}/llama.cpp"
    if os.path.exists(checkout):
        shutil.rmtree(checkout)
    os.makedirs(checkout)
    subprocess.check_call(["git", "init"], cwd=checkout)
    subprocess.check_call(
        ["git", "remote", "add", "origin", LLAMA_CPP_REPO], cwd=checkout
    )
    subprocess.check_call(
        ["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_COMMIT],
        cwd=checkout,
    )
    subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if actual != LLAMA_CPP_COMMIT:
        raise RuntimeError(
            f"llama.cpp checkout mismatch: wanted {LLAMA_CPP_COMMIT}, got {actual}"
        )
    print(f"Building llama-quantize from llama.cpp {actual}...")
    subprocess.check_call(["cmake", "-B", "build"], cwd=checkout)
    subprocess.check_call(
        ["cmake", "--build", "build", "--target", "llama-quantize", "-j4"],
        cwd=checkout,
    )
    return checkout


def _ensure_tokenizer_files(fused_dir: str):
    """Restore tokenizer files that save_pretrained occasionally omits."""
    import os
    import shutil

    from huggingface_hub import hf_hub_download, login

    login(token=os.environ["HF_TOKEN"])
    base_model_id = "google/functiongemma-270m-it"
    for fname in ["tokenizer.model", "added_tokens.json", "special_tokens_map.json"]:
        if os.path.exists(os.path.join(fused_dir, fname)):
            continue
        try:
            src = hf_hub_download(base_model_id, fname)
            shutil.copy2(src, os.path.join(fused_dir, fname))
            print(f"  Copied {fname} from Hugging Face Hub")
        except Exception as exc:
            print(f"  WARNING: could not download {fname}: {exc}")


def _convert_fused(
    fused_dir: str,
    locale: str,
    output_dir: str,
    quant_types=("Q4_K_M",),
    retain_f16: bool = False,
):
    """Convert one fused checkpoint and quantise every requested variant."""
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)
    _ensure_tokenizer_files(fused_dir)
    llama_cpp = _prepare_llama_cpp()

    f16_dir = output_dir if retain_f16 else WORK_DIR
    f16_path = f"{f16_dir}/ari-functiongemma-{locale}-f16.gguf"
    print("Converting to GGUF F16...")
    subprocess.check_call(
        [
            sys.executable,
            f"{llama_cpp}/convert_hf_to_gguf.py",
            fused_dir,
            "--outfile",
            f16_path,
            "--outtype",
            "f16",
        ]
    )

    files = {}
    if retain_f16:
        files["f16"] = {
            "path": f16_path,
            "bytes": os.path.getsize(f16_path),
        }
    for quant_type in quant_types:
        suffix = quant_type.lower()
        quant_path = f"{output_dir}/ari-functiongemma-{locale}-{suffix}.gguf"
        print(f"Quantising to {quant_type}...")
        subprocess.check_call(
            [
                f"{llama_cpp}/build/bin/llama-quantize",
                f16_path,
                quant_path,
                quant_type,
            ]
        )
        files[suffix] = {
            "path": quant_path,
            "bytes": os.path.getsize(quant_path),
        }
        print(f"  {suffix}: {files[suffix]['bytes'] / 1024 / 1024:.1f} MiB")

    metadata = {
        "locale": locale,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    metadata_path = Path(output_dir) / "quantization-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={OUTPUT_DIR: volume},
)
def train(engine_ref: str = "main", skills_ref: str = "main", tools_ref: str = "main", locale: str = "en"):
    import json
    import os
    from pathlib import Path

    # Ensure cargo is on PATH (installed during image build at /root/.cargo/bin)
    cargo_bin = "/root/.cargo/bin"
    if cargo_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

    os.makedirs(WORK_DIR, exist_ok=True)

    # Clone repos. Refs default to main; a branch can be passed so an
    # experiment can be trained without landing on main first (and so the
    # nightly, which passes no refs, is unaffected).
    print(f"Cloning ari-engine @ {engine_ref}...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--branch", engine_ref, ARI_ENGINE_REPO, f"{WORK_DIR}/ari-engine"]
    )
    print(f"Cloning ari-skills @ {skills_ref}...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--branch", skills_ref, ARI_SKILLS_REPO, f"{WORK_DIR}/ari-skills"]
    )
    print(f"Cloning ari-tools @ {tools_ref}...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--branch", tools_ref, ARI_TOOLS_REPO, f"{WORK_DIR}/ari-tools"]
    )

    # Verify cargo is available
    print(f"PATH: {os.environ.get('PATH', 'NOT SET')}")
    subprocess.check_call(["which", "cargo"])
    subprocess.check_call(["cargo", "--version"])

    # Generate dataset
    print("Generating dataset...")
    env = os.environ.copy()
    env["ARI_ENGINE_DIR"] = f"{WORK_DIR}/ari-engine"
    env["ARI_SKILLS_DIR"] = f"{WORK_DIR}/ari-skills"
    env["PATH"] = os.environ["PATH"]
    dataset_path = f"{WORK_DIR}/dataset.jsonl"
    with open(dataset_path, "w") as f:
        subprocess.check_call(
            [sys.executable, f"{WORK_DIR}/ari-tools/functiongemma/generate-dataset.py",
             "--locale", locale,
             # Frame×slot banks — the Google mobile-actions-scale volume
             # recipe. Deterministic expansion from committed, reviewed JSON;
             # fails loudly if a router-eligible skill has no bank.
             "--augment", f"{WORK_DIR}/ari-tools/functiongemma/corpus"],
            stdout=f,
            env=env,
        )
    line_count = sum(1 for _ in open(dataset_path))
    # Content hash in the run log: generation is deterministic given its
    # inputs (verified 2026-07-22, incl. across PYTHONHASHSEED values), so
    # two runs with different hashes ATE DIFFERENT INPUTS — the 2026-07-20
    # "nondeterminism" scare was bank-surgery commits landing 3 minutes
    # before a dispatch. This line makes the next such diff self-diagnosing.
    import hashlib
    corpus_sha = hashlib.sha256(open(dataset_path, "rb").read()).hexdigest()[:16]
    print(f"Dataset: {line_count} samples sha256:{corpus_sha}")

    # Load and split
    samples = []
    with open(dataset_path) as f:
        for line in f:
            samples.append(json.loads(line))
    train_data = [s for s in samples if s["metadata"] == "train"]
    eval_data = [s for s in samples if s["metadata"] == "eval"]
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # HF login
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    # Load model
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    base_model_id = "google/functiongemma-270m-it"
    print(f"Loading {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.config.pad_token_id = processor.pad_token_id
    print(f"Model: {base_model.num_parameters():,} parameters")

    # Format for TRL
    print("Formatting dataset...")
    train_formatted = []
    eval_formatted = []
    for data_list, out_list in [(train_data, train_formatted), (eval_data, eval_formatted)]:
        for s in data_list:
            try:
                full = processor.apply_chat_template(
                    s["messages"], tools=s["tools"], tokenize=False,
                    add_generation_prompt=False,
                )
                prompt = processor.apply_chat_template(
                    s["messages"][:-1], tools=s["tools"], tokenize=False,
                    add_generation_prompt=True,
                )
                completion = full[len(prompt):]
                if completion.strip():
                    out_list.append({"prompt": prompt, "completion": completion})
            except Exception:
                pass
    print(f"  Formatted: {len(train_formatted)} train, {len(eval_formatted)} eval")

    # Train
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    train_ds = Dataset.from_list(train_formatted)
    eval_ds = Dataset.from_list(eval_formatted)

    # 2 epochs is the value every good model this project has produced was
    # trained at (effective batch 4 x 8 = 32; ~630 samples -> ~18 steps/epoch
    # -> ~36 optimizer steps).
    #
    # It was briefly raised to 5 on 2026-07-19, to compensate for optimizer
    # steps lost when the keyword-hit filter halved the corpus. That made
    # results much WORSE (en positives 65%->40%, it 65%->30% with a total
    # abstention collapse), which falsified the undertraining hypothesis and
    # showed the smaller corpus itself was the problem. The filter is now
    # default-off, the corpus is back to full size, and so is this.
    #
    # If corpus size changes materially, re-derive this from target optimizer
    # steps rather than leaving it fixed — that coupling is what made the
    # earlier measurement uninterpretable.
    # Batch geometry: 4 × 8-accum (effective 32) — measured 2026-07-19 as
    # the right call, twice over. Do not "optimise" this without new
    # evidence:
    #   - batch 32×1 OOM'd instantly (run 29699674091): a single 41.16 GiB
    #     allocation. The wall is not the 270M's weights, it is the LOGITS
    #     tensor — Gemma's 262,144-token vocabulary makes loss computation
    #     materialise batch × seq × 262144 floats.
    #   - batch 8×4 fit (peak 33.4 GiB of 40 — far above naive estimates,
    #     so no more headroom than that) but delivered ZERO speedup: the
    #     same vocab projection dominates runtime and scales with tokens,
    #     not micro-pass count. Trained model was statistically equivalent
    #     (local eval 82%/35% vs 84%/40%, within cross-run noise), so it
    #     was reverted to keep the canonical config identical to every
    #     measured baseline.
    # The real lever, if training speed ever matters, is a fused/chunked
    # cross-entropy (Liger-style) that never materialises the full logits
    # tensor — that kills the OOM wall and the runtime wall together.
    config_kwargs = dict(
        output_dir=f"{WORK_DIR}/training",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        packing=False,
        optim="adamw_torch_fused",
        bf16=True,
        completion_only_loss=True,
        # Must stay well below the total step count or the run emits no logs and
        # no intermediate eval at all — which is exactly what happened at 50/100
        # against a ~14-step run.
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    try:
        training_config = SFTConfig(max_length=1536, **config_kwargs)
    except TypeError:
        training_config = SFTConfig(max_seq_length=1536, **config_kwargs)

    trainer = SFTTrainer(
        model=base_model,
        args=training_config,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    trainer.train()
    # Headroom as a number, not a vibe — the batch-32 geometry is safe only
    # while this stays comfortably under the card's 40GB.
    import torch
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM: {peak_gb:.1f} GiB of 40 GiB")

    # Save fused model (both to working dir and volume for recovery)
    fused_dir = f"{WORK_DIR}/fused"
    trainer.save_model(fused_dir)
    processor.save_pretrained(fused_dir)
    print(f"Model saved to {fused_dir}")

    # Also save to volume so we can retry conversion without retraining
    import shutil as _shutil
    fused_backup = f"{OUTPUT_DIR}/fused-{locale}"
    if os.path.exists(fused_backup):
        _shutil.rmtree(fused_backup)
    _shutil.copytree(fused_dir, fused_backup)
    volume.commit()
    print(f"Fused model backed up to volume")

    metadata = _convert_fused(fused_dir, locale, OUTPUT_DIR)
    q4 = metadata["files"]["q4_k_m"]
    print(f"Final model: {q4['path']} ({q4['bytes'] / 1024 / 1024:.0f} MiB)")

    volume.commit()
    print("Done. Model saved to Modal volume 'ari-functiongemma-output'.")


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={OUTPUT_DIR: volume},
)
def convert_only_fn(locale: str = "en"):
    """Retry just the GGUF conversion using the fused model saved to the volume."""
    import os
    import shutil

    fused_backup = f"{OUTPUT_DIR}/fused-{locale}"
    if not os.path.exists(fused_backup):
        raise RuntimeError("No fused model on volume. Run full training first.")

    os.makedirs(WORK_DIR, exist_ok=True)
    fused_dir = f"{WORK_DIR}/fused"
    if not os.path.exists(fused_dir):
        shutil.copytree(fused_backup, fused_dir)
    print(f"Fused model restored from volume to {fused_dir}")

    metadata = _convert_fused(fused_dir, locale, OUTPUT_DIR)
    q4 = metadata["files"]["q4_k_m"]
    print(f"Final model: {q4['path']} ({q4['bytes'] / 1024 / 1024:.0f} MiB)")
    volume.commit()


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={OUTPUT_DIR: volume},
)
def quant_sweep_fn(locale: str = "it"):
    """Quantise a retained fused checkpoint without spending GPU time."""
    import os
    import shutil

    fused_backup = f"{OUTPUT_DIR}/fused-{locale}"
    if not os.path.exists(fused_backup):
        raise RuntimeError(
            f"No fused-{locale} checkpoint on the Modal volume. Run training first."
        )

    os.makedirs(WORK_DIR, exist_ok=True)
    fused_dir = f"{WORK_DIR}/fused"
    if os.path.exists(fused_dir):
        shutil.rmtree(fused_dir)
    shutil.copytree(fused_backup, fused_dir)
    print(f"Fused model restored from {fused_backup}")

    sweep_dir = f"{OUTPUT_DIR}/quant-sweep-{locale}"
    if os.path.exists(sweep_dir):
        shutil.rmtree(sweep_dir)
    _convert_fused(
        fused_dir,
        locale,
        sweep_dir,
        quant_types=SWEEP_QUANTS,
        retain_f16=True,
    )
    volume.commit()
    print(f"Quant sweep saved to {sweep_dir}")


@app.local_entrypoint()
def main(
    convert_only: bool = False,
    quant_sweep: bool = False,
    engine_ref: str = "main",
    skills_ref: str = "main",
    tools_ref: str = "main",
    locale: str = "en",
):
    import os as _os
    if convert_only and quant_sweep:
        raise ValueError("--convert-only and --quant-sweep are mutually exclusive")

    if convert_only:
        print(f"Retrying GGUF conversion only ({locale})...")
        convert_only_fn.remote(locale=locale)
    elif quant_sweep:
        print(f"Quantising retained fused checkpoint ({locale})...")
        quant_sweep_fn.remote(locale=locale)
    else:
        print(f"Launching training on Modal (engine={engine_ref} skills={skills_ref} tools={tools_ref} locale={locale})...")
        train.remote(engine_ref=engine_ref, skills_ref=skills_ref, tools_ref=tools_ref, locale=locale)

    import subprocess as sp
    _os.makedirs("./output", exist_ok=True)
    if quant_sweep:
        filenames = [
            f"ari-functiongemma-{locale}-{suffix}.gguf"
            for suffix in ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")
        ] + ["quantization-metadata.json"]
        print("Downloading quant sweep from Modal volume...")
        for filename in filenames:
            sp.check_call(
                [
                    "modal",
                    "volume",
                    "get",
                    "ari-functiongemma-output",
                    f"quant-sweep-{locale}/{filename}",
                    "./output/",
                    "--force",
                ]
            )
        print("Done. Quant sweep is in ./output/")
    else:
        gguf_name = f"ari-functiongemma-{locale}-q4_k_m.gguf"
        print("Downloading model from Modal volume...")
        sp.check_call(
            [
                "modal",
                "volume",
                "get",
                "ari-functiongemma-output",
                gguf_name,
                "./output/",
                "--force",
            ]
        )
        print(f"Done. Model: ./output/{gguf_name}")
