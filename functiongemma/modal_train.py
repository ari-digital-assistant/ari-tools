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
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
        "echo 'export PATH=/root/.cargo/bin:$PATH' >> /root/.bashrc",
        "/root/.cargo/bin/rustc --version",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

ARI_ENGINE_REPO = "https://github.com/ari-digital-assistant/ari-engine.git"
ARI_SKILLS_REPO = "https://github.com/ari-digital-assistant/ari-skills.git"
ARI_TOOLS_REPO = "https://github.com/ari-digital-assistant/ari-tools.git"

OUTPUT_DIR = "/output"
WORK_DIR = "/work"


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={OUTPUT_DIR: volume},
)
def train():
    import json
    import os
    from pathlib import Path

    # Ensure cargo is on PATH (installed during image build at /root/.cargo/bin)
    cargo_bin = "/root/.cargo/bin"
    if cargo_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

    os.makedirs(WORK_DIR, exist_ok=True)

    # Clone repos
    print("Cloning ari-engine...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", ARI_ENGINE_REPO, f"{WORK_DIR}/ari-engine"]
    )
    print("Cloning ari-skills...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", ARI_SKILLS_REPO, f"{WORK_DIR}/ari-skills"]
    )
    print("Cloning ari-tools...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", ARI_TOOLS_REPO, f"{WORK_DIR}/ari-tools"]
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
            [sys.executable, f"{WORK_DIR}/ari-tools/functiongemma/generate-dataset.py"],
            stdout=f,
            env=env,
        )
    line_count = sum(1 for _ in open(dataset_path))
    print(f"Dataset: {line_count} samples")

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

    config_kwargs = dict(
        output_dir=f"{WORK_DIR}/training",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        packing=False,
        optim="adamw_torch_fused",
        bf16=True,
        completion_only_loss=True,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        report_to="none",
    )
    try:
        training_config = SFTConfig(max_length=512, **config_kwargs)
    except TypeError:
        training_config = SFTConfig(max_seq_length=512, **config_kwargs)

    trainer = SFTTrainer(
        model=base_model,
        args=training_config,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    trainer.train()

    # Save fused model
    fused_dir = f"{WORK_DIR}/fused"
    trainer.save_model(fused_dir)
    processor.save_pretrained(fused_dir)
    print(f"Model saved to {fused_dir}")

    # Convert to GGUF
    print("Cloning llama.cpp for GGUF conversion...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", f"{WORK_DIR}/llama.cpp"]
    )
    print("Building llama-quantize...")
    subprocess.check_call(["cmake", "-B", "build"], cwd=f"{WORK_DIR}/llama.cpp")
    subprocess.check_call(
        ["cmake", "--build", "build", "--target", "llama-quantize", "-j4"],
        cwd=f"{WORK_DIR}/llama.cpp",
    )

    f16_path = f"{WORK_DIR}/ari-functiongemma-f16.gguf"
    q4_path = f"{OUTPUT_DIR}/ari-functiongemma-q4_k_m.gguf"

    print("Converting to GGUF F16...")
    subprocess.check_call([
        sys.executable,
        f"{WORK_DIR}/llama.cpp/convert_hf_to_gguf.py",
        fused_dir,
        "--outfile", f16_path,
        "--outtype", "f16",
    ])

    print("Quantising to Q4_K_M...")
    subprocess.check_call([
        f"{WORK_DIR}/llama.cpp/build/bin/llama-quantize",
        f16_path,
        q4_path,
        "Q4_K_M",
    ])

    size_mb = os.path.getsize(q4_path) / 1024 / 1024
    print(f"Final model: {q4_path} ({size_mb:.0f} MB)")

    volume.commit()
    print("Done. Model saved to Modal volume 'ari-functiongemma-output'.")


@app.local_entrypoint()
def main():
    print("Launching training on Modal...")
    train.remote()

    print("Downloading model from Modal volume...")
    import subprocess as sp
    sp.check_call([
        "modal", "volume", "get",
        "ari-functiongemma-output",
        "ari-functiongemma-q4_k_m.gguf",
        "--output", "./output/ari-functiongemma-q4_k_m.gguf",
    ])
    print("Done. Model: ./output/ari-functiongemma-q4_k_m.gguf")
