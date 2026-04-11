#!/usr/bin/env python3
"""
Fine-tune FunctionGemma 270M for Ari skill routing.

Designed to run on a clean AWS Deep Learning AMI (PyTorch + CUDA
pre-installed). Generates the dataset fresh from generate-dataset.py
(which pulls Ari skill paraphrases + Google's mobile-actions), then
trains, then exports GGUF.

Usage:
    python3 train.py --hf-token $HF_TOKEN --output-dir ./output

Expects generate-dataset.py to be in the same directory.

Output: <output-dir>/ari-functiongemma-q4_k_m.gguf
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def install_deps():
    """Install Python deps not in the Deep Learning AMI."""
    print("Installing Python dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.57.1",
        "trl==0.25.1",
        "datasets",
        "huggingface_hub",
        "gguf",
    ])


def generate_dataset(dest: Path):
    """Run generate-dataset.py to build the training data fresh."""
    print("Generating dataset...")
    script = Path(__file__).parent / "generate-dataset.py"
    with open(dest, "w") as f:
        subprocess.check_call(
            [sys.executable, str(script)],
            stdout=f,
        )
    line_count = sum(1 for _ in open(dest))
    print(f"  {line_count} samples")


def hf_login(token: str):
    from huggingface_hub import login
    login(token=token)


def load_dataset_split(jsonl_path: Path):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))
    train = [s for s in samples if s["metadata"] == "train"]
    eval_ = [s for s in samples if s["metadata"] == "eval"]
    return train, eval_


def format_for_trl(samples, processor):
    """Convert messages/tools format to prompt/completion strings."""
    out = []
    for s in samples:
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
            out.append({"prompt": prompt, "completion": completion})
    return out


def train_model(
    base_model_id: str,
    train_data,
    eval_data,
    output_dir: Path,
):
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor, AutoModelForCausalLM
    from trl import SFTTrainer, SFTConfig

    print(f"Loading base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.config.pad_token_id = processor.pad_token_id

    print(f"Model loaded: {base_model.num_parameters():,} parameters")

    print("Formatting dataset...")
    train_formatted = format_for_trl(train_data, processor)
    eval_formatted = format_for_trl(eval_data, processor)
    print(f"  train: {len(train_formatted)}, eval: {len(eval_formatted)}")

    # Compute max sequence length from a sample
    max_len = 0
    for s in train_formatted[:1000]:
        tokens = processor.encode(s["prompt"] + s["completion"])
        max_len = max(max_len, len(tokens))
    max_seq_len = min(max_len + 100, 2048)
    print(f"  max sequence length: {max_seq_len}")

    train_ds = Dataset.from_list(train_formatted)
    eval_ds = Dataset.from_list(eval_formatted)

    config_kwargs = dict(
        output_dir=str(output_dir / "training"),
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
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        report_to="none",
    )
    # max_length / max_seq_length naming changed across TRL versions
    try:
        training_config = SFTConfig(max_length=max_seq_len, **config_kwargs)
    except TypeError:
        training_config = SFTConfig(max_seq_length=max_seq_len, **config_kwargs)

    trainer = SFTTrainer(
        model=base_model,
        args=training_config,
        processing_class=processor,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    trainer.train()

    fused_dir = output_dir / "fused"
    print(f"Saving fused model to {fused_dir}")
    trainer.save_model(str(fused_dir))
    processor.save_pretrained(str(fused_dir))

    return fused_dir


def convert_to_gguf(fused_dir: Path, output_dir: Path) -> Path:
    """Convert HF model to GGUF F16, then quantise to Q4_K_M."""
    llama_cpp_dir = output_dir / "llama.cpp"
    if not llama_cpp_dir.exists():
        print("Cloning llama.cpp...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(llama_cpp_dir),
        ])

    print("Building llama-quantize...")
    subprocess.check_call(
        ["cmake", "-B", "build"], cwd=llama_cpp_dir,
    )
    subprocess.check_call(
        ["cmake", "--build", "build", "--target", "llama-quantize", "-j4"],
        cwd=llama_cpp_dir,
    )

    f16_path = output_dir / "ari-functiongemma-f16.gguf"
    print(f"Converting to GGUF F16: {f16_path}")
    subprocess.check_call([
        sys.executable,
        str(llama_cpp_dir / "convert_hf_to_gguf.py"),
        str(fused_dir),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ])

    q4_path = output_dir / "ari-functiongemma-q4_k_m.gguf"
    print(f"Quantising to Q4_K_M: {q4_path}")
    subprocess.check_call([
        str(llama_cpp_dir / "build" / "bin" / "llama-quantize"),
        str(f16_path),
        str(q4_path),
        "Q4_K_M",
    ])

    print(f"Final model size: {q4_path.stat().st_size / 1024 / 1024:.0f} MB")
    return q4_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--base-model", default="google/functiongemma-270m-it")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    if not args.skip_install:
        install_deps()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.jsonl"
    generate_dataset(dataset_path)

    hf_login(args.hf_token)

    train_data, eval_data = load_dataset_split(dataset_path)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    fused_dir = train_model(
        args.base_model, train_data, eval_data, output_dir,
    )

    gguf_path = convert_to_gguf(fused_dir, output_dir)
    print(f"\nDone: {gguf_path}")


if __name__ == "__main__":
    main()
