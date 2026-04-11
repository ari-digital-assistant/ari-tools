# ari-tools

Operational scripts for the [Ari digital assistant](https://github.com/ari-digital-assistant/ari).

This is the place for things that aren't part of the engine, the Android app,
or the skill registry — but are needed to keep the project running. Build
pipelines, training scripts, registry maintenance, that sort of thing.

## Layout

```
ari-tools/
└── functiongemma/        — fine-tuning pipeline for the FunctionGemma router
    ├── generate-dataset.py    Build the training JSONL from Ari skills + mobile-actions
    ├── train.py               Standalone training script (runs on GPU instance)
    ├── launch-aws.sh          Launch a spot instance, train, download result, terminate
    └── finetune-colab.ipynb   Colab notebook (alternative to AWS path)
```

## functiongemma

See `functiongemma/` for the FunctionGemma fine-tuning pipeline.

Quick start (manual run):

```bash
# One-time: copy the env template and fill in your HF token
cp .env.example .env
$EDITOR .env

# Run the training pipeline
./functiongemma/launch-aws.sh
```

That's it. The launch script:
1. Spins up an AWS spot instance (g5.xlarge, A10G GPU, ~$0.15/hr)
2. Clones this repo on the instance
3. Regenerates the training dataset fresh from current Ari skill descriptions
4. Fine-tunes FunctionGemma 270M (~20 minutes on A10G)
5. Quantises to GGUF Q4_K_M
6. SCPs the result back to `./output/ari-functiongemma-q4_k_m.gguf`
7. Terminates the instance

Cost per run: ~$0.10. Final model: ~240MB.

Prerequisites:
- `aws` CLI configured (`aws configure list`)
- `HF_TOKEN` env var (HuggingFace token with Gemma license accepted)
- An EC2 G5 instance quota in your region (request via AWS Service Quotas if needed)
