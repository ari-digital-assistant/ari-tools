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

Fine-tuning pipeline for FunctionGemma 270M, the optional skill router that
catches paraphrases the keyword matcher misses.

Quick start (manual run):

```bash
./functiongemma/launch-aws.sh
```

That's it. The launch script:
1. Spins up an AWS spot instance (g6.xlarge, L4 GPU, ~$0.30/hr spot)
2. Clones ari-tools and ari-engine on the instance
3. Reads the HF token from AWS Secrets Manager (via instance profile)
4. Regenerates the training dataset fresh from the current Ari skill descriptions
5. Fine-tunes FunctionGemma 270M (~20 minutes on L4)
6. Quantises to GGUF Q4_K_M
7. SCPs the result back to `./output/ari-functiongemma-q4_k_m.gguf`
8. Terminates the instance

Cost per run: ~$0.10-0.15. Final model: ~240MB.

### Prerequisites

- `aws` CLI configured (or GitHub OIDC federation if running from Actions)
- AWS region of your choice has an EC2 G instance vCPU quota > 0
- The following AWS resources created once, out of band:
  - OIDC provider for `token.actions.githubusercontent.com` (for Actions runs)
  - IAM role `ari-functiongemma-training` (assumed by GitHub Actions)
  - IAM role + instance profile `ari-functiongemma-instance` (attached to the
    spot instance, grants Secrets Manager read)
  - Secrets Manager secret `ari-functiongemma/hf-token` holding the HuggingFace
    token with Gemma model license accepted

The HuggingFace token is **not** stored in `.env`, GitHub secrets, or
environment variables at runtime. It lives only in Secrets Manager, and the
spot instance fetches it at boot via the instance profile's credentials.
