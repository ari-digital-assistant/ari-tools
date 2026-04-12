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
    ├── modal_train.py         Train on Modal (recommended — one command, no infra)
    ├── train.py               Standalone training script (runs on any GPU instance)
    ├── launch-aws.sh          Launch an AWS spot instance for training (legacy)
    ├── eval.py                Quick eval harness for a GGUF model against Ari test cases
    └── finetune-colab.ipynb   Colab notebook (alternative, manual)
```

After a training run, evaluate the model:

```bash
pip install llama-cpp-python
python3 functiongemma/eval.py ./output/ari-functiongemma-q4_k_m.gguf
```

The eval harness runs 24 hand-picked test cases covering easy utterances,
paraphrases, and negative examples (general-knowledge questions that
shouldn't match any skill). Prints a score per category.

## functiongemma

Fine-tuning pipeline for FunctionGemma 270M, the optional skill router that
catches paraphrases the keyword matcher misses.

### Quick start with Modal (recommended)

```bash
pip install modal
modal setup                                                    # one-time: authenticate
modal secret create huggingface HF_TOKEN=hf_your_token_here    # one-time
modal run functiongemma/modal_train.py                         # train
```

That's it. Modal spins up an A10G GPU, clones ari-engine + ari-skills,
generates the dataset fresh, trains, quantises to GGUF Q4_K_M, and
downloads the result to `./output/ari-functiongemma-q4_k_m.gguf`.

Cost: ~$1.50-2.00 per run. Free tier gives $30/month (~20 runs).

### Alternative: AWS (legacy)

```bash
./functiongemma/launch-aws.sh
```

Requires AWS CLI, OIDC federation, Secrets Manager secret, instance profile,
VPC, and G-instance vCPU quota. See the script header for full prerequisites.
