#!/usr/bin/env python3
"""
Publish on-device model manifests to ari-tools floating releases.

The Ari Android app's auto-update layer polls
`github.com/ari-digital-assistant/ari-tools/releases/download/<tag>/manifest.json`
once a day for each installed model and offers updates when the
manifest version differs from the on-disk sidecar. This script is the
publisher half — three GitHub Actions workflows call it (one per
model category).

Usage:
  publish_manifest.py llm \\
      --hf-repo unsloth/gemma-3-1b-it-GGUF \\
      --hf-file gemma-3-1b-it-Q4_K_M.gguf \\
      --release-tag llm-small-latest

  publish_manifest.py stt-bundle \\
      --hf-repo csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06 \\
      --hf-files encoder.onnx,decoder.onnx,joiner.onnx,tokens.txt \\
      --release-tag stt-kroko-latest

  publish_manifest.py functiongemma \\
      --gguf output/ari-functiongemma-q4_k_m.gguf \\
      --version 2026.04.28-r17 \\
      --gguf-url https://github.com/.../functiongemma-2026.04.28-r17/.../...gguf \\
      --release-tag functiongemma-latest

Dedup is content-based: each kind compares the newly-fetched file SHA(s)
against the SHA(s) in the existing manifest (if any) and exits 0 with no
republish if nothing changed. The version string is for display
(`{date}-{sha-short}`) but the comparison is on the SHAs themselves, so
re-running on a different day with no upstream change is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests


HF_API = "https://huggingface.co/api/models"
GH_REPO = "ari-digital-assistant/ari-tools"


def hf_tree(repo: str) -> list[dict]:
    """List files at HF repo HEAD with their LFS metadata."""
    r = requests.get(f"{HF_API}/{repo}/tree/main", timeout=30)
    r.raise_for_status()
    return r.json()


def hf_file_meta(repo: str, filename: str) -> dict:
    """Return {url, sha256, size_bytes} for a single file in the repo."""
    for entry in hf_tree(repo):
        if entry.get("path") == filename:
            # LFS-tracked files (all GGUFs / .onnx) carry a real SHA-256
            # under entry.lfs.sha256; small files like tokens.txt are
            # inline blobs whose `oid` is a git blob SHA-1, not a file
            # SHA-256, so fall back to a direct download for hashing.
            if entry.get("lfs") and entry["lfs"].get("sha256"):
                return {
                    "url": f"https://huggingface.co/{repo}/resolve/main/{filename}",
                    "sha256": entry["lfs"]["sha256"],
                    "size_bytes": entry["size"],
                }
            return _hash_inline_file(repo, filename, entry.get("size"))
    raise SystemExit(f"file '{filename}' not found in HF repo {repo}")


def _hash_inline_file(repo: str, filename: str, size_hint: Optional[int]) -> dict:
    """Fetch a small (non-LFS) file and SHA-256 it client-side."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    sha = hashlib.sha256(r.content).hexdigest()
    return {"url": url, "sha256": sha, "size_bytes": size_hint or len(r.content)}


def existing_manifest(release_tag: str) -> Optional[dict]:
    """Download manifest.json from the floating release, if present."""
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run(
            [
                "gh", "release", "download", release_tag,
                "--repo", GH_REPO,
                "--pattern", "manifest.json",
                "--dir", td,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return None
        path = Path(td) / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            # Corrupt manifest on the release — treat as absent so we
            # republish a clean one. Loud failure would leave the
            # release stuck; better to self-heal.
            print(f"warn: existing manifest.json is unparseable, will republish",
                  file=sys.stderr)
            return None


def replace_release_asset(release_tag: str, manifest: dict) -> None:
    """Delete the old manifest.json (if any) and upload the new one."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        # Delete-then-upload because GH has no atomic asset replace.
        # `--clobber` on upload also overwrites, but explicit delete
        # makes the operation visible in the audit log.
        subprocess.run(
            ["gh", "release", "delete-asset", release_tag, "manifest.json",
             "--repo", GH_REPO, "--yes"],
            check=False, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["gh", "release", "upload", release_tag, path,
             "--repo", GH_REPO, "--clobber"],
            check=True,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y.%m.%d")


def publish_llm(args: argparse.Namespace) -> int:
    meta = hf_file_meta(args.hf_repo, args.hf_file)

    prev = existing_manifest(args.release_tag)
    prev_sha = (prev or {}).get("sha256", "")
    if prev_sha and prev_sha.lower() == meta["sha256"].lower():
        print(f"{args.release_tag}: no change (sha {meta['sha256'][:12]}…)")
        return 0

    version = f"{today_yyyymmdd()}-{meta['sha256'][:7]}"
    manifest = {
        "version": version,
        "url": meta["url"],
        "sha256": meta["sha256"],
        "size_bytes": meta["size_bytes"],
        "released_at": now_iso(),
    }
    replace_release_asset(args.release_tag, manifest)
    print(f"{args.release_tag}: published version={version}")
    return 0


def publish_stt_bundle(args: argparse.Namespace) -> int:
    file_names = [f.strip() for f in args.hf_files.split(",") if f.strip()]
    if not file_names:
        raise SystemExit("--hf-files must list at least one file")
    metas = {name: hf_file_meta(args.hf_repo, name) for name in file_names}

    prev = existing_manifest(args.release_tag)
    prev_files = {
        entry["name"]: entry["sha256"].lower()
        for entry in ((prev or {}).get("files") or [])
    }
    new_files = {name: m["sha256"].lower() for name, m in metas.items()}
    if prev_files and prev_files == new_files:
        print(f"{args.release_tag}: no change ({len(new_files)} files)")
        return 0

    # Joint dedup hash for the version string. Stable across re-runs as
    # long as no file changes, regardless of date.
    joint = hashlib.sha256(
        ",".join(f"{name}:{sha}" for name, sha in sorted(new_files.items())).encode()
    ).hexdigest()
    version = f"{today_yyyymmdd()}-{joint[:7]}"

    manifest = {
        "version": version,
        "released_at": now_iso(),
        "files": [
            {
                "name": name,
                "url": metas[name]["url"],
                "sha256": metas[name]["sha256"],
                "size_bytes": metas[name]["size_bytes"],
            }
            for name in file_names
        ],
    }
    replace_release_asset(args.release_tag, manifest)
    print(f"{args.release_tag}: published version={version} ({len(new_files)} files)")
    return 0


def publish_functiongemma(args: argparse.Namespace) -> int:
    """For FunctionGemma, the GGUF is already published to a versioned
    release by the training workflow. This script step computes SHA-256
    of the local artifact and replaces the floating-release manifest.
    No content-based dedup — every successful training run emits a fresh
    manifest, by design.
    """
    gguf = Path(args.gguf)
    if not gguf.is_file():
        raise SystemExit(f"GGUF not found at {gguf}")

    h = hashlib.sha256()
    with gguf.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)

    manifest = {
        "version": args.version,
        "url": args.gguf_url,
        "sha256": h.hexdigest(),
        "size_bytes": gguf.stat().st_size,
        "released_at": now_iso(),
    }
    replace_release_asset(args.release_tag, manifest)
    print(f"{args.release_tag}: published version={args.version} "
          f"({manifest['size_bytes']} bytes)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Publish model manifests to ari-tools floating releases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="kind", required=True)

    p_llm = sub.add_parser("llm", help="single-file HF-hosted LLM tier")
    p_llm.add_argument("--hf-repo", required=True)
    p_llm.add_argument("--hf-file", required=True)
    p_llm.add_argument("--release-tag", required=True)

    p_stt = sub.add_parser("stt-bundle", help="multi-file HF-hosted STT bundle")
    p_stt.add_argument("--hf-repo", required=True)
    p_stt.add_argument(
        "--hf-files", required=True,
        help="comma-separated file names (e.g. encoder.onnx,decoder.onnx,...)",
    )
    p_stt.add_argument("--release-tag", required=True)

    p_fg = sub.add_parser(
        "functiongemma",
        help="locally-trained GGUF — point manifest at an already-published versioned release",
    )
    p_fg.add_argument("--gguf", required=True, help="path to local GGUF file")
    p_fg.add_argument("--version", required=True)
    p_fg.add_argument("--gguf-url", required=True,
                      help="public URL where the GGUF is hosted")
    p_fg.add_argument("--release-tag", required=True)

    args = p.parse_args()
    if args.kind == "llm":
        return publish_llm(args)
    if args.kind == "stt-bundle":
        return publish_stt_bundle(args)
    if args.kind == "functiongemma":
        return publish_functiongemma(args)
    raise SystemExit(f"unknown kind: {args.kind}")


if __name__ == "__main__":
    sys.exit(main())
