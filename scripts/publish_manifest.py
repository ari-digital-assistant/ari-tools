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

Dedup is content-based: each kind compares the newly-fetched file SHA(s)
against the SHA(s) in the existing manifest (if any) and exits 0 with no
republish if nothing changed. The version string is for display
(`{date}-{sha-short}`) but the comparison is on the SHAs themselves, so
re-running on a different day with no upstream change is a no-op.

LLM tiers are served from our own copy, not from Hugging Face: each GGUF
is mirrored to the `heyari-models` S3 bucket and the manifest points at
`https://heyari.dev/models/...` (CloudFront). The key carries the SHA, so
an object never changes once written and can be cached forever. Hugging
Face throttled multi-GB downloads (ari-android#36), and its `resolve/main`
URL changes content under us whenever upstream re-uploads, failing the
app's checksum until the next nightly run caught up.
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


GH_REPO = "ari-digital-assistant/ari-tools"

# heyari.dev's CloudFront distribution routes /models/* to this bucket.
MODELS_BUCKET = "heyari-models"
MODELS_BASE_URL = "https://heyari.dev"

# The line each upstream licence asks redistributors to ship. Gemma's
# Terms of Use require this exact sentence in a NOTICE file.
LICENCE_NOTICES = {
    "gemma": "Gemma is provided under and subject to the Gemma Terms of Use "
             "found at ai.google.dev/gemma/terms",
    "apache-2.0": "Licensed under the Apache License, Version 2.0: "
                  "https://www.apache.org/licenses/LICENSE-2.0",
}

_session: Optional[requests.Session] = None


def hf_session() -> requests.Session:
    """Shared session for Hugging Face calls.

    Bearer auth (when HF_TOKEN is set) raises our rate-limit ceiling; the
    retry adapter rides out transient 429s/5xx on the `/resolve/` download
    path, honouring any `Retry-After` header HF sends back. Note we no
    longer touch the `/api/models/.../tree` metadata endpoint at all (see
    hf_file_meta) — that JSON API enforces a tight *per-IP* limit shared
    across GitHub's egress IP pool, which a token cannot lift and which
    handed us the nightly 429s. HEAD and GET both go through the resolve
    CDN path instead.
    """
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=8,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        token = os.environ.get("HF_TOKEN")
        if token:
            s.headers["Authorization"] = f"Bearer {token}"
        _session = s
    return _session


def hf_file_meta(repo: str, filename: str) -> dict:
    """Return {url, sha256, size_bytes} for a single file in the repo.

    The hash and size come straight off a HEAD against the file's
    `/resolve/` download URL: HF answers LFS/Xet-backed files with a
    redirect carrying `x-linked-etag` (the content SHA-256) and
    `x-linked-size` (byte count). This deliberately avoids the
    `/api/models/.../tree` JSON endpoint, whose per-IP rate-limit —
    shared across every CI runner on GitHub's egress IP pool — is what
    handed us the nightly 429s that no token or backoff could ride out.
    The `/resolve/` path is the high-traffic download infra and tolerates
    CI bursts the metadata API rejects. As a bonus it also kills the old
    fallback's habit of downloading the whole multi-GB blob just to hash
    it.

    Plain git blobs (e.g. a small `tokens.txt`) also expose an
    `x-linked-etag`, but there it's the git SHA-1 of the blob, not a
    content SHA-256. HF used to omit `x-linked-size` for those, and we
    keyed on that; by October 2026 it sent the size for plain blobs too,
    and both STT manifests shipped a SHA-1 the app could never match. The
    etag's length is the only reliable marker: 64 hex chars is a SHA-256,
    anything else gets downloaded and hashed client-side.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    r = hf_session().head(url, allow_redirects=False, timeout=30)
    if r.status_code == 404:
        raise SystemExit(f"file '{filename}' not found in HF repo {repo}")
    r.raise_for_status()

    etag = (r.headers.get("x-linked-etag") or "").strip('"')
    size = r.headers.get("x-linked-size")
    if len(etag) == 64 and size:
        return {
            "url": url,
            "sha256": etag,
            "size_bytes": int(size),
        }
    return _hash_inline_file(repo, filename, None)


def _hash_inline_file(repo: str, filename: str, size_hint: Optional[int]) -> dict:
    """Fetch a small (non-LFS) file and SHA-256 it client-side."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    r = hf_session().get(url, timeout=60)
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
    """Delete the old manifest.json (if any) and upload the new one.

    `gh release upload` uses the file's basename as the asset name, so
    we have to write into a temp directory under the literal name
    `manifest.json` rather than letting `tempfile.mkstemp` pick a random
    name. Otherwise the asset shows up on the release as something like
    `tmp0abcd123.json` and the app's manifest URL 404s.
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        # Self-heal: create the floating release on first publish for a
        # new model. `gh release create` exits non-zero when the tag
        # already exists, which is the common case — we ignore that.
        # Without this, adding a new model to the workflow matrix would
        # need a one-shot manual `gh release create` step.
        #
        # --prerelease because these are machine-consumed pointers, not
        # releases for humans: the app fetches
        # `/releases/download/<tag>/manifest.json` by exact tag and never
        # asks for "latest". Without the flag a floating tag can outrank a
        # real versioned release for the repo's "Latest release" badge,
        # which is how the existing tags drifted into a mix of true and
        # false.
        subprocess.run(
            ["gh", "release", "create", release_tag,
             "--repo", GH_REPO,
             "--title", release_tag,
             "--prerelease",
             "--notes",
             f"Floating release tracking the latest manifest.json for {release_tag}.",
             "--target", "main"],
            check=False, stderr=subprocess.DEVNULL,
        )
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


def s3_object_exists(key: str) -> bool:
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", MODELS_BUCKET, "--key", key],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return True
    if "404" in result.stderr or "Not Found" in result.stderr:
        return False
    raise SystemExit(f"head-object s3://{MODELS_BUCKET}/{key} failed: {result.stderr.strip()}")


def mirror_to_s3(meta: dict, key: str, notice: str) -> None:
    """Copy a Hugging Face file to S3 at `key`, with a NOTICE beside it.

    The NOTICE goes up last, so its presence means both landed and a re-run
    after a half-finished mirror redoes the whole thing. The bytes are
    hashed on the way down and must match the SHA the manifest will carry:
    a mirror that serves something else would fail every install.
    """
    notice_key = f"{key.rsplit('/', 1)[0]}/NOTICE"
    if s3_object_exists(notice_key):
        print(f"already mirrored: s3://{MODELS_BUCKET}/{key}")
        return

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / key.rsplit("/", 1)[1]
        digest = hashlib.sha256()
        with hf_session().get(meta["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != meta["sha256"].lower():
            raise SystemExit(
                f"downloaded {meta['url']} hashes to {digest.hexdigest()}, "
                f"expected {meta['sha256']}"
            )

        subprocess.run(
            ["aws", "s3", "cp", str(path), f"s3://{MODELS_BUCKET}/{key}",
             "--content-type", "application/octet-stream",
             "--cache-control", "public, max-age=31536000, immutable",
             "--only-show-errors"],
            check=True,
        )
        notice_path = Path(td) / "NOTICE"
        notice_path.write_text(notice)
        subprocess.run(
            ["aws", "s3", "cp", str(notice_path), f"s3://{MODELS_BUCKET}/{notice_key}",
             "--content-type", "text/plain; charset=utf-8",
             "--cache-control", "public, max-age=31536000, immutable",
             "--only-show-errors"],
            check=True,
        )
    print(f"mirrored to s3://{MODELS_BUCKET}/{key}")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y.%m.%d")


def publish_llm(args: argparse.Namespace) -> int:
    meta = hf_file_meta(args.hf_repo, args.hf_file)
    key = f"models/llm/{meta['sha256'].lower()}/{args.hf_file}"
    url = f"{MODELS_BASE_URL}/{key}"

    prev = existing_manifest(args.release_tag) or {}
    same_bytes = prev.get("sha256", "").lower() == meta["sha256"].lower()
    if same_bytes and prev.get("url") == url:
        print(f"{args.release_tag}: no change (sha {meta['sha256'][:12]}…)")
        return 0

    notice = (
        f"{args.hf_file}\n"
        f"Mirrored unchanged from https://huggingface.co/{args.hf_repo} "
        f"for the Ari digital assistant.\n\n"
        f"{LICENCE_NOTICES[args.licence]}\n"
    )
    # Before the manifest moves, so the app is never pointed at a file
    # that isn't there yet.
    mirror_to_s3(meta, key, notice)

    if same_bytes and prev.get("version"):
        # Same file, new host. Keeping the version means installed copies
        # aren't offered an "update" that would download identical bytes.
        version = prev["version"]
        released_at = prev.get("released_at") or now_iso()
    else:
        version = f"{today_yyyymmdd()}-{meta['sha256'][:7]}"
        released_at = now_iso()
    manifest = {
        "version": version,
        "url": url,
        "sha256": meta["sha256"],
        "size_bytes": meta["size_bytes"],
        "released_at": released_at,
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
    p_llm.add_argument("--licence", required=True, choices=sorted(LICENCE_NOTICES))

    p_stt = sub.add_parser("stt-bundle", help="multi-file HF-hosted STT bundle")
    p_stt.add_argument("--hf-repo", required=True)
    p_stt.add_argument(
        "--hf-files", required=True,
        help="comma-separated file names (e.g. encoder.onnx,decoder.onnx,...)",
    )
    p_stt.add_argument("--release-tag", required=True)

    args = p.parse_args()
    if args.kind == "llm":
        return publish_llm(args)
    if args.kind == "stt-bundle":
        return publish_stt_bundle(args)
    raise SystemExit(f"unknown kind: {args.kind}")


if __name__ == "__main__":
    sys.exit(main())
