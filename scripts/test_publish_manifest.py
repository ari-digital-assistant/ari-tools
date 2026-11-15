#!/usr/bin/env python3
"""Tests for publish_manifest's HF metadata fetch.

Run: python3 scripts/test_publish_manifest.py

These pin the behaviour that keeps the nightly `Publish * manifests`
crons green: the file hash/size must come from a HEAD against the
`/resolve/` download path (headers `x-linked-etag` / `x-linked-size`),
NOT from the `/api/models/.../tree` JSON endpoint whose per-IP rate
limit — shared across GitHub's egress IP pool — handed us the nightly
429s. The metadata API must never be touched for an LFS file, and we
must never download a multi-GB blob just to hash it.
"""

import unittest
from unittest import mock

import publish_manifest as pm


class FakeResponse:
    def __init__(self, status_code=200, headers=None, content=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"raise_for_status on {self.status_code}")


class HfFileMetaTests(unittest.TestCase):
    def setUp(self):
        # Reset the module-level memoised session between tests.
        pm._session = None

    def _install_session(self, head=None, get=None):
        sess = mock.Mock()
        sess.head = head or mock.Mock()
        sess.get = get or mock.Mock()
        pm._session = sess
        return sess

    def test_lfs_file_reads_hash_from_head_no_api_no_download(self):
        # HF serves LFS/Xet files as a 302 carrying the content SHA-256
        # in x-linked-etag (quoted) and the size in x-linked-size.
        head = mock.Mock(return_value=FakeResponse(
            status_code=302,
            headers={
                "x-linked-etag": '"9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"',
                "x-linked-size": "3106736256",
            },
        ))
        sess = self._install_session(head=head)

        meta = pm.hf_file_meta("unsloth/gemma-4-E2B-it-GGUF",
                               "gemma-4-E2B-it-Q4_K_M.gguf")

        self.assertEqual(
            meta["sha256"],
            "9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d",
        )
        self.assertEqual(meta["size_bytes"], 3106736256)
        self.assertEqual(
            meta["url"],
            "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
        )
        # The HEAD must target the resolve path, not the throttled API,
        # and must not follow the redirect into the CDN (the headers we
        # need live on the 302 itself).
        (called_url,), kwargs = head.call_args
        self.assertIn("/resolve/main/", called_url)
        self.assertNotIn("/api/models", called_url)
        self.assertFalse(kwargs.get("allow_redirects", True))
        # Never download the blob to hash it.
        sess.get.assert_not_called()

    def test_non_lfs_blob_with_sha1_etag_falls_back_to_download(self):
        # The trap: HF's resolve HEAD on a plain git blob (tokens.txt)
        # returns x-linked-etag as the git SHA-1, not a content SHA-256,
        # and since October 2026 it sends x-linked-size alongside it.
        # Headers below are what HF really returned for Kroko's tokens.txt.
        # We must ignore that etag and hash the bytes ourselves, or the
        # manifest ends up with a 40-char SHA-1 the app can never match.
        body = b"hello tokens"
        import hashlib
        expected = hashlib.sha256(body).hexdigest()
        head = mock.Mock(return_value=FakeResponse(
            status_code=307,
            headers={
                "x-linked-etag": '"14a984cd7fdc0cabd277d6d0979f1b6561cbf1f1"',
                "x-linked-size": "6310",
            },
        ))
        get = mock.Mock(return_value=FakeResponse(status_code=200, content=body))
        self._install_session(head=head, get=get)

        meta = pm.hf_file_meta("some/repo", "tokens.txt")

        self.assertEqual(meta["sha256"], expected)
        self.assertEqual(len(meta["sha256"]), 64)
        self.assertEqual(meta["size_bytes"], len(body))
        get.assert_called_once()

    def test_missing_file_raises(self):
        head = mock.Mock(return_value=FakeResponse(status_code=404))
        self._install_session(head=head)
        with self.assertRaises(SystemExit):
            pm.hf_file_meta("some/repo", "nope.gguf")


class PublishLlmTests(unittest.TestCase):
    """LLM manifests point at our CloudFront mirror, never at Hugging Face."""

    SHA = "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"
    HF_URL = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
    MIRROR_URL = f"https://heyari.dev/models/llm/{SHA}/gemma-4-E2B-it-Q4_K_M.gguf"

    def _run(self, prev):
        args = pm.argparse.Namespace(
            hf_repo="unsloth/gemma-4-E2B-it-GGUF",
            hf_file="gemma-4-E2B-it-Q4_K_M.gguf",
            release_tag="llm-medium-latest",
            licence="apache-2.0",
        )
        meta = {"url": self.HF_URL, "sha256": self.SHA, "size_bytes": 3106738272}
        with mock.patch.object(pm, "hf_file_meta", return_value=meta), \
             mock.patch.object(pm, "existing_manifest", return_value=prev), \
             mock.patch.object(pm, "mirror_to_s3") as mirror, \
             mock.patch.object(pm, "replace_release_asset") as publish:
            self.assertEqual(pm.publish_llm(args), 0)
        return mirror, publish

    def test_already_mirrored_is_a_no_op(self):
        mirror, publish = self._run({"version": "v1", "sha256": self.SHA, "url": self.MIRROR_URL})
        mirror.assert_not_called()
        publish.assert_not_called()

    def test_same_bytes_on_hugging_face_move_keeping_the_version(self):
        # The one-off move: nothing changed upstream, so installed copies
        # must not be offered an update, but the URL has to change.
        prev = {"version": "2026.07.18-740185b", "sha256": self.SHA, "url": self.HF_URL,
                "released_at": "2026-07-18T04:13:57Z"}
        mirror, publish = self._run(prev)
        mirror.assert_called_once()
        (_, manifest), _ = publish.call_args
        self.assertEqual(manifest["url"], self.MIRROR_URL)
        self.assertEqual(manifest["version"], "2026.07.18-740185b")
        self.assertEqual(manifest["released_at"], "2026-07-18T04:13:57Z")

    def test_new_upstream_bytes_get_a_new_version_and_the_mirror_url(self):
        prev = {"version": "old", "sha256": "0" * 64, "url": "https://heyari.dev/models/llm/old/x.gguf"}
        mirror, publish = self._run(prev)
        (meta, key, notice), _ = mirror.call_args
        self.assertEqual(key, f"models/llm/{self.SHA}/gemma-4-E2B-it-Q4_K_M.gguf")
        self.assertIn("Apache License", notice)
        (_, manifest), _ = publish.call_args
        self.assertEqual(manifest["url"], self.MIRROR_URL)
        self.assertTrue(manifest["version"].endswith("-740185b"))
        self.assertNotEqual(manifest["version"], "old")


if __name__ == "__main__":
    unittest.main()
