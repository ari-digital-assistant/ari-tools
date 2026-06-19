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
        # still returns x-linked-etag — but it's the git SHA-1, NOT a
        # content SHA-256, and there's no x-linked-size. We must ignore
        # that etag and hash the bytes ourselves, or the manifest ends up
        # with a 40-char SHA-1 the app can never match.
        body = b"hello tokens"
        import hashlib
        expected = hashlib.sha256(body).hexdigest()
        head = mock.Mock(return_value=FakeResponse(
            status_code=307,
            headers={"x-linked-etag": '"14a984cd7fdc0cabd277d6d0979f1b6561cbf1f1"'},
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


if __name__ == "__main__":
    unittest.main()
