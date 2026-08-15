"""Process-portable identity for evaluated graph extents."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_extent_hash(cell_ids: Iterable[str]) -> int:
    """Return a deterministic identity for an admitted cell set.

    Python's built-in hash of strings is intentionally salted per
    interpreter process, so ``hash(frozenset(cell_ids))`` cannot be
    compared across saved reports. This function preserves set semantics
    while using a domain-separated BLAKE2b digest. Each UTF-8 identifier
    is length-prefixed, avoiding delimiter ambiguities, and duplicates do
    not alter the result.

    The integer return type retains the existing report and cluster-key
    surface. It is an identity token, not a security attestation; file
    provenance continues to use the corpus SHA-256 manifests.
    """
    digest = hashlib.blake2b(
        digest_size=16,
        person=b'corroborate-ext',
    )
    for cell_id in sorted(frozenset(cell_ids)):
        encoded = cell_id.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, byteorder='big'))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), byteorder='big')


__all__ = ['stable_extent_hash']
