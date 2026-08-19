"""Process-portable grouping key for evaluated graph extents."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_extent_hash(cell_ids: Iterable[str]) -> int:
    """Return a compact grouping key for string cell identifiers.

    Python's built-in hash of strings is intentionally salted per
    interpreter process, so ``hash(frozenset(cell_ids))`` cannot be
    compared across processes. This function preserves set semantics
    with a domain-separated BLAKE2b digest. Each UTF-8 identifier is
    length-prefixed, avoiding delimiter ambiguities, and duplicates do
    not alter the key.

    Only the de-duplicated identifier strings participate: row values,
    row multiplicity, missing identifiers, and identifier namespaces do
    not. Equality is therefore useful only as a compact, dataset-relative
    graph-grouping hint. It is not evidence identity, provenance,
    chronology, an integrity attestation, or an admission criterion.
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
