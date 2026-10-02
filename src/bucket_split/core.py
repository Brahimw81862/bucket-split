"""Stable A/B bucketing by hashing the subject, not the clock.

The goal: given the same subject string and the same set of buckets, return
the same bucket every time, forever, on every machine. We achieve that by
hashing the subject with FNV-1a (a self-contained, deterministic, non-
cryptographic hash) and mapping the resulting integer onto [0, 1) using a
fixed divisor. That gives us a uniform float in the half-open unit interval,
which we then reduce modulo the number of buckets.

Design choices worth stating plainly:

* FNV-1a, not hashlib. hashlib is standard-library but its available
  algorithms vary by platform and Python build ("md5" is universal, but
  reading it as a 128-bit integer and halving it is more machinery than the
  problem deserves). FNV-1a is ~40 lines, has no platform variance, and its
  statistical properties are more than good enough for splitting traffic.

* The hash is reduced to [0, 1) by dividing by 2**64, not by taking hash()
  modulo the bucket count directly. Going through a unit interval lets us
  support non-uniform bucket weights (e.g. 50/30/20) with the same code path
  as uniform buckets, and it keeps the reduction independent of the bucket
  count so adding a bucket doesn't reshuffle every existing subject.

* We hash the UTF-8 bytes of the subject. Python's built-in hash() is salted
  per process and therefore useless for stable bucketing; str.__hash__ also
  varies across Python versions. Encoding to bytes and hashing those makes the
  output reproducible across processes and Python versions.
"""

from __future__ import annotations

__all__ = ["bucket_for", "bucket_distribution", "assign"]


_FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325
_FNV_PRIME_64 = 0x100000001B3
_MASK_64 = (1 << 64) - 1
_DIVISOR = float(1 << 64)


def _fnv1a_64(data: bytes) -> int:
    """Return the 64-bit FNV-1a hash of *data*.

    Implemented inline rather than reaching for hashlib because FNV-1a has a
    fixed specification with no platform or build variance, which is exactly
    the property a stable bucketing function needs.
    """
    h = _FNV_OFFSET_BASIS_64
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME_64) & _MASK_64
    return h


def _unit_value(subject: str) -> float:
    """Map *subject* to a float in the half-open interval [0, 1).

    The float is obtained by dividing the raw 64-bit hash by 2**64. Because the
    hash is an integer in [0, 2**64) and the divisor is exactly representable
    as a float, the result is always in [0, 1) and is deterministic for any
    given subject string across processes and Python versions.
    """
    if not isinstance(subject, str):
        raise TypeError("subject must be str, not {!r}".format(type(subject).__name__))
    h = _fnv1a_64(subject.encode("utf-8"))
    return h / _DIVISOR


def _validate_weights(weights):
    if not weights:
        raise ValueError("weights must be a non-empty sequence")
    total = 0.0
    for w in weights:
        if not isinstance(w, (int, float)):
            raise TypeError("weights must be numeric, got {!r}".format(type(w).__name__))
        if isinstance(w, bool):
            raise TypeError("weights must be numeric, not bool")
        if w <= 0 or w != w:  # w != w is the NaN check
            raise ValueError("every weight must be a positive finite number")
        if w == float("inf"):
            raise ValueError("every weight must be a positive finite number")
        total += w
    if total <= 0:
        raise ValueError("weights must sum to a positive number")


def bucket_distribution(weights):
    """Build the cumulative-weight boundary table used by :func:`assign`.

    *weights* is a sequence of positive numbers describing the relative size of
    each bucket. The function returns a list of cumulative upper bounds in [0,
    1], scaled so the final boundary is exactly 1.0. Passing the result to
    :func:`assign` is the intended use; the table is returned mostly so callers
    can inspect it (and so tests can assert on it).

    We precompute the cumulative table once so that :func:`assign` is O(log n)
    in the number of buckets rather than O(n). For typical A/B tests with two
    or three buckets this hardly matters, but building the table per call would
    be wasteful and would also make it impossible to reuse the same table
    across millions of subjects.
    """
    _validate_weights(weights)
    total = float(sum(weights))
    cumulative = []
    running = 0.0
    for w in weights:
        running += float(w) / total
        cumulative.append(running)
    # Guard against floating-point drift leaving the final boundary at
    # 0.9999999999999998 instead of 1.0, which would mean a subject whose unit
    # value lands in the last sliver could fall off the end.
    cumulative[-1] = 1.0
    return cumulative


def assign(subject, cumulative):
    """Return the index of the bucket *subject* falls into.

    *subject* is the stable identifier of the entity being bucketed (user id,
    session id, etc.). *cumulative* is the table produced by
    :func:`bucket_distribution`.

    We walk the cumulative table from left to right and return the index of the
    first boundary that the subject's unit value is strictly less than. The
    half-open semantics matter: a value equal to a boundary belongs to the
    next bucket, not the one ending at that boundary. Because the unit value
    is always strictly less than 1.0 and the final boundary is pinned to 1.0,
    every subject is guaranteed to land in some bucket.
    """
    if not isinstance(cumulative, list) or not cumulative:
        raise TypeError("cumulative must be a non-empty list from bucket_distribution()")
    v = _unit_value(subject)
    for i, bound in enumerate(cumulative):
        if v < bound:
            return i
    # Defensive fallback. With a well-formed table this is unreachable because
    # cumulative[-1] == 1.0 and v < 1.0. We keep it so a corrupted table can
    # never raise IndexError; the worst case is the last bucket getting one
    # extra subject, which is a safe failure mode for an A/B tool.
    return len(cumulative) - 1


def bucket_for(subject, weights):
    """One-shot helper: build the table and assign *subject* in one call.

    Equivalent to ``assign(subject, bucket_distribution(weights))`` but avoids
    making the caller build the table themselves. If you are bucketing many
    subjects against the same weights, call :func:`bucket_distribution` once
    and :func:`assign` per subject instead — rebuilding the table per subject
    is wasteful and also makes it impossible to reuse the same table
    across millions of subjects.
    """
    return assign(subject, bucket_distribution(weights))
