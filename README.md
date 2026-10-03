# Bucket Split

Stable A/B bucketing by hashing the subject string, not the clock. Given the same subject and the same bucket weights, `bucket_for` returns the same bucket index on every machine, in every process, forever.

## Usage

```python
from bucket_split import bucket_for, bucket_distribution, assign

# One-shot: build the table and assign in a single call.
bucket = bucket_for("user-42", [1, 1])   # -> 0 or 1

# Hot path: build the cumulative table once, assign many subjects.
cum = bucket_distribution([50, 30, 20])  # -> [0.5, 0.8, 1.0]
for subject in ["user-1", "user-2", "user-3"]:
    idx = assign(subject, cum)           # -> 0, 1, or 2
```

`bucket_for(subject, weights)` is sugar for `assign(subject, bucket_distribution(weights))`. Use the two-call form when you are bucketing many subjects against the same weights so you don't rebuild the table per subject.

## Why this exists

The problem: you need to split traffic into buckets in a way that is stable across restarts, machines, and Python versions, but you don't want to store the assignment in a database. Hashing the subject identifier gives you that for free — the hash *is* the storage.

The trade-off: this is non-cryptographic hashing (FNV-1a) over the subject's UTF-8 bytes, reduced to `[0, 1)` by dividing the 64-bit hash by `2**64`, then walked against a cumulative weight table. FNV-1a is chosen over `hashlib` because its specification is fixed and has no platform or build variance — `hashlib`'s available algorithms differ across Python builds, and Python's built-in `hash()` is salted per process. The cost is that FNV-1a is not cryptographically strong, so a hostile caller who knows the algorithm could craft subjects that cluster into one bucket. If that is your threat model, this is the wrong library.

## The awkward edge

Adding a bucket does *not* leave every existing subject where it was. Going through a unit interval means most subjects stay put when you grow the number of buckets, but some must move — that is the whole point of adding a bucket. If you need perfect stickiness when reshaping an experiment, keep a per-subject assignment table instead of hashing.

Weights are positive finite numbers only. `0`, negatives, `NaN`, `inf`, and `bool` (which is a subclass of `int` in Python) are all rejected. The empty string is a legal subject; it hashes to a definite value and buckets accordingly.
