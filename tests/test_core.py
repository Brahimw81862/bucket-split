import unittest

from bucket_split import bucket_for, bucket_distribution, assign


class TestStability(unittest.TestCase):
    """The whole point of the library: same input, same output, forever."""

    def test_same_subject_same_bucket_across_calls(self):
        w = [1, 1]
        first = bucket_for("user-42", w)
        for _ in range(50):
            self.assertEqual(bucket_for("user-42", w), first)

    def test_different_subjects_can_share_a_bucket(self):
        # With two buckets we expect roughly half of a small set to land in each,
        # but we deliberately do NOT assert on distribution shape here — that
        # would be a statistical test and therefore non-deterministic. We only
        # assert that every subject maps to a valid bucket.
        w = [1, 1]
        for s in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            idx = bucket_for(s, w)
            self.assertIn(idx, (0, 1))


class TestWeights(unittest.TestCase):
    def test_uniform_two_buckets_covers_all_indices(self):
        cum = bucket_distribution([1, 1])
        self.assertEqual(len(cum), 2)
        self.assertAlmostEqual(cum[0], 0.5)
        self.assertEqual(cum[1], 1.0)

    def test_weighted_buckets(self):
        cum = bucket_distribution([1, 3])
        self.assertAlmostEqual(cum[0], 0.25)
        self.assertEqual(cum[1], 1.0)

    def test_three_way_split(self):
        cum = bucket_distribution([2, 2, 1])
        self.assertAlmostEqual(cum[0], 0.4)
        self.assertAlmostEqual(cum[1], 0.8)
        self.assertEqual(cum[2], 1.0)

    def test_integer_weights_normalized(self):
        cum = bucket_distribution([10, 10, 10])
        self.assertAlmostEqual(cum[0], 1.0 / 3.0)
        self.assertAlmostEqual(cum[1], 2.0 / 3.0)
        self.assertEqual(cum[2], 1.0)

    def test_float_weights(self):
        cum = bucket_distribution([0.5, 0.5])
        self.assertAlmostEqual(cum[0], 0.5)
        self.assertEqual(cum[1], 1.0)

    def test_single_bucket_always_zero(self):
        cum = bucket_distribution([7])
        self.assertEqual(cum, [1.0])
        self.assertEqual(assign("anything", cum), 0)
        self.assertEqual(assign("", cum), 0)

    def test_final_boundary_is_exactly_one(self):
        # Many weights that don't divide cleanly could leave the last boundary
        # at 0.9999... due to float drift. The implementation pins it to 1.0;
        # this test makes that guarantee explicit.
        cum = bucket_distribution([3, 3, 3, 3, 3, 3, 3])
        self.assertEqual(cum[-1], 1.0)


class TestAssign(unittest.TestCase):
    def test_reuses_precomputed_table(self):
        cum = bucket_distribution([1, 1, 1])
        result = assign("alpha", cum)
        # Same subject, same table, same answer — building the table once and
        # assigning many subjects is the supported hot path.
        self.assertEqual(assign("alpha", cum), result)

    def test_subjects_span_all_buckets(self):
        # Construct subjects known to hit each bucket of a three-way split so
        # that the assign() walk actually exercises every branch. We find them
        # by brute force in the test rather than hardcoding, which keeps the
        # test honest with respect to whatever hash the implementation uses.
        cum = bucket_distribution([1, 1, 1])
        seen = set()
        for i in range(2000):
            seen.add(assign("subject-{}".format(i), cum))
        self.assertEqual(seen, {0, 1, 2})


class TestValidation(unittest.TestCase):
    def test_empty_weights_rejected(self):
        with self.assertRaises(ValueError):
            bucket_distribution([])

    def test_zero_weight_rejected(self):
        with self.assertRaises(ValueError):
            bucket_distribution([1, 0])

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            bucket_distribution([1, -1])

    def test_nan_weight_rejected(self):
        with self.assertRaises(ValueError):
            bucket_distribution([1, float("nan")])

    def test_inf_weight_rejected(self):
        with self.assertRaises(ValueError):
            bucket_distribution([1, float("inf")])

    def test_bool_weight_rejected(self):
        # bool is a subclass of int in Python, so a naive isinstance check would
        # silently accept True/False as weights. The implementation rejects
        # them explicitly; this test pins that.
        with self.assertRaises(TypeError):
            bucket_distribution([True, 1])

    def test_non_numeric_weight_rejected(self):
        with self.assertRaises(TypeError):
            bucket_distribution(["a", 1])

    def test_non_string_subject_rejected(self):
        cum = bucket_distribution([1, 1])
        with self.assertRaises(TypeError):
            assign(123, cum)

    def test_none_subject_rejected(self):
        cum = bucket_distribution([1, 1])
        with self.assertRaises(TypeError):
            assign(None, cum)

    def test_empty_string_subject_is_valid(self):
        # The empty string is a perfectly legal subject: it encodes to zero
        # bytes, hashes to a definite value, and buckets accordingly. We allow
        # it rather than special-casing a rejection, because callers using
        # user ids that are sometimes empty will want consistent behaviour.
        cum = bucket_distribution([1, 1])
        idx = assign("", cum)
        self.assertIn(idx, (0, 1))

    def test_empty_cumulative_rejected(self):
        with self.assertRaises(TypeError):
            assign("user", [])

    def test_non_list_cumulative_rejected(self):
        with self.assertRaises(TypeError):
            assign("user", (0.5, 1.0))


class TestEncoding(unittest.TestCase):
    def test_unicode_subject_is_stable(self):
        w = [1, 1]
        first = bucket_for("用户-42", w)
        self.assertEqual(bucket_for("用户-42", w), first)

    def test_unicode_subject_valid_bucket(self):
        cum = bucket_distribution([1, 1, 1])
        idx = assign("\u00e9", cum)
        self.assertIn(idx, (0, 1, 2))


class TestIndependence(unittest.TestCase):
    def test_adding_a_bucket_does_not_reshuffle_everyone(self):
        # This is the core reason we go through a unit interval instead of
        # taking hash % n. When we grow from 2 buckets to 3, the subjects that
        # were in bucket 0 of the 2-way split should mostly still be in bucket
        # 0 of the 3-way split. We assert that at least 30% of them stay put —
        # a strict "all" assertion would be false by design (some subjects must
        # move when you add a bucket), so we pick a concrete threshold and
        # state it here rather than pretending the property is exact.
        subjects = ["user-{}".format(i) for i in range(1000)]
        cum2 = bucket_distribution([1, 1])
        cum3 = bucket_distribution([1, 1, 1])
        stayed = sum(
            1 for s in subjects
            if assign(s, cum2) == 0 and assign(s, cum3) == 0
        )
        zero_in_two = sum(1 for s in subjects if assign(s, cum2) == 0)
        self.assertGreater(zero_in_two, 0)
        self.assertGreater(stayed / zero_in_two, 0.3)


if __name__ == "__main__":
    unittest.main()
