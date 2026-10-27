"""Prevent numerical mismatches from passing the public-data Quickstart."""
import unittest
from scripts.quickstart import compare_results


class ReproducibilityTests(unittest.TestCase):
    def test_roundoff_is_accepted_but_result_changes_are_rejected(self):
        compare_results({'value': [0.5 + 1e-13, 3]}, {'value': [0.5, 3]})
        for candidate in [{'value': [0.6, 3]}, {'value': [0.5]}, {'other': [0.5, 3]}]:
            with self.assertRaises(ValueError):
                compare_results(candidate, {'value': [0.5, 3]})

    def test_nonfinite_results_cannot_pass(self):
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                compare_results(value, 0.5)


if __name__ == '__main__':
    unittest.main()
