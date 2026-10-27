"""Validate the release without a private workbook or surgical videos."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from data.source_data_io import read_public_tables
from scripts.validate_source_data import validate
from scripts.analyze_predictions import main as analyze

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/source_data'


class PublicSourceDataTests(unittest.TestCase):
    def test_complete_release_and_referential_integrity(self):
        result = validate(DATA)
        self.assertEqual(result['csv_tables'], 24)
        self.assertEqual(result['total_exported_records'], 8563)
        self.assertEqual(result['procedures'], 196)
        self.assertEqual(result['clip_predictions'], 4581)
        self.assertEqual(result['source_workbook_sha256'], 'ae9516771048394df8acf2ee1dcba19cf0eaa335fbfc875e12a4c48da18a7c19')

    def test_linkage_removed_without_dropping_procedures(self):
        tables, _ = read_public_tables(DATA)
        self.assertEqual(len({r['video_id'] for r in tables['data_split']}), 196)
        self.assertNotIn('surgeon_id', tables['data_split'][0])
        self.assertEqual(len(tables['rater_scores']), 196)
        self.assertTrue(all(r['hospital'] in ['Centre A', 'Centre B', 'Centre C'] for r in tables['data_split']))

    def test_original_flags_and_score_fields_are_preserved(self):
        tables, _ = read_public_tables(DATA)
        sais = [r for r in tables['procedure_predictions'] if r['model'] == 'SAIS']
        self.assertEqual(len(sais), 196)
        self.assertTrue(all(r['pred_grs'] is not None for r in sais))
        with (DATA/'recorded_summaries/statistical_tests.csv').open(newline='') as f:
            original = list(csv.DictReader(f))
        self.assertEqual(sum(r['source_significant_unadjusted'] == 'true' for r in original), 5)
        self.assertAlmostEqual(float(original[0]['p_value_unadjusted']), 0.021270751953125)

    def test_tampered_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'data'
            shutil.copytree(DATA, target)
            with (target/'procedure_predictions.csv').open('a') as f:
                f.write('corrupt\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_public_tables(target)

    def test_analysis_runs_from_public_tables_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'analysis.json'
            analyze(SimpleNamespace(data=str(DATA), xlsx=None, output=str(output),
                                    seed=20261020, bootstrap=20, permutations=99))
            report = json.loads(output.read_text())
            self.assertEqual(report['split_counts'], dict(train=117, val=39, test=40))
            self.assertEqual(report['models']['ArthroSkill-MC']['test']['correct'], 35)
            self.assertNotIn('mae', report['models']['SAIS']['test'])
            self.assertAlmostEqual(report['rater_agreement']['icc_a1'], 0.8228854072112503)
            self.assertTrue(all(c['p_holm'] > .05 for c in report['paired_test_comparisons']))


if __name__ == '__main__':
    unittest.main()
