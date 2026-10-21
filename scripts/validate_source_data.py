"""Verify public source-data integrity, joins and optional workbook equivalence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import re
from collections import Counter
from data.source_data_io import read_public_tables, read_workbook_tables, sha256


def validate(root, xlsx=None):
    root=Path(root).resolve();tables,manifest=read_public_tables(root)
    dictionary=json.loads((root/'data_dictionary.json').read_text())
    if manifest['covered_source_worksheets']!=list(range(1,23)):
        raise ValueError('Incomplete V5 worksheet coverage')
    for file in manifest['files']:
        path=root/file['path']
        if not path.resolve().is_relative_to(root) or sha256(path)!=file['sha256']:
            raise ValueError(f'File integrity mismatch: {file["path"]}')
        if not path.read_text(encoding='utf-8').isascii():
            raise ValueError(f'Non-English export text requires review: {file["path"]}')
        if re.search(r'\bSUR\d+\b|/Users/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',path.read_text(encoding='utf-8')):
            raise ValueError(f'Identity linkage or local/contact metadata detected: {file["path"]}')
        if file['path'].endswith('.csv'):
            with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            spec=dictionary['tables'][file['path']]
            if len(rows)!=file['rows'] or len(rows)!=spec['rows']:raise ValueError('CSV row count mismatch')
            key=spec['primary_key']
            if key and len({tuple(r[k] for k in key) for r in rows})!=len(rows):raise ValueError('Duplicate primary key')
    split=tables['data_split'];reference={r['video_id']:r for r in split}
    if len(reference)!=196 or len(split)!=196:raise ValueError('Expected 196 unique procedures')
    if Counter(r['dataset_split'] for r in split)!=Counter(train=117,val=39,test=40):raise ValueError('Unexpected procedure partition')
    if any('surgeon_id' in r for r in split):raise ValueError('Surgeon linkage field must be withheld')
    ratings=tables['rater_scores']
    if len(ratings)!=196 or {r['video_id'] for r in ratings}!=set(reference):raise ValueError('Incomplete independent ratings')
    review_count=0
    for row in ratings:
        for rater in [1,2]:
            values=[row[f'rater_{rater}_dimension_{d}'] for d in range(1,7)]
            if not all(1<=v<=5 and int(v)==v for v in values) or sum(values)!=row[f'rater_{rater}_total']:raise ValueError('Invalid independent dimension/total scores')
        one,two=row['rater_1_total'],row['rater_2_total']
        if one-two!=row['rater_difference'] or (one+two)/2!=row['consensus_grs']:raise ValueError('Consensus or rater difference mismatch')
        if row['consensus_grs']!=reference[row['video_id']]['grs_score']:raise ValueError('Consensus differs across source tables')
        review_count+=abs(one-two)>=6
    if review_count!=32:raise ValueError('Expected 32 score-gap review procedures')
    predictions=tables['procedure_predictions'];models=Counter(r['model'] for r in predictions)
    if len(models)!=8 or len(predictions)!=1568 or any(n!=196 for n in models.values()):raise ValueError('Incomplete eight-model prediction table')
    for row in predictions:
        ref=reference[row['video_id']]
        if (row['split'],row['hospital'],row['surgery_type'],row['true_label'],row['true_grs'])!=(ref['dataset_split'],ref['hospital'],ref['surgery_type'],ref['skill_level'],ref['grs_score']):raise ValueError('Prediction reference metadata mismatch')
        if row['correct']!=(row['true_label']==row['pred_label']):raise ValueError('Incorrect classification flag')
        if abs(abs(row['residual'])-row['abs_error'])>1e-10 or abs(row['pred_grs']-row['true_grs']-row['residual'])>0.000501:raise ValueError('Prediction/error precision exceeds recorded rounding tolerance')
    annotation=tables['visibility_annotations'];clips=tables['clip_predictions']
    test_ids={r['video_id'] for r in split if r['dataset_split']=='test'}
    keys={(r['video_id'],r['clip_idx']):r for r in annotation}
    if len(annotation)!=1527 or len(keys)!=1527 or {r['video_id'] for r in annotation}!=test_ids:raise ValueError('Incomplete test visibility annotations')
    if Counter(r['final_severity'] for r in annotation)!=Counter({0:910,1:411,2:206}):raise ValueError('Unexpected final severity counts')
    if any(r['final_severity']!=r['annotator_a_severity'] for r in annotation):raise ValueError('Unexpected final severity assignment')
    if len(clips)!=4581:raise ValueError('Expected three predictions for each annotated clip')
    model_keys={}
    for row in clips:
        key=row['video_id'],row['clip_idx'];ann=keys[key];ref=reference[row['video_id']]
        model_keys.setdefault(row['model'],set()).add(key)
        if row['severity']!=ann['final_severity'] or row['true_label']!=ref['skill_level'] or row['correct']!=(row['true_label']==row['pred_label']):raise ValueError('Clip annotation/prediction join mismatch')
    if set(model_keys)!=set(['Complete','Naive concat','Unmasked fusion']) or any(k!=set(keys) for k in model_keys.values()):raise ValueError('Incomplete paired clip-model coverage')
    loso=tables['loso_predictions']
    if len(loso)!=196 or {r['video_id'] for r in loso}!=set(reference):raise ValueError('Incomplete LOSO procedure coverage')
    if sum(r['correct'] for r in loso)!=126:raise ValueError('Unexpected LOSO correct count')
    if len(tables['training_runs'])!=6:raise ValueError('Expected six aggregate training-run records')
    with (root/'exclusions.csv').open(newline='',encoding='utf-8') as f:excluded={r['video_id'] for r in csv.DictReader(f)}
    if excluded!=set(['P009','P019','P047','P138']) or excluded&set(reference):raise ValueError('Incorrect exclusion inventory')
    if xlsx:
        if sha256(xlsx)!=manifest['source_workbook_sha256']:raise ValueError('Source workbook checksum differs from the release')
        if read_workbook_tables(xlsx)!=tables:raise ValueError('Public primary values differ from the original workbook after documented de-identification')
    return dict(csv_tables=len(dictionary['tables']),total_exported_records=sum(t['rows'] for t in dictionary['tables'].values()),
                procedures=196,procedure_predictions=1568,visibility_annotations=1527,clip_predictions=4581,
                independent_rater_records=196,loso_procedures=196,training_runs=6,exclusions=4,
                workbook_equivalence='verified' if xlsx else 'not requested',source_workbook_sha256=manifest['source_workbook_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/source_data');p.add_argument('--xlsx');a=p.parse_args()
    print(json.dumps(validate(a.data,a.xlsx),indent=2))
