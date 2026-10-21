"""Read the public English source tables or the original V5 workbook."""
import csv
import hashlib
import json
from pathlib import Path

# Worksheet ordinals and data row offsets are fixed by the V5 source schema.
# The private workbook's localised worksheet labels are not needed for export.
TABLES = {
    'procedure_predictions': (2, 1, ['video_id','hospital','split','surgery_type','true_label','true_grs','model','pred_label','pred_grs','correct','abs_error','residual']),
    'data_split': (3, 1, ['video_id','hospital','surgery_type','surgeon_id','skill_level','grs_score','dataset_split']),
    'visibility_annotations': (9, 1, ['video_id','clip_idx','degradation_type','severity','degraded_ratio','annotator_a_severity','annotator_a_type','annotator_b_severity','annotator_b_type','final_severity','n_clips_total','n_clips_annotated','sampling_ratio']),
    'clip_predictions': (14, 1, ['video_id','clip_idx','model','true_label','pred_label','correct','severity','degradation_type','n_clips_total','n_clips_annotated','sampling_ratio']),
    'rater_scores': (15, 2, ['video_id']+[f'rater_1_dimension_{d}' for d in range(1,7)]+['rater_1_total']+[f'rater_2_dimension_{d}' for d in range(1,7)]+['rater_2_total','rater_difference','consensus_grs']),
    'loso_predictions': (19, 2, ['video_id','hospital','surgery_type','true_label','true_grs','pred_label','pred_grs','grade_distance','correct']),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_workbook_tables(path):
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    if len(wb.worksheets) != 22:
        raise ValueError('Expected the 22-worksheet V5 source-data schema')
    result = {}
    for name, (index, start, header) in TABLES.items():
        rows = list(wb.worksheets[index].values)
        if rows[start-1][0] != 'video_id' or len(rows[start-1]) != len(header):
            raise ValueError(f'Unexpected V5 table schema: {name}')
        result[name] = [dict(zip(header, row)) for row in rows[start:] if row and row[0]]
    for row in result['data_split']:
        row.pop('surgeon_id')
    runs=[]; current=None
    for row in list(wb.worksheets[8].values)[1:]:
        if row[0]: current=row[0]
        if row[1] in [3407,3433,2025]:
            runs.append(dict(model=current,seed=int(row[1]),validation_accuracy=row[2],
                             test_accuracy=row[3],mae=None if row[4]=='-' else row[4],
                             rmse=None if row[5]=='-' else row[5],pearson_r=None if row[6]=='-' else row[6],
                             run_role='main' if row[1]==3407 else 'repeat'))
    result['training_runs']=runs
    wb.close()
    return result


def read_public_tables(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    schema_entry=next(r for r in manifest['files'] if r['path']=='data_dictionary.json')
    if sha256(root/'data_dictionary.json')!=schema_entry['sha256']:
        raise ValueError('Public data dictionary checksum mismatch')
    schema=json.loads((root/'data_dictionary.json').read_text())
    result={}
    for name in [*TABLES, 'training_runs']:
        filename=name+'.csv'; path=root/filename
        entry=next(r for r in manifest['files'] if r['path']==filename)
        if sha256(path)!=entry['sha256']:
            raise ValueError(f'Public table checksum mismatch: {filename}')
        definitions=schema['tables'][filename]['columns']
        with path.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames != [d['name'] for d in definitions]:
                raise ValueError(f'Public table schema mismatch: {filename}')
            rows=[]
            for source in reader:
                row={}
                for column in definitions:
                    key=column['name']; value=source[key]; kind=column['type']
                    if value=='': value=None
                    elif kind=='integer': value=int(value)
                    elif kind=='number': value=float(value)
                    elif kind=='boolean':
                        if value not in ['true','false']: raise ValueError(f'Invalid Boolean in {filename}')
                        value=value=='true'
                    row[key]=value
                rows.append(row)
        if len(rows)!=entry['rows']: raise ValueError(f'Public row count mismatch: {filename}')
        result[name]=rows
    return result, manifest
