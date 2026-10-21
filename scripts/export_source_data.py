"""Export an English, values-only public package from the V5 source workbook.

The source workbook is read without modification. Study values are preserved;
only an unnecessary surgeon-linkage field and workbook internals are withheld.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import math
import shutil
from openpyxl import load_workbook
from data.source_data_io import read_workbook_tables, sha256, TABLES

ROOT = Path(__file__).resolve().parents[1]

COVER = {
1:'ArthroSkill-MC source-data package',
2:'Occlusion-aware multimodal surgical skill assessment in arthroscopy; n = 196 videos; 3 anonymised centres',
4:'The source package contains two components:',
5:'All results for the updated stratified split: train 117, validation 39, test 40; split seed 20241012; including visibility strata and training-seed repeats.',
6:'Annotation and LOSO component: original ratings, ICC, exclusions and three LOSO folds; the source workbook states that these protocols and data were retained.',
7:'The source workbook states that earlier split predictions, statistical analyses, SAIS calibration and seed experiments were superseded and are not included.',
9:'Key metrics for the updated split',10:'Test accuracy',11:'Full-cohort accuracy',
12:'Test Pearson correlation for GRS',13:'Test GRS MAE / RMSE',14:'SAIS test accuracy',15:'SAIS full-cohort accuracy',
16:'Pooled three-fold LOSO accuracy',17:'Pooled LOSO GRS MAE',18:'Visibility strata: good / mild / severe',19:'Inter-rater ICC(A,1)',
21:'Public table index',22:'Experiment report, procedure predictions, data split, model metrics and statistical tests',
23:'Test and validation procedure detail snapshots',24:'Visibility annotations, recorded tier summaries, agreement, degradation counts and clip predictions',
25:'Independent rater scores, recorded ICC calculation and scoring protocol',26:'LOSO procedure predictions and recorded summaries',
27:'Excluded recording log: 200 acquired, 196 included',28:'Recorded cohort and performance summaries',
30:'The source workbook anonymises hospitals as Centre A, Centre B and Centre C.'}

COVER_VALUE = {18:'89.8% / 88.3% / 71.8% (1,527 annotated test clips)',
22:'procedure_predictions.csv; data_split.csv; recorded_summaries/experiment_report.csv; recorded_summaries/model_metrics.csv; recorded_summaries/statistical_tests.csv',
23:'recorded_summaries/test_details.csv; recorded_summaries/validation_details.csv',
24:'visibility_annotations.csv; clip_predictions.csv; recorded_summaries/visibility_tier_metrics.csv; recorded_summaries/annotation_agreement.csv; recorded_summaries/degradation_counts.csv; recorded_summaries/clip_summary.csv',
25:'rater_scores.csv; recorded_summaries/icc_calculation.csv; recorded_summaries/scoring_protocol.csv',
26:'loso_predictions.csv; recorded_summaries/loso_summary.csv; recorded_summaries/loso_error_summary.csv',
27:'exclusions.csv',28:'recorded_summaries/cohort_summary.csv'}

SCORING = [
('rating_design','Two arthroscopic surgeons at associate-chief level or above independently viewed each complete recording and rated six GRS dimensions from 1 to 5, blinded to each other\'s ratings.'),
('rater_background','Rater 1 had 15 years and rater 2 had 11 years of arthroscopy experience. The source note states that neither participated in model development or results analysis.'),
('consensus_rule','Average the independent totals, retaining 0.5-point granularity. Differences of at least 6 total-score points trigger joint review and a decision to retain or revise the ratings.'),
('reliability','Two-way random-effects, single-rater absolute-agreement ICC(A,1) = 0.823; the source note refers to the manuscript for the confidence interval and labels 0.75-0.90 as good agreement.'),
('procedures_requiring_review','32 / 196 (16.3%)')]

ICC = [
('n_procedures',196,'Number of procedures'),('n_raters',2,'Number of independent raters'),
('grand_mean',19.3112,'Grand mean of all independent pre-consensus total ratings'),
('ss_rows',16610.0306,'k * sum((row_mean - grand_mean)^2)'),
('ss_columns',6.898,'n * sum((column_mean - grand_mean)^2)'),
('ss_error',1615.102,'SST - SSR - SSC'),
('degrees_of_freedom','195 / 1 / 195','n-1 / k-1 / (n-1)*(k-1)'),
('ms_rows',85.1796,'SSR / df_rows'),('ms_columns',6.898,'SSC / df_columns'),('ms_error',8.2826,'SSE / df_error'),
('icc_a1',0.8229,'(MSR-MSE)/(MSR+(k-1)*MSE+(k/n)*(MSC-MSE))')]

EXCLUSIONS = [
('P009','batch01','Encoding corruption prevented frame decoding with ffmpeg.','Two retries failed; no inference prediction was produced.'),
('P019','batch01','Duration was 6.8 seconds, below the 10-second temporal-input eligibility threshold.','Valid temporal features could not be extracted; no prediction was produced.'),
('P047','batch01','File corruption with missing footage from the first half of the procedure.','Loading failed; no prediction was produced.'),
('P138','batch03','Frame rate did not meet the input specification.','Resampling did not resolve the mismatch; inference was abandoned.')]

DESCRIPTIONS = {
'video_id':'Study procedure pseudonym. This is not a medical-record number or a released identity mapping.',
'hospital':'Anonymised hospital category: Centre A, Centre B or Centre C. No site-identity key is distributed.',
'surgery_type':'Joint category: knee or shoulder.', 'split':'Fixed procedure partition: train, val or test.',
'dataset_split':'Fixed procedure partition: train, val or test.', 'skill_level':'Consensus grade: novice, intermediate or expert.',
'grs_score':'Consensus total GRS, in native 6-30 points; mean of two independent totals.',
'true_label':'Consensus grade: novice, intermediate or expert.', 'true_grs':'Consensus total GRS, in native 6-30 points.',
'model':'Recorded model or ablation name. Complete in clip tables denotes ArthroSkill-MC.',
'pred_label':'Recorded classifier grade; independent of the regression total.',
'pred_grs':'Total-score prediction as recorded, typically at 0.001-point precision. SAIS values are retained for source audit only; SAIS regression is not a manuscript-supported comparator endpoint.',
'correct':'Whether the recorded predicted grade equals the reference grade.',
'abs_error':'Absolute GRS error as recorded. It may retain more precision than pred_grs; it is not overwritten from rounded predictions.',
'residual':'Predicted minus reference GRS as recorded, potentially at greater precision than pred_grs.',
'clip_idx':'Zero-based annotated clip index within a procedure; not an original video frame number.',
'degradation_type':'Recorded final degradation category: none, blood, blur, fog, out_of_view or overexposure.',
'severity':'Recorded severity tier: 0 good, 1 mild degradation, 2 severe degradation.',
'final_severity':'Released severity label used for primary visibility analyses; all released labels coincide with annotator A.',
'degraded_ratio':'Recorded fraction labelled as degraded within the annotated clip; this is not a source-video prevalence estimate.',
'annotator_a_severity':'Independent severity from annotator A, coded 0/1/2.',
'annotator_b_severity':'Independent severity from annotator B, coded 0/1/2.',
'annotator_a_type':'Independent degradation type from annotator A.', 'annotator_b_type':'Independent degradation type from annotator B.',
'n_clips_total':'Number of quality-screened inference clips recorded for this procedure.',
'n_clips_annotated':'Number of clips selected for visibility annotation in this procedure.',
'sampling_ratio':'Recorded annotated-clip count divided by the quality-screened inference-clip count.',
'rater_1_total':'Sum of the six independent dimension scores from rater 1, in 6-30 points.',
'rater_2_total':'Sum of the six independent dimension scores from rater 2, in 6-30 points.',
'rater_difference':'Signed rater 1 total minus rater 2 total. Review eligibility uses the absolute difference of at least 6.',
'consensus_grs':'Mean of the two independent total ratings; native 6-30 points at 0.5-point granularity.',
'grade_distance':'Absolute ordinal difference between reference and predicted grade, with novice/intermediate/expert coded 0/1/2.',
'seed':'Recorded training seed; 3407 is the designated main run, with repeats 3433 and 2025.',
'validation_accuracy':'Recorded best validation procedure accuracy, as a fraction.',
'test_accuracy':'Recorded test procedure accuracy, as a fraction.',
'mae':'Recorded aggregate total-GRS mean absolute error in points; blank when not reported.',
'rmse':'Recorded aggregate total-GRS root mean squared error in points; blank when not reported.',
'pearson_r':'Recorded Pearson correlation of predicted and reference GRS; blank when not reported.',
'run_role':'main for designated seed 3407; repeat for the two additional fitting runs.',
'acquisition_batch':'Non-identifying acquisition-batch label retained from the exclusion log.',
'reason':'English translation of the recorded exclusion reason.', 'retry_outcome':'English translation of the recorded retry or inference outcome.',
'source_row':'One-based row number in the original worksheet.',
'field':'English field or statistic label in a recorded summary.', 'value':'Recorded value or English translation, as text when the source has mixed types.',
'section':'Recorded report section.', 'explanation':'English translation of the recorded definition or formula description.',
'threshold':'Threshold text recorded in the source summary; not a preregistered decision rule.',
'status':'Pass or missing status as recorded in the source summary.',
'percentage':'Recorded percentage on the 0-100 scale, rather than a 0-1 fraction.',
'additional_value':'Additional recorded value identified by the field label and source row.',
'accuracy':'Recorded classification accuracy as a fraction.', 'n':'Recorded number of procedures.',
'p_value_unadjusted':'Recorded exact McNemar p value without seven-comparison Holm adjustment.',
'source_significant_unadjusted':'Original Boolean significance flag using the unadjusted p value; not the manuscript family-wise conclusion.',
'source_significance_label':'Original unadjusted significance label; not an adjusted inference.',
'both_correct':'Paired procedures correctly classified by both complete model and comparator.',
'main_only':'Paired procedures correct only for the complete model.',
'base_only':'Paired procedures correct only for the comparator.',
'both_wrong':'Paired procedures incorrect for both models.',
'fold':'LOSO held-out-hospital fold or pooled result.', 'n_correct':'Recorded count of correct classifications.',
'accuracy_recorded':'Original accuracy text or proportion, without recalculation.',
'p_vs_naive_as_recorded':'Original visibility-summary p value; its calculation is not specified in that sheet and is not substituted for the procedure-cluster analysis.',
'p_vs_unmasked_as_recorded':'Original visibility-summary p value; not substituted for the procedure-cluster analysis.',
'notes':'English explanatory text or source note.'}
for r in [1,2]:
    for d in range(1,7): DESCRIPTIONS[f'rater_{r}_dimension_{d}']=f'Independent rater {r} score on GRS dimension {d}, an integer from 1 to 5. Dimension definitions are in README.md.'


def english(value):
    if isinstance(value,str):
        value=value.replace('\u00d7',' x ').replace('\u2248','approximately ').replace('\u2265','>=').replace('\u00b1','+/-').replace('\u00b7',';').replace('\u2212','-').replace('\u2013','-').replace('\u2014','-')
        if not value.isascii(): raise ValueError(f'Untranslated source text encountered: {value!r}')
    return value


def kind(values):
    values=[v for v in values if v is not None and v!='']
    if values and all(isinstance(v,bool) for v in values): return 'boolean'
    if values and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values):
        return 'integer' if all(isinstance(v,int) for v in values) else 'number'
    return 'string'


def dump_table(root, path, header, rows, registry, sheet, role, note='', key=None):
    rows=[[english(v) for v in row] for row in rows]
    for row in rows:
        if len(row)!=len(header): raise ValueError(f'Nonrectangular export: {path}')
        if any(isinstance(v,float) and not math.isfinite(v) for v in row): raise ValueError('Non-finite source value')
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(header)
        for row in rows:writer.writerow(['true' if v is True else 'false' if v is False else '' if v is None else v for v in row])
    columns=[]
    for i,name in enumerate(header):
        description=DESCRIPTIONS.get(name)
        if description is None:
            if name.startswith('accuracy_'): description='Recorded accuracy for '+name[len('accuracy_'):].replace('_',' ')+'.'
            elif name.startswith('n_clips_'): description='Recorded clip count for '+name[len('n_clips_'):].replace('_',' ')+'.'
            elif name.startswith('severity_'): description='Recorded clip count in visibility severity '+name[len('severity_'):]+'.'
            else: raise ValueError(f'Missing column description: {path}:{name}')
        columns.append(dict(name=name,type=kind([r[i] for r in rows]),description=description))
    registry[path]=dict(role=role,source_worksheet_index=sheet+1,rows=len(rows),columns=columns,
                        primary_key=key or [],notes=note)


def export(xlsx, output):
    if sha256(xlsx)!='67147246593aab0e6861f243a0f0ae37fc78aa831495ed210ef0c0211a688659':
        raise ValueError('Source workbook differs from the reviewed V5 release; review new data before publishing')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for path in output.rglob('*.csv'):
        raise FileExistsError(f'Refusing to mix data releases in {output}; use an empty output directory')
    tables=read_workbook_tables(xlsx);wb=load_workbook(xlsx,read_only=True,data_only=True);registry={}
    for name,rows in tables.items():
        header=list(rows[0]); index=TABLES[name][0] if name in TABLES else 8
        note='Recorded study outputs; not predictions from newly trained repository models.'
        if name=='procedure_predictions':note+=' SAIS regression fields are retained as recorded for audit, although the manuscript comparator is classification only. Error fields may have greater precision than pred_grs.'
        keys={'procedure_predictions':['video_id','model'],'data_split':['video_id'],
              'rater_scores':['video_id'],'loso_predictions':['video_id'],
              'visibility_annotations':['video_id','clip_idx'],'clip_predictions':['video_id','clip_idx','model'],
              'training_runs':['model','seed']}
        dump_table(output,name+'.csv',header,[[r[k] for k in header] for r in rows],registry,index,'primary',note,keys[name])
    dump_table(output,'exclusions.csv',['video_id','acquisition_batch','reason','retry_outcome'],EXCLUSIONS,registry,17,'primary','English translation of four excluded-recording entries.',['video_id'])
    def archive(path,index,start=1,header=None,notes='Recorded summary snapshot; underlying primary tables are preferred for inference.'):
        rows=list(wb.worksheets[index].values)
        return dump_table(output,'recorded_summaries/'+path,header or list(rows[start-1]),[list(r) for r in rows[start:] if r and r[0] not in [None,'']],registry,index,'recorded_summary',notes)
    archive('model_metrics.csv',4,notes='Original rounded aggregate metrics, including SAIS regression fields not reported as comparator endpoints in the manuscript.')
    archive('statistical_tests.csv',5,header=['model','both_correct','main_only','base_only','both_wrong','p_value_unadjusted','source_significant_unadjusted','source_significance_label'],notes='Original significance flags use unadjusted p values. All seven comparisons exceed 0.05 after Holm adjustment.')
    archive('test_details.csv',6);archive('validation_details.csv',7)
    archive('visibility_tier_metrics.csv',10,header=['severity','accuracy_Complete','accuracy_Naive_concat','accuracy_Unmasked_fusion','n_clips_Complete','n_clips_Naive_concat','n_clips_Unmasked_fusion','p_vs_naive_as_recorded','p_vs_unmasked_as_recorded'],notes='Original p values are retained without asserting their test method. Use procedure-cluster reanalysis for manuscript inference.')
    archive('degradation_counts.csv',12,header=['degradation_type','severity_0','severity_1','severity_2','n_clips_total','percentage'])
    archive('clip_summary.csv',13,header=['severity','model','n_correct','n_clips_total','accuracy'])
    rows=list(wb.worksheets[0].values)
    cover=[[i,COVER[i],COVER_VALUE[i] if i in COVER_VALUE else english(r[1])] for i,r in enumerate(rows,1) if i in COVER]
    dump_table(output,'recorded_summaries/package_overview.csv',['source_row','field','value'],cover,registry,0,'recorded_summary','Translated overview and public file crosswalk. These are source-workbook notes, not repository or publication dates.')
    rows=list(wb.worksheets[1].values);section='';report=[]
    for i,r in enumerate(rows[1:],2):
        if not r[0]: continue
        if r[1] in ['',None] and str(r[0]).isupper():section=r[0];continue
        report.append([i,section,r[0],r[1]])
    dump_table(output,'recorded_summaries/experiment_report.csv',['source_row','section','field','value'],report,registry,1,'recorded_summary','Unverified source report text. Its unadjusted significance claims, trend claim, approximately 48 million raw frames and approximately 99.5% filtering claim are not adopted by the current analysis. Administrative date/version/status are source fields only.')
    rows=list(wb.worksheets[8].values)
    seed_summary=[[r[0],r[3],None if r[4]=='-' else r[4],None if r[5]=='-' else r[5],None if r[6]=='-' else r[6],'Classification only' if r[0]=='SAIS' else ''] for r in rows[9:11]]
    dump_table(output,'recorded_summaries/training_seed_summary.csv',['model','test_accuracy','mae','rmse','pearson_r','notes'],seed_summary,registry,8,'recorded_summary','Original aggregate mean +/- sample-SD text; individual run rows are in training_runs.csv.')
    rows=list(wb.worksheets[11].values)
    dump_table(output,'recorded_summaries/annotation_agreement.csv',['field','value','threshold','status'],[list(r) for r in rows[1:]],registry,11,'recorded_summary','Rounded source annotations summary. Current bootstrap intervals are computed from visibility_annotations.csv.')
    # Keep source numeric values, even when a rounded or old summary conflicts.
    rows=list(wb.worksheets[16].values)
    icc_rows=[[field,rows[i+2][1],explanation] for i,(field,_,explanation) in enumerate(ICC)]
    icc_rows.append(['source_note',None,'The source note reports 0.823 and cites Shrout and Fleiss (1979) and Koo and Li (2016).'])
    dump_table(output,'recorded_summaries/icc_calculation.csv',['field','value','explanation'],icc_rows,registry,16,'recorded_summary','Values retained as recorded; raw-score reanalysis gives ICC(A,1) approximately 0.822885.')
    dump_table(output,'recorded_summaries/scoring_protocol.csv',['field','value'],SCORING,registry,18,'recorded_summary','Translated protocol notes. The manuscript reports that review retained both independent ratings unchanged.')
    rows=list(wb.worksheets[20].values)
    fold_rows=[[r[0],r[1] if r[0]!='Pooled' else 'All held-out hospitals',r[2],r[3],r[4],r[5],r[6],''] for r in rows[2:6]]
    fold_rows.append(['Source note','',None,None,'',None,None,'The source note attributes Pearson correlation and MAE to Python calculations and accuracy/error counts to formulas; these statements are retained as recorded, not independently asserted.'])
    dump_table(output,'recorded_summaries/loso_summary.csv',['fold','hospital','n','n_correct','accuracy_recorded','pearson_r','mae','notes'],fold_rows,registry,20,'recorded_summary','The Centre B source MAE is recorded as 4.59, while the rounded per-procedure predictions yield about 4.60; preserve both with their provenance.')
    errors=[['adjacent_grade_errors',rows[8][6]],['two_grade_errors',rows[9][6]],['total_errors',rows[10][6]],['two_grade_fraction',rows[11][6]]]
    dump_table(output,'recorded_summaries/loso_error_summary.csv',['field','value'],errors,registry,20,'recorded_summary')
    rows=list(wb.worksheets[21].values)
    labels={2:'Hospital composition',6:'Joint composition',9:'Skill-grade composition',13:'Stratified random split; seed 20241012',17:'Performance on the updated split',18:'ArthroSkill-MC test accuracy',19:'ArthroSkill-MC full-cohort accuracy',20:'SAIS test accuracy',21:'SAIS full-cohort accuracy',22:'Main-model test GRS Pearson correlation',23:'Main-model test GRS MAE / RMSE',24:'Inter-rater ICC(A,1)'}
    cohort=[[i,labels.get(i,r[0]),r[1],r[2],r[3]] for i,r in enumerate(rows,1) if i>=2 and r[0]]
    for r in cohort:
        if r[0]==17:r[2:]=['Correct','Total','Accuracy']
    dump_table(output,'recorded_summaries/cohort_summary.csv',['source_row','field','value','additional_value','notes'],cohort,registry,21,'recorded_summary','The source overview records ICC 0.822; other source sheets and raw-score reanalysis support reporting 0.823. This snapshot is not silently corrected.')
    dictionary=dict(format='arthroskill_mc_public_source_data_v1',missing_values='Empty CSV cells represent missing/not-reported values; never imputed as zero.',tables=registry)
    (output/'data_dictionary.json').write_text(json.dumps(dictionary,indent=2,ensure_ascii=True)+'\n')
    files=[]
    for path,spec in registry.items():files.append(dict(path=path,rows=spec['rows'],sha256=sha256(output/path),role=spec['role']))
    files.append(dict(path='data_dictionary.json',sha256=sha256(output/'data_dictionary.json'),role='metadata'))
    for name in ['README.md','TERMS.md']:
        source=ROOT/'data/source_data'/name
        target=output/name
        if source.is_file() and source.resolve()!=target.resolve():shutil.copyfile(source,target)
        if target.is_file():files.append(dict(path=name,sha256=sha256(target),role='documentation'))
    manifest=dict(title='ArthroSkill-MC public source data',release_version='v5-public-1',
                  description='De-identified ratings, partitions and recorded procedure/clip predictions for statistical reanalysis.',
                  maintainer='Ning Xi',related_code='https://github.com/Irene-NingXi/ArthroSkill-MC',doi=None,
                  keywords=['arthroscopy','surgical skill assessment','GRS','visibility','multi-centre'],
                  source_workbook_sha256=sha256(xlsx),
                  source_workbook_version_label='V5 filename; the internal experiment report records v6.0 (Visibility-stratified).',
                  format='arthroskill_mc_public_source_data_v1',encoding='UTF-8',csv_line_ending='LF',
                  release_date='2026-10-21',data_reuse_terms='See TERMS.md; no new data-specific open licence assigned',files=files,
                  source_worksheet_count=22,covered_source_worksheets=sorted({s['source_worksheet_index'] for s in registry.values()}),
                  withheld=['surgeon_id linkage column','Original workbook container, properties, formatting and any non-tabular internals','Raw recordings and any identity or hospital-name linkage key'],
                  transformations=['English table and field names','English translation of narrative notes','CSV Boolean encoding true/false','Blank cells for not-reported training-run metrics originally marked with a dash','Removal of surgeon linkage IDs; procedure pseudonyms retained for joins','Recorded summaries kept separate from primary observations; no numerical source value corrected'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=True)+'\n');wb.close()
    print(json.dumps(dict(tables=len(registry),rows=sum(s['rows'] for s in registry.values()),covered_worksheets=manifest['covered_source_worksheets']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--xlsx',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(a.xlsx,a.output)
