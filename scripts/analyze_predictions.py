"""Reanalyse the public source tables or V5 workbook without fitting vision models.

Procedures are the resampling/exchange units. No clip-independent tests are used.
"""
import argparse
import sys
import hashlib
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import stats
from data.source_data_io import read_public_tables, read_workbook_tables

LABEL = {'novice': 0, 'intermediate': 1, 'expert': 2}


def table(wb, name):
    rows = list(wb[name].values)
    start = next(i for i, row in enumerate(rows) if row and row[0] == 'video_id')
    header = rows[start]
    return [dict(zip(header, row)) for row in rows[start+1:] if row and row[0]]


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    return out.tolist()


def interval(values):
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def wilson(k, n):
    z = stats.norm.ppf(.975)
    mid = (k/n + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(mid-half), float(mid+half)]


def ordinal(y, p):
    cm = np.zeros((3, 3))
    np.add.at(cm, (y, p), 1)
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.full(3, np.nan), where=cm.sum(1)>0)
    weights = (np.arange(3)[:, None]-np.arange(3)[None, :])**2
    expected = np.outer(cm.sum(1), cm.sum(0))/len(y)
    denom = (weights*expected).sum()
    return np.array([np.nanmean(recall), 1-(weights*cm).sum()/denom if denom else np.nan, np.abs(y-p).mean()])


def summary(rows, rng, draws, pred='pred_label', score='pred_grs', truth='true_label', true_score='true_grs'):
    y = np.array([LABEL[r[truth]] for r in rows])
    p = np.array([LABEL[r[pred]] for r in rows])
    n = len(rows)
    result = dict(n=n, correct=int((y==p).sum()), accuracy=float((y==p).mean()),
                  accuracy_ci=wilson(int((y==p).sum()), n),
                  adjacent_errors=int((np.abs(y-p)==1).sum()), cross_grade_errors=int((np.abs(y-p)==2).sum()))
    point = ordinal(y, p)
    ix = rng.integers(n, size=(draws, n))
    boot = np.array([ordinal(y[i], p[i]) for i in ix])
    for j, name in enumerate(['balanced_accuracy', 'quadratic_grade_kappa', 'mean_grade_distance']):
        result[name] = float(point[j]); result[name+'_ci'] = interval(boot[:, j])
    if all(r.get(score) is not None for r in rows):
        t = np.array([float(r[true_score]) for r in rows])
        q = np.array([float(r[score]) for r in rows])
        e = q-t
        result.update(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e*e).mean())), bias=float(e.mean()))
        for name, v in [('mae', np.abs(e[ix]).mean(1)), ('rmse', np.sqrt((e[ix]**2).mean(1))), ('bias', e[ix].mean(1))]:
            result[name+'_ci'] = interval(v)
        if t.std()>0 and q.std()>0:
            corr = float(np.corrcoef(t,q)[0,1]); result['pearson_r'] = corr
            z = np.arctanh(np.clip(corr,-1+1e-15,1-1e-15))
            result['pearson_r_ci'] = np.tanh(z+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(n-3)).tolist()
    return result


def icc(data):
    n,k = data.shape
    mean = data.mean(); row = data.mean(1); col = data.mean(0)
    msr = k*((row-mean)**2).sum()/(n-1)
    msc = n*((col-mean)**2).sum()/(k-1)
    mse = ((data-row[:,None]-col[None,:]+mean)**2).sum()/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)


def swaps_p(differences, denominator, rng, permutations):
    observed = abs(differences.sum()/denominator)
    extreme = 0
    for start in range(0, permutations, 2000):
        signs = rng.choice([-1, 1], size=(min(2000, permutations-start), len(differences)))
        extreme += int((np.abs(signs@differences/denominator)>=observed-1e-12).sum())
    return (extreme+1)/(permutations+1)


def visibility(annotations, predictions, rng, draws, permutations, severity='final_severity'):
    keys = [(r['video_id'],r['clip_idx']) for r in annotations]
    if len(set(keys)) != len(keys): raise ValueError('Duplicate visibility annotation')
    annotation = {key:r for key,r in zip(keys,annotations)}
    ids = sorted({r['video_id'] for r in annotations}); index = {v:i for i,v in enumerate(ids)}
    models = ['Complete','Naive concat','Unmasked fusion']
    counts = np.zeros((len(ids),3)); correct = np.zeros((3,len(ids),3))
    seen = set()
    for r in predictions:
        key = (r['video_id'],r['clip_idx']); a = annotation[key]; m = models.index(r['model']); t = int(a[severity]); i = index[r['video_id']]
        if (m,key) in seen: raise ValueError('Duplicate clip prediction')
        seen.add((m,key)); correct[m,i,t] += r['true_label']==r['pred_label']
        if m==0: counts[i,t] += 1
    if len(seen)!=len(keys)*3: raise ValueError('Missing paired clip predictions')
    ix = rng.integers(len(ids),size=(draws,len(ids)))
    denominator = counts[ix].sum(1)
    results=[]; raw=[]
    for t in range(3):
        entry=dict(severity=t,n_clips=int(counts[:,t].sum()),n_procedures=int((counts[:,t]>0).sum()),models={},comparisons={})
        for m,name in enumerate(models):
            entry['models'][name]=dict(correct=int(correct[m,:,t].sum()),accuracy=float(correct[m,:,t].sum()/counts[:,t].sum()),ci=interval(correct[m,ix,t].sum(1)/denominator[:,t]))
        for m in (1,2):
            d=correct[0,:,t]-correct[m,:,t]
            pv=swaps_p(d,counts[:,t].sum(),rng,permutations); raw.append(pv)
            entry['comparisons'][models[m]]=dict(gain=float(d.sum()/counts[:,t].sum()),ci=interval(d[ix].sum(1)/denominator[:,t]),p_raw=pv)
        results.append(entry)
    adjusted=iter(holm(raw))
    for entry in results:
        for comparison in entry['comparisons'].values(): comparison['p_holm']=next(adjusted)
    contrasts=[]
    eligible=(counts[:,0]>0)&(counts[:,2]>0)
    for m in (1,2):
        gains=(correct[0,eligible]-correct[m,eligible])/np.maximum(counts[eligible],1)
        d=gains[:,2]-gains[:,0]; boot=rng.integers(len(d),size=(draws,len(d)))
        contrasts.append(dict(comparator=models[m],n_procedures=len(d),mean_gain_contrast=float(d.mean()),ci=interval(d[boot].mean(1)),p_raw=swaps_p(d,len(d),rng,permutations)))
    for c,pv in zip(contrasts,holm([c['p_raw'] for c in contrasts])): c['p_holm']=pv
    return dict(tiers=results,severe_minus_good=contrasts)


def annotation_agreement(annotations, draws):
    ids=sorted({r['video_id'] for r in annotations}); index={v:i for i,v in enumerate(ids)}
    cm=np.zeros((len(ids),3,3)); same=np.zeros(len(ids)); counts=np.zeros(len(ids))
    for r in annotations:
        i=index[r['video_id']];a=int(r['annotator_a_severity']);b=int(r['annotator_b_severity'])
        cm[i,a,b]+=1;counts[i]+=1;same[i]+=r['annotator_a_type']==r['annotator_b_type']
    def kappa(c,power):
        weights=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:])**power
        expected=np.outer(c.sum(1),c.sum(0))/c.sum()
        return 1-(c*weights).sum()/(expected*weights).sum()
    rng=np.random.default_rng(20261021);ix=rng.integers(len(ids),size=(draws,len(ids)))
    boot_cm=cm[ix].sum(1)
    return dict(n_clips=len(annotations),quadratic_kappa=float(kappa(cm.sum(0),2)),
                quadratic_kappa_ci=interval([kappa(c,2) for c in boot_cm]),
                linear_kappa=float(kappa(cm.sum(0),1)),type_agreement=float(same.sum()/counts.sum()),
                type_agreement_ci=interval(same[ix].sum(1)/counts[ix].sum(1)),
                one_level_disagreements=int(sum(abs(r['annotator_a_severity']-r['annotator_b_severity'])==1 for r in annotations)),
                two_level_disagreements=int(sum(abs(r['annotator_a_severity']-r['annotator_b_severity'])==2 for r in annotations)))


def hospital_exact(rows):
    from math import comb
    sites=sorted({r['hospital'] for r in rows})
    totals=[sum(r['hospital']==site for r in rows) for site in sites]
    failures=[sum(r['hospital']==site and r['true_label']!=r['pred_label'] for r in rows) for site in sites]
    k=sum(failures);den=comb(sum(totals),k)
    def probability(f):
        return np.prod([comb(n,x) for n,x in zip(totals,f)])/den
    observed=probability(failures);pv=0
    for x in range(min(totals[0],k)+1):
        for y in range(min(totals[1],k-x)+1):
            z=k-x-y
            if 0<=z<=totals[2]:
                prob=probability([x,y,z])
                if prob<=observed+1e-15:pv+=prob
    return dict(sites=sites,n=totals,errors=failures,p_two_sided=float(pv))


def main(a):
    if getattr(a,'data',None):
        inputs, manifest = read_public_tables(a.data)
        source_digest = manifest['source_workbook_sha256']
    else:
        inputs = read_workbook_tables(a.xlsx)
        source_digest = hashlib.sha256(Path(a.xlsx).read_bytes()).hexdigest()
    rows=inputs['procedure_predictions']; splits=inputs['data_split']
    if len(splits)!=196 or len({r['video_id'] for r in splits})!=196: raise ValueError('Expected 196 unique procedures')
    split_by={r['video_id']:r['dataset_split'] for r in splits}
    by_model={}
    for r in rows:
        if r['split']!=split_by[r['video_id']]: raise ValueError('Split mismatch')
        by_model.setdefault(r['model'],[]).append(r)
    for model,records in by_model.items():
        if len(records)!=196 or len({r['video_id'] for r in records})!=196: raise ValueError(f'Incomplete model: {model}')
    if a.bootstrap<2 or a.permutations<1: raise ValueError('Positive resampling counts required')
    rng=np.random.default_rng(a.seed)
    report=dict(source_sha256=source_digest,
                analysis_seed=a.seed,bootstrap_samples=a.bootstrap,random_swaps=a.permutations,
                split_counts={s:sum(r['dataset_split']==s for r in splits) for s in ['train','val','test']},models={})
    for model,records in by_model.items():
        report['models'][model]={split:summary([r for r in records if split=='all' or r['split']==split],rng,a.bootstrap,score=None if model=='SAIS' else 'pred_grs') for split in ['test','all']}
        if model=='SAIS':
            report['models'][model]['regression_status']='Source score fields retained for audit; not reported as manuscript comparator endpoints.'
    main_rows={r['video_id']:r for r in by_model['ArthroSkill-MC'] if r['split']=='test'}
    comparisons=[]
    for model,records in by_model.items():
        if model=='ArthroSkill-MC': continue
        paired={r['video_id']:r for r in records if r['split']=='test'}
        differences=[]; b=c=0
        for key,r in main_rows.items():
            s=paired[key]
            if r['true_label']!=s['true_label'] or r['true_grs']!=s['true_grs']: raise ValueError('Unpaired references')
            x=int(r['true_label']==r['pred_label']); y=int(s['true_label']==s['pred_label'])
            b+=x>y; c+=y>x; differences.append(x-y)
        d=np.array(differences); ix=rng.integers(len(d),size=(a.bootstrap,len(d)))
        comparisons.append(dict(comparator=model,main_only=b,comparator_only=c,gain=float(d.mean()),ci=interval(d[ix].mean(1)),p_raw=float(stats.binomtest(b,b+c,.5).pvalue) if b+c else 1.))
    for c,pv in zip(comparisons,holm([c['p_raw'] for c in comparisons])): c['p_holm']=pv
    report['paired_test_comparisons']=comparisons
    rating=inputs['rater_scores']; data=np.array([[r['rater_1_total'],r['rater_2_total']] for r in rating],float)
    for r in rating:
        for k in [1,2]:
            if sum(r[f'rater_{k}_dimension_{d}'] for d in range(1,7))!=r[f'rater_{k}_total']: raise ValueError('Rater total mismatch')
        if (r['rater_1_total']+r['rater_2_total'])/2!=r['consensus_grs']: raise ValueError('Consensus mismatch')
    rrng=np.random.default_rng(a.seed)
    report['rater_agreement']=dict(icc_a1=float(icc(data)),ci=interval([icc(data[i]) for i in rrng.integers(len(data),size=(a.bootstrap,len(data)))]),review_count=int((np.abs(data[:,0]-data[:,1])>=6).sum()))
    loso=inputs['loso_predictions']
    report['loso']={site:summary([r for r in loso if site=='pooled' or r['hospital']==site],rng,a.bootstrap) for site in ['Centre A','Centre B','Centre C','pooled']}
    annotations=inputs['visibility_annotations']; clips=inputs['clip_predictions']
    if {r['video_id'] for r in annotations} != set(main_rows): raise ValueError('Visibility annotations must cover the fixed test procedures')
    if len(annotations)!=1527: raise ValueError('Expected 1527 annotated test clips')
    report['annotation_agreement']=annotation_agreement(annotations,a.bootstrap)
    report['hospital_exact']=hospital_exact(list(main_rows.values()))
    report['visibility']=visibility(annotations,clips,rng,a.bootstrap,a.permutations)
    report['visibility_annotator_b']=visibility(annotations,clips,rng,a.bootstrap,a.permutations,'annotator_b_severity')
    runs=[{key:value for key,value in r.items() if key!='run_role'} for r in inputs['training_runs']]
    report['training_repeats']=dict(runs=runs,summary={m:dict(mean=float(np.mean([r['test_accuracy'] for r in runs if r['model']==m])),sample_sd=float(np.std([r['test_accuracy'] for r in runs if r['model']==m],ddof=1))) for m in sorted({r['model'] for r in runs})})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(split_counts=report['split_counts'],test=report['models']['ArthroSkill-MC']['test'],rater=report['rater_agreement']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--xlsx');group.add_argument('--data',help='Public English source-data directory');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20261020);p.add_argument('--bootstrap',type=int,default=20000);p.add_argument('--permutations',type=int,default=99999)
    main(p.parse_args())
