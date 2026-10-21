"""Export the authoritative split and averaged dimension scores to English CSV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
from data.source_data_io import read_public_tables, read_workbook_tables


def main(a):
    tables = read_public_tables(a.data)[0] if getattr(a,'data',None) else read_workbook_tables(a.xlsx)
    records=tables['data_split']
    ratings={r['video_id']:r for r in tables['rater_scores']}
    if len(records)!=196 or len(ratings)!=196: raise ValueError('Expected 196 included procedures')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['video_id','video_path','split','centre','joint','label_cls','label_reg']);writer.writeheader()
        for r in records:
            row=ratings[r['video_id']]
            dims=[(float(row[f'rater_1_dimension_{i}'])+float(row[f'rater_2_dimension_{i}']))/2 for i in range(1,7)]
            total=(float(row['rater_1_total'])+float(row['rater_2_total']))/2
            if total!=r['grs_score'] or abs(sum(dims)-total)>1e-9: raise ValueError('Consensus score mismatch')
            writer.writerow(dict(video_id=r['video_id'],video_path=str(Path(a.videos)/f"{r['video_id']}{a.extension}"),split=r['dataset_split'],centre=r['hospital'].replace('Centre ',''),joint=r['surgery_type'],label_cls=['novice','intermediate','expert'].index(r['skill_level']),label_reg=';'.join(map(str,dims+[total]))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True);group.add_argument('--xlsx');group.add_argument('--data');p.add_argument('--videos',required=True);p.add_argument('--output',required=True);p.add_argument('--extension',default='.mp4');main(p.parse_args())
