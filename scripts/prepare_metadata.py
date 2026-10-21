"""Export the authoritative split and averaged dimension scores to English CSV."""
import argparse
import csv
from pathlib import Path
from openpyxl import load_workbook


def main(a):
    wb=load_workbook(a.xlsx,read_only=True,data_only=True)
    splits=list(wb['Data Split'].values); header=splits[0]
    records=[dict(zip(header,row)) for row in splits[1:] if row[0]]
    ratings={row[0]:row for row in list(wb['原始评分表'].values)[2:] if row[0]}
    if len(records)!=196 or len(ratings)!=196: raise ValueError('Expected 196 included procedures')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['video_id','video_path','split','centre','joint','label_cls','label_reg']);writer.writeheader()
        for r in records:
            row=ratings[r['video_id']]
            dims=[(float(row[i])+float(row[i+7]))/2 for i in range(1,7)]
            total=(float(row[7])+float(row[14]))/2
            if total!=r['grs_score'] or abs(sum(dims)-total)>1e-9: raise ValueError('Consensus score mismatch')
            writer.writerow(dict(video_id=r['video_id'],video_path=str(Path(a.videos)/f"{r['video_id']}{a.extension}"),split=r['dataset_split'],centre=r['hospital'].replace('Centre ',''),joint=r['surgery_type'],label_cls=['novice','intermediate','expert'].index(r['skill_level']),label_reg=';'.join(map(str,dims+[total]))))
    wb.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--xlsx',required=True);p.add_argument('--videos',required=True);p.add_argument('--output',required=True);p.add_argument('--extension',default='.mp4');main(p.parse_args())
