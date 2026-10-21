"""Build quality-screened clips with YOLOv8, DeepSORT and native GRS labels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import cv2
import numpy as np
import torch
from models.motion_branch import KinematicFeatureExtractor


def extract_tracks(video_path, detector, tracker, extractor, size, brightness=30, blur=100):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise ValueError(f'Cannot decode {video_path}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if abs(fps-25)>0.1:
        cap.release(); raise ValueError(f'Expected 25 fps, received {fps}; resample explicitly before extraction')
    frames=[];motions=[];confs=[];indices=[];trajectories={};frame_no=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        source_index=frame_no;frame_no+=1
        frame=cv2.resize(frame,(size[1],size[0]),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if gray.mean()<brightness or cv2.Laplacian(gray,cv2.CV_64F).var()<blur: continue
        result=detector(frame,verbose=False,conf=0.001)[0]
        detections=[]
        for box in result.boxes:
            x,y,r,b=box.xyxy[0].detach().cpu().tolist();score=float(box.conf[0]);cls=int(box.cls[0])
            detections.append(([x,y,r-x,b-y],score,cls))
        tracks=tracker.update_tracks(detections,frame=frame)
        visible={}
        for tr in tracks:
            if not tr.is_confirmed() or tr.time_since_update!=0: continue
            tid=tr.track_id
            # Do not join stale history across a missing observation.
            history=trajectories.get(tid,[])
            if history and history[-1][0]!=source_index-1: history=[]
            history.append((source_index,list(tr.to_ltrb())))
            trajectories[tid]=history
            visible[tid]=[box for _,box in history]
        feature,_=extractor.extract(visible,size)
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));motions.append(feature)
        confs.append(max((d[1] for d in detections),default=0.));indices.append(source_index)
    cap.release()
    if not frames: raise ValueError(f'No retained frames in {video_path}')
    frames=np.asarray(frames,dtype=np.float32)
    mean=frames.mean(axis=(0,1,2),keepdims=True);std=frames.std(axis=(0,1,2),keepdims=True)+1e-6
    return (frames-mean)/std,np.asarray(motions),confs,indices


def main(a):
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    detector=YOLO(a.weights);extractor=KinematicFeatureExtractor()
    with open(a.metadata,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    seen=set()
    for row in rows:
        vid=row['video_id'];split=row['split']
        if vid in seen or split not in ['train','val','test']: raise ValueError('Duplicate procedure or invalid split')
        seen.add(vid)
        labels=[float(v) for v in row['label_reg'].split(';')]
        if len(labels)!=7 or not all(1<=v<=5 for v in labels[:6]) or not 6<=labels[6]<=30:
            raise ValueError('Expected six native 1-5 dimension scores and a 6-30 total')
        frames,motion,conf,indices=extract_tracks(row['video_path'],detector,DeepSort(max_age=30,n_init=2),extractor,(a.height,a.width),a.brightness,a.blur)
        target=Path(a.output)/split/'clips';target.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(frames)-a.clip_len+1,a.clip_len):
            end=start+a.clip_len
            payload=dict(frames=torch.from_numpy(frames[start:end]).permute(3,0,1,2).contiguous(),motion=torch.from_numpy(motion[start:end]).float(),detection_conf=torch.tensor(conf[start:end]),label_cls=int(row['label_cls']),label_reg=torch.tensor(labels),score_units='grs_points',video_id=vid,centre=row['centre'],source_frame_indices=indices[start:end])
            torch.save(payload,target/f'{vid}_clip{start//a.clip_len:04d}.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);p.add_argument('--weights',required=True,help='Arthroscopy detector fitted only on eligible training procedures')
    p.add_argument('--clip-len',type=int,default=16);p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=640)
    p.add_argument('--brightness',type=float,default=30);p.add_argument('--blur',type=float,default=100)
    main(p.parse_args())
