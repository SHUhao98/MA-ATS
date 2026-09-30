"""Motion-aware adaptive temporal smoothing for xyxy detections."""
from collections import defaultdict, deque
import numpy as np

class TemporalSmoother:
    def __init__(self, alpha=0.7, max_history=5, min_confidence=0.25, enable_class_aware=True):
        self.alpha=float(alpha); self.max_history=int(max_history); self.min_confidence=float(min_confidence); self.enable_class_aware=enable_class_aware; self.reset()
    def reset(self):
        self.box_history=defaultdict(lambda: deque(maxlen=self.max_history)); self.conf_history=defaultdict(lambda: deque(maxlen=self.max_history)); self.cls_history=defaultdict(lambda: deque(maxlen=self.max_history)); self.frame_count=0
    def reset_track(self, track_id):
        for h in (self.box_history,self.conf_history,self.cls_history): h.pop(int(track_id),None)
    @staticmethod
    def _motion(a,b):
        ca=np.array([(a[0]+a[2])/2,(a[1]+a[3])/2]); cb=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
        area=max(float((a[2]-a[0])*(a[3]-a[1])),0.0)
        return float(np.linalg.norm(ca-cb)/(np.sqrt(area)+1e-6))
    @staticmethod
    def _adaptive_alpha(m):
        if m<0.05: return 0.3
        if m>0.5: return 0.9
        return 0.3+(m-0.05)*0.6/0.45
    def smooth_boxes_adaptive(self, boxes, confidences, class_ids, track_ids=None):
        boxes=np.asarray(boxes); confidences=np.asarray(confidences); class_ids=np.asarray(class_ids)
        if len(boxes)==0: return boxes, confidences
        ids=np.arange(len(boxes)) if track_ids is None else np.asarray(track_ids); out_b=boxes.copy(); out_c=confidences.copy()
        for i,(box,conf,cls,tid) in enumerate(zip(boxes,confidences,class_ids,ids)):
            if conf<self.min_confidence: continue
            bh,ch,lh=self.box_history[int(tid)],self.conf_history[int(tid)],self.cls_history[int(tid)]
            if self.enable_class_aware and lh and lh[-1]!=cls: bh.clear(); ch.clear(); lh.clear()
            if bh:
                a=self._adaptive_alpha(self._motion(box,bh[-1])); out_b[i]=a*box+(1-a)*bh[-1]; out_c[i]=a*conf+(1-a)*ch[-1]
            bh.append(out_b[i].copy()); ch.append(float(out_c[i])); lh.append(cls)
        self.frame_count+=1; return out_b,out_c
    def smooth_boxes(self, boxes, confidences, class_ids, track_ids=None):
        old=self._adaptive_alpha; self._adaptive_alpha=staticmethod(lambda _:self.alpha)
        try: return self.smooth_boxes_adaptive(boxes,confidences,class_ids,track_ids)
        finally: self._adaptive_alpha=old

MAATSTemporalSmoother=TemporalSmoother
