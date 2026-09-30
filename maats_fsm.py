"""Four-state confidence-weighted hysteresis FSM for event triggering."""
import numpy as np

MAATS_FSM_PARAMS=dict(conf_threshold=0.25,score_max=5.0,conf_weight=1.0,decay=0.2,gap_decay=0.05,T_on=1.0,T_off=0.3,N_conf=2,gap=1,cooldown_frames=3,min_event_length=3)

class EnhancedFSM:
    IDLE,CANDIDATE,ACTIVE,COOLDOWN=0,1,2,3
    def __init__(self,**kw):
        p=MAATS_FSM_PARAMS.copy(); p.update(kw); self.__dict__.update(p); assert self.T_on>self.T_off; self.reset()
    def reset(self):
        self.state=0; self.score=0.; self.consecutive_dets=0; self.gap_counter=0; self.cooldown_counter=0; self.frame_count=0; self.score_history=[]; self.state_history=[]; self.trigger_signal=[]; self.event_boundaries=[]; self._start=-1
    def process_frame(self,effective_confidence):
        valid=effective_confidence>0.; prev=self.state; self.consecutive_dets=self.consecutive_dets+1 if valid else 0; inc=self.conf_weight*effective_confidence if valid else 0.
        if self.state in (0,1):
            self.score=min(self.score_max,self.score+inc) if valid else max(0.,self.score-self.decay)
            if self.score>=self.T_on and self.consecutive_dets>=self.N_conf: self.state=2
            elif self.score<=0: self.state=0
            else: self.state=1
        elif self.state==2:
            if valid: self.score=min(self.score_max,self.score+inc); self.gap_counter=0
            else:
                self.gap_counter+=1; self.score=max(0.,self.score-(self.gap_decay if self.gap_counter<=self.gap else self.decay))
            if self.score<=self.T_off and self.gap_counter>=self.gap: self.state=3; self.cooldown_counter=self.cooldown_frames
        else:
            self.cooldown_counter-=1; self.score=min(self.score_max,self.score+inc*.5) if valid else max(0.,self.score-self.decay)
            if self.cooldown_counter<=0: self.state=2 if self.score>=self.T_on else (1 if self.score>0 else 0)
        self.score_history.append(self.score); self.state_history.append(self.state); trig=int(self.state==2); self.trigger_signal.append(trig)
        if prev!=2 and self.state==2: self._start=self.frame_count
        if prev==2 and self.state!=2 and self._start>=0: self.event_boundaries.append((self._start,self.frame_count-1)); self._start=-1
        self.frame_count+=1; return trig,self.state
    def get_trigger_signal(self): return np.asarray(self.trigger_signal)
    def get_active_state(self): return np.asarray(self.state_history)
    def get_score_history(self): return np.asarray(self.score_history)
    def get_params(self): return {k:getattr(self,k) for k in MAATS_FSM_PARAMS}
