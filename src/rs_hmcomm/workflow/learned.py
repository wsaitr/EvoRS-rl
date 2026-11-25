"""Trainable addressed packet compression, temporal fusion and state value."""
import hashlib
import math
import torch
from torch import nn
import torch.nn.functional as F


def query_features(text,width=64):
    vector=torch.zeros(width)
    for word in text.casefold().split():
        digest=hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:4],'little')%width]+=1 if digest[4]&1 else -1
    return F.normalize(vector,dim=0)


def spatial_grid(count,height,width):
    if min(count,height,width)<=0:raise ValueError('invalid spatial grid')
    pairs=[(h,count//h) for h in range(1,count+1) if count%h==0]
    return min(pairs,key=lambda hw:abs(math.log((hw[0]/hw[1])/(height/width))))


def spatial_allocations(count,sources):
    """Equal source allocation in powers of two, preserving total wire slots."""
    if not isinstance(count,int) or isinstance(count,bool) or count<1 or count&(count-1):raise ValueError('unsupported spatial slot budget')
    if sources<1 or sources>count:raise ValueError('too many sources for spatial slots')
    values=[1]*sources;remaining=count-sources
    while remaining:
        eligible=[i for i,v in enumerate(values) if v<=remaining]
        index=min(eligible,key=lambda i:(values[i],i))
        remaining-=values[index];values[index]*=2
    return tuple(values)


class AddressedCodec(nn.Module):
    """Queries depend on question and recipient; output is in receiver space."""
    def __init__(self,hidden,max_tokens=16,width=64,roles=5):
        super().__init__();self.hidden=hidden;self.width=width;self.max_tokens=max_tokens
        self.query=nn.Parameter(torch.randn(max_tokens,width)*.02)
        self.role=nn.Embedding(roles,width);self.context=nn.Linear(width,width)
        self.key=nn.Linear(hidden,width,bias=False)
        self.projection_down=nn.Linear(hidden,width,bias=False)
        self.projection_up=nn.Linear(width,hidden,bias=False)
        nn.init.normal_(self.projection_up.weight,std=1e-3)

    def forward(self,tokens,question,recipient,count,*,layout=None,query_offset=0,positions=None,spatial_prior=True):
        if (layout is None and count not in {4,16,64}) or count<1 or query_offset<0 or count+query_offset>self.max_tokens:raise ValueError('unsupported compression grid')
        tokens=tokens.float();device=tokens.device
        context=self.context(query_features(question,self.width).to(device))+self.role.weight[recipient]
        q=self.query[query_offset:query_offset+count]+context
        logits=q@self.key(tokens).T/math.sqrt(self.width)
        if layout is not None and spatial_prior:
            height,width=layout
            if positions is None and height*width!=tokens.shape[0]:raise ValueError('source grid and token count disagree')
            qh,qw=spatial_grid(count,height,width)
            def centers(h,w):
                yy,xx=torch.meshgrid((torch.arange(h,device=device)+.5)/h,(torch.arange(w,device=device)+.5)/w,indexing='ij')
                return torch.stack((yy,xx),dim=-1).reshape(-1,2)
            source_centers=centers(height,width) if positions is None else positions.to(device=device,dtype=logits.dtype)
            if source_centers.shape!=(tokens.shape[0],2):raise ValueError("source positions and tokens disagree")
            offsets=(centers(qh,qw)[:,None,:]-source_centers[None,:,:])*logits.new_tensor((qh,qw))
            logits=logits-4*offsets.square().sum(dim=-1)
        weights=torch.softmax(logits,dim=-1)
        pooled=weights@tokens
        return pooled+self.projection_up(self.projection_down(pooled))


class TemporalFusion(nn.Module):
    """[before; after; signed difference; elapsed-time/missing-date encoding]."""
    def __init__(self,hidden,width=64,time_width=16):
        super().__init__();self.time_width=time_width
        self.down=nn.Linear(3*hidden+time_width+1,width)
        self.up=nn.Linear(width,hidden,bias=False);nn.init.normal_(self.up.weight,std=1e-3)

    def forward(self,before,after,*,elapsed_days=None):
        if before.shape!=after.shape:raise ValueError('temporal tokens must be registered on the same grid')
        t=0.0 if elapsed_days is None else math.copysign(math.log1p(abs(elapsed_days)),elapsed_days)
        frequencies=torch.exp(torch.arange(0,self.time_width,2,device=before.device,dtype=torch.float32)*(-math.log(10000)/self.time_width))
        encoding=torch.stack((torch.sin(t*frequencies),torch.cos(t*frequencies)),dim=-1).flatten()
        encoding=torch.cat((encoding,encoding.new_tensor([float(elapsed_days is None)])))
        x=torch.cat((before.float(),after.float(),after.float()-before.float(),encoding.expand(before.shape[0],-1)),dim=-1)
        return (before.float()+after.float())/2+self.up(F.gelu(self.down(x)))


class StateValue(nn.Module):
    def __init__(self,width=16):
        super().__init__();self.layers=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,state):return self.layers(state.float()).squeeze(-1)


def state_vector(*,coverage,model_calls,tool_calls,evidence_count,candidate_count,missing_count,role,invalid_actions,communication_bytes):
    values=[coverage,model_calls/64,tool_calls/32,evidence_count/32,candidate_count/16,
            missing_count/8,invalid_actions/16,communication_bytes/16777216,1.0]
    values += [float(role==r) for r in ('controller','visual','knowledge','reasoning','verification')]
    values += [float(candidate_count>0),float(coverage>=.999)]
    return tuple(values)
