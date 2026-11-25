"""Label-free task requirements and source-grouped semantic evidence."""
from dataclasses import dataclass, asdict
import math
from .evidence import _normalized_bbox


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    query: str
    kind: str
    scope: str
    frame_ids: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self):
        if not self.requirement_id or not self.query.strip():raise ValueError('empty requirement')
        if self.kind not in {'object','count','relation','change','scene'}:raise ValueError('unknown requirement kind')
        if self.scope not in {'local','global'}:raise ValueError('unknown requirement scope')
        if not self.frame_ids or len(set(self.frame_ids))!=len(self.frame_ids):raise ValueError('invalid required frames')
        if not math.isfinite(self.weight) or self.weight<=0:raise ValueError('invalid requirement weight')

    def to_dict(self):return asdict(self)


def parse_requirements(value, task, *, limit=6):
    if set(value)!={'requirements'} or not isinstance(value['requirements'],list):raise ValueError('expected requirements list')
    if not 1<=len(value['requirements'])<=limit:raise ValueError('requirement count out of bounds')
    allowed={f.frame_id for f in task.frames};result=[]
    for index,row in enumerate(value['requirements']):
        if set(row)!={'query','kind','scope','frame_ids'}:raise ValueError('invalid requirement fields')
        if not isinstance(row['query'],str) or not isinstance(row['frame_ids'],list):raise ValueError('invalid requirement types')
        if set(row['frame_ids'])-allowed:raise ValueError('requirement references unknown dates')
        if row['kind']=='change' and len(row['frame_ids'])<2:raise ValueError('change requires two dates')
        result.append(Requirement(f'req-{index}',row['query'],row['kind'],row['scope'],tuple(row['frame_ids'])))
    return tuple(result)


class SemanticLedger:
    """Repeated views from one source do not become independent votes.

    Support is an estimated entailment score. References are never used here.
    Semantic confidence and geometric inspection remain separate signals.
    """
    scores={'supported':1.0,'uncertain':0.25,'contradicted':0.0}
    def __init__(self,requirements):
        self.requirements=tuple(requirements);self.assessments={}

    def record(self,requirement_id,evidence_id,source_group,verdict,*,global_valid=False,pair_valid=False,observed_evidence_ids=()):
        requirement=next((r for r in self.requirements if r.requirement_id==requirement_id),None)
        if requirement is None or verdict not in self.scores:raise ValueError('unknown requirement or verdict')
        if not evidence_id or not source_group:raise ValueError('missing assessment provenance')
        if verdict=='supported' and (requirement.scope=='global' and not global_valid or requirement.kind=='change' and not pair_valid):
            verdict='uncertain'
        observed=tuple(observed_evidence_ids)
        if len(set(observed))!=len(observed) or any(not isinstance(x,str) or not x for x in observed):raise ValueError('invalid observation provenance')
        supersedes=[]
        if observed:
            for old_id,ratings in self.assessments.items():
                old=ratings.get(requirement_id)
                if old and old['source_group']==source_group and old.get('observed_evidence_ids') and set(old['observed_evidence_ids'])<set(observed):supersedes.append(old_id)
        self.assessments.setdefault(evidence_id,{})[requirement_id]={
            'source_group':source_group,'verdict':verdict,'score':self.scores[verdict],
            'global_valid':global_valid,'pair_valid':pair_valid,
            'observed_evidence_ids':list(observed),'supersedes':supersedes}

    def support(self,selected=None):
        rows=self.assessments if selected is None else {k:v for k,v in self.assessments.items() if k in selected}
        result={}
        for r in self.requirements:
            groups={};contradicted=False
            superseded={old for ratings in rows.values() for old in ratings.get(r.requirement_id,{}).get('supersedes',())}
            for evidence_id,ratings in rows.items():
                if evidence_id in superseded:continue
                rating=ratings.get(r.requirement_id)
                if rating is None:continue
                contradicted |= rating['verdict']=='contradicted'
                group=rating['source_group'];groups[group]=max(groups.get(group,0.0),rating['score'])
            result[r.requirement_id]={'support':max(groups.values(),default=0.0),
                'source_groups':len(groups),'contradicted':contradicted}
        return result

    def coverage(self,selected=None):
        ratings=self.support(selected);total=sum(r.weight for r in self.requirements)
        return sum(r.weight*ratings[r.requirement_id]['support'] for r in self.requirements)/max(total,1e-8)

    def sufficient(self,selected=None):
        ratings=self.support(selected)
        return bool(ratings) and all(v['support']>=.999 and not v['contradicted'] for v in ratings.values())

    def missing(self,selected=None):
        ratings=self.support(selected)
        return [r.to_dict() for r in self.requirements if ratings[r.requirement_id]['support']<.999 or ratings[r.requirement_id]['contradicted']]


def bbox_iou(left,right):
    a=_normalized_bbox(left);b=_normalized_bbox(right)
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(union,1e-8)


class ObjectRegistry:
    """Merge overlapping-tile detections in source coordinates, retaining lineage."""
    def __init__(self,threshold=.5):self.threshold=threshold;self.objects=[];self.temporal_links=[]
    def add(self,category,bbox,frame_id,evidence_id,*,reference_bbox=None):
        if not isinstance(category,str) or not category.strip():raise ValueError('missing category')
        bbox=_normalized_bbox(bbox)
        for item in self.objects:
            if item['frame_id']==frame_id and item['category'].casefold()==category.casefold() and bbox_iou(item['bbox'],bbox)>=self.threshold:
                if evidence_id not in item['evidence_ids']:item['evidence_ids'].append(evidence_id)
                return item['object_id']
        item={'object_id':f'object-{len(self.objects)}','category':category,'bbox':bbox,'frame_id':frame_id,
              'evidence_ids':[evidence_id],'reference_bbox':reference_bbox}
        self.objects.append(item)
        if reference_bbox is not None:
            for old in self.objects[:-1]:
                if old['frame_id']!=frame_id and old['reference_bbox'] is not None and old['category'].casefold()==category.casefold():
                    overlap=bbox_iou(old['reference_bbox'],reference_bbox)
                    if overlap>=self.threshold:self.temporal_links.append({'before':old['object_id'],'after':item['object_id'],'iou':overlap})
        return item['object_id']

    def to_dict(self):
        from copy import deepcopy
        relations=[]
        for index,a in enumerate(self.objects):
            for b in self.objects[index+1:]:
                if a['frame_id']!=b['frame_id']:continue
                ax,ay=(a['bbox'][0]+a['bbox'][2])/2,(a['bbox'][1]+a['bbox'][3])/2
                bx,by=(b['bbox'][0]+b['bbox'][2])/2,(b['bbox'][1]+b['bbox'][3])/2
                relations.append({'subject':a['object_id'],'object':b['object_id'],
                    'horizontal':'left' if ax<bx else 'right' if ax>bx else 'aligned',
                    'vertical':'above' if ay<by else 'below' if ay>by else 'aligned',
                    'distance_normalized':math.hypot(ax-bx,ay-by),'coordinate_system':'source image, y increases down'})
        # A saved event must not acquire detections or lineage from later steps.
        return deepcopy({'objects':self.objects,'temporal_links':self.temporal_links,'spatial_relations':relations})
