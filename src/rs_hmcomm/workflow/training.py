"""On-policy whole-workflow optimization with completion-only gradients."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, asdict

import torch


def normalize_answer(value):
    value = unicodedata.normalize("NFKC",value or "").strip().casefold()
    return re.sub(r"[^\w\s]","",value).strip()


@dataclass(frozen=True)
class WorkflowReward:
    correctness: float
    coverage: float
    model_cost: float
    tool_cost: float
    communication_cost: float
    invalid_cost: float
    total: float

    def to_dict(self):
        return asdict(self)


def score_episode(episode, reference, failure_penalty=0.0):
    """Development exact-match reward; not an official open-ended scorer.

    ``failure_penalty`` is subtracted when the episode did not reach a
    validated finish, so an invalid trajectory stays strictly below any
    finished trajectory instead of only losing its coverage bonus.
    """
    if failure_penalty < 0 or not math.isfinite(failure_penalty):
        raise ValueError("nonnegative finite failure penalty required")
    from .metrics import task_score
    correct = task_score(episode.answer,reference) if episode.stop_reason=="finished" else 0.0
    model_cost = .01*episode.cost.model_calls
    tool_cost = .005*episode.cost.tool_calls
    communication_cost = .005*episode.cost.message_latent_bytes/1048576
    invalid_cost = .05*episode.cost.invalid_actions
    total = correct+.1*episode.coverage-model_cost-tool_cost-communication_cost-invalid_cost
    if episode.stop_reason!="finished":
        total -= failure_penalty
    return WorkflowReward(correct,episode.coverage,model_cost,tool_cost,communication_cost,invalid_cost,total)


class WorkflowTrainer:
    """PPO clipping over every sampled role completion in one executed episode.

    The episode return is shared by controller AND expert/tool/communication
    decisions. A pre-update moving baseline reduces variance. Prompt tokens,
    images, references and tool observations are not response targets.
    """
    def __init__(self, policy, *, clip=.2, epochs=1, max_grad_norm=1.0, kl_coefficient=.01):
        if not 0<clip<1 or epochs<1 or max_grad_norm<=0:
            raise ValueError("invalid optimizer settings")
        self.policy = policy
        self.clip,self.epochs,self.max_grad_norm,self.kl_coefficient = clip,epochs,max_grad_norm,kl_coefficient
        self.baseline = 0.0
        self.updates = 0

    def update(self, episode, reference, failure_penalty=0.0):
        reward = score_episode(episode,reference,failure_penalty=failure_penalty)
        records = episode.decisions
        if not records or any(r.old_logprobs.numel()!=r.output_ids.numel() or r.output_ids.numel()==0 for r in records):
            raise ValueError("training requires sampled on-policy completion logprobs for EVERY turn")
        if any(r.model_identity!=self.policy.identity for r in records):
            raise ValueError("model identity changed between rollout and update")
        before = self.policy.parameter_hash()
        if getattr(self.policy,"rollout_parameter_hash",before)!=before:
            raise ValueError("parameters changed after on-policy rollout")
        advantage = reward.total-self.baseline
        if not math.isfinite(advantage):
            raise ValueError("nonfinite return")
        counts = {};grad_by_role = {};loss_value = 0.0
        params = [p for p in self.policy.model.parameters() if p.requires_grad]
        self.policy.model.train()
        for _ in range(self.epochs):
            self.policy.optimizer.zero_grad(set_to_none=True)
            for record in records:
                logp = self.policy.logprobs(record)
                old = record.old_logprobs.to(logp.device)
                difference = logp-old
                ratio = difference.exp()
                if not torch.isfinite(ratio).all():
                    raise ValueError("nonfinite probability ratio")
                unclipped = ratio*advantage
                clipped = ratio.clamp(1-self.clip,1+self.clip)*advantage
                approx_kl = ratio-1-difference
                loss = (-torch.minimum(unclipped,clipped)+self.kl_coefficient*approx_kl).mean()/len(records)
                # Per-role contribution is measured before accumulation, not inferred from a global hash.
                previous = [None if p.grad is None else p.grad.detach().clone() for p in params]
                loss.backward()
                norm2 = 0.0
                for p,prev in zip(params,previous):
                    if p.grad is not None:
                        contribution = p.grad if prev is None else p.grad-prev
                        norm2 += float(contribution.float().square().sum())
                role = record.role
                grad_by_role[role] = grad_by_role.get(role,0.0)+math.sqrt(norm2)
                counts[role] = counts.get(role,0)+1
                loss_value += float(loss.detach())
                del logp,old,ratio,loss,previous
            norm = torch.nn.utils.clip_grad_norm_(params,self.max_grad_norm,error_if_nonfinite=True)
            if not math.isfinite(float(norm)):
                raise ValueError("nonfinite gradients")
            self.policy.optimizer.step()
        self.policy.model.eval()
        self.baseline = .9*self.baseline+.1*reward.total
        self.updates += 1
        after = self.policy.parameter_hash()
        return {"reward":reward.to_dict(),"advantage":advantage,"loss":loss_value,
                "roles_trained":counts,"gradient_norm_by_role":grad_by_role,
                "parameters_before":before,"parameters_after":after,"parameters_changed":before!=after,
                "update":self.updates,"optimizer":"completion-token PPO clipping with episode return",
                "reference_in_policy_observations":False}


class PaperWorkflowTrainer(WorkflowTrainer):
    """Joint completion PPO, learned state value, GAE and projected budget duals.

    Potential has a zero terminal value: shaping cannot reward an unfinished
    episode simply for inspecting many views. Each decision owns the actual
    cost incurred before the next decision, including its tool/routing effects.
    """
    def __init__(self,policy,*,gamma=.99,gae_lambda=.95,shaping=.1,
                 cost_targets=(32.,8.,8.),dual_rate=.001,initial_duals=(.01,.005,.005),quality_scale=1.0,**kwargs):
        super().__init__(policy,**kwargs)
        if not 0<gamma<=1 or not 0<=gae_lambda<=1 or dual_rate<0:
            raise ValueError('invalid return or dual settings')
        if len(cost_targets)!=3 or any(x<=0 for x in cost_targets):raise ValueError('invalid budget targets')
        if len(initial_duals)!=3 or any(not math.isfinite(x) or x<0 for x in initial_duals):raise ValueError('invalid initial budget duals')
        if not math.isfinite(quality_scale) or quality_scale<=0:raise ValueError('invalid quality scale')
        self.quality_scale=quality_scale
        self.gamma,self.gae_lambda,self.shaping=gamma,gae_lambda,shaping
        self.cost_targets=tuple(cost_targets);self.dual_rate=dual_rate
        self.initial_duals=tuple(initial_duals);self.duals=list(initial_duals)

    @staticmethod
    def costs(value):
        return (value['model_calls'],value['tool_calls'],(value['message_latent_bytes']+value['message_structure_bytes'])/1048576)

    def state_dict(self):
        return {'baseline':self.baseline,'updates':self.updates,'duals':self.duals,
            'initial_duals':self.initial_duals,
            'gamma':self.gamma,'gae_lambda':self.gae_lambda,'shaping':self.shaping,
            'cost_targets':self.cost_targets,'dual_rate':self.dual_rate,'quality_scale':self.quality_scale}

    def load_state_dict(self,value):
        for key in ('gamma','gae_lambda','shaping','dual_rate'):
            if value[key]!=getattr(self,key):raise ValueError('trainer configuration mismatch: '+key)
        if tuple(value['cost_targets'])!=self.cost_targets:raise ValueError('budget target mismatch')
        if tuple(value.get('initial_duals',(.01,.005,.005)))!=self.initial_duals:raise ValueError('initial budget objective mismatch')
        if value.get('quality_scale',1.0)!=self.quality_scale:raise ValueError('quality objective mismatch')
        self.baseline=value['baseline'];self.updates=value['updates'];self.duals=list(value['duals'])

    def transitions(self,episode,reference):
        records=episode.decisions
        from .metrics import task_score
        terminal=task_score(episode.answer,reference) if episode.stop_reason=='finished' else 0.0
        rewards=[];details=[]
        for i,record in enumerate(records):
            following=records[i+1] if i+1<len(records) else None
            before=record.cost_before;after=following.cost_before if following else episode.cost.to_dict()
            if before is None or len(record.state_features)!=16:raise ValueError('missing step state/cost')
            delta=[b-a for a,b in zip(self.costs(before),self.costs(after))]
            if any(x<0 for x in delta):raise ValueError('cost history is not monotone')
            penalty=sum(l*c for l,c in zip(self.duals,delta))
            invalid=.05*(after['invalid_actions']-before['invalid_actions'])
            potential_next=following.potential_before if following else 0.0
            shaped=self.shaping*(self.gamma*potential_next-record.potential_before)
            external=self.quality_scale*terminal if following is None else 0.0
            rewards.append(external-penalty-invalid+shaped)
            details.append({'external':external,'cost_delta':delta,'cost_penalty':penalty,
                'invalid_penalty':invalid,'potential_before':record.potential_before,
                'potential_next':potential_next,'shaping':shaped})
        advantages=[0.]*len(records);last=0.
        for i in reversed(range(len(records))):
            next_value=records[i+1].old_value if i+1<len(records) else 0.
            residual=rewards[i]+self.gamma*next_value-records[i].old_value
            last=residual+self.gamma*self.gae_lambda*last;advantages[i]=last
        targets=[a+r.old_value for a,r in zip(advantages,records)]
        if any(not math.isfinite(x) for x in (*rewards,*advantages,*targets)):raise ValueError('nonfinite step return')
        return rewards,advantages,targets,details

    def policy_objective(self,record,logp,advantage):
        difference=logp-record.old_logprobs.to(logp.device)
        ratio=difference.exp()
        if not torch.isfinite(ratio).all():raise ValueError('nonfinite ratio')
        surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-self.clip,1+self.clip)*advantage)
        return -surrogate.mean()+self.kl_coefficient*(ratio-1-difference).mean()

    def update(self,episode,reference):
        records=episode.decisions
        if not records or any(r.old_logprobs.numel()!=r.output_ids.numel() or not r.output_ids.numel() for r in records):
            raise ValueError('every sampled completion requires old likelihoods')
        before=self.policy.parameter_hash()
        if any(r.model_identity!=self.policy.identity for r in records) or self.policy.rollout_parameter_hash!=before:
            raise ValueError('not an on-policy rollout')
        rewards,advantages,targets,details=self.transitions(episode,reference)
        params=self.policy.trainable_parameters();decoder=[p for p in self.policy.model.parameters() if p.requires_grad]
        counts={};grad_by_role={};component_norms={};loss_total=0.
        self.policy.model.train();self.policy.auxiliary.train()
        for _ in range(self.epochs):
            self.policy.optimizer.zero_grad(set_to_none=True)
            for record,advantage,target in zip(records,advantages,targets):
                logp=self.policy.logprobs(record)
                policy_loss=self.policy_objective(record,logp,advantage)
                evaluator = getattr(self.policy, 'record_value', None)
                value=evaluator(record) if evaluator else self.policy.value(record.state_features)
                old=value.new_tensor(record.old_value);target=value.new_tensor(target)
                clipped_value=old+(value-old).clamp(-self.clip,self.clip)
                value_loss=torch.maximum((value-target).square(),(clipped_value-target).square())*.5
                loss=(policy_loss+.5*value_loss)/len(records)
                previous=[None if p.grad is None else p.grad.detach().clone() for p in decoder]
                loss.backward();norm2=0.
                for p,prev in zip(decoder,previous):
                    if p.grad is not None:norm2+=float((p.grad if prev is None else p.grad-prev).float().square().sum())
                grad_by_role[record.role]=grad_by_role.get(record.role,0)+math.sqrt(norm2)
                counts[record.role]=counts.get(record.role,0)+1;loss_total+=float(loss.detach())
                del previous,logp,policy_loss,loss
            for name,module in self.policy.auxiliary.items():
                component_norms[name]=math.sqrt(sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None))
            norm=torch.nn.utils.clip_grad_norm_(params,self.max_grad_norm,error_if_nonfinite=True)
            if not math.isfinite(float(norm)):raise ValueError('nonfinite gradient')
            self.policy.optimizer.step()
        self.policy.model.eval();self.policy.auxiliary.eval()
        spent=self.costs(episode.cost.to_dict());previous_duals=list(self.duals)
        self.duals=[max(0.,l+self.dual_rate*(c-b)/b) for l,c,b in zip(self.duals,spent,self.cost_targets)]
        self.updates+=1;after=self.policy.parameter_hash();reward=score_episode(episode,reference)
        return {'reward':reward.to_dict(),'loss':loss_total,'step_rewards':rewards,'advantages':advantages,
            'value_targets':targets,'transitions':details,'duals_before':previous_duals,'duals_after':self.duals,
            'roles_trained':counts,'gradient_norm_by_role':grad_by_role,'auxiliary_gradient_norms':component_norms,
            'parameters_before':before,'parameters_after':after,'parameters_changed':before!=after,'update':self.updates,
            'optimizer':'joint completion PPO + state value + GAE + potential shaping + projected budget duals',
            'reference_in_policy_observations':False}
