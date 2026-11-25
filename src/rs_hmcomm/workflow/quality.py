from . import qwen_backbone
"""Explicit next-protocol interventions; the historical runtime stays unchanged.

These are development hypotheses, not an assertion of performance improvement.
"""
import math
import torch
from torch import nn
from .training import PaperWorkflowTrainer
from .learned import AddressedCodec, query_features, spatial_grid
from .qwen_policy import QwenWorkflowPolicy
from .paper_runtime import PaperWorkflowRuntime
from .runtime import answer_instruction, answer_valid


class StableAddressedCodec(AddressedCodec):
    """Bounded semantic attention prevents random native feature scale domination."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(2.)))

    def forward(self, tokens, question, recipient, count, *, layout=None,
                query_offset=0, positions=None, spatial_prior=True):
        if count < 1 or query_offset < 0 or query_offset+count > self.max_tokens:
            raise ValueError('unsupported slot allocation')
        tokens = tokens.float()
        context = self.context(query_features(question, self.width).to(tokens.device))
        context = context+self.role.weight[recipient]
        q = torch.nn.functional.normalize(self.query[query_offset:query_offset+count]+context, dim=-1)
        k = torch.nn.functional.normalize(self.key(tokens), dim=-1)
        logits = (q@k.T)*self.log_temperature.exp().clamp(max=16.)
        if layout is not None and spatial_prior:
            h, w = layout
            qh, qw = spatial_grid(count, h, w)
            def centers(height, width):
                y, x = torch.meshgrid((torch.arange(height, device=tokens.device)+.5)/height,
                                     (torch.arange(width, device=tokens.device)+.5)/width, indexing='ij')
                return torch.stack((y, x), dim=-1).reshape(-1, 2)
            source = centers(h, w) if positions is None else positions.to(tokens.device)
            if source.shape != (tokens.shape[0], 2):
                raise ValueError('source grid mismatch')
            distance = (centers(qh, qw)[:, None]-source[None])*logits.new_tensor((qh, qw))
            logits = logits-4*distance.square().sum(-1)
        pooled = torch.softmax(logits, dim=-1)@tokens
        return pooled+self.projection_up(self.projection_down(pooled))


class QualityWorkflowPolicy(QwenWorkflowPolicy):
    """Role instructions remain distinct under the intermediate-action schemas."""
    responsibilities = {
        'controller': 'Plan the shortest sufficient workflow. Spend calls on new evidence rather than repeated administration. Delegate a question requiring visual facts to the visual expert; use knowledge only when domain rules are needed.',
        'visual': 'Inspect actual pixels. Distinguish visible observations from uncertain interpretation. Do not invent objects or measurements. Request a detail view when small objects cannot be resolved.',
        'knowledge': 'Supply only relevant domain definitions, units, physical constraints or formulas. You are not an independent visual witness; never turn a plausible prior into an observed fact.',
        'reasoning': 'Solve the original question using the actual evidence and calculations. Preserve counts, units, dates and option meanings. Object metadata is a sparse index, never a complete inventory or a count. Do not copy an earlier claim without checking its evidence.',
        'verification': 'Check the assigned requirement or candidate against actual evidence. A negative answer can be correct. Repetition by another role is not corroboration; identify a concrete contradiction before rejecting.'}

    def __init__(self, *args, stable_codec=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stable_codec = stable_codec
        if stable_codec:
            codec = StableAddressedCodec(qwen_backbone.text_config(self.base).hidden_size,
                                        max_tokens=self.compressed_tokens).to(self.device)
            self.auxiliary['codec'] = codec
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=lr)
        # Incompatible codec weights must never masquerade as an old checkpoint.
        import hashlib
        self.identity = hashlib.sha256((self.identity+'quality-role-v1').encode()).hexdigest()

    def prepare(self, role, prompt, packet_ids, *, output_format='json'):
        if not getattr(self, 'role_priors', True):
            return QwenWorkflowPolicy.prepare(self, role, prompt, packet_ids, output_format=output_format)
        return super().prepare(role, self.responsibilities[role]+'\n'+prompt,
                               packet_ids, output_format=output_format)

    def save(self, path):
        super().save(path)
        import json
        from pathlib import Path
        (Path(path)/'quality.json').write_text(json.dumps({
            'stable_codec': isinstance(self.auxiliary['codec'], StableAddressedCodec),
            'role_priors': getattr(self, 'role_priors', True),
            'codec_capacity': self.auxiliary['codec'].max_tokens,
            'max_pixels': self.processor.image_processor.max_pixels}, indent=2))

    def load(self, path, *, resume_training=False):
        import json
        from pathlib import Path
        config = json.loads((Path(path)/'quality.json').read_text())
        if config['stable_codec'] != isinstance(self.auxiliary['codec'], StableAddressedCodec):
            raise ValueError('quality codec architecture mismatch')
        if config['codec_capacity'] != self.auxiliary['codec'].max_tokens:
            raise ValueError('quality codec capacity mismatch')
        if config['role_priors'] != getattr(self, 'role_priors', True):
            raise ValueError('quality role prompt mismatch')
        if config['max_pixels'] != self.processor.image_processor.max_pixels:
            raise ValueError('quality observation resolution mismatch')
        return super().load(path, resume_training=resume_training)


class QualityWorkflowRuntime(PaperWorkflowRuntime):
    """Closed-answer verification first predicts without seeing the candidate."""
    def __init__(self, *args, blind_verification=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.blind_verification = blind_verification
        self._blind_mismatch = None

    def _expert(self, role, context, packets, selected):
        self._blind_mismatch = None
        if (role == 'verification' and self.blind_verification and
                self.task.answer_format in {'yes_no', 'integer', 'choice'}):
            candidate = self.state.answer_candidates[-1]
            blind = answer_valid(self._call('verification',
                'Independently answer the ORIGINAL task from the attached actual views. '
                'No candidate answer or other expert claim is provided. '+answer_instruction(self.task.answer_format),
                packets, {}, output_format='text'), self.task.answer_format)
            self._blind_mismatch = blind.casefold() != candidate.answer.casefold()
            self.events.append({'kind': 'blind_verification', 'independent_answer': blind,
                                'candidate_id': candidate.candidate_id,
                                'agreement': not self._blind_mismatch,
                                'reference_used': False})
        try:
            return super()._expert(role, context, packets, selected)
        finally:
            self._blind_mismatch = None

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        if (role == 'verification' and self._blind_mismatch and
                objective.startswith('Independently check the candidate')):
            context = dict(context)
            context['legal_action_examples'] = [a for a in context['legal_action_examples']
                                               if a['decision'] != 'accept']
            objective += ' Your independent closed-form answer disagreed with this candidate. Reject or request new evidence; agreement cannot be fabricated.'
        return super()._call(role, objective, packet_ids, context, **kwargs)


class QualityWorkflowTrainer(PaperWorkflowTrainer):
    """Use dimensionless per-budget costs, preserving whole-workflow PPO/GAE."""
    def state_dict(self):
        return {**super().state_dict(), 'cost_units': 'budget_fraction_v1'}

    def load_state_dict(self, value):
        if value.get('cost_units') != 'budget_fraction_v1':
            raise ValueError('cost objective mismatch')
        super().load_state_dict(value)

    def transitions(self, episode, reference):
        original = self.duals
        try:
            self.duals = [d/b for d, b in zip(original, self.cost_targets)]
            result = super().transitions(episode, reference)
        finally:
            self.duals = original
        return result

    def update(self, episode, reference):
        result = super().update(episode, reference)
        result['cost_units'] = 'fraction of declared model/tool/wire budget; not raw calls'
        result['objective_terminal_quality'] = self.quality_scale
        return result
