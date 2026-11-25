"""Export bounded checks on the published winner; keep private drafts private."""
import json

from rs_hmcomm.contracts import Evidence
from .adaptive_fitness import AdaptiveFitnessRuntime
from .runtime import question_choices


def selected_search_feedback(events, alternatives, *, max_groups=4):
    """Join existing private fitness receipts to the selected winner only.

    A checker disagreement is a model observation, not a factual verdict. The
    function receives no reference labels and never exports the private trace.
    """
    if not 1 <= max_groups <= 12:
        raise ValueError('bounded source-group count required')
    searches = [e for e in events if e['kind'] == 'private_evolution']
    if not searches:
        return None
    if len(searches) != 1:
        raise ValueError('exactly one reasoning transaction required')
    selected = searches[0]['selected']
    answer, private_id = selected['payload']['answer'], selected['candidate_id']
    if answer not in alternatives:
        raise ValueError('published winner outside public answer space')
    pending, matched = [], None
    for event in events:
        if event['kind'] == 'independent_private_fitness':
            pending.append(event)
        elif event['kind'] == 'private_source_fitness':
            if event['candidate_id'] == private_id:
                if matched is not None:
                    raise ValueError('ambiguous selected fitness receipt')
                matched = pending
                verdicts = event['source_verdicts']
            pending = []
    if matched is None:
        return None  # Historical/non-independent search has no such replies.
    checks = []
    seen = set()
    for event in matched:
        other = event['independent_answer']; group = event['source_group']
        if (event['candidate_answer'] != answer or other not in [*alternatives, 'uncertain']
                or event['reference_used'] or event['candidate_disclosed_in_prompt']
                or event['model_agreement_is_ground_truth'] or group in seen):
            raise ValueError('invalid or duplicate selected-source check')
        verdict = 'uncertain' if other == 'uncertain' else 'supported' if other == answer else 'contradicted'
        if event['mapped_verdict'] != verdict or verdicts.get(group) != verdict:
            raise ValueError('selected source verdict disagrees with actual reply')
        seen.add(group)
        checks.append({'source_group': group, 'independent_answer': other,
            'relation_to_candidate': 'insufficient' if other == 'uncertain' else 'agreement' if other == answer else 'disagreement',
            'decision_index': event['decision_index']})
    if set(verdicts) != seen:
        raise ValueError('selected source-group coverage incomplete')
    return {'candidate_answer': answer, 'source_group_checks': checks[:max_groups],
        'total_source_groups': len(checks), 'omitted_source_groups': max(0, len(checks)-max_groups),
        'model_checks_are_ground_truth': False, 'new_visual_observation_claimed': False,
        'semantic_support_granted': False, 'external_reference_used': False,
        'interpretation': 'Agreement or disagreement between model readings of the same authorized source; not independent ground truth or extra votes.'}


class PublicSearchFeedbackRuntime(AdaptiveFitnessRuntime):
    def __init__(self, *args, public_search_feedback=False, **kwargs):
        super().__init__(*args, **kwargs)
        if public_search_feedback and not self.independent_fitness:
            raise ValueError('public search feedback requires candidate-hidden independent fitness')
        self.public_search_feedback = public_search_feedback

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        if (self.public_search_feedback and role == 'verification'
                and objective.startswith('Evaluate the support contributed by this ORIGINAL SOURCE GROUP to the private candidate.')):
            # Public reviews name a candidate. They must never enter a call that
            # claims to hide that candidate, including text/compressed modes.
            context = dict(context or {})
            context['evidence'] = [e for e in context.get('evidence', [])
                                   if e.get('kind') != 'workflow_search_feedback']
        return super()._call(role, objective, packet_ids, context, **kwargs)

    def _expert(self, role, context, packets, selected):
        start, count = len(self.events), len(self.state.answer_candidates)
        result = super()._expert(role, context, packets, selected)
        if not self.public_search_feedback or role != 'reasoning':
            return result
        alternatives = ([v.split('.', 1)[0].strip() for v in question_choices(self.task.question)]
                        if self.task.answer_format == 'choice' else ['Yes', 'No'] if self.task.answer_format == 'yes_no' else [])
        if not alternatives:
            return result
        receipt = selected_search_feedback(self.events[start:], alternatives)
        if receipt is None:
            return result
        added = self.state.answer_candidates[count:]
        if len(added) != 1 or added[0].answer != receipt['candidate_answer'] or tuple(added[0].evidence_ids) != tuple(selected):
            raise ValueError('selected feedback must bind to the actual newly published candidate')
        candidate = added[0]
        receipt = {**receipt, 'target_candidate_id': candidate.candidate_id,
                   'source_evidence_ids': list(selected)}
        eid = 'search-feedback-'+candidate.candidate_id
        # This explicit public export is an ordinary Workspace record. A later
        # expert receives it only through the usual authorized, billed route.
        self._commit('reasoning', evidence=(Evidence(eid, 'workflow_search_feedback',
            json.dumps(receipt, ensure_ascii=False), 'reasoning', self.step, payload=receipt),))
        self.events.append({'kind': 'public_search_feedback_export', 'evidence_id': eid,
            **receipt, 'private_trace_exported': False, 'extra_model_calls': 0})
        return result

    def _controller_decide(self, context, legal):
        if self.public_search_feedback:
            context = dict(context)
            context['search_review_rule'] = (
                'Search feedback is bound to its target_candidate_id and original sources. '
                'Read the actual agreement, disagreement and insufficient-detail receipts. '
                'They are fallible model checks, not factual certification or additional votes. '
                'Resolve important disagreement by revising or acquiring useful evidence when affordable. '
                'No receipt automatically invalidates a candidate or forces a different answer.')
        return super()._controller_decide(context, legal)

    def _initialize_task(self):
        self.events.append({'kind': 'public_search_feedback_protocol', 'enabled': self.public_search_feedback,
            'extra_model_calls': 0, 'semantic_support_granted': False, 'reference_used': False,
            'action_space_changed': False})
        return super()._initialize_task()
