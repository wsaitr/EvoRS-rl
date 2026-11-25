"""Bound fallible source-check replies to the actual private parent evaluation.

This consumes no label, chooses no answer and publishes no private draft.
Checks are grouped by original source; repeated views cannot create extra votes.
"""
import json


def _field(record, key):
    return record[key] if isinstance(record, dict) else getattr(record, key)


def parent_source_feedback(events, records, parent_id, parent_answer, alternatives, *, max_groups=4):
    if (not isinstance(parent_id, str) or not parent_id or parent_answer not in alternatives
            or len(set(alternatives)) != len(alternatives) or not alternatives
            or type(max_groups) is not int or not 1 <= max_groups <= 4):
        raise ValueError('invalid private parent or public alternatives')
    boundary = max((i+1 for i,e in enumerate(events) if e['kind']=='private_evolution'), default=0)
    matches = [i for i in range(boundary,len(events)) if events[i]['kind']=='private_source_fitness'
               and events[i]['candidate_id']==parent_id]
    if len(matches)!=1:
        raise ValueError('parent must have one evaluation in this private transaction')
    end = matches[0]
    start = max((i+1 for i in range(boundary,end) if events[i]['kind']=='private_source_fitness'), default=boundary)
    verdicts = events[end]['source_verdicts']
    seen = set()
    checks = []
    for event_index in range(start,end):
        event = events[event_index]
        if event['kind']!='independent_private_fitness':
            continue
        answer, group, decision_index = event['independent_answer'], event['source_group'], event['decision_index']
        if (event['candidate_answer'] != parent_answer or event['reference_used']
                or event['candidate_disclosed_in_prompt'] or event['model_agreement_is_ground_truth']
                or answer not in [*alternatives,'uncertain'] or not isinstance(group,str) or not group
                or group in seen or type(decision_index) is not int or not 0<=decision_index<len(records)):
            raise ValueError('invalid, leaking or repeated-source parent check')
        record=records[decision_index]
        action=json.loads(_field(record,'text'))
        if (_field(record,'role')!='verification' or action!={'independent_answer':answer}
                or list(_field(record,'packet_ids'))!=event['packet_ids']):
            raise ValueError('parent check differs from actual saved verification completion')
        verdict='uncertain' if answer=='uncertain' else 'supported' if answer==parent_answer else 'contradicted'
        if event['mapped_verdict']!=verdict or verdicts.get(group)!=verdict:
            raise ValueError('parent verdict does not match its actual check')
        seen.add(group)
        checks.append({'source_group':group,'independent_answer':answer,
            'relation_to_parent':'insufficient' if answer=='uncertain' else 'agreement' if answer==parent_answer else 'disagreement',
            'decision_index':decision_index,'event_index':event_index,
            'prior_summaries_disclosed':event['prior_summaries_disclosed']})
    if not checks or set(verdicts)!=seen:
        raise ValueError('parent source coverage missing or incomplete')
    return {'parent_candidate_id':parent_id,'parent_answer':parent_answer,
        'source_group_checks':checks[:max_groups],'total_source_groups':len(checks),
        'omitted_source_groups':max(0,len(checks)-max_groups),'fitness_event_index':end,
        'model_checks_are_ground_truth':False,'external_reference_used':False,
        'new_visual_observation_claimed':False,'semantic_support_granted':False,
        'interpretation':'These are fallible readings of the same authorized original sources, not independent facts or votes.'}


def feedback_wire_bytes(receipt):
    return len(json.dumps(receipt,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))
