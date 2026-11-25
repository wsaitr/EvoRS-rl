"""Explicit overview reference during expert routing; no ground truth inspection."""
from .quality import QualityWorkflowRuntime
from .object_labels import usable_object_label


class ContextAnchoredRuntime(QualityWorkflowRuntime):
    def __init__(self, *args, anchor_global=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_global = anchor_global

    def _anchored(self, role, selected):
        if (self.anchor_global and selected and role in {'reasoning', 'verification'} and
                any(e.evidence_id == 'overview' for e in self.state.evidence)):
            return tuple(dict.fromkeys(('overview', *selected)))
        return tuple(selected)

    def _dispatch_key(self, role, selected, mode):
        return super()._dispatch_key(role, self._anchored(role, selected), mode)

    def _possible_modes(self, role, selected, *, include_dispatched=False):
        # Include the real anchor in budget/slot feasibility before offering
        # a route. It is not a free extra image appended after accounting.
        selected = self._anchored(role, selected)
        modes = super()._possible_modes(role, selected, include_dispatched=include_dispatched)
        if 'tokens' in modes:
            lookup = {e.evidence_id: e for e in self.state.evidence}
            closure = self._support_closure([lookup[e] for e in selected])
            packets = {p for e in closure for p in e.artifact_refs}
            count = sum(self.policy.packets[p].artifact.shape[0] for p in packets)
            if count > self.max_visual_tokens: modes.remove('tokens')
        return modes

    def _route(self, role, selected, mode):
        anchored = self._anchored(role, selected)
        if tuple(selected) != anchored:
            self.events.append({'kind': 'global_context_anchor', 'role': role,
                'original_selected': list(selected), 'actual_selected': list(anchored),
                'reference_labels_used': False})
        return super()._route(role, list(anchored), mode)

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        objects = role == 'visual' and 'Extract at most FOUR question-relevant visible objects.' in objective
        if objects:
            kwargs['json_field_schema']=[['objects','array']]
            objective = ('Localize at most four question-relevant objects in the ACTUAL attached views. '
                'Return a JSON object with key objects, a list of object records. '
                'Each record has category (actual observed class string), crop_index (zero-based attached-view index), '
                'and bbox (four normalized coordinates x1,y1,x2,y2 within that view, with x1<x2 and y1<y2). '
                'Coordinates must come from visible object extents; no example coordinates or invented objects. '
                'Return an empty objects list if reliable localization is impossible. '
                'This sparse index is not an exhaustive count.')
        result = super()._call(role, objective, packet_ids, context, **kwargs)
        if objects and isinstance(result, dict):
            if any(isinstance(row, dict) and not usable_object_label(row.get('category')) for row in result.get('objects', [])):
                # The caller retains the visual observation and logs this optional
                # metadata failure. It does not commit a fake spatial object.
                raise ValueError('object category is an ungrounded schema placeholder')
        return result
