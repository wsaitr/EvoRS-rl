"""Exact cap-aware lazy completion feasibility; masks replay without live state."""
from functools import lru_cache
from .typed_measurement_actions import TypedMeasurementWorkflowRuntime


class FeasiblePointerMixin:
    def _completion_checker(self, role, ids, required, prefix=()):
        ids=tuple(ids);required=frozenset(required);cap=self.pointer_limit
        version=self.state.version
        @lru_cache(maxsize=4096)
        def finish(selected):
            assert self.state.version==version
            return bool(selected and required<=set(selected) and self._possible_modes(role,selected))
        @lru_cache(maxsize=4096)
        def visit(selected, remaining):
            if len(selected)>cap:return False
            if finish(selected):return True
            if len(selected)==cap:return False
            prunable=getattr(self,'_completion_prunable',None)
            if prunable is not None and prunable(role,selected):return False
            for index,eid in enumerate(remaining):
                if visit(tuple(sorted((*selected,eid))),remaining[index+1:]):return True
            return False
        def feasible(selected, ending=False):
            selected=tuple(sorted(set(selected)))
            if ending:return finish(selected) and len(selected)<=cap
            mandatory=tuple(sorted(set(selected)|required))
            if not set(mandatory)<=set(ids) or len(mandatory)>cap:return False
            return visit(mandatory,tuple(eid for eid in ids if eid not in mandatory))
        return feasible

    def _feasible_subset(self, role, prefix, optional, required=()):
        if self._sequential_pointer():
            return super()._feasible_subset(role,prefix,optional,required)
        ids=tuple(dict.fromkeys((*prefix,*optional)))
        return self._completion_checker(role,ids,required)(prefix)

    def _pointer_selection_context(self,role,ids,required):
        feasible=self._completion_checker(role,ids,required)
        if not feasible(()):raise ValueError('no cap-bounded executable evidence subset')
        return {'_pointer_feasibility':feasible,
            'pointer_feasibility_snapshot':{'workspace_version':self.state.version,
            'recipient':role,'remaining_wire_bytes':self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes,
            'maximum_evidence_ids':self.pointer_limit,'constraint':'at least one undispatched affordable mode including private search and verification'}}


class FeasiblePointerWorkflowRuntime(FeasiblePointerMixin,TypedMeasurementWorkflowRuntime):
    pass
