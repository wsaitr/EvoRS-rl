from . import qwen_backbone
"""Budget complete private search before offering its initial disclosure."""
import json
from .rejection_feedback import RejectionFeedbackRuntime


class TransactionBudgetRuntime(RejectionFeedbackRuntime):
    def _route_bound(self,role,selected,mode):
        selected=self._anchored(role,selected)
        lookup={e.evidence_id:e for e in self.state.evidence}
        closed=self._support_closure([lookup[e] for e in selected])
        packets={p for e in closed for p in e.artifact_refs}
        # Include the full registry/graph conservatively; actual authorization
        # may disclose fewer nodes. Metadata is estimated, actual route caps stay.
        metadata=len(json.dumps({'evidence':[e.to_dict() for e in closed],
            'graph':[n.to_dict() for n in self.state.spatial_nodes],
            'objects':self.objects.to_dict()},ensure_ascii=False).encode())
        overhead=max(65536,2*metadata+8192)
        if mode=='text':return overhead
        if mode=='tokens':return overhead+sum(self.policy.packets[p].artifact.byte_size for p in packets)
        return overhead+self.policy.compressed_tokens*qwen_backbone.text_config(self.policy.base).hidden_size*4

    def _search_groups(self,selected):
        lookup={e.evidence_id:e for e in self.state.evidence}
        closed=self._support_closure([lookup[e] for e in self._anchored('reasoning',selected)])
        groups={}
        for ev in closed:
            if ev.kind!='workflow_visual':continue
            key=','.join(sorted(f.descriptor()['sha256'] for f in self.task.frames
                if f.frame_id in ev.payload.get('frame_ids',())))
            groups.setdefault(key,[]).append(ev.evidence_id)
        return list(groups.values())

    def _completion_prunable(self, role, selected):
        """Necessary monotone wire bound, ignoring dispatch/pair prerequisites.

        Supersets can introduce a registered pair or an unused signature, so
        their absence never justifies pruning. Source closure, wire size and
        source-group fitness costs can only grow for the frozen workspace.
        """
        if not hasattr(self,'max_packet_bytes') or not hasattr(self,'cost'):
            return False  # No resource information means no proven exclusion.
        if self._packet_completion_prunable(role, selected):
            return True
        if (role!='reasoning' or not self.mechanisms.evolution
                or getattr(self,'adaptive_private_search',False)):
            # Other branches use a different parent feasibility predicate;
            # adding its missing metadata reserve would alter that language.
            return False
        remaining=self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes
        modes=[m for m in self.mechanisms.modes
               if (m!='compressed' or self.mechanisms.compression)
               and (m!='temporal' or (self.mechanisms.temporal and self._workspace_temporal_possible()))]
        if not modes:return True
        search=True
        groups=self._search_groups(selected) if search else []
        evaluations=1+self.mechanisms.generations*(self.mechanisms.population_size-1) if search else 0
        # Use every configured mode for the closing lower bound; a future
        # superset may enable temporal communication currently unavailable.
        closing=min(self._route_bound('verification',selected,m) for m in modes) if search else 0
        return all(self._route_bound(role,selected,m)+closing+
                   evaluations*sum(self._route_bound('verification',group,m) for group in groups)
                   >remaining for m in modes)

    def _packet_completion_prunable(self, role, selected):
        """Exclude supersets only using the parent's monotone packet limits.

        Do not impose transaction metadata reserves on non-search roles, and
        do not use missing pairs or previously used signatures as exclusions:
        adding evidence may satisfy those prerequisites.
        """
        if (not hasattr(getattr(self,'state',None),'evidence') or
                not hasattr(getattr(self,'policy',None),'packets')):
            return False
        lookup={e.evidence_id:e for e in self.state.evidence}
        closed=self._support_closure([lookup[e] for e in self._anchored(role,selected)])
        packets={p for e in closed for p in e.artifact_refs}
        remaining=self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes
        raw=sum(self.policy.packets[p].artifact.byte_size for p in packets)
        slots=sum(self.policy.packets[p].artifact.shape[0] for p in packets)
        config=getattr(getattr(self.policy,'base',None),'config',None)
        compressed=getattr(self.policy,'compressed_tokens',16)
        encoded=compressed*getattr(config,'hidden_size',3584)*4
        for mode in self.mechanisms.modes:
            if mode=='text':return False
            if mode=='tokens':
                if raw<=remaining and slots<=getattr(self,'max_visual_tokens',float('inf')):
                    return False
            elif mode=='compressed':
                if not self.mechanisms.compression:continue
                if encoded<=remaining and (not getattr(self.policy,'spatial_codec',False) or len(packets)<=compressed):
                    return False
            elif mode=='temporal':
                if self.mechanisms.temporal and encoded<=remaining and self._workspace_temporal_possible():return False
            else:
                return False  # An unknown parent's mode has no proven bound.
        return True

    def _workspace_temporal_possible(self):
        """A frozen workspace with no pair cannot gain one by selecting more IDs.

        Only the known monotone parent predicate supports this proof. Unknown
        overrides and incomplete state information retain the conservative bound.
        A pair absent from the current prefix may still exist elsewhere.
        """
        from .paper_runtime import PaperWorkflowRuntime
        state=getattr(self,'state',None)
        if (not hasattr(state,'evidence') or not hasattr(state,'version') or
                getattr(getattr(self,'_find_pair',None),'__func__',None) is not PaperWorkflowRuntime._find_pair):
            return True
        cached=getattr(self,'_temporal_workspace_bound',None)
        if cached is not None and cached[0] is state and cached[1]==state.version:
            return cached[2]
        try:
            possible=self._find_pair(tuple(e.evidence_id for e in state.evidence)) is not None
        except (ValueError,KeyError,TypeError):
            return True
        self._temporal_workspace_bound=(state,state.version,possible)
        return possible

    def _possible_modes(self,role,selected,*,include_dispatched=False):
        modes=super()._possible_modes(role,selected,include_dispatched=include_dispatched)
        if role!='reasoning' or not self.mechanisms.evolution:return modes
        groups=self._search_groups(selected)
        evaluations=1+self.mechanisms.generations*(self.mechanisms.population_size-1)
        remaining=self.max_packet_bytes-self.cost.message_latent_bytes-self.cost.message_structure_bytes
        verification_modes=super()._possible_modes('verification',selected,include_dispatched=True)
        if not verification_modes:return []
        closing=min(self._route_bound('verification',selected,m) for m in verification_modes)
        feasible=[]
        for mode in modes:
            # A globally legal temporal route may be illegal for a single-source
            # fitness group. Test each actual group, with the same anchor rule.
            if any(mode not in super(TransactionBudgetRuntime,self)._possible_modes(
                'verification',group,include_dispatched=True) for group in groups):continue
            expected=self._route_bound('reasoning',selected,mode)+closing
            expected+=evaluations*sum(self._route_bound('verification',group,mode) for group in groups)
            if expected<=remaining:feasible.append(mode)
        return feasible
