"""Shared trainable Qwen policy for ALL workflow roles and native ROI tokens.

Unlike pooled-vector diagnostics, transmitted merger outputs are inserted into
actual image-token positions of the receiving language model. The frozen vision
encoder is run once per unique processed crop, never again by the receiver.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import time
from typing import Any

import torch
from rs_hmcomm.artifacts import ArtifactKind, InMemoryArtifactStore, VisualArtifact
from .evidence import image_digest
from .json_protocol import complete_object,ObjectRootGrammar
from .generation_attention import generation_bounded_attention
from . import qwen_backbone


@contextmanager
def generation_last_position(head):
    """Generation consumes only the last position; retain full decoder/KV context."""
    handle = head.register_forward_pre_hook(
        lambda _module, args: (args[0][:, -1:, :], *args[1:]))
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def generation_chunked_mlp(modules, chunk_size=256):
    """Bound tokenwise feed-forward temporaries without changing attention/KV."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive feed-forward chunk size required')
    originals = []
    try:
        for module in modules:
            original = module.forward
            originals.append((module, original))
            def forward(states, *args, _original=original, **kwargs):
                if states.ndim != 3 or states.shape[1] <= chunk_size:
                    return _original(states, *args, **kwargs)
                return torch.cat([_original(chunk, *args, **kwargs)
                                  for chunk in states.split(chunk_size, dim=1)], dim=1)
            module.forward = forward
        yield
    finally:
        for module, original in originals:
            module.forward = original


def generation_cache_options(device, sampling):
    # OffloadedCache requires a GPU; PPO sampling/replay keeps its original path.
    return {'cache_implementation': 'offloaded'} if torch.device(device).type == 'cuda' and not sampling else {}


@dataclass
class NativePacket:
    artifact: VisualArtifact
    grid: tuple[int, int, int]
    label: str
    recipe: dict | None = None


@dataclass
class TokenDecision:
    role: str
    prompt: str
    packet_ids: tuple[str, ...]
    input_ids: Any
    output_ids: Any
    old_logprobs: Any
    text: str
    latency_ms: float
    model_identity: str
    constraint_options: tuple[str, ...] = ()
    state_features: tuple[float, ...] = ()
    cost_before: dict | None = None
    potential_before: float = 0.0
    old_value: float = 0.0
    json_object_constraint: bool = False
    json_syntax_constraint: bool = False
    json_syntax_token_budget: int = 0
    json_field_schema: list | None = None
    evidence_pointer_constraint: dict | None = None
    evidence_pointer_token_masks: list | None = None
    value_context: Any = None
    value_context_provenance: dict | None = None
    json_field_backend: str = 'python'
    behaviour_likelihood: dict | None = None

    def descriptor(self):
        result = {"role":self.role,"prompt":self.prompt,"packet_ids":list(self.packet_ids),
                "input_ids":self.input_ids.tolist(),"output_ids":self.output_ids.tolist(),
                "old_logprobs":self.old_logprobs.tolist(),"text":self.text,
                "latency_ms":self.latency_ms,"model_identity":self.model_identity,
                "constraint_options":list(self.constraint_options),'state_features':list(self.state_features),
                'cost_before':self.cost_before,'potential_before':self.potential_before,'old_value':self.old_value,
                'json_object_constraint':self.json_object_constraint,
                'json_syntax_constraint':self.json_syntax_constraint,
                'json_syntax_token_budget':self.json_syntax_token_budget,
                'json_field_schema':self.json_field_schema,
                'json_field_backend':self.json_field_backend,
                'evidence_pointer_constraint':self.evidence_pointer_constraint,
                'evidence_pointer_token_masks':self.evidence_pointer_token_masks}
        if self.value_context is not None:
            result['value_context'] = self.value_context.tolist()
            result['value_context_provenance'] = self.value_context_provenance
        if self.behaviour_likelihood is not None:
            result['behaviour_likelihood'] = self.behaviour_likelihood
        return result


class ActionTrie:
    """Finite legal-action grammar. Both sampling and likelihood use this mask."""
    def __init__(self, tokenizer, options):
        self.root = {}
        for option in options:
            node = self.root
            for token in tokenizer.encode(option,add_special_tokens=False)+[tokenizer.eos_token_id]:
                node = node.setdefault(token,{})

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            if int(token) not in node:
                raise ValueError("completion violates recorded legal-action grammar")
            node = node[int(token)]
        if not node:
            raise ValueError("tokens requested after completed action")
        return tuple(node)


class QwenWorkflowPolicy:
    roles = ("controller","visual","knowledge","reasoning","verification")

    def __init__(self, model_path, *, device="cuda:0", revision=None, max_pixels=65536,
                 max_new_tokens=192, lora_rank=8, learning_rate=2e-5, model=None, processor=None,
                 paper_mechanisms=False, compressed_tokens=16,spatial_codec=False,spatial_variant="both",
                 inference_only_sharded=False, pretrained_only=False, field_decoder_backend='python',
                 backbone_family='qwen2_5_vl', training_sharded=False):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if backbone_family not in qwen_backbone.SUPPORTED:
            raise ValueError('explicit supported backbone family required')
        if type(training_sharded) is not bool or (training_sharded and inference_only_sharded):
            raise ValueError('training and inference dispatch are separate explicit modes')
        if training_sharded and (model is None or pretrained_only):
            raise ValueError('training dispatch requires an already loaded trainable model')
        if type(pretrained_only) is not bool:
            raise ValueError('pretrained-only inference must be explicit')
        if pretrained_only and paper_mechanisms:
            raise ValueError('pretrained-only inference cannot create learned auxiliary modules')
        self.pretrained_only = pretrained_only
        if field_decoder_backend not in ('python', 'xgrammar-0.2.8-fields-v1'):
            raise ValueError('explicit supported field decoder backend required')
        self.field_decoder_backend = field_decoder_backend
        self.device = torch.device(device)
        if type(inference_only_sharded) is not bool:
            raise ValueError('sharded inference must be explicitly enabled')
        if inference_only_sharded and model is None:
            raise ValueError('supply an already dispatched model for sharded inference')
        self.inference_only_sharded = inference_only_sharded
        self.max_new_tokens = max_new_tokens
        self.processor = processor or AutoProcessor.from_pretrained(
            model_path,local_files_only=True,max_pixels=max_pixels)
        loader = Qwen2_5_VLForConditionalGeneration
        if backbone_family == 'qwen3_5':
            from transformers import Qwen3_5ForConditionalGeneration
            loader = Qwen3_5ForConditionalGeneration
        self.base = model or loader.from_pretrained(
            model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa")
        if qwen_backbone.family(self.base) != backbone_family:
            raise ValueError('requested family differs from actual checkpoint configuration')
        self.backbone_family = backbone_family
        self.training_sharded = training_sharded
        if inference_only_sharded or training_sharded:
            from .cuda_dispatch import validate_cuda_dispatch
            self.inference_device_map = validate_cuda_dispatch(
                getattr(self.base,'hf_device_map',None), entry_device=self.device,
                embedding_device=self.base.get_input_embeddings().weight.device,
                visual_devices={str(p.device) for p in qwen_backbone.visual_module(self.base).parameters()})
        else:
            self.base.to(self.device)
        # Decoder adapters receive gradients from controller and all expert turns.
        # Restrict target modules to decoder; vision encoder and merger stay frozen.
        if pretrained_only:
            if getattr(self.base, 'peft_config', None):
                raise ValueError('supply an original backbone without project adapters')
            self.model = self.base
            self.model.requires_grad_(False)
        else:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(r=lora_rank,lora_alpha=2*lora_rank,lora_dropout=0.0,
                                target_modules=qwen_backbone.lora_targets(self.base),task_type="CAUSAL_LM")
            self.model = get_peft_model(self.base,config)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            self.model.enable_input_require_grads()
        self.paper_mechanisms=paper_mechanisms;self.compressed_tokens=compressed_tokens
        self.spatial_codec=spatial_codec
        if spatial_variant not in {"both","source","position"}:raise ValueError("invalid spatial codec variant")
        self.spatial_variant=spatial_variant
        self.auxiliary=None
        if paper_mechanisms:
            from .learned import AddressedCodec,TemporalFusion,StateValue
            hidden_size=qwen_backbone.text_config(self.base).hidden_size
            self.auxiliary=torch.nn.ModuleDict({'codec':AddressedCodec(hidden_size,max_tokens=compressed_tokens),
                'temporal':TemporalFusion(hidden_size),'value':StateValue()}).to(self.device)
        self.optimizer = None if pretrained_only else torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=learning_rate)
        if self.auxiliary is not None:self.optimizer.add_param_group({'params':self.auxiliary.parameters()})
        self.identity = hashlib.sha256(json.dumps({"config":self.base.config.to_dict(),
                            "revision":revision,"model_path":str(model_path)},sort_keys=True).encode()).hexdigest()
        self.store = InMemoryArtifactStore()
        self.packets: dict[str,NativePacket] = {}
        self.records: list[TokenDecision] = []
        self.sampling = False
        self.cache: dict[str,NativePacket] = {}
        self.vision_encodes = 0
        self.receiver_visual_forwards = 0
        self._encoding = False
        def count_receiver(_module, _inputs):
            if not self._encoding:
                self.receiver_visual_forwards += 1
        self._vision_audit_hook = qwen_backbone.visual_module(self.base).register_forward_pre_hook(count_receiver)

    def begin_episode(self, *, training=False):
        if training and getattr(self, 'pretrained_only', False):
            raise ValueError('pretrained-only policy cannot train')
        if training and getattr(self,'inference_only_sharded',False):
            raise ValueError('CUDA-dispatched candidate supports inference only')
        self.records = []
        self.failed_generations = []
        self.sampling = training
        self.model.eval()
        self.rollout_parameter_hash = self.parameter_hash() if training else None
        # Release previous task evidence; records remain owned by returned episode.
        self.store = InMemoryArtifactStore()
        self.packets = {}
        self.cache = {}

    def encode(self, image, *, artifact_id, source_image_ref, roi, label):
        processed = self.processor.image_processor(images=image,return_tensors="pt")
        grid = tuple(int(v) for v in processed["image_grid_thw"][0])
        key = image_digest(image)+str(grid)
        cached = self.cache.get(key)
        if cached is None:
            self._encoding = True
            try:
                with torch.no_grad():
                    visual=qwen_backbone.visual_module(self.base)
                    values = qwen_backbone.visual_merger_output(self.base,
                        processed["pixel_values"].to(self.device,visual.dtype),
                        processed["image_grid_thw"].to(self.device)).float().cpu()
            finally:
                self._encoding = False
            if values.ndim != 2 or not torch.isfinite(values).all():
                raise ValueError("vision encoder produced invalid native tokens")
            expected = grid[0]*grid[1]*grid[2]//self.base.config.vision_config.spatial_merge_size**2
            if values.shape != (expected,qwen_backbone.text_config(self.base).hidden_size):
                raise ValueError("native visual token shape disagrees with receiver")
            self.vision_encodes += 1
            data = values.contiguous().numpy().tobytes()
        else:
            data = self.store.get(cached.artifact.artifact_id).data
            values = torch.frombuffer(bytearray(data),dtype=torch.float32).reshape(cached.artifact.shape)
        descriptor = VisualArtifact(artifact_id,ArtifactKind.LATENT,"application/x-evors-float32-le",
            tuple(values.shape),"float32",len(data),source_image_ref,roi,self.identity,
            "qwen35_visual_merger" if qwen_backbone.family(self.base)=='qwen3_5' else "qwen25_visual_merger",
            "native_identity",self.roles,hashlib.sha256(data).hexdigest())
        self.store.put(descriptor,data)
        packet = NativePacket(descriptor,grid,label)
        self.packets[artifact_id] = packet
        self.cache[key] = packet
        return descriptor

    def _features(self, packet_ids):
        blocks = []
        encoder_identity = getattr(self, 'replay_visual_encoder_identity', self.identity)
        for pid in packet_ids:
            packet = self.packets[pid]
            if packet.artifact.encoder != encoder_identity:
                raise ValueError("native token encoder/receiver identity mismatch")
            if packet.recipe is not None:
                blocks.append(self._recipe_features(packet.recipe))
            else:
                stored = self.store.get(pid)
                blocks.append(torch.frombuffer(bytearray(stored.data),dtype=torch.float32).reshape(stored.artifact.shape).to(self.device))
        if not blocks:
            return None
        return torch.cat(blocks).to(self.device,self.base.dtype)

    def _recipe_features(self,recipe):
        sources=[]
        for pid in recipe['sources']:
            stored=self.store.get(pid)
            sources.append(torch.frombuffer(bytearray(stored.data),dtype=torch.float32).reshape(stored.artifact.shape).to(self.device))
        tokens=torch.cat(sources)
        if recipe['mode']=='temporal':
            if len(sources)!=2:raise ValueError('temporal fusion requires exactly two registered sources')
            # Resample the registered grids rather than pairing unrelated token indices.
            grids=[self.packets[p].grid for p in recipe['sources']]
            merge=self.base.config.vision_config.spatial_merge_size
            target=(min(g[1] for g in grids)//merge,min(g[2] for g in grids)//merge)
            aligned=[]
            for source,grid in zip(sources,grids):
                image=source.reshape(grid[1]//merge,grid[2]//merge,-1).permute(2,0,1).unsqueeze(0)
                resized=torch.nn.functional.interpolate(image,size=target,mode='bilinear',align_corners=False)
                aligned.append(resized[0].permute(1,2,0).reshape(-1,source.shape[-1]))
            tokens=self.auxiliary['temporal'](*aligned,elapsed_days=recipe.get('elapsed_days'))
        layout=None;positions=None
        if recipe.get('spatial'):
            merge=self.base.config.vision_config.spatial_merge_size
            layout=target if recipe['mode']=='temporal' else (self.packets[recipe['sources'][0]].grid[1]//merge,self.packets[recipe['sources'][0]].grid[2]//merge)
        if recipe.get('spatial') and recipe['mode']=='compressed' and len(sources)>1:
            coordinates=[]
            for pid in recipe['sources']:
                grid=self.packets[pid].grid;h,w=grid[1]//merge,grid[2]//merge
                yy,xx=torch.meshgrid((torch.arange(h,device=self.device)+.5)/h,(torch.arange(w,device=self.device)+.5)/w,indexing='ij')
                coordinates.append(torch.stack((yy,xx),dim=-1).reshape(-1,2))
            positions=torch.cat(coordinates)
        return self.auxiliary['codec'](tokens,recipe['question'],recipe['recipient'],recipe['count'],
            layout=layout,query_offset=recipe.get('query_offset',0),positions=positions,spatial_prior=recipe.get('spatial_prior',True))

    def disclose(self,packet_ids,*,recipient,question,mode='compressed',elapsed_days=None,count=None):
        if mode=='tokens':return tuple(packet_ids)
        if self.auxiliary is None or mode not in {'compressed','temporal'}:raise ValueError('learned packet mode unavailable')
        if not packet_ids:raise ValueError('disclosure requires visual sources')
        count=self.compressed_tokens if count is None else count
        if mode=='temporal' and len(packet_ids)!=2:raise ValueError('temporal mode requires a registered pair')
        if mode=='temporal' and getattr(self,'spatial_codec',False):
            # A fused mean alone cannot identify which acquisition contained a
            # feature. Keep ordered source anchors within the SAME wire budget.
            if count not in (4,16,64) or count>self.compressed_tokens:
                raise ValueError('unsupported temporal slot budget')
            anchor=count//4
            before=self._store_disclosure([packet_ids[0]],recipient,question,'compressed',
                None,anchor,spatial=True,query_offset=0)
            after=self._store_disclosure([packet_ids[1]],recipient,question,'compressed',
                None,anchor,spatial=True,query_offset=anchor)
            fused=self._store_disclosure(packet_ids,recipient,question,mode,elapsed_days,
                count-2*anchor,spatial=True,query_offset=2*anchor)
            return (before,after,fused)
        if getattr(self,'spatial_codec',False) and getattr(self,'spatial_variant','both')!='position' and mode=='compressed':
            from .learned import spatial_allocations
            result=[];offset=0
            for source,slots in zip(packet_ids,spatial_allocations(count,len(packet_ids))):
                result.append(self._store_disclosure([source],recipient,question,mode,elapsed_days,slots,spatial=True,query_offset=offset))
                offset+=slots
            return tuple(result)
        return (self._store_disclosure(packet_ids,recipient,question,mode,elapsed_days,count,spatial=getattr(self,'spatial_codec',False)),)

    def _store_disclosure(self,packet_ids,recipient,question,mode,elapsed_days,count,*,spatial=False,query_offset=0):
        recipe={'sources':list(packet_ids),'recipient':self.roles.index(recipient),'question':question,
                'mode':mode,'elapsed_days':elapsed_days,'count':count,'spatial':spatial,'query_offset':query_offset,'spatial_prior':getattr(self,'spatial_variant','both')!='source'}
        with torch.no_grad():values=self._recipe_features(recipe).float().cpu()
        data=values.contiguous().numpy().tobytes();pid=f'addressed-{len(self.packets)}'
        side=int(count**.5);merge=self.base.config.vision_config.spatial_merge_size
        qh,qw=side,side
        if spatial:
            from .learned import spatial_grid
            grid=self.packets[packet_ids[0]].grid
            height,width=grid[1]//merge,grid[2]//merge
            if mode=='temporal':
                height=min(self.packets[p].grid[1]//merge for p in packet_ids)
                width=min(self.packets[p].grid[2]//merge for p in packet_ids)
            qh,qw=spatial_grid(count,height,width)
        source=self.packets[packet_ids[0]].artifact
        artifact=VisualArtifact(pid,ArtifactKind.LATENT,'application/x-evors-float32-le',tuple(values.shape),
            'float32',len(data),source.source_image_ref,source.roi,self.identity,'addressed_attention',
            'learned_codec:'+self.parameter_hash(),(recipient,),hashlib.sha256(data).hexdigest())
        self.store.put(artifact,data)
        self.packets[pid]=NativePacket(artifact,(1,qh*merge,qw*merge),
            f'{mode} evidence from '+', '.join(self.packets[p].label for p in packet_ids),recipe)
        return pid

    def _system_instruction(self, role, *, output_format="json"):
        """Prompt hook; the default preserves the historical messages verbatim."""
        responsibility = {
            "controller":"You schedule work. NEVER answer the image question. Select an action from the provided examples. Your output must contain the role field.",
            "visual":"You choose image observations or describe crops. Follow the current required action schema; do not substitute a final answer.",
            "knowledge":"Provide interpretation guidance with evidence references.",
            "reasoning":"Propose a short semantic answer with evidence references.",
            "verification":"Check the given candidate and return an accept, reject or needs_evidence decision with reasons."}[role]
        if self.paper_mechanisms:
            responsibility='Complete ONLY the currently assigned action. Use its requested schema, including intermediate requirement, pointer and fitness actions. Never substitute another action or expose reference labels.'
        format_instruction = "Return ONLY the requested JSON object without additional fields." if output_format=="json" else "Return only the requested concise text. Do not include JSON, markdown fences or internal IDs."
        return f"You are the {role} expert in a remote-sensing workflow. {responsibility} {format_instruction}"

    def prepare(self, role, prompt, packet_ids, *, output_format="json"):
        if role not in self.roles:
            raise ValueError("unknown workflow role")
        content = []
        for pid in packet_ids:
            packet = self.packets[pid]
            if role not in packet.artifact.compatible_consumers:
                raise PermissionError("packet not authorized for this consumer")
            content += [{"type":"text","text":packet.label},{"type":"image"}]
        content.append({"type":"text","text":prompt})
        messages = [{"role":"system","content":self._system_instruction(role, output_format=output_format)},
                    {"role":"user","content":content}]
        template_options={'enable_thinking':False} if qwen_backbone.family(self.base)=='qwen3_5' else {}
        rendered = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_options)
        if packet_ids:
            rendered = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**template_options)
            parts = rendered.split("<|image_pad|>")
            if len(parts) != len(packet_ids)+1:
                raise ValueError("chat template image placeholder mismatch")
            rendered = parts[0]+"".join("<|image_pad|>"*self.packets[pid].artifact.shape[0]+suffix
                                      for pid,suffix in zip(packet_ids,parts[1:]))
        ids = self.processor.tokenizer(rendered,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device)
        return ids

    def _model_inputs(self, ids, packet_ids, *, for_generation=False):
        embeds = self.model.get_input_embeddings()(ids)
        features = self._features(packet_ids)
        intervention=getattr(self,'payload_intervention','none')
        if features is not None and intervention!='none':
            if intervention=='zero':features=torch.zeros_like(features)
            elif intervention=='permute':features=features.flip(0)
            else:raise ValueError('unknown payload intervention')
        image_mask = ids == self.base.config.image_token_id
        if features is not None:
            if int(image_mask.sum()) != features.shape[0]:
                raise ValueError("image placeholders and transmitted visual tokens disagree")
            embeds = embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(embeds),features.to(embeds.dtype))
        elif image_mask.any():
            raise ValueError("image placeholders have no visual features")
        grids=torch.tensor([self.packets[p].grid for p in packet_ids],device=self.device) if packet_ids else None
        return qwen_backbone.native_inputs(self.base,ids,embeds,grids,for_generation=for_generation)

    def _pointer_grammar(self, specification, feasibility=None):
        from .evidence_pointer_grammar import PointerTokenLexicon, EvidencePointerGrammar
        if not hasattr(self, 'pointer_token_lexicon'):
            self.pointer_token_lexicon = PointerTokenLexicon(self.processor.tokenizer)
        return EvidencePointerGrammar(self.pointer_token_lexicon, specification, feasibility)

    def _field_grammar(self,fields):
        if getattr(self, 'field_decoder_backend', 'python') != 'python':
            from .compiled_json_fields import CompiledFieldGrammar
            from .json_fields import field_schema_key
            import xgrammar as xgr
            if not hasattr(self, 'compiled_field_compiler'):
                vocabulary = qwen_backbone.text_config(self.base).vocab_size
                info = xgr.TokenizerInfo.from_huggingface(self.processor.tokenizer,
                    vocab_size=vocabulary, stop_token_ids=[self.processor.tokenizer.eos_token_id])
                self.compiled_field_compiler = xgr.GrammarCompiler(info, max_threads=2)
                self.compiled_field_grammars = {}
            key = field_schema_key(fields)
            if key not in self.compiled_field_grammars:
                self.compiled_field_grammars[key] = CompiledFieldGrammar(
                    self.compiled_field_compiler, fields, qwen_backbone.text_config(self.base).vocab_size)
            return self.compiled_field_grammars[key]
        from .json_fields import JSONFieldGrammar,field_schema_key
        from .json_syntax import JSONSyntaxGrammar
        if not hasattr(self,'json_syntax_grammar'):self.json_syntax_grammar=JSONSyntaxGrammar(self.processor.tokenizer)
        if not hasattr(self,'json_field_grammars'):self.json_field_grammars={}
        key=field_schema_key(fields)
        if key not in self.json_field_grammars:self.json_field_grammars[key]=JSONFieldGrammar(self.processor.tokenizer,fields,self.json_syntax_grammar)
        return self.json_field_grammars[key]

    def decide(self, role, prompt, packet_ids=(), *, allowed_actions=None, output_format="json", evidence_pointer_constraint=None, pointer_feasibility=None,json_field_schema=None):
        if json_field_schema is not None:
            if allowed_actions or evidence_pointer_constraint is not None or output_format!='json':raise ValueError('field schema requires its own JSON response')
            prompt+='\nUse exactly these fields and types IN THIS ORDER at each declared object level: '+json.dumps(json_field_schema)+'. A type {"object": [...]} declares required nested fields; {"array": TYPE, "maximum": N} declares each item type and maximum length; {"tuple": [TYPE, ...]} declares an exact-length array. number and integer are JSON numeric scalars, never quoted strings or booleans. numeric_expression is an ASCII JSON string containing decimal/scientific numbers, + - * / ** %, parentheses, and sqrt/abs/min/max/round scalar calls; no variables or GIS operations. Values must come from your assigned task and evidence.'
        packet_ids = tuple(packet_ids)
        ids = self.prepare(role,prompt,packet_ids,output_format=output_format)
        started = time.perf_counter()
        self.model.eval()
        options = tuple(json.dumps(a,separators=(",",":")) for a in allowed_actions or ())
        processors = None
        stopping=None
        if evidence_pointer_constraint is not None:
            if options or output_format != 'json':
                raise ValueError('pointer constraint requires its own JSON response')
            from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
            grammar = self._pointer_grammar(evidence_pointer_constraint, pointer_feasibility)
            prompt_length = ids.shape[1]
            class PointerMask(LogitsProcessor):
                def __call__(self, current_ids, scores):
                    allowed = grammar.allowed(current_ids[0,prompt_length:].tolist())
                    masked = torch.full_like(scores,float('-inf'))
                    masked[0,list(allowed)] = scores[0,list(allowed)]
                    return masked
            class CompletePointer(StoppingCriteria):
                def __call__(self, current_ids, scores, **kwargs):
                    return grammar.complete(current_ids[0,prompt_length:].tolist())
            processors = LogitsProcessorList([PointerMask()])
            stopping = StoppingCriteriaList([CompletePointer()])
        elif options:
            from transformers import LogitsProcessor, LogitsProcessorList
            trie = ActionTrie(self.processor.tokenizer,options)
            prompt_length = ids.shape[1]
            class GrammarMask(LogitsProcessor):
                def __call__(self, current_ids, scores):
                    masked = torch.full_like(scores,float("-inf"))
                    allowed = trie.allowed(current_ids[0,prompt_length:].tolist())
                    masked[0,list(allowed)] = scores[0,list(allowed)]
                    return masked
            processors = LogitsProcessorList([GrammarMask()])
        elif output_format=='json':
            from transformers import StoppingCriteria,StoppingCriteriaList,LogitsProcessor,LogitsProcessorList
            tokenizer=self.processor.tokenizer;prompt_length=ids.shape[1]
            if self.paper_mechanisms or json_field_schema is not None:
                from .json_syntax import JSONSyntaxGrammar
                if not hasattr(self,'json_syntax_grammar'):self.json_syntax_grammar=JSONSyntaxGrammar(tokenizer)
                grammar=self._field_grammar(json_field_schema) if json_field_schema is not None else self.json_syntax_grammar
                syntax_token_budget=self.max_new_tokens
                class ObjectRootMask(LogitsProcessor):
                    def __call__(self,current_ids,scores):
                        allowed=grammar.allowed(current_ids[0,prompt_length:].tolist(),token_budget=syntax_token_budget)
                        if allowed is None:return scores
                        masked=torch.full_like(scores,float('-inf'));masked[0,list(allowed)]=scores[0,list(allowed)]
                        return masked
                processors=LogitsProcessorList([ObjectRootMask()])
            class CompleteJSONObject(StoppingCriteria):
                def __call__(self,current_ids,scores,**kwargs):
                    text=tokenizer.decode(current_ids[0,prompt_length:],skip_special_tokens=True).strip()
                    return complete_object(text) is not None
            stopping=StoppingCriteriaList([CompleteJSONObject()])
        with torch.no_grad():
            sampling_args = {"temperature":1.0,"top_p":1.0,"top_k":0} if self.sampling else {}
            # Qwen 4.51 projects the entire prefill into the vocabulary although
            # generation reads only logits[:, -1]. This can exceed 24GB on long
            # workflow context. Remove the hook before PPO likelihood replay.
            with qwen_backbone.generation_context(self.base,self.device,self.sampling):
                from .generation_failures import tracked_generate
                generated = tracked_generate(self,self._model_inputs(ids,packet_ids,for_generation=True),role=role,prompt=prompt,packet_ids=packet_ids,started=started,
                    max_new_tokens=self.max_new_tokens,do_sample=self.sampling,
                    repetition_penalty=1.0,use_cache=True,logits_processor=processors,stopping_criteria=stopping,
                    **sampling_args, **qwen_backbone.cache_options(self.base,self.device,self.sampling))
            output = generated[:,ids.shape[1]:].detach().cpu()
            text = self.processor.tokenizer.decode(output[0],skip_special_tokens=True)
            # Evaluate the unwarped on-policy log probabilities used by the trainer.
            record = TokenDecision(role,prompt,packet_ids,ids.detach().cpu(),output,
                torch.empty(0),text,(time.perf_counter()-started)*1000,self.identity,options)
            record.json_object_constraint=bool((self.paper_mechanisms or json_field_schema is not None) and output_format=='json' and not options and evidence_pointer_constraint is None)
            record.json_syntax_constraint=record.json_object_constraint
            record.json_syntax_token_budget=self.max_new_tokens if record.json_syntax_constraint else 0
            record.json_field_schema=json.loads(json.dumps(json_field_schema)) if json_field_schema is not None else None
            if json_field_schema is not None:
                record.json_field_backend = getattr(self, 'field_decoder_backend', 'python')
            if evidence_pointer_constraint is not None:
                record.evidence_pointer_constraint=json.loads(json.dumps(evidence_pointer_constraint))
                if pointer_feasibility is not None:
                    prefix=[]; masks=[]
                    for token in output[0].tolist():
                        allowed=list(grammar.allowed(prefix))
                        if token not in allowed:raise ValueError('rollout feasibility mask changed')
                        masks.append(allowed);prefix.append(token)
                    record.evidence_pointer_token_masks=masks
                    grammar.feasibility=None
            if self.sampling:
                record.old_logprobs = self.logprobs(record).detach().cpu()
        self.records.append(record)
        if output_format=='json' and not options:
            framed=complete_object(text)
            if framed is not None:return framed
        return text

    def logprobs(self, record):
        if getattr(record, 'json_field_schema', None) is not None:
            if getattr(record, 'json_field_backend', 'python') != getattr(self, 'field_decoder_backend', 'python'):
                raise ValueError('field decoder backend changed between rollout and likelihood replay')
        ids = torch.cat((record.input_ids,record.output_ids),dim=1).to(self.device)
        start = record.input_ids.shape[1]-1
        # Keep the entire decoder context, but project only positions predicting
        # response targets. Qwen 4.51 lacks logits_to_keep; projecting every prompt
        # state creates a large unused vocabulary tensor and its backward buffer.
        head=self.base.lm_head
        handle=head.register_forward_pre_hook(
            lambda _module,args:(args[0][:,start:-1],*args[1:]))
        try:
            result=self.model(**self._model_inputs(ids,record.packet_ids),use_cache=False)
        finally:
            handle.remove()
        logits=result.logits.float()
        return self._completion_logprobs(logits,record)

    def _completion_logprobs(self,logits,record):
        """Apply the saved reply distribution after selecting decoder positions."""
        targets = record.output_ids.to(self.device)
        if logits.shape[:2]!=targets.shape:raise ValueError('completion projection length mismatch')
        if getattr(record, 'evidence_pointer_token_masks', None) is not None:
            masks=record.evidence_pointer_token_masks
            if len(masks)!=targets.shape[1]:raise ValueError('missing pointer feasibility mask')
            mask=torch.full_like(logits,float('-inf'))
            for index,(token,allowed) in enumerate(zip(targets[0].tolist(),masks)):
                if not allowed or token not in allowed:raise ValueError('pointer outside saved feasibility mask')
                mask[0,index,allowed]=0
            logits=logits+mask
        elif getattr(record, 'evidence_pointer_constraint', None) is not None:
            grammar=self._pointer_grammar(record.evidence_pointer_constraint)
            prefix=[];mask=torch.full_like(logits,float('-inf'))
            for index,token in enumerate(targets[0].tolist()):
                allowed=grammar.allowed(prefix)
                if token not in allowed:
                    raise ValueError('recorded evidence pointer violates its rollout grammar')
                mask[0,index,list(allowed)]=0
                prefix.append(token)
            logits=logits+mask
        elif record.constraint_options:
            trie = ActionTrie(self.processor.tokenizer,record.constraint_options)
            prefix = []
            mask = torch.full_like(logits,float("-inf"))
            for index,token in enumerate(targets[0].tolist()):
                allowed = trie.allowed(prefix)
                if token not in allowed:
                    raise ValueError("recorded action is outside its grammar")
                mask[0,index,list(allowed)] = 0
                prefix.append(token)
            logits = logits+mask
        elif getattr(record,'json_syntax_constraint',False):
            from .json_syntax import JSONSyntaxGrammar
            if not hasattr(self,'json_syntax_grammar'):self.json_syntax_grammar=JSONSyntaxGrammar(self.processor.tokenizer)
            grammar=self._field_grammar(record.json_field_schema) if getattr(record,'json_field_schema',None) is not None else self.json_syntax_grammar
            prefix=[];mask=torch.full_like(logits,float('-inf'))
            for index,token in enumerate(targets[0].tolist()):
                allowed=grammar.allowed(prefix,token_budget=getattr(record,'json_syntax_token_budget',0) or None)
                if token not in allowed:raise ValueError('recorded action violates JSON syntax')
                mask[0,index,list(allowed)]=0
                prefix.append(token)
            logits=logits+mask
        elif record.json_object_constraint:
            if not hasattr(self,'object_root_grammar'):self.object_root_grammar=ObjectRootGrammar(self.processor.tokenizer)
            prefix=[];mask=torch.zeros_like(logits)
            for index,token in enumerate(targets[0].tolist()):
                allowed=self.object_root_grammar.allowed(prefix)
                if allowed is not None:
                    if token not in allowed:raise ValueError('recorded action violates JSON object root')
                    mask[0,index,:]=float('-inf');mask[0,index,list(allowed)]=0
                prefix.append(token)
            logits=logits+mask
        return torch.log_softmax(logits,dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name,p in self.model.named_parameters():
            if p.requires_grad or getattr(self, 'pretrained_only', False):
                digest.update(name.encode());digest.update(p.detach().float().cpu().numpy().tobytes())
        if self.auxiliary is not None:
            for name,p in self.auxiliary.named_parameters():
                digest.update(('auxiliary.'+name).encode());digest.update(p.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]+([] if self.auxiliary is None else list(self.auxiliary.parameters()))

    def value(self,features):
        if self.auxiliary is None:raise ValueError('state value requires paper mechanisms')
        return self.auxiliary['value'](torch.tensor(features,device=self.device))

    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.optimizer.state_dict(),str(path)+"/optimizer.pt")
        if self.auxiliary is not None:torch.save(self.auxiliary.state_dict(),str(path)+"/auxiliary.pt")
        from pathlib import Path
        (Path(path)/"policy.json").write_text(json.dumps({"model_identity":self.identity,
            "parameter_hash":self.parameter_hash(),"roles":self.roles,"paper_mechanisms":self.paper_mechanisms,
            "compressed_tokens":self.compressed_tokens,"spatial_codec":getattr(self,'spatial_codec',False),"spatial_variant":getattr(self,'spatial_variant',"both")},indent=2))

    def load(self, path, *, resume_training=False):
        """Load an adapter; resuming also restores the exact optimizer state."""
        if getattr(self, 'pretrained_only', False):
            raise ValueError('pretrained-only policy cannot load project checkpoints')
        from pathlib import Path
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        path = Path(path)
        metadata = json.loads((path/"policy.json").read_text())
        if bool(metadata.get('paper_mechanisms'))!=self.paper_mechanisms:raise ValueError('checkpoint mechanism configuration mismatch')
        if self.paper_mechanisms and metadata.get('compressed_tokens')!=self.compressed_tokens:raise ValueError('checkpoint codec specification mismatch')
        if bool(metadata.get('spatial_codec',False))!=getattr(self,'spatial_codec',False):raise ValueError('checkpoint spatial codec mismatch')
        if getattr(self,'spatial_codec',False) and metadata.get('spatial_variant','both')!=getattr(self,'spatial_variant','both'):raise ValueError('checkpoint spatial variant mismatch')
        if metadata["model_identity"]!=self.identity or tuple(metadata["roles"])!=self.roles:
            raise ValueError("checkpoint encoder/backbone or role identity mismatch")
        set_peft_model_state_dict(self.model,load_peft_weights(str(path),device=str(self.device)))
        if self.auxiliary is not None:self.auxiliary.load_state_dict(torch.load(path/'auxiliary.pt',map_location=self.device,weights_only=True))
        if self.parameter_hash()!=metadata["parameter_hash"]:
            raise ValueError("checkpoint adapter parameter hash mismatch")
        if resume_training:
            self.optimizer.load_state_dict(torch.load(path/"optimizer.pt",map_location=self.device,weights_only=True))
        self.loaded_checkpoint_receipt = {
            'path': str(path.resolve()), 'parameter_hash': metadata['parameter_hash'],
            'metadata_SHA256': hashlib.sha256((path/'policy.json').read_bytes()).hexdigest(),
            'optimizer_resumed': bool(resume_training)}
        return metadata
