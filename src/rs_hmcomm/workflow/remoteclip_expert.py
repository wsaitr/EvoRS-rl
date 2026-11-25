"""Frozen author RemoteCLIP crop diagnostic; cosine scores are uncalibrated.

No label correction, benchmark reference input or workflow benefit is implied.
"""
import hashlib
import json
from pathlib import Path
import time

TEMPLATES = ('a satellite image of a {}.', 'an aerial image of a {}.',
             'a {} seen from above.')


def validate_labels(labels):
    if (not isinstance(labels, list) or not 2 <= len(labels) <= 64
            or any(not isinstance(label, str) or not label.strip() or len(label) > 80 for label in labels)
            or len({label.strip().casefold() for label in labels}) != len(labels)):
        raise ValueError('two to 64 distinct, explicit nonempty class labels required')


class RemoteClipExpert:
    def __init__(self, model_path):
        import torch
        import open_clip
        path = Path(model_path)
        metadata = json.loads((path/'evors_remoteclip_provenance.json').read_text())
        record = next(row for row in metadata['files'] if row['file'] == 'RemoteCLIP-ViT-B-32.pt')
        digest = hashlib.sha256()
        with (path/record['file']).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError('RemoteCLIP weights changed')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
        state = torch.load(path/record['file'], map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to('cpu').eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.identity = {'repository': metadata['repository'], 'revision': metadata['revision'],
            'weight_sha256': record['sha256'], 'architecture': 'ViT-B-32', 'device': 'cpu',
            'pretraining_overlap_audit_pending': True}

    def classify(self, image, labels):
        import torch
        validate_labels(labels)
        started = time.perf_counter()
        prompts = [template.format(label) for label in labels for template in TEMPLATES]
        with torch.inference_mode():
            text = self.model.encode_text(self.tokenizer(prompts))
            text = torch.nn.functional.normalize(text.float(), dim=-1)
            bank = torch.nn.functional.normalize(text.reshape(len(labels), len(TEMPLATES), -1).mean(1), dim=-1)
            visual = self.model.encode_image(self.preprocess(image.convert('RGB')).unsqueeze(0))
            visual = torch.nn.functional.normalize(visual.float(), dim=-1)
            similarities = (visual @ bank.T)[0]
        if not torch.isfinite(similarities).all():
            raise ValueError('nonfinite crop similarities')
        values = similarities.tolist(); order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
        return {'labels': labels, 'cosine_similarities': values, 'highest_similarity_label': labels[order[0]],
            'top_two_margin': values[order[0]]-values[order[1]], 'scores_are_calibrated_probabilities': False,
            'templates': list(TEMPLATES), 'model_identity': self.identity,
            'latency_ms': (time.perf_counter()-started)*1000, 'reference_used': False,
            'ground_truth_claim': False}
