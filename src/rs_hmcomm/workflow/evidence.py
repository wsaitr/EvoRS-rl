"""Dated observations and registered crops; no labels enter this module."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image
from rs_hmcomm.spatial.representation import _normalize_bbox


def _normalized_bbox(value):
    result = _normalize_bbox(value)
    if result is None:
        raise ValueError("bbox is required")
    return result


@dataclass(frozen=True)
class ImageFrame:
    frame_id: str
    timestamp: str
    image: Image.Image
    # Maps normalized reference coordinates to normalized coordinates in this frame.
    reference_to_frame: tuple[float, ...] | None = None
    alignment_confidence: float = 0.0
    source: str = ""

    def __post_init__(self):
        if not self.frame_id or not self.timestamp or not isinstance(self.image, Image.Image):
            raise ValueError("frame requires an ID, timestamp and PIL image")
        if not math.isfinite(self.alignment_confidence) or not 0 <= self.alignment_confidence <= 1:
            raise ValueError("alignment confidence must be in [0,1]")
        if self.reference_to_frame is not None:
            m = tuple(float(x) for x in self.reference_to_frame)
            if len(m) != 6 or not all(math.isfinite(x) for x in m):
                raise ValueError("reference_to_frame must be six finite affine coefficients")
            if abs(m[0]*m[4]-m[1]*m[3]) < 1e-8:
                raise ValueError("singular alignment")
            object.__setattr__(self, "reference_to_frame", m)

    def descriptor(self):
        return {"frame_id": self.frame_id, "timestamp": self.timestamp,
                "size": list(self.image.size), "source": self.source,
                "sha256": image_digest(self.image),
                "reference_to_frame": self.reference_to_frame,
                "alignment_confidence": self.alignment_confidence}


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def image_bytes(image):
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def crop_frame(frame: ImageFrame, bbox):
    x1,y1,x2,y2 = _normalized_bbox(bbox)
    w,h = frame.image.size
    return frame.image.crop((math.floor(x1*w),math.floor(y1*h),math.ceil(x2*w),math.ceil(y2*h)))


def observation_encodable(frames, action, *, min_side=28):
    """Check native crop sizes before offering a tool action to the policy.

    No enlargement or synthetic detail is introduced. Paired crops use the
    reference ROI dimensions used by paired_crops; grid cells are checked
    separately, including floor/ceil pixel rounding.
    """
    frame_by_id={f.frame_id:f for f in frames}
    x1,y1,x2,y2=_normalized_bbox(action['bbox']);op=action['op']
    if op=='paired_crop':
        return all(math.ceil((x2-x1)*f.image.width)>=min_side and
                   math.ceil((y2-y1)*f.image.height)>=min_side for f in frames)
    frame=frame_by_id[action['frame_id']]
    boxes=[(x1,y1,x2,y2)]
    if op=='grid':
        xm,ym=(x1+x2)/2,(y1+y2)/2
        boxes=[(x1,y1,xm,ym),(xm,y1,x2,ym),(x1,ym,xm,y2),(xm,ym,x2,y2)]
    return all(math.ceil(c*frame.image.width)-math.floor(a*frame.image.width)>=min_side and
               math.ceil(d*frame.image.height)-math.floor(b*frame.image.height)>=min_side
               for a,b,c,d in boxes)


def paired_crops(frames: tuple[ImageFrame, ...], bbox, *, min_confidence=0.8):
    """Warp the SAME reference ROI into every date; reject uncertain alignment.

    Known registration may be supplied by a dataset or a separate estimator.
    Identity is never assumed merely because image dimensions match.
    """
    x1,y1,x2,y2 = _normalized_bbox(bbox)
    if len(frames) < 2 or len({f.timestamp for f in frames}) != len(frames):
        raise ValueError("temporal pairing requires at least two distinct dates")
    result = []
    for frame in frames:
        if frame.reference_to_frame is None or frame.alignment_confidence < min_confidence:
            raise ValueError(f"alignment unavailable/uncertain: {frame.frame_id}")
        a,b,c,d,e,f = frame.reference_to_frame
        points = [(a*x+b*y+c,d*x+e*y+f) for x,y in ((x1,y1),(x1,y2),(x2,y1),(x2,y2))]
        if any(not (0<=px<=1 and 0<=py<=1) for px,py in points):
            raise ValueError("paired ROI extends outside a source frame")
        w,h = frame.image.size
        ow,oh = max(1,math.ceil((x2-x1)*w)), max(1,math.ceil((y2-y1)*h))
        coefficients = (a*(x2-x1)*w/ow,b*(y2-y1)*w/oh,(a*x1+b*y1+c)*w,
                        d*(x2-x1)*h/ow,e*(y2-y1)*h/oh,(d*x1+e*y1+f)*h)
        image = frame.image.transform((ow,oh),Image.Transform.AFFINE,coefficients,
                                      resample=Image.Resampling.BILINEAR)
        native_bbox = (min(x for x,y in points),min(y for x,y in points),
                       max(x for x,y in points),max(y for x,y in points))
        result.append((frame,image,native_bbox))
    return tuple(result)


@dataclass(frozen=True)
class WorkflowTask:
    sample_id: str
    question: str
    frames: tuple[ImageFrame, ...]
    scene_id: str
    answer_format: str = "free_text"

    def __post_init__(self):
        if not self.sample_id or not self.question or not self.scene_id or not self.frames:
            raise ValueError("task requires sample_id, question, scene_id and frames")
        if len({f.frame_id for f in self.frames}) != len(self.frames):
            raise ValueError("duplicate frame IDs")
        if self.answer_format not in {"free_text","yes_no","integer","choice","bbox"}:
            raise ValueError("unsupported answer format")


def infer_answer_format(question):
    """Question-only protocol; never infer an output type from a reference label."""
    import re
    if re.match(r"^\s*how many\b",question,re.I):
        return "integer"
    if re.match(r"^\s*(is|are|was|were|does|do|did|has|have|can)\b",question,re.I):
        # Alternative questions ("Is X larger or smaller?") are not polar
        # questions. A permissive phrase format also admits Yes/No when "or"
        # connects objects in an otherwise polar query; never force a wrong type.
        if re.search(r"\bor\b(?!\s+not\b)",question,re.I):
            return "free_text"
        return "yes_no"
    return "free_text"


def load_workflow_manifest(path: str | Path):
    """Yield (label-free task, reference). References are only used by trainer/scorer.

    Multi-date rows contain frames:[{frame_id,timestamp,image,reference_to_frame,
    alignment_confidence}], and scene_id. Single-image legacy manifests work too.
    """
    import json
    path = Path(path).resolve()
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row["sample_id"])
        if sid in seen:
            raise ValueError("duplicate sample_id")
        seen.add(sid)
        values = row.get("frames") or [{"frame_id":"t0","timestamp":"unspecified",
                                       "image":row.get("image_ref",row.get("image"))}]
        frames = []
        for value in values:
            image_path = Path(value["image"])
            if not image_path.is_absolute():
                image_path = path.parent/image_path
            with Image.open(image_path) as opened:
                image = opened.convert("RGB").copy()
            frames.append(ImageFrame(value["frame_id"],value["timestamp"],image,
                                     value.get("reference_to_frame"),value.get("alignment_confidence",0),
                                     str(image_path.resolve())))
        answer_format=row.get("answer_format",infer_answer_format(row["question"]))
        if answer_format in {"short_phrase","short_text"}:answer_format="free_text"
        yield (WorkflowTask(sid,row["question"],tuple(frames),str(row.get("scene_id",frames[0].source)),
                            answer_format),
               row.get("reference",row.get("reference_answer",row.get("answer",""))))


def check_disjoint(train, evaluation):
    """Reject scene/source/pixel overlap including temporal/crop derivatives."""
    def keys(rows):
        scenes = {task.scene_id for task,_ in rows}
        sources = {f.source for task,_ in rows for f in task.frames if f.source}
        pixels = {image_digest(f.image) for task,_ in rows for f in task.frames}
        return scenes,sources,pixels
    left,right = keys(train),keys(evaluation)
    if any(a & b for a,b in zip(left,right)):
        raise ValueError("training/evaluation scenes or source images overlap")
