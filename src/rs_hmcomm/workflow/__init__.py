"""Whole-workflow learning, dated evidence and native visual-token communication.

Torch/Transformers are imported only by the model/trainer modules.
"""
from .evidence import ImageFrame, WorkflowTask, load_workflow_manifest, paired_crops

__all__ = ["ImageFrame", "WorkflowTask", "load_workflow_manifest", "paired_crops"]
