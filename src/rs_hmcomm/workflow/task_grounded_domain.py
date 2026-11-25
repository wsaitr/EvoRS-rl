"""Task-aligned requirements and existing certificate-based first attempts."""
from .remote_sensing_analysis import RemoteSensingAnalysisRuntime
from .early_reasoning import EarlyReasoningRuntime


class TaskGroundedDomainRuntime(RemoteSensingAnalysisRuntime, EarlyReasoningRuntime):
    def _controller_role_choices(self, choices, context):
        legal = super()._controller_role_choices(choices, context)
        # Reuse source-bound first-attempt/correction logic, not an answer shortcut.
        return EarlyReasoningRuntime._controller_role_choices(self, legal, context)

    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        if role == 'controller' and 'Write ONE distinct visual fact' in objective:
            objective = objective.replace('ONE distinct visual fact', 'ONE distinct task-relevant evidence requirement')
            objective += (
                ' Cover every explicitly requested output, not just one subquestion. For '
                'multi-output tasks separate localization or segmentation, object inventory, '
                'thresholded counts, physical unit conversion and fractions when requested. '
                'A fraction requires both its numerator and a source-supported denominator. '
                'Every/all/total targets require global coverage, even when each object is '
                'small; a local crop does not establish a complete inventory. Distinguish '
                'actual mask pixel area from bounding-box area and measured values from '
                'visual estimates. Do not invent measurements or additional deliverables. '
                'Distinguish identifying a depicted spacecraft from identifying the sensor '
                'that acquired an Earth image and from comparing spectral-band properties. '
                'Use task-relevant visible design, labels, acquisition metadata or physical '
                'properties as appropriate; do not require an unmentioned wavelength or '
                'assume that a displayed RGB image reveals its original spectral response. '
                'If spectral properties are explicitly requested, identify the relevant '
                'band or acquisition property without inventing its measured value. '
                'For distances distinguish object localization, public scale and path versus '
                'straight-line geometry. For change require corresponding dated regions. '
                'Do not assume any option is correct or any object is present.')
        elif role == 'controller' and 'Does a distinct requirement remain?' in objective:
            objective += (' Check that the requirements cover the requested quantity or '
                          'choice among statements, rather than only target visibility. Check each '
                          'explicitly requested output and its scope; do not terminate '
                          'decomposition after satisfying only a count when the question '
                          'also asks for a conversion or percentage. '
                          'Do not add an answer, speculative fact or redundant paraphrase.')
        return super()._call(role, objective, packet_ids, context, **kwargs)
