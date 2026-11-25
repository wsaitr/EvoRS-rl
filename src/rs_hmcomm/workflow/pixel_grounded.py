"""Explicit native-canvas localization, never guessed coordinate conventions."""
import math
from .budgeted_assessment import BudgetedAssessmentRuntime
from .runtime import schema
from .object_labels import usable_object_label


def normalized_pixel_objects(value, canvases):
    schema(value, {'objects'})
    if not isinstance(value['objects'],list) or len(value['objects'])>4:raise ValueError('invalid object list')
    result=[]
    for row in value['objects']:
        schema(row,{'category','crop_index','bbox_pixels'})
        category=row['category'];index=row['crop_index'];bbox=row['bbox_pixels']
        if not usable_object_label(category):
            raise ValueError('ungrounded object category')
        if type(index)!=int or not 0<=index<len(canvases):raise ValueError('unknown actual canvas')
        width,height=canvases[index]
        if not isinstance(bbox,list) or len(bbox)!=4 or any(type(v) not in {int,float} or not math.isfinite(v) for v in bbox):
            raise ValueError('invalid pixel box')
        x1,y1,x2,y2=bbox
        if not 0<=x1<x2<=width or not 0<=y1<y2<=height:raise ValueError('pixel box outside actual canvas')
        result.append({'category':category.strip(),'crop_index':index,'bbox':[x1/width,y1/height,x2/width,y2/height]})
    return {'objects':result}


class PixelGroundedRuntime(BudgetedAssessmentRuntime):
    def _call(self, role, objective, packet_ids=(), context=None, **kwargs):
        if role!='visual' or 'Extract at most FOUR question-relevant visible objects.' not in objective:
            return super()._call(role,objective,packet_ids,context,**kwargs)
        patch=self.policy.base.config.vision_config.patch_size
        canvases=[]
        for pid in packet_ids:
            packet=self.policy.packets[pid]
            if packet.recipe is not None:raise ValueError('object localization requires an actual native view')
            _,height,width=packet.grid
            canvases.append((width*patch,height*patch))
        objective=('Localize at most four question-relevant objects in the ACTUAL attached views. '
            'Return JSON with key objects, a list of records containing category (the specific class visible in the image; never copy the field description), crop_index '
            '(zero-based attached view), and bbox_pixels (x1,y1,x2,y2 in that view\'s declared pixel canvas). '
            'Use the canvas dimensions in context. Do not use normalized coordinates or a guessed universal canvas. '
            'Return an empty objects list when reliable localization is impossible. This is not an exhaustive count.')
        context={**(context or {}),'actual_pixel_canvases':[
            {'crop_index':index,'width':width,'height':height} for index,(width,height) in enumerate(canvases)]}
        kwargs['json_field_schema']=[['objects',{'array':{'object':[
            ['category','string'],['crop_index','integer'],
            ['bbox_pixels',{'tuple':['number','number','number','number']}]]},'maximum':4}]]
        value=super()._call(role,objective,packet_ids,context,**kwargs)
        result=normalized_pixel_objects(value,canvases)
        self.events.append({'kind':'native_pixel_coordinate_conversion','actual_canvases':canvases,
                            'normalization_guessed':False,'reference_used':False})
        return result
