"""Real benchmark tools. Never replay annotated bounding boxes or tool outputs."""
import ast
import math
import operator
from pathlib import Path


def calculate(expression, variables=None, *, rounding=False):
    """A bounded arithmetic AST; no eval, attributes, imports or filesystem access."""
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                 ast.USub: operator.neg, ast.UAdd: operator.pos}
    functions = {'sqrt': lambda x: x**.5, 'abs': abs, 'min': min, 'max': max}
    if rounding:
        def bounded_round(value, digits=None):
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError('round requires a finite scalar')
            if digits is not None and (type(digits) != int or abs(digits) > 12):
                raise ValueError('invalid rounding precision')
            return round(value) if digits is None else round(value, digits)
        functions['round'] = bounded_round
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ValueError('invalid arithmetic expression; use numeric values and supported operators') from error
    if len(list(ast.walk(tree))) > 128:
        raise ValueError('expression too long')
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            if not math.isfinite(node.value) or abs(node.value) > 1e12:
                raise ValueError('invalid numeric literal')
            return node.value
        if isinstance(node, ast.Name) and node.id in (variables or {}):
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (not isinstance(right, (int, float)) or abs(right) > 8):
                raise ValueError('unbounded exponent')
            return operators[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
            return functions[node.func.id](*(visit(a) for a in node.args))
        raise ValueError('unsupported arithmetic syntax')
    value = visit(tree)
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError('nonfinite result')
    return value


class BenchmarkToolbox:
    """Task-local asset authorization; native raster values retain georeferencing."""
    def __init__(self, assets, policy, *, output):
        self.assets = {k: Path(v).resolve() for k, v in assets.items()}
        self.policy = policy
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = []

    def path(self, name):
        if name not in self.assets:
            raise ValueError('asset is not authorized for this task')
        path = self.assets[name]
        if not path.is_file():
            raise ValueError('asset has not been downloaded')
        return path

    def detection_prompt(self, question, packets, sizes):
        return question+'\nReturn JSON {"objects":[{"label":"category","bbox":[x1,y1,x2,y2]}]}. Boxes are normalized in the ORIGINAL image. Return up to 32 visible objects; do not guess invisible objects.'

    def parse_detections(self, raw, packets, sizes):
        from .runtime import strict_json, schema
        from .evidence import _normalized_bbox
        result = strict_json(raw); schema(result, {'objects'})
        if not isinstance(result['objects'], list) or len(result['objects']) > 32:
            raise ValueError('invalid detections')
        for row in result['objects']:
            schema(row, {'label', 'bbox'}); _normalized_bbox(row['bbox'])
        result['original_image_sizes'] = sizes
        return result

    def execute(self, action):
        tool = action['tool']
        if tool == 'list_assets':
            result = list(self.assets)
        elif tool == 'calculate':
            result = calculate(action['expression'])
        elif tool in {'inspect', 'detect'}:
            from PIL import Image
            names = action.get('assets') or [action['asset']]
            if not 1 <= len(names) <= 4:
                raise ValueError('inspect accepts one to four actual views')
            if tool == 'detect' and len(names) != 1:
                raise ValueError('detect requires one explicit source image')
            packets = []
            sizes = []
            for name in names:
                with Image.open(self.path(name)) as im:
                    if im.mode not in {'RGB', 'RGBA', 'L', 'P'}:
                        raise ValueError('scientific raster requires native numeric tools')
                    im = im.convert('RGB'); sizes.append(list(im.size))
                    pid = f'benchmark-view-{len(self.policy.packets)}'
                    self.policy.encode(im, artifact_id=pid, source_image_ref=str(self.path(name)),
                                       roi=(0, 0, 1, 1), label=name)
                    packets.append(pid)
            prompt = action.get('question', 'Describe the question-relevant visible facts.')
            if tool == 'detect':
                prompt = self.detection_prompt(prompt, packets, sizes)
            validator = getattr(self, 'validate_visual_packets', None)
            if validator is not None: validator(packets)
            raw = self.policy.decide('visual', prompt, tuple(packets), output_format='json' if tool == 'detect' else 'text')
            if tool == 'detect':
                result = self.parse_detections(raw, packets, sizes)
            else:
                result = {'observations': raw, 'original_image_sizes': sizes}
        elif tool in {'raster_info', 'raster_statistics', 'raster_expression'}:
            import rasterio
            import numpy as np
            from contextlib import ExitStack
            names = action.get('assets') or [action['asset']]
            with ExitStack() as stack:
                sources = [stack.enter_context(rasterio.open(self.path(n))) for n in names]
                first = sources[0]
                metadata = {'width': first.width, 'height': first.height, 'bands': first.count,
                            'crs': str(first.crs), 'transform': list(first.transform), 'nodata': first.nodata}
                if tool == 'raster_info':
                    result = metadata
                else:
                    if any((s.width, s.height, s.crs, s.transform) != (first.width, first.height, first.crs, first.transform) for s in sources):
                        raise ValueError('unaligned rasters: explicit registration required')
                    bands = action.get('bands', [1]*len(sources))
                    if len(bands) != len(sources) or any(type(b) != int or not 1 <= b <= s.count for b, s in zip(bands, sources)):
                        raise ValueError('invalid raster bands')
                    if len(sources) > 6:
                        raise ValueError('at most six raster operands')
                    count = 0; total = 0.; squares = 0.; lower = math.inf; upper = -math.inf
                    for y in range(0, first.height, 512):
                        for x in range(0, first.width, 512):
                            window = rasterio.windows.Window(x, y, min(512, first.width-x), min(512, first.height-y))
                            operands = {chr(97+i): s.read(b, window=window, masked=True).astype('float64')
                                        for i, (s, b) in enumerate(zip(sources, bands))}
                            values = calculate(action['expression'], operands) if tool == 'raster_expression' else operands['a']
                            values = np.ma.asarray(values).compressed(); values = values[np.isfinite(values)]
                            if values.size:
                                count += values.size; total += float(values.sum()); squares += float(np.square(values).sum())
                                lower = min(lower, float(values.min())); upper = max(upper, float(values.max()))
                    if not count: raise ValueError('no valid raster pixels')
                    result = {**metadata, 'valid_pixels': int(count), 'min': lower, 'max': upper,
                              'mean': total/count, 'sum': total, 'std': math.sqrt(max(0., squares/count-(total/count)**2))}
        else:
            raise ValueError('unknown benchmark tool')
        self.events.append({'kind': 'executed_tool', 'action': action, 'output': result,
                            'origin': 'actual asset / model / deterministic arithmetic; never reference trace'})
        return result
