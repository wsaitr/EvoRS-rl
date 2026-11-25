"""Real annotation artifacts and calibrated native raster tools; no label replay."""
import hashlib
import math
from pathlib import Path
from .benchmark_tools import BenchmarkToolbox

CALIBRATION_PROFILES = {
    # Explicit caller-selected assumptions from the public benchmark tool.
    # Never apply these merely because a filename contains NDVI or LST.
    'earth_modis_ndvi': (.0001, 0., 'dimensionless'),
    'earth_modis_lst': (.02, 0., 'K')}


def finite_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


class EvidenceBenchmarkToolbox(BenchmarkToolbox):
    def validate_visual_packets(self, packets):
        tokens = sum(self.policy.packets[p].artifact.shape[0] for p in packets)
        if tokens > 1024: raise ValueError('declared tool visual budget exceeded')

    def execute(self, action):
        tool = action['tool']
        if tool == 'annotate_boxes':
            from PIL import Image, ImageDraw
            from .evidence import _normalized_bbox
            name = action['asset']; boxes = action['boxes']
            if not isinstance(boxes, list) or not 1 <= len(boxes) <= 32:
                raise ValueError('one to 32 predicted boxes required')
            # Validate every prediction before writing any artifact.
            normalized = []
            for row in boxes:
                if set(row)-{'bbox', 'label'} or 'bbox' not in row: raise ValueError('invalid box fields')
                box = _normalized_bbox(row['bbox'])
                label = row.get('label', '')
                if not isinstance(label, str) or len(label) > 160: raise ValueError('invalid annotation label')
                normalized.append({'bbox': list(box), 'label': label})
            with Image.open(self.path(name)) as opened:
                if opened.mode not in {'RGB', 'RGBA', 'L', 'P'}: raise ValueError('annotation requires display imagery')
                original = opened.convert('RGB'); annotated = original.copy()
            draw = ImageDraw.Draw(annotated); width, height = original.size
            for row in normalized:
                x1, y1, x2, y2 = row['bbox']
                rectangle = [min(width-1, round(x1*width)), min(height-1, round(y1*height)),
                             min(width-1, round(x2*width)), min(height-1, round(y2*height))]
                draw.rectangle(rectangle, outline=(255, 0, 0), width=max(1, min(width, height)//200))
                if row['label']: draw.text((rectangle[0], rectangle[1]), row['label'], fill=(255, 0, 0))
            dest = self.output/f'annotation-{len(self.events):04d}.png'; annotated.save(dest)
            result = {'artifact': str(dest.resolve()), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                'source_asset': name, 'source_file_sha256': hashlib.sha256(self.path(name).read_bytes()).hexdigest(),
                'original_size': [width, height], 'boxes': normalized, 'artifact_type': 'predicted_box_overlay',
                'ground_truth_used': False}
        elif tool in {'calibrated_raster_info', 'calibrated_raster_statistics'}:
            result = self.raster(action)
        elif tool == 'raster_series_statistics':
            names = action['assets']
            if not isinstance(names, list) or not 1 <= len(names) <= 128 or len(set(names)) != len(names):
                raise ValueError('one to 128 distinct actual rasters required')
            result = {'series': [dict(asset=name, **self.raster({'tool': 'calibrated_raster_statistics',
                'asset': name, 'band': action.get('band', 1), 'calibration': action.get('calibration', 'metadata')}))
                for name in names], 'spatial_registration_assumed': False,
                'date_source': 'asset names retained; no invented acquisition dates'}
        else:
            return super().execute(action)
        self.events.append({'kind': 'executed_tool', 'action': action, 'output': result,
                            'origin': 'actual asset / deterministic transformation; no reference trace'})
        return result

    def raster(self, action):
        import rasterio
        import numpy as np
        with rasterio.open(self.path(action['asset'])) as source:
            metadata = {'width': source.width, 'height': source.height, 'bands': source.count,
                'crs': str(source.crs), 'transform': list(source.transform), 'bounds': list(source.bounds),
                'nodata': finite_or_none(source.nodata),
                'nodata_nonfinite': source.nodata is not None and not math.isfinite(source.nodata),
                'scales': [finite_or_none(v) for v in source.scales],
                'offsets': [finite_or_none(v) for v in source.offsets],
                'units': list(source.units), 'descriptions': list(source.descriptions),
                'dtypes': list(source.dtypes), 'tags': source.tags(),
                'affine_pixel_area_crs_units_squared': abs(source.transform.a*source.transform.e-source.transform.b*source.transform.d),
                'metric_area_inferred': False}
            if action['tool'] == 'calibrated_raster_info': return metadata
            band = action.get('band', 1); calibration = action.get('calibration', 'metadata')
            if type(band) != int or not 1 <= band <= source.count: raise ValueError('invalid band')
            if calibration not in {'metadata', 'raw', *CALIBRATION_PROFILES}: raise ValueError('unknown explicit calibration')
            profile = CALIBRATION_PROFILES.get(calibration)
            scale, offset = profile[:2] if profile else ((source.scales[band-1], source.offsets[band-1])
                                                        if calibration == 'metadata' else (1., 0.))
            if not math.isfinite(scale) or not math.isfinite(offset): raise ValueError('nonfinite calibration metadata')
            count = 0; mean = 0.; m2 = 0.; total = 0.; lower = math.inf; upper = -math.inf
            # Merge block moments to avoid catastrophic variance cancellation.
            for y in range(0, source.height, 512):
                for x in range(0, source.width, 512):
                    window = rasterio.windows.Window(x, y, min(512, source.width-x), min(512, source.height-y))
                    values = source.read(band, window=window, masked=True).astype('float64').compressed()
                    values = values[np.isfinite(values)]*scale+offset
                    values = values[np.isfinite(values)]
                    if not values.size: continue
                    n = int(values.size); block_mean = float(values.mean()); delta = block_mean-mean
                    m2 += float(np.square(values-block_mean).sum())+delta*delta*count*n/(count+n)
                    mean += delta*n/(count+n); count += n; total += float(values.sum())
                    lower = min(lower, float(values.min())); upper = max(upper, float(values.max()))
            if not count: raise ValueError('no valid pixels')
            return {**metadata, 'band': band, 'calibration': calibration, 'applied_scale': scale,
                'applied_offset': offset, 'physical_units': profile[2] if profile else source.units[band-1] or 'unknown',
                'calibration_profile_source': 'Earth-Agent ca1dfaef750654331308a434966d307f1728fd90; Index.py compute_tvdi' if profile else 'raster metadata or raw',
                'profile_applicability_verified': False if profile else None,
                'units_inferred_from_filename': False, 'valid_pixels': count, 'mean': mean, 'sum': total,
                'min': lower, 'max': upper, 'std': math.sqrt(m2/count)}
