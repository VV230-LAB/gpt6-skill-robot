"""Read-only nominal camera arithmetic extracted from frozen R3C1 helpers.

USD camera axes are +x right, +y up and -z forward. Coordinates are metres
in environment_origin, using current measured link6 EEF poses. Pixel matches
and any plane height come from the caller. Nominal geometry and numeric
quality do not establish object identity, contact, calibration or action authority.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

from tools.geometry_config import load_config


CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def vector(value, length):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (length,) or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw):
        raise ValueError(f'Expected {length} finite numbers')
    try:
        v = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f'Expected {length} finite numbers') from error
    if not np.isfinite(v).all():
        raise ValueError(f'Expected {length} finite numbers')
    return v


def pose_matrix(position, quaternion_wxyz):
    p = vector(position, 3)
    q = vector(quaternion_wxyz, 4)
    norm = math.hypot(*q)
    if abs(norm - 1) > 1e-3:
        raise ValueError('Expected unit wxyz quaternion')
    w, x, y, z = q / norm
    t = np.eye(4)
    t[:3, 3] = p
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                 [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                 [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
    return t


def intrinsics(camera, config):
    if camera not in CAMERAS:
        raise ValueError('Unsupported camera')
    return np.array(config['cameras'][camera]['K'], dtype=float)


def camera_matrix(packet, camera, config):
    intrinsics(camera, config)
    if camera == 'cam_high':
        return np.array(config['head_to_environment_usd'], dtype=float)
    eef = packet['current_eef']['left' if camera == 'cam_left_wrist' else 'right']
    if eef.get('frame') != config['eef_frame'] or eef.get('link') != config['eef_link']:
        raise ValueError('Expected measured configured EEF frame/link')
    return pose_matrix(eef['position'], eef['quaternion_wxyz']) @ np.array(config['eef_to_wrist_usd'])


def png_size(path):
    """Read dimensions only after checking PNG integrity and decoding RGB pixels."""
    try:
        with Image.open(path) as image:
            if image.format != 'PNG' or image.mode != 'RGB':
                raise ValueError('Expected a public RGB PNG image')
            image.verify()
        # verify() checks chunks/checksums; load() also checks compressed pixel data.
        with Image.open(path) as image:
            image.load()
            return list(image.size)
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError('Cannot decode public RGB PNG image: ' + str(error)) from error


def indexed_paths(root, pattern):
    paths = []
    request = pattern.startswith('request')
    for p in root.glob(pattern):
        match = re.fullmatch(r'request_(\d+)\.json', p.name) if request else re.fullmatch(r'\d+', p.parent.name)
        if match:
            paths.append((int(match.group(1) if request else match.group()), p))
    if not paths:
        raise ValueError('Missing current public request or observation')
    if len({i for i, _ in paths}) != len(paths):
        raise ValueError('Ambiguous public request or observation index')
    return max(paths, key=lambda item: item[0])


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError('Expected a nonempty public file path')
    return Path(value)


def _read_object(path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite JSON number: ' + value)
        return number
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Ambiguous duplicate public JSON key: ' + key)
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(), parse_constant=invalid_constant,
                           parse_float=finite_float, object_pairs_hook=unique_object)
    except RecursionError as error:
        raise ValueError('Public JSON nesting exceeds the supported parser depth') from error
    if not isinstance(value, dict):
        raise ValueError('Expected a public JSON object')
    return value


def load_current(controller_root, *, episode_id, request_id, step_id, config, packet_path=None):
    """Read a fresh native RoboDojo packet; never query a controller or simulator.

    The caller must explicitly load a configuration with load_config(path).
    This adapter supports only the three native cameras and environment_origin/link6.
    """
    if (not isinstance(config, dict) or config.get('robot_profile') != 'robodojo'
            or config.get('eef_frame') != 'environment_origin' or config.get('eef_link') != 'link6'
            or not isinstance(config.get('cameras'), dict) or set(config['cameras']) != set(CAMERAS)):
        raise ValueError('Unsupported configured public observation adapter')
    if (not isinstance(episode_id, str) or not episode_id or not isinstance(request_id, str) or not request_id
            or type(step_id) is not int or step_id < 0):
        raise ValueError('Explicit nonempty episode/request and integer step binding required')
    root = _path(controller_root).resolve()
    index, request_path = indexed_paths(root, 'request_*.json')
    obs_index, observation_path = indexed_paths(root / 'observations', '*/observation.json')
    if index != obs_index:
        raise ValueError('Latest observation has no current request; episode may be ended')
    expected_directory = root / 'observations' / f'{index:03d}'
    expected_observation = expected_directory / 'observation.json'
    expected_request = root / f'request_{index:03d}.json'
    # Compare against lexical expected paths to reject aliases of the current directory.
    if observation_path.resolve() != expected_observation or request_path.resolve() != expected_request:
        raise ValueError('Public path alias escapes current observation/request')
    if packet_path is not None and _path(packet_path).resolve() != expected_observation:
        raise ValueError('Packet is not the current observation')
    request = _read_object(request_path)
    packet = _read_object(observation_path)
    binding = dict(episode_id=episode_id, request_id=request_id, step_id=step_id)
    for obj in (request, packet):
        if any(obj.get(k) != v or (k == 'step_id' and type(obj.get(k)) is not int) for k, v in binding.items()):
            raise ValueError('Current episode/request/step binding mismatch')
        if obj.get('robot_profile') != config['robot_profile'] or obj.get('rollout_finished') is not False:
            raise ValueError('Expected live robodojo public observation')
        if _path(obj.get('observation_path')).resolve() != expected_observation:
            raise ValueError('Observation path binding mismatch')
    if _path(packet.get('request_path')).resolve() != expected_request:
        raise ValueError('Request path binding mismatch')
    for field in ('current_eef', 'images', 'task'):
        if request.get(field) != packet.get(field):
            raise ValueError('Request and observation public state disagree')
    if 'task' in packet and (not isinstance(packet['task'], str) or not packet['task'].strip()):
        raise ValueError('Expected a nonempty public task name')
    eefs = packet.get('current_eef')
    if not isinstance(eefs, dict):
        raise ValueError('Expected current measured EEF objects')
    for arm in ('left', 'right'):
        eef = eefs.get(arm)
        if not isinstance(eef, dict) or eef.get('frame') != config['eef_frame'] or eef.get('link') != config['eef_link']:
            raise ValueError('Wrong EEF frame/link')
        pose_matrix(eef.get('position'), eef.get('quaternion_wxyz'))
    images = packet.get('images')
    if (not isinstance(images, list) or len(images) != 3 or any(not isinstance(x, dict) for x in images)
            or any(x.get('camera') not in CAMERAS for x in images)
            or {x.get('camera') for x in images} != set(CAMERAS)):
        raise ValueError('Expected the three synchronized public RGB cameras')
    for item in images:
        expected = expected_directory / (item['camera'] + '.png')
        if (_path(item.get('path')).resolve() != expected or expected.resolve() != expected
                or png_size(expected) != config['cameras'][item['camera']]['resolution']):
            raise ValueError('RGB must be the current configured-resolution PNG without aliases')
    current = {k: packet[k] for k in ('episode_id', 'request_id', 'step_id', 'current_eef', 'images', 'observation_path')}
    if 'task' in packet:
        current['task'] = packet['task']
    return current


def original_pixel(packet, camera, uv, pixel_space, config):
    intrinsics(camera, config)
    uv = vector(uv, 2)
    native_size = config['cameras'][camera]['resolution']
    size = native_size
    preview_path = None
    if pixel_space == 'preview':
        matches = [x for x in packet['images'] if x['camera'] == camera]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous current camera')
        item = matches[0]
        directory = _path(packet['observation_path']).resolve().parent
        if 'codex_preview' in item:
            descriptor = item['codex_preview']
            if not isinstance(descriptor, dict) or not descriptor:
                raise ValueError('Expected a current preview descriptor object')
            preview = _path(descriptor.get('path'))
            size = descriptor.get('size')
            if descriptor.get('original_size') != native_size:
                raise ValueError('Preview original dimensions disagree with native config')
        else:
            candidates = list((directory / 'codex_previews').glob(camera + '.max*.png'))
            if len(candidates) != 1:
                raise ValueError('Expected exactly one current preview; use original pixel space otherwise')
            preview = candidates[0]
            size = png_size(preview)
        if (preview.resolve() != preview.absolute() or preview.resolve().parent != directory / 'codex_previews'
                or not re.fullmatch(re.escape(camera) + r'\.max\d+\.png', preview.name)):
            raise ValueError('Preview is outside the current observation or uses an alias')
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('Expected two positive integer preview dimensions')
        actual = png_size(preview)
        edge = int(re.search(r'\.max(\d+)\.png$', preview.name).group(1))
        if actual != size or max(size) != edge or any(a > b for a, b in zip(size, native_size)):
            raise ValueError('Preview dimensions/descriptor disagree')
        if abs(size[0] / size[1] - native_size[0] / native_size[1]) > 1 / min(size):
            raise ValueError('Unexpected preview aspect ratio')
        preview_path = str(preview.resolve())
    elif pixel_space != 'original':
        raise ValueError('Explicit original or preview pixel space required')
    if np.any(uv < 0) or np.any(uv >= size):
        raise ValueError('Pixel is outside the selected image')
    converted = uv * np.array(native_size) / np.array(size)
    return converted, dict(pixel_space=pixel_space, input_uv=uv.tolist(), input_size=size,
                          original_uv=converted.tolist(), original_size=native_size, preview_path=preview_path)


def _unit(value):
    v = vector(value, 3)
    scale = float(np.max(np.abs(v)))
    if scale == 0:
        raise ValueError('Zero ray direction')
    scaled = v / scale
    return scaled / np.linalg.norm(scaled)


def ray(packet, camera, uv, config):
    uv = vector(uv, 2)
    K = intrinsics(camera, config)
    width, height = config['cameras'][camera]['resolution']
    if np.any(uv < 0) or uv[0] >= width or uv[1] >= height:
        raise ValueError('Expected current full-resolution pixel within image')
    C = camera_matrix(packet, camera, config)
    d = C[:3, :3] @ np.array([(uv[0]-K[0, 2])/K[0, 0], -(uv[1]-K[1, 2])/K[1, 1], -1.])
    return C[:3, 3], _unit(d)


def project(packet, camera, point, config):
    C = camera_matrix(packet, camera, config)
    v = C[:3, :3].T @ (vector(point, 3)-C[:3, 3])
    if v[2] >= -1e-12:
        return None
    K = intrinsics(camera, config)
    return [float(K[0, 2] + K[0, 0]*v[0]/-v[2]), float(K[1, 2] - K[1, 1]*v[1]/-v[2])]


def closest_rays(origin_a, direction_a, origin_b, direction_b):
    p = vector(origin_a, 3)
    s = vector(origin_b, 3)
    d = _unit(direction_a)
    v = _unit(direction_b)
    A = np.column_stack([d, -v])
    U, singular, Vh = np.linalg.svd(A, full_matrices=False)
    sine = float(np.linalg.norm(np.cross(d, v)))
    near = sine < 1e-4
    result = dict(near_parallel=near, sin_ray_angle=sine,
        condition_number=None if singular[-1] < 1e-15 else float(singular[0]/singular[-1]),
        singular_values=singular.tolist(), behind_camera=None, ray_depths_m=None,
        ray_gap_m=None, point_environment_m=None, usable_geometry=False)
    if near:
        return result
    depths = Vh.T @ ((U.T @ (s-p)) / singular)
    a = p + depths[0]*d
    b = s + depths[1]*v
    behind = bool(np.any(depths <= 0))
    result.update(behind_camera=behind, ray_depths_m=depths.tolist(), ray_gap_m=float(np.linalg.norm(a-b)),
                  point_environment_m=((a+b)/2).tolist(), usable_geometry=not behind)
    return result


def triangulate(packet, camera_a, uv_a, camera_b, uv_b, config):
    if camera_a == camera_b:
        raise ValueError('Triangulation requires distinct synchronized camera views')
    p, d = ray(packet, camera_a, uv_a, config)
    s, v = ray(packet, camera_b, uv_b, config)
    result = closest_rays(p, d, s, v)
    result['reprojection_error_px'] = None
    if result['point_environment_m'] is not None:
        projections = [project(packet, c, result['point_environment_m'], config) for c in (camera_a, camera_b)]
        if all(x is not None for x in projections):
            result['reprojection_error_px'] = [float(np.linalg.norm(np.array(x)-uv)) for x, uv in zip(projections, (uv_a, uv_b))]
        else:
            result.update(behind_camera=True, usable_geometry=False)
    return result


def ray_plane_intersection(origin, direction, z):
    """Intersect a ray with caller-supplied environment z; never infer a scene plane."""
    p = vector(origin, 3)
    d = _unit(direction)
    height = float(vector([z], 1)[0])
    parallel = bool(abs(d[2]) < 1e-12)
    result = dict(plane_z_m=height, parallel_to_plane=parallel, behind_camera=None,
                  ray_depth_m=None, point_environment_m=None, usable_geometry=False)
    if parallel:
        return result
    depth = float((height-p[2]) / d[2])
    behind = depth <= 0
    result.update(behind_camera=behind, ray_depth_m=depth, point_environment_m=(p+depth*d).tolist(),
                  usable_geometry=not behind)
    return result


def compute(command, *, config_path, episode_id, request_id, step_id,
            controller_root=None, packet_path=None, pixel_space=None,
            camera=None, uv=None, camera_a=None, uv_a=None,
            camera_b=None, uv_b=None, point=None, z=None):
    """Compute against a fresh public packet and recheck it before returning.

    Use this entry point for Python policies at an idle observation boundary.
    Pixel-consuming commands and project require explicit original/preview space;
    matrices does not. Numeric quality never establishes physical correspondence.
    Invalid inputs/state raise ValueError; file errors may raise OSError.
    """
    if command not in ('matrices', 'ray', 'triangulate', 'project', 'plane-intersection'):
        raise ValueError('Unsupported geometry command')
    if pixel_space not in ('original', 'preview') and not (command == 'matrices' and pixel_space is None):
        raise ValueError('Explicit original or preview pixel space required')
    if (controller_root is None) == (packet_path is None):
        raise ValueError('Choose exactly one controller_root or packet_path')
    config = load_config(config_path)
    if packet_path is not None:
        path = _path(packet_path).absolute()
        if len(path.parents) < 3:
            raise ValueError('Expected a current native packet path')
        root = path.parents[2]
    else:
        root = controller_root
    params = dict(episode_id=episode_id, request_id=request_id, step_id=step_id,
                  config=config, packet_path=packet_path)
    packet = load_current(root, **params)
    result = dict(binding={k: packet[k] for k in ('episode_id', 'request_id', 'step_id')},
        config_id=config['config_id'], frame=config['eef_frame'], no_action=True, actual_acks=0,
        model='fixed public nominal camera geometry; caller supplied pixel correspondence and plane, not object truth',
        quality_scope='numeric geometry only; no contact, calibration or action authorization claim', pixel_space=pixel_space)
    if command == 'matrices':
        result['cameras'] = {c: dict(camera_to_environment_usd=camera_matrix(packet, c, config).tolist(),
            K=intrinsics(c, config).tolist(), resolution=config['cameras'][c]['resolution']) for c in CAMERAS}
    elif command in ('ray', 'plane-intersection'):
        original_uv, mapping = original_pixel(packet, camera, uv, pixel_space, config)
        p, d = ray(packet, camera, original_uv, config)
        result.update(camera=camera, pixel_mapping=mapping, origin_environment_m=p.tolist(), direction_environment=d.tolist())
        if command == 'plane-intersection':
            result.update(ray_plane_intersection(p, d, z))
    elif command == 'project':
        projected = project(packet, camera, point, config)
        result.update(camera=camera, point_environment_m=vector(point, 3).tolist(), original_uv=projected,
                      behind_camera=projected is None, usable_geometry=projected is not None)
        if pixel_space == 'preview' and projected is not None:
            # Reuse the verified preview dimensions, also for off-image projections.
            _, mapping = original_pixel(packet, camera, [0, 0], 'preview', config)
            result.update(projected_uv=(np.array(projected) * np.array(mapping['input_size']) / np.array(mapping['original_size'])).tolist(),
                          pixel_mapping=mapping)
        else:
            result['projected_uv'] = projected
    else:
        a, ma = original_pixel(packet, camera_a, uv_a, pixel_space, config)
        b, mb = original_pixel(packet, camera_b, uv_b, pixel_space, config)
        result.update(triangulate(packet, camera_a, a, camera_b, b, config),
                      cameras=[camera_a, camera_b], pixel_mapping=[ma, mb])
    if load_current(root, **params) != packet:
        raise ValueError('Current public observation changed during geometry calculation')
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--packet', type=Path, help='Current native observations/NNN/observation.json')
    location.add_argument('--controller-root', type=Path)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--step-id', required=True, type=int)
    parser.add_argument('--pixel-space', choices=('original', 'preview'),
                        help='Required except for matrices')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('matrices')
    ray_parser = commands.add_parser('ray')
    ray_parser.add_argument('--camera', choices=CAMERAS, required=True)
    ray_parser.add_argument('--uv', nargs=2, type=float, required=True)
    tri_parser = commands.add_parser('triangulate')
    for suffix in ('a', 'b'):
        tri_parser.add_argument('--camera-' + suffix, choices=CAMERAS, required=True)
        tri_parser.add_argument('--uv-' + suffix, nargs=2, type=float, required=True)
    project_parser = commands.add_parser('project')
    project_parser.add_argument('--camera', choices=CAMERAS, required=True)
    project_parser.add_argument('--point', nargs=3, type=float, required=True)
    plane_parser = commands.add_parser('plane-intersection')
    plane_parser.add_argument('--camera', choices=CAMERAS, required=True)
    plane_parser.add_argument('--uv', nargs=2, type=float, required=True)
    plane_parser.add_argument('--z', type=float, required=True)
    options = vars(parser.parse_args(argv))
    command = options.pop('command')
    options['config_path'] = options.pop('config')
    options['packet_path'] = options.pop('packet')
    print(json.dumps(compute(command, **options), allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError, OverflowError, RuntimeError) as error:
        print(json.dumps(dict(error=str(error), no_action=True, actual_acks=0)), file=sys.stderr)
        sys.exit(2)
