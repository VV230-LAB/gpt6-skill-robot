"""CPU-only checks using public packet-shaped fixtures and real temporary PNGs."""
import copy
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'tools/configs/robodojo-x5-nominal.json'


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'tools/geometry.py').is_file(), 'reusable geometry helper is missing')
        self.g = importlib.import_module('tools.geometry')
        self.config = self.g.load_config(CONFIG)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'controller'
        self.directory = self.root / 'observations/000'
        self.directory.mkdir(parents=True)
        self.packet = dict(episode_id='synthetic-new-episode', request_id='fresh-request', step_id=0,
            robot_profile='robodojo', rollout_finished=False,
            observation_path=str(self.directory / 'observation.json'),
            request_path=str(self.root / 'request_000.json'),
            current_eef={arm: dict(position=[0, 0, 1], quaternion_wxyz=[1, 0, 0, 0],
                frame='environment_origin', link='link6') for arm in ('left', 'right')}, images=[])
        for camera, spec in self.config['cameras'].items():
            p = self.directory / (camera + '.png')
            Image.new('RGB', tuple(spec['resolution'])).save(p)
            self.packet['images'].append(dict(camera=camera, path=str(p)))
        self.save()

    def save(self):
        (self.directory / 'observation.json').write_text(json.dumps(self.packet))
        (self.root / 'request_000.json').write_text(json.dumps(self.packet))

    def load(self, **kwargs):
        params = dict(episode_id='synthetic-new-episode', request_id='fresh-request', step_id=0, config=self.config)
        params.update(kwargs)
        return self.g.load_current(self.root, **params)

    def cli(self, *command, packet=False, pixel_space='original'):
        return [sys.executable, '-B', '-m', 'tools.geometry', '--config', str(CONFIG),
            '--packet' if packet else '--controller-root', str(self.directory / 'observation.json') if packet else str(self.root),
            '--episode-id', self.packet['episode_id'], '--request-id', self.packet['request_id'],
            '--step-id', str(self.packet['step_id']), '--pixel-space', pixel_space, *command]

    def test_public_nominal_matrices_preserve_source_arithmetic(self):
        def T(p, r):
            t = np.eye(4)
            t[:3, 3] = p
            t[:3, :3] = r
            return t
        mount = T([.057, 0, 0], Rotation.from_euler('xyz', [0, 0, -3.1416]).as_matrix()) @ T(
            [-.0275, 0, .05], Rotation.from_euler('xyz', [0, .3491, math.pi]).as_matrix()) @ T(
            [0, 0, .001], Rotation.from_euler('XYZ', [0, -80, -90], degrees=True).as_matrix())
        head = T([0, -.41, 1.308], Rotation.from_euler('XYZ', [30, 0, 0], degrees=True).as_matrix())
        np.testing.assert_allclose(self.config['eef_to_wrist_usd'], mount, atol=1e-15)
        np.testing.assert_allclose(self.g.camera_matrix(self.packet, 'cam_high', self.config), head, atol=1e-15)
        np.testing.assert_allclose(self.g.camera_matrix(self.packet, 'cam_left_wrist', self.config), T([0, 0, 1], np.eye(3)) @ mount, atol=1e-15)
        np.testing.assert_allclose(self.g.intrinsics('cam_high', self.config), [[640*10/22.212, 0, 320], [0, 480*10/14.266, 240], [0, 0, 1]])

    def test_wxyz_pose_rotation_sign_and_invalid_values(self):
        a = self.g.pose_matrix([1, 2, 3], [math.sqrt(.5), 0, 0, math.sqrt(.5)])
        np.testing.assert_allclose(a @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-14)
        np.testing.assert_allclose(a, self.g.pose_matrix([1, 2, 3], [-math.sqrt(.5), 0, 0, -math.sqrt(.5)]))
        for q in ([0, 0, 0, 0], [2, 0, 0, 0], [float('nan'), 0, 0, 1], [True, 0, 0, 0], ['1', 0, 0, 0]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.g.pose_matrix([1, 2, 3], q)

    def test_principal_ray_usd_forward_and_image_down(self):
        origin, direction = self.g.ray(self.packet, 'cam_high', [320, 240], self.config)
        C = self.g.camera_matrix(self.packet, 'cam_high', self.config)
        np.testing.assert_allclose(origin, C[:3, 3])
        np.testing.assert_allclose(direction, -C[:3, 2])
        _, lower = self.g.ray(self.packet, 'cam_high', [320, 260], self.config)
        self.assertLess(float(lower @ C[:3, 1]), 0)

    def synthetic_pair(self):
        head = self.g.camera_matrix(self.packet, 'cam_high', self.config)
        wrist = head.copy()
        wrist[:3, 3] += [.15, 0, 0]
        eef = wrist @ np.linalg.inv(np.array(self.config['eef_to_wrist_usd']))
        q = Rotation.from_matrix(eef[:3, :3]).as_quat()
        self.packet['current_eef']['left'].update(position=eef[:3, 3].tolist(), quaternion_wxyz=q[[3, 0, 1, 2]].tolist())
        point = (head @ [0, -.05, -.8, 1])[:3]
        a = self.g.project(self.packet, 'cam_high', point, self.config)
        b = self.g.project(self.packet, 'cam_left_wrist', point, self.config)
        return point, a, b

    def test_synthetic_projection_and_svd_recovery(self):
        expected, a, b = self.synthetic_pair()
        r = self.g.triangulate(self.packet, 'cam_high', a, 'cam_left_wrist', b, self.config)
        np.testing.assert_allclose(r['point_environment_m'], expected, atol=1e-12)
        self.assertLess(r['ray_gap_m'], 1e-12)
        self.assertTrue(r['usable_geometry'])
        self.assertFalse(r['near_parallel'])
        self.assertLess(max(r['reprojection_error_px']), 1e-10)
        self.assertTrue(all(t > 0 for t in r['ray_depths_m']))

    def test_skew_rays_report_residual(self):
        _, a, b = self.synthetic_pair()
        b[1] += 8
        r = self.g.triangulate(self.packet, 'cam_high', a, 'cam_left_wrist', b, self.config)
        self.assertGreater(r['ray_gap_m'], .005)
        self.assertGreater(max(r['reprojection_error_px']), 1)

    def test_triangulated_midpoint_behind_a_camera_is_unusable(self):
        self.packet['current_eef']['left'].update(
            position=[.48411914906202635, -.28969286115892534, .6067681732539083],
            quaternion_wxyz=[.7738357642524696, .5470661635470113,
                             -.2868238543811186, .14010317349621307])
        result = self.g.triangulate(self.packet, 'cam_high', [255.62682297531617, 48.36388118880682],
                                    'cam_left_wrist', [93.53244301240879, 270.6422974253665], self.config)
        self.assertTrue(all(depth > 0 for depth in result['ray_depths_m']))
        self.assertIsNone(self.g.project(self.packet, 'cam_left_wrist', result['point_environment_m'], self.config))
        self.assertFalse(result['usable_geometry'])
        self.assertTrue(result['behind_camera'])
        self.assertIsNone(result['reprojection_error_px'])

    def test_parallel_and_backward_intersections_are_unusable(self):
        for direction in ([0, 0, -1], [1e-8, 0, -1]):
            r = self.g.closest_rays([0, 0, 0], [0, 0, -1], [1, 0, 0], direction)
            self.assertTrue(r['near_parallel'])
            self.assertFalse(r['usable_geometry'])
            self.assertIsNone(r['point_environment_m'])
            json.dumps(r, allow_nan=False)
        r = self.g.closest_rays([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])
        self.assertTrue(r['behind_camera'])
        self.assertFalse(r['usable_geometry'])
        with self.assertRaises(ValueError):
            self.g.closest_rays([0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0])

    def test_explicit_plane_intersection_is_numeric_only(self):
        r = self.g.ray_plane_intersection([1, 2, 3], [0, 0, -2], .5)
        np.testing.assert_allclose(r['point_environment_m'], [1, 2, .5])
        self.assertAlmostEqual(r['ray_depth_m'], 2.5)
        self.assertEqual(r['plane_z_m'], .5)
        self.assertTrue(r['usable_geometry'])
        self.assertFalse(r['parallel_to_plane'])
        self.assertFalse(r['behind_camera'])
        json.dumps(r, allow_nan=False)

    def test_plane_requires_finite_z_and_reports_degeneracy(self):
        for direction in ([1, 0, 0], [1, 0, 1e-13]):
            r = self.g.ray_plane_intersection([0, 0, 1], direction, 0)
            self.assertTrue(r['parallel_to_plane'])
            self.assertFalse(r['usable_geometry'])
            self.assertIsNone(r['point_environment_m'])
        r = self.g.ray_plane_intersection([0, 0, 1], [0, 0, 1], 0)
        self.assertTrue(r['behind_camera'])
        self.assertFalse(r['usable_geometry'])
        for z in (None, True, '0', float('nan'), float('inf')):
            with self.subTest(z=z), self.assertRaises(ValueError):
                self.g.ray_plane_intersection([0, 0, 1], [0, 0, -1], z)
        with self.assertRaises(ValueError):
            self.g.ray_plane_intersection([0, 0, 1], [0, 0, 0], 0)

    def test_projection_rejects_nonfinite_point_and_reports_behind(self):
        C = self.g.camera_matrix(self.packet, 'cam_high', self.config)
        self.assertIsNone(self.g.project(self.packet, 'cam_high', C[:3, 3] + C[:3, 2], self.config))
        with self.assertRaises(ValueError):
            self.g.project(self.packet, 'cam_high', [0, 0, float('nan')], self.config)

    def test_binding_requires_explicit_config_and_valid_ids(self):
        with self.assertRaises(TypeError):
            self.g.load_current(self.root, episode_id='e', request_id='r', step_id=0)
        for kwargs in (dict(episode_id='old'), dict(request_id='old'), dict(step_id=1), dict(step_id=False), dict(episode_id=1)):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'binding'):
                self.load(**kwargs)
        self.assertEqual(self.load()['step_id'], 0)

    def test_newer_request_or_terminal_observation_rejects_current(self):
        newer = dict(self.packet, request_id='newer', step_id=1)
        (self.root / 'request_001.json').write_text(json.dumps(newer))
        with self.assertRaises(ValueError):
            self.load()
        (self.root / 'request_001.json').unlink()
        later = self.root / 'observations/001'
        later.mkdir()
        (later / 'observation.json').write_text(json.dumps(dict(self.packet, rollout_finished=True, step_id=1)))
        with self.assertRaises(ValueError):
            self.load()

    def test_packet_request_state_and_current_descriptor_must_match(self):
        for field, changed in (('current_eef', {'left': {}}), ('images', []), ('observation_path', str(self.root / 'old.json'))):
            v = copy.deepcopy(self.packet)
            v[field] = changed
            (self.directory / 'observation.json').write_text(json.dumps(v))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load()
        self.save()
        v = dict(self.packet, request_path=str(self.root / 'old.json'))
        (self.directory / 'observation.json').write_text(json.dumps(v))
        with self.assertRaises(ValueError):
            self.load()

    def test_invalid_packet_types_nonfinite_eef_and_unsupported_state(self):
        original = copy.deepcopy(self.packet)
        for mutation in ('profile', 'frame', 'link', 'ended', 'nan', 'string_position', 'bool_position', 'eef_list', 'images_dict', 'step_bool', 'camera_duplicate'):
            self.packet = copy.deepcopy(original)
            if mutation == 'profile': self.packet['robot_profile'] = 'other'
            elif mutation == 'ended': self.packet['rollout_finished'] = True
            elif mutation == 'nan': self.packet['current_eef']['left']['position'][0] = float('nan')
            elif mutation == 'string_position': self.packet['current_eef']['left']['position'][0] = '0'
            elif mutation == 'bool_position': self.packet['current_eef']['left']['position'][0] = False
            elif mutation == 'eef_list': self.packet['current_eef'] = []
            elif mutation == 'images_dict': self.packet['images'] = {}
            elif mutation == 'step_bool': self.packet['step_id'] = False
            elif mutation == 'camera_duplicate': self.packet['images'][1]['camera'] = 'cam_high'
            else: self.packet['current_eef']['left'][mutation] = 'other'
            self.save()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load()
        for value in ([], None, 1):
            (self.directory / 'observation.json').write_text(json.dumps(value))
            (self.root / 'request_000.json').write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()

    def test_configured_resolution_is_checked_against_real_png(self):
        c = copy.deepcopy(self.config)
        c['cameras']['cam_high']['resolution'] = [800, 600]
        with self.assertRaises(ValueError):
            self.load(config=c)
        Image.new('RGB', (800, 600)).save(self.directory / 'cam_high.png')
        self.assertEqual(self.load(config=c)['step_id'], 0)
        (self.directory / 'cam_high.png').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load(config=c)

    def test_truncated_and_non_rgb_current_images_are_rejected(self):
        path = self.directory / 'cam_high.png'
        original = path.read_bytes()
        path.write_bytes(original[:24])
        with self.assertRaisesRegex(ValueError, 'PNG'):
            self.load()
        for mode in ('L', 'RGBA'):
            Image.new(mode, (640, 480)).save(path)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'RGB'):
                self.load()

    def test_corrupt_preview_is_rejected_even_with_matching_size_header(self):
        path = self.make_preview()
        path.write_bytes(path.read_bytes()[:24])
        with self.assertRaisesRegex(ValueError, 'PNG'):
            self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'preview', self.config)

    def test_oversized_png_keeps_structured_failure_for_both_clis(self):
        import struct
        import zlib
        def chunk(kind, payload):
            return (struct.pack('>I', len(payload)) + kind + payload
                    + struct.pack('>I', zlib.crc32(kind + payload)))
        header = struct.pack('>IIBBBBB', 20000, 20000, 8, 2, 0, 0, 0)
        (self.directory / 'cam_high.png').write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
                                                    + chunk(b'IDAT', b'') + chunk(b'IEND', b''))
        commands = [('geometry', self.cli('matrices')),
            ('initialize', [sys.executable, '-B', '-m', 'tools.initialize', '--config', str(CONFIG),
                '--controller-root', str(self.root), '--episode-id', self.packet['episode_id'],
                '--request-id', self.packet['request_id'], '--step-id', '0',
                '--task', 'stack_bowls', '--layout', '0'])]
        for entry, command in commands:
            with self.subTest(entry=entry):
                run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                result = json.loads(run.stderr if entry == 'geometry' else run.stdout)
                self.assertTrue(result['no_action'])
                self.assertEqual(result['actual_acks'], 0)
                if entry == 'initialize':
                    self.assertEqual(result['status'], 'CURRENT_INVALID')
                self.assertNotIn('Traceback', run.stderr)

    def test_request_and_observation_task_must_match_when_present(self):
        self.packet['task'] = 'stack_bowls'
        self.save()
        self.assertEqual(self.load().get('task'), 'stack_bowls')
        request = self.root / 'request_000.json'
        changed = json.loads(request.read_text())
        changed['task'] = 'stack_blocks'
        request.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.load()

    def test_ambiguous_and_overflowing_public_json_rejected(self):
        raw = json.dumps(self.packet)
        for changed in (raw[:-1] + ', "step_id": 0}', raw[:-1] + ', "extra": 1e999}'):
            (self.directory / 'observation.json').write_text(changed)
            (self.root / 'request_000.json').write_text(changed)
            with self.subTest(changed=changed[-45:]), self.assertRaises(ValueError):
                self.load()

    def test_foreign_packet_image_request_and_directory_aliases_rejected(self):
        wrong = self.root / 'observations/old/observation.json'
        wrong.parent.mkdir()
        wrong.write_text(json.dumps(self.packet))
        with self.assertRaises(ValueError):
            self.load(packet_path=wrong)
        for p in (self.directory / 'cam_high.png', self.root / 'request_000.json'):
            foreign = self.root.parent / ('foreign-' + p.name)
            p.rename(foreign)
            p.symlink_to(foreign)
            with self.subTest(path=p), self.assertRaises(ValueError):
                self.load()
            p.unlink()
            foreign.rename(p)
        original = self.directory
        alternate = self.root / 'observations/alias'
        original.rename(alternate)
        original.symlink_to(alternate, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_numeric_request_and_observation_aliases_rejected(self):
        request = self.root / 'request_0.json'
        request.write_text(json.dumps(self.packet))
        with self.assertRaises(ValueError):
            self.load()
        request.unlink()
        other = self.root / 'observations/0'
        other.mkdir()
        (other / 'observation.json').write_text(json.dumps(self.packet))
        with self.assertRaises(ValueError):
            self.load()

    def test_bad_pixels_unsupported_camera_and_duplicate_pair_rejected(self):
        for uv in ([-1, 2], [640, 2], [2, float('inf')], [1, 2, 3], ['1', 2], [True, 2]):
            with self.subTest(uv=uv), self.assertRaises(ValueError):
                self.g.ray(self.packet, 'cam_high', uv, self.config)
        with self.assertRaises(ValueError):
            self.g.ray(self.packet, 'other', [1, 2], self.config)
        with self.assertRaises(ValueError):
            self.g.triangulate(self.packet, 'cam_high', [320, 240], 'cam_high', [321, 240], self.config)

    def make_preview(self, edge=480, size=(480, 360)):
        directory = self.directory / 'codex_previews'
        directory.mkdir(exist_ok=True)
        p = directory / f'cam_high.max{edge}.png'
        Image.new('RGB', size).save(p)
        return p

    def test_preview_scales_only_verified_current_images(self):
        p = self.make_preview()
        uv, mapping = self.g.original_pixel(self.packet, 'cam_high', [240, 180], 'preview', self.config)
        np.testing.assert_allclose(uv, [320, 240])
        self.assertEqual(mapping['input_size'], [480, 360])
        self.packet['images'][0]['codex_preview'] = dict(path=str(p), size=[480, 360], original_size=[640, 480])
        uv, _ = self.g.original_pixel(self.packet, 'cam_high', [300, 150], 'preview', self.config)
        np.testing.assert_allclose(uv, [400, 200])
        self.packet['images'][0]['codex_preview']['size'] = [240, 180]
        with self.assertRaises(ValueError):
            self.g.original_pixel(self.packet, 'cam_high', [240, 150], 'preview', self.config)

    def test_missing_ambiguous_foreign_and_symlink_previews_rejected(self):
        with self.assertRaises(ValueError):
            self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'preview', self.config)
        p = self.make_preview()
        self.make_preview(320, (320, 240))
        with self.assertRaises(ValueError):
            self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'preview', self.config)
        foreign = self.root / 'old/codex_previews/cam_high.max480.png'
        foreign.parent.mkdir(parents=True)
        Image.new('RGB', (480, 360)).save(foreign)
        item = self.packet['images'][0]
        item['codex_preview'] = dict(path=str(foreign), size=[480, 360], original_size=[640, 480])
        with self.assertRaises(ValueError):
            self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'preview', self.config)
        p.unlink()
        p.symlink_to(foreign)
        item['codex_preview']['path'] = str(p)
        with self.assertRaises(ValueError):
            self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'preview', self.config)

    def test_malformed_preview_descriptor_and_pixel_space_rejected(self):
        p = self.make_preview()
        for descriptor in ([], 'bad', {}, dict(path=str(p), size=[480, 360], original_size=[320, 240]), dict(path=str(p), size=['480', 360], original_size=[640, 480])):
            self.packet['images'][0]['codex_preview'] = descriptor
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'preview', self.config)
        with self.assertRaises(ValueError):
            self.g.original_pixel(self.packet, 'cam_high', [100, 100], 'unknown', self.config)

    def test_cli_all_operations_bound_finite_json_without_writes(self):
        point, a, b = self.synthetic_pair()
        self.save()
        operations = [('matrices',), ('ray', '--camera', 'cam_high', '--uv', '320', '240'),
            ('triangulate', '--camera-a', 'cam_high', '--uv-a', *map(str, a), '--camera-b', 'cam_left_wrist', '--uv-b', *map(str, b)),
            ('project', '--camera', 'cam_high', '--point', *map(str, point)),
            ('plane-intersection', '--camera', 'cam_high', '--uv', '320', '240', '--z', '.5')]
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        for operation in operations:
            run = subprocess.run(self.cli(*operation, packet=True), cwd=ROOT, capture_output=True, text=True)
            with self.subTest(operation=operation):
                self.assertEqual(run.returncode, 0, run.stderr)
                result = json.loads(run.stdout)
                self.assertEqual(result['binding'], {k: self.packet[k] for k in ('episode_id', 'request_id', 'step_id')})
                self.assertEqual(result['frame'], 'environment_origin')
                self.assertEqual(result['config_id'], self.config['config_id'])
                self.assertIs(result['no_action'], True)
                self.assertEqual(result['actual_acks'], 0)
                json.dumps(result, allow_nan=False)
                if operation[0] == 'project':
                    np.testing.assert_allclose(result['original_uv'], a)
                if operation[0] == 'plane-intersection':
                    self.assertEqual(result['plane_z_m'], .5)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_cli_invalid_input_is_structured_no_action_failure(self):
        for operation in (('ray', '--camera', 'cam_high', '--uv', 'nan', '1'),
                          ('plane-intersection', '--camera', 'cam_high', '--uv', '320', '240', '--z', 'inf')):
            run = subprocess.run(self.cli(*operation), cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertEqual(run.stdout, '')
            error = json.loads(run.stderr)
            self.assertIs(error['no_action'], True)
            self.assertEqual(error['actual_acks'], 0)
        command = self.cli('matrices')
        del command[command.index('--config'):command.index('--config')+2]
        self.assertEqual(subprocess.run(command, cwd=ROOT, capture_output=True, text=True).returncode, 2)

    def test_cli_rechecks_binding_and_public_state_before_output(self):
        original_ray = self.g.ray
        def changed(*args, **kwargs):
            result = original_ray(*args, **kwargs)
            self.packet['current_eef']['left']['position'][0] += .1
            self.save()
            return result
        # Deterministic file mutation in flight; no robot or physics is mocked.
        with patch.object(self.g, 'ray', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed'):
            self.g.main(self.cli('ray', '--camera', 'cam_high', '--uv', '320', '240')[4:])

    def compute(self, command, **arguments):
        self.assertTrue(callable(getattr(self.g, 'compute', None)), 'Missing current-bound Python compute API')
        return self.g.compute(command, config_path=CONFIG, controller_root=self.root,
            episode_id=self.packet['episode_id'], request_id=self.packet['request_id'],
            step_id=self.packet['step_id'], **arguments)

    def test_python_compute_matches_cli_for_all_operations(self):
        point, a, b = self.synthetic_pair()
        self.save()
        cases = [('matrices', {}, ()),
            ('ray', dict(camera='cam_high', uv=[320, 240]), ('--camera', 'cam_high', '--uv', '320', '240')),
            ('triangulate', dict(camera_a='cam_high', uv_a=a, camera_b='cam_left_wrist', uv_b=b),
             ('--camera-a', 'cam_high', '--uv-a', *map(str, a), '--camera-b', 'cam_left_wrist', '--uv-b', *map(str, b))),
            ('project', dict(camera='cam_high', point=point), ('--camera', 'cam_high', '--point', *map(str, point))),
            ('plane-intersection', dict(camera='cam_high', uv=[320, 240], z=.5),
             ('--camera', 'cam_high', '--uv', '320', '240', '--z', '.5'))]
        for command, arguments, flags in cases:
            with self.subTest(command=command):
                result = self.compute(command, pixel_space='original', **arguments)
                cli = subprocess.run(self.cli(command, *flags), cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(cli.returncode, 0, cli.stderr)
                self.assertEqual(result, json.loads(cli.stdout))

    def test_python_compute_rejects_state_change_during_calculation(self):
        original_ray = self.g.ray
        def changed(*args, **kwargs):
            result = original_ray(*args, **kwargs)
            self.packet['current_eef']['left']['position'][0] += .1
            self.save()
            return result
        with patch.object(self.g, 'ray', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed'):
            self.compute('ray', pixel_space='original', camera='cam_high', uv=[320, 240])

    def test_matrices_does_not_require_pixel_space(self):
        result = self.compute('matrices')
        self.assertEqual(set(result['cameras']), set(self.g.CAMERAS))
        command = self.cli('matrices')
        i = command.index('--pixel-space')
        del command[i:i+2]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(result, json.loads(run.stdout))

    def test_python_compute_requires_explicit_pixel_space_and_known_command(self):
        for command, arguments in [('ray', dict(camera='cam_high', uv=[320, 240])),
                                   ('project', dict(camera='cam_high', point=[0, 0, 0]))]:
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'pixel'):
                self.compute(command, **arguments)
        with self.assertRaisesRegex(ValueError, 'command'):
            self.compute('unknown')


if __name__ == '__main__':
    unittest.main()
