"""Offline initialization contracts; synthetic public state, no simulator."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'tools/configs/robodojo-x5-nominal.json'


class InitializeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'tools/initialize.py').is_file(), 'initialization entry is missing')
        self.init = importlib.import_module('tools.initialize')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def current(self):
        root = self.directory / 'controller'
        obs = root / 'observations/000'
        obs.mkdir(parents=True)
        packet = dict(episode_id='new-episode', request_id='new-request', step_id=0,
            robot_profile='robodojo', rollout_finished=False,
            observation_path=str(obs / 'observation.json'), request_path=str(root / 'request_000.json'),
            current_eef={arm: dict(frame='environment_origin', link='link6', position=[0., 0., 1.],
                quaternion_wxyz=[1., 0., 0., 0.]) for arm in ('left', 'right')}, images=[])
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            path = obs / (camera + '.png')
            Image.new('RGB', (640, 480)).save(path)
            packet['images'].append(dict(camera=camera, path=str(path)))
        (obs / 'observation.json').write_text(json.dumps(packet))
        (root / 'request_000.json').write_text(json.dumps(packet))
        return dict(controller_root=root, episode_id='new-episode', request_id='new-request',
                    step_id=0, task='stack_bowls', layout=0)

    def test_config_only_does_not_claim_current_or_contact_readiness(self):
        result = self.init.initialize(CONFIG, config_only=True)
        self.assertEqual(result['status'], 'CONFIG_READY')
        self.assertIsNone(result['binding'])
        self.assertFalse(result['physical_calibration_verified'])
        self.assertFalse(result['motion_authorized'])
        self.assertEqual(result['actual_acks'], 0)

    def test_default_reports_dynamic_missing_fields_without_inventing_values(self):
        result = self.init.initialize(CONFIG)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertIn('request_id', result['missing'])
        self.assertIn('controller_root_or_packet', result['missing'])
        self.assertIn('task', result['missing'])
        self.assertIsNone(result['binding'])

    def test_template_reports_static_gaps(self):
        result = self.init.initialize(ROOT / 'tools/configs/template.json', config_only=True)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertTrue(result['missing'])

    def test_current_matching_task_scope_is_ready_nominal_and_read_only(self):
        arguments = self.current()
        before = {str(p): p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        result = self.init.initialize(CONFIG, **arguments)
        self.assertEqual(result['status'], 'READY_NOMINAL', result)
        self.assertEqual(result['binding']['request_id'], 'new-request')
        self.assertIn('cam_left_wrist', result['cameras'])
        self.assertTrue(result['limitations'])
        self.assertFalse(result['motion_authorized'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.directory.rglob('*') if p.is_file()})

    def test_current_public_task_must_match_the_requested_task_when_present(self):
        arguments = self.current()
        request = Path(arguments['controller_root']) / 'request_000.json'
        packet = json.loads(request.read_text())
        for task, status in (('stack_bowls', 'READY_NOMINAL'),
                             ('stack_blocks', 'CURRENT_INVALID')):
            with self.subTest(task=task):
                packet['task'] = task
                request.write_text(json.dumps(packet))
                Path(packet['observation_path']).write_text(json.dumps(packet))
                result = self.init.initialize(CONFIG, **arguments)
                self.assertEqual(result['status'], status, result)

    def test_overly_nested_public_json_is_a_current_invalid_report(self):
        arguments = self.current()
        request = Path(arguments['controller_root']) / 'request_000.json'
        observation = Path(json.loads(request.read_text())['observation_path'])
        nested = '{"nested":' + '[' * (sys.getrecursionlimit() + 100) + '0'
        nested += ']' * (sys.getrecursionlimit() + 100) + '}'
        for path in (request, observation):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    path.write_text(nested)
                    result = self.init.initialize(CONFIG, **arguments)
                    self.assertEqual(result['status'], 'CURRENT_INVALID', result)
                    self.assertTrue(result['errors'])
                    self.assertIsNone(result['binding'])
                finally:
                    path.write_bytes(original)

    def test_new_task_or_layout_is_not_silently_treated_as_validated(self):
        arguments = self.current()
        for changes in (dict(task='stack_blocks'), dict(layout=1)):
            with self.subTest(changes=changes):
                result = self.init.initialize(CONFIG, **(arguments | changes))
                self.assertEqual(result['status'], 'OUT_OF_SCOPE', result)

    def test_old_request_and_native_terminal_are_not_ready(self):
        arguments = self.current()
        result = self.init.initialize(CONFIG, **(arguments | dict(request_id='historical-request')))
        self.assertEqual(result['status'], 'CURRENT_INVALID', result)
        request = Path(arguments['controller_root']) / 'request_000.json'
        packet = json.loads(request.read_text())
        packet['rollout_finished'] = True
        request.write_text(json.dumps(packet))
        Path(packet['observation_path']).write_text(json.dumps(packet))
        result = self.init.initialize(CONFIG, **arguments)
        self.assertEqual(result['status'], 'CURRENT_INVALID', result)

    def test_config_only_rejects_ignored_current_arguments(self):
        result = self.init.initialize(CONFIG, config_only=True, request_id='must-not-be-ignored')
        self.assertEqual(result['status'], 'INVALID')

    def test_duplicate_config_keys_are_rejected_consistently_with_geometry(self):
        path = self.directory / 'duplicate.json'
        original = CONFIG.read_text()
        path.write_text('{"config_id":"ignored-duplicate",' + original.lstrip()[1:])
        result = self.init.initialize(path, config_only=True)
        self.assertEqual(result['status'], 'INVALID')

    def test_present_but_empty_preview_metadata_is_not_ignored(self):
        arguments = self.current()
        request = Path(arguments['controller_root']) / 'request_000.json'
        packet = json.loads(request.read_text())
        packet['images'][0]['codex_preview'] = {}
        request.write_text(json.dumps(packet))
        Path(packet['observation_path']).write_text(json.dumps(packet))
        result = self.init.initialize(CONFIG, **arguments)
        self.assertEqual(result['status'], 'CURRENT_INVALID')

    def test_missing_and_malformed_config_are_json_reports(self):
        path = self.directory / 'config.json'
        for body in (None, '{invalid', 'null'):
            if body is not None:
                path.write_text(body)
            result = self.init.initialize(path, config_only=True)
            self.assertEqual(result['status'], 'INVALID')
            json.dumps(result, allow_nan=False)

    def test_cli_exit_codes_and_json_are_usable_without_live_robot(self):
        command = [sys.executable, '-B', '-m', 'tools.initialize', '--config', str(CONFIG)]
        ready = subprocess.run(command + ['--config-only'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(ready.returncode, 0, ready.stderr)
        self.assertEqual(json.loads(ready.stdout)['status'], 'CONFIG_READY')
        gaps = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(gaps.returncode, 2, gaps.stderr)
        self.assertEqual(json.loads(gaps.stdout)['status'], 'INCOMPLETE')
        self.assertNotIn('Traceback', gaps.stderr)

    def test_cli_escapes_isolated_surrogates_in_json_strings(self):
        config = json.loads(CONFIG.read_text())
        config['config_id'] = 'nominal-\ud800'
        path = self.directory / 'surrogate-config.json'
        path.write_text(json.dumps(config))
        command = [sys.executable, '-B', '-m', 'tools.initialize', '--config', str(path)]
        for options, code, status in ((['--config-only'], 0, 'CONFIG_READY'),
                                      ([], 2, 'INCOMPLETE')):
            with self.subTest(status=status):
                process = subprocess.run(command + options, cwd=ROOT, capture_output=True,
                                         text=True, env=os.environ | {'PYTHONIOENCODING': 'utf-8:strict'})
                self.assertEqual(process.returncode, code, process.stderr)
                result = json.loads(process.stdout)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['config_id'], config['config_id'])
                self.assertNotIn('Traceback', process.stderr)


if __name__ == '__main__':
    unittest.main()
