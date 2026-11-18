"""Inspect explicit geometry configuration and bind a fresh public observation.

This command only reads files and prints JSON. Readiness is nominal numerical
readiness, never physical calibration, contact verification or motion authority.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .geometry_config import inspect_config, parse_config


def initialize(config_path, *, config_only=False, task=None, layout=None,
               controller_root=None, episode_id=None, request_id=None,
               step_id=None, packet_path=None):
    """Return missing inputs or current-bound nominal geometry without mutation."""
    result = dict(status='INVALID', no_action=True, actual_acks=0,
                  motion_authorized=False, physical_calibration_verified=False,
                  config_id=None, config_sha256=None, config_report=None,
                  binding=None, cameras={}, missing=[], errors=[], limitations=[])
    try:
        raw = Path(config_path).read_bytes()
        config = parse_config(raw)
        report = inspect_config(config)
        result.update(config_sha256=hashlib.sha256(raw).hexdigest(), config_report=report,
                      status=report['status'], missing=list(report['missing']),
                      errors=list(report['errors']), limitations=list(report['limitations']))
        if not report['ready']:
            return result
        result['config_id'] = config['config_id']
        result['validated_task_layouts'] = config['validated_task_layouts']
        supplied = dict(task=task, layout=layout, controller_root=controller_root,
                        packet_path=packet_path, episode_id=episode_id,
                        request_id=request_id, step_id=step_id)
        if type(config_only) is not bool:
            raise ValueError('config_only must be a boolean')
        if config_only:
            if any(value is not None for value in supplied.values()):
                raise ValueError('--config-only cannot ignore task or current binding arguments')
            result['status'] = 'CONFIG_READY'
            result['limitations'].append('Current observation and task scope have not been bound.')
            return result
        result['missing'] = [name for name, value in supplied.items()
                             if value is None and name not in ('controller_root', 'packet_path')]
        if controller_root is None and packet_path is None:
            result['missing'].append('controller_root_or_packet')
        if result['missing']:
            result['status'] = 'INCOMPLETE'
            return result
        if controller_root is not None and packet_path is not None:
            raise ValueError('Choose controller_root or packet_path, not both')
        if type(task) is not str or not task.strip():
            raise ValueError('task must be a nonempty string')
        if type(layout) is not int or layout < 0:
            raise ValueError('layout must be a nonnegative integer')
        if layout not in config['validated_task_layouts'].get(task, []):
            result.update(status='OUT_OF_SCOPE', errors=[
                'Task/layout is outside the declared applicability of this configuration; '
                'supply an independently justified configuration for the new scope.'])
            return result
        # Importing the adapter is deliberately postponed until current binding is requested.
        from . import geometry
        root = (Path(controller_root) if controller_root is not None
                else Path(packet_path).resolve().parents[2])
        binding_args = dict(config=config, episode_id=episode_id, request_id=request_id,
                            step_id=step_id, packet_path=packet_path)
        try:
            packet = geometry.load_current(root, **binding_args)
            if 'task' in packet and packet['task'] != task:
                raise ValueError('Current public task does not match the requested task')
            cameras = {}
            for camera, camera_config in config['cameras'].items():
                image = next(item for item in packet['images'] if item['camera'] == camera)
                if 'codex_preview' in image:
                    geometry.original_pixel(packet, camera, [0., 0.], 'preview', config)
                cameras[camera] = dict(
                    resolution=camera_config['resolution'], K=camera_config['K'],
                    camera_to_environment_usd=geometry.camera_matrix(packet, camera, config).tolist())
            if geometry.load_current(root, **binding_args) != packet:
                raise ValueError('Current public state changed during initialization')
        except (ValueError, TypeError, KeyError, OSError, IndexError, StopIteration) as error:
            result.update(status='CURRENT_INVALID', errors=[str(error)])
            return result
        result.update(status='READY_NOMINAL', cameras=cameras,
                      binding={key: packet[key] for key in ('episode_id', 'request_id', 'step_id')},
                      task=task, layout=layout, frame=config['eef_frame'])
        result['limitations'].append(
            'Current binding is a snapshot; each later calculation must bind its latest request again.')
        return result
    except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
        result.update(status='INVALID', errors=[str(error)])
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--task')
    parser.add_argument('--layout', type=int)
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--controller-root', type=Path)
    location.add_argument('--packet', type=Path)
    parser.add_argument('--episode-id')
    parser.add_argument('--request-id')
    parser.add_argument('--step-id', type=int)
    args = parser.parse_args(argv)
    result = initialize(args.config, config_only=args.config_only, task=args.task,
                        layout=args.layout, controller_root=args.controller_root,
                        packet_path=args.packet, episode_id=args.episode_id,
                        request_id=args.request_id, step_id=args.step_id)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result['status'] in ('CONFIG_READY', 'READY_NOMINAL') else 2


if __name__ == '__main__':
    raise SystemExit(main())
