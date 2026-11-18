"""Validate explicit nominal geometry data without inferring calibration.

CONFIG_READY means that the supported numeric configuration is complete. It
does not certify camera calibration, visual correspondences or grasp success.
This module reads no research source, simulator or robot state at runtime.
"""
import json
import math
from pathlib import Path
import re


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SCHEMA = "tools.camera-geometry.v1"
SUPPORTED = {
    "schema": SCHEMA,
    "robot_profile": "robodojo",
    "eef_frame": "environment_origin",
    "eef_link": "link6",
    "camera_convention": "USD: +x right, +y up, -z forward",
    "length_unit": "m",
    "quaternion_order": "wxyz",
    "projection_model": "pinhole_no_distortion",
}
DESCRIPTIVE = ("config_id", "applicability", "intrinsics_convention")
LIMITATIONS = (
    "CONFIG_READY validates explicit numeric configuration only; it does not verify physical calibration or grasp success.",
    "Nominal public geometry does not establish TCP/contact offsets, object geometry, depth or a scene plane.",
    "Callers must bind current public observations and establish matching visual points; no correspondence is inferred.",
    "Provenance and task/layout scope are declarations; source files are not read or authenticated at runtime.",
)


def _json_safe(value):
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def inspect_config(value):
    """Return a JSON-safe report; never fill absent or unknown parameters.

    Paths use dotted object keys and bracketed list indices. If several kinds
    of issue coexist, INVALID takes precedence over UNSUPPORTED, then
    INCOMPLETE. Unsupported declarations are also explained in ``errors``.
    """
    missing, invalid, unsupported = [], [], []
    known = {}

    def absent(item, path):
        if item is None or (isinstance(item, str) and not item.strip()):
            missing.append(path)
            return True
        return False

    def text(item, path, expected=None):
        if absent(item, path):
            return
        if type(item) is not str:
            invalid.append(path + ": expected a nonempty string")
        elif expected is not None and item != expected:
            unsupported.append(path + ": unsupported declaration")

    def matrix(item, path, size):
        if absent(item, path):
            return None
        if type(item) is not list or len(item) != size:
            invalid.append(path + f": expected a {size}x{size} matrix")
            return None
        good = True
        for i, row in enumerate(item):
            if row is None:
                missing.append(f"{path}[{i}]")
                good = False
                continue
            if type(row) is not list or len(row) != size:
                invalid.append(f"{path}[{i}]: expected {size} numbers")
                good = False
                continue
            for j, number in enumerate(row):
                if absent(number, f"{path}[{i}][{j}]"):
                    good = False
                elif not _finite_number(number):
                    invalid.append(f"{path}[{i}][{j}]: expected a finite number")
                    good = False
        return [[float(x) for x in row] for row in item] if good else None

    def transform(item, path):
        mat = matrix(item, path, 4)
        if mat is None:
            return
        if mat[3] != [0.0, 0.0, 0.0, 1.0]:
            invalid.append(path + ": last row must be [0, 0, 0, 1]")
        rotation = [row[:3] for row in mat[:3]]
        if any(not math.isclose(sum(rotation[k][i] * rotation[k][j] for k in range(3)),
                                float(i == j), rel_tol=0, abs_tol=1e-6)
               for i in range(3) for j in range(3)):
            invalid.append(path + ": rotation must be orthonormal within 1e-6")
        a, b, c = rotation
        determinant = (a[0] * (b[1]*c[2] - b[2]*c[1])
                       - a[1] * (b[0]*c[2] - b[2]*c[0])
                       + a[2] * (b[0]*c[1] - b[1]*c[0]))
        if not math.isclose(determinant, 1.0, rel_tol=0, abs_tol=1e-6):
            invalid.append(path + ": rotation determinant must be +1 within 1e-6")

    def camera(item, path):
        if absent(item, path):
            return
        if type(item) is not dict:
            invalid.append(path + ": expected a camera object")
            return
        size = item.get("resolution")
        if not absent(size, path + ".resolution"):
            if type(size) is not list or len(size) != 2:
                invalid.append(path + ".resolution: expected [width, height]")
            else:
                for i, number in enumerate(size):
                    if not absent(number, f"{path}.resolution[{i}]") and (type(number) is not int or number <= 0):
                        invalid.append(f"{path}.resolution[{i}]: expected a positive integer")
        mat = matrix(item.get("K"), path + ".K", 3)
        if mat is not None:
            if mat[0][0] <= 0 or mat[1][1] <= 0:
                invalid.append(path + ".K: fx and fy must be positive")
            if mat[2] != [0.0, 0.0, 1.0] or mat[1][0] != 0.0:
                invalid.append(path + ".K: expected pinhole last row [0, 0, 1] and K[1][0]=0")
            if mat[0][1] != 0.0:
                unsupported.append(path + ".K[0][1]: nonzero skew is unsupported")

    def provenance(item):
        if absent(item, "provenance"):
            return
        if type(item) is not dict:
            invalid.append("provenance: expected an object")
            return
        status = item.get("status")
        if status == "incomplete":
            missing.append("provenance.status")
        else:
            text(status, "provenance.status", "nominal_public_geometry")
        text(item.get("note"), "provenance.note")
        sources = item.get("sources")
        if absent(sources, "provenance.sources"):
            return
        if type(sources) is not list:
            invalid.append("provenance.sources: expected a nonempty list")
            return
        if not sources:
            missing.append("provenance.sources")
        seen = set()
        for i, source in enumerate(sources):
            path = f"provenance.sources[{i}]"
            if absent(source, path):
                continue
            if type(source) is not dict:
                invalid.append(path + ": expected path and sha256")
                continue
            source_path = source.get("path")
            text(source_path, path + ".path")
            digest = source.get("sha256")
            text(digest, path + ".sha256")
            if isinstance(digest, str) and digest.strip() and not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                invalid.append(path + ".sha256: expected a 64-digit hexadecimal hash")
            if isinstance(source_path, str) and source_path.strip():
                if source_path in seen:
                    invalid.append(path + ".path: duplicate provenance source")
                seen.add(source_path)

    def scope(item):
        if absent(item, "validated_task_layouts"):
            return
        if type(item) is not dict:
            invalid.append("validated_task_layouts: expected a task to layout-list object")
            return
        if not item:
            missing.append("validated_task_layouts")
        for task, layouts in item.items():
            path = "validated_task_layouts." + str(task)
            if type(task) is not str or not task.strip():
                invalid.append("validated_task_layouts: task names must be nonempty strings")
            if absent(layouts, path):
                continue
            if type(layouts) is not list:
                invalid.append(path + ": expected a list of nonnegative integer layouts")
            elif not layouts:
                missing.append(path)
            else:
                seen = set()
                for i, layout in enumerate(layouts):
                    if absent(layout, f"{path}[{i}]"):
                        continue
                    if type(layout) is not int or layout < 0:
                        invalid.append(f"{path}[{i}]: expected a nonnegative integer")
                    elif layout in seen:
                        invalid.append(f"{path}[{i}]: duplicate layout")
                    else:
                        seen.add(layout)

    if not _json_safe(value):
        invalid.append("$: expected JSON data with finite numbers")
    if type(value) is not dict:
        invalid.append("$: expected a configuration object")
    else:
        for field, expected in SUPPORTED.items():
            text(value.get(field), field, expected)
        for field in DESCRIPTIVE:
            text(value.get(field), field)
        transform(value.get("head_to_environment_usd"), "head_to_environment_usd")
        transform(value.get("eef_to_wrist_usd"), "eef_to_wrist_usd")
        cameras = value.get("cameras")
        if not absent(cameras, "cameras"):
            if type(cameras) is not dict:
                invalid.append("cameras: expected an object keyed by supported camera names")
            else:
                if set(cameras) - set(CAMERAS):
                    unsupported.append("cameras: unsupported camera set")
                for name in CAMERAS:
                    camera(cameras.get(name), "cameras." + name)
        provenance(value.get("provenance"))
        scope(value.get("validated_task_layouts"))
        for field in (*SUPPORTED, *DESCRIPTIVE, "provenance", "validated_task_layouts"):
            if field in value and _json_safe(value[field]):
                known[field] = json.loads(json.dumps(value[field], allow_nan=False))

    status = ("INVALID" if invalid else "UNSUPPORTED" if unsupported else
              "INCOMPLETE" if missing else "CONFIG_READY")
    return {"status": status, "ready": status == "CONFIG_READY",
            "missing": list(dict.fromkeys(missing)), "errors": invalid + unsupported,
            "known": known, "limitations": list(LIMITATIONS)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def parse_config(raw):
    """Parse JSON text/bytes with duplicate-key rejection, without filling gaps.

    The parsed value is deliberately not configuration-validated. Pass it to
    inspect_config for a structured report, including missing parameters.
    """
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Cannot parse geometry configuration: " + str(exc)) from exc


def load_config(path):
    """Read an explicit JSON path and return validated data, or raise ValueError."""
    try:
        value = parse_config(Path(path).read_bytes())
    except (OSError, UnicodeError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Cannot load geometry configuration: " + str(exc)) from exc
    report = inspect_config(value)
    if not report["ready"]:
        details = report["errors"] + ["missing " + field for field in report["missing"]]
        raise ValueError(report["status"] + ": " + "; ".join(details))
    return value
