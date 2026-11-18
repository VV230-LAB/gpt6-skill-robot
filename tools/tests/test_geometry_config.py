"""CPU-only configuration contract tests; no research code is imported."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tools/configs/robodojo-x5-nominal.json"
TEMPLATE = ROOT / "tools/configs/template.json"
SOURCE = ROOT / "research/stack-bowls-r-three-content/geometry-config.json"


class GeometryConfigTests(unittest.TestCase):
    def api(self):
        path = ROOT / "tools/geometry_config.py"
        self.assertTrue(path.is_file(), "configuration validator is not implemented")
        spec = importlib.util.spec_from_file_location("geometry_config_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def nominal(self):
        self.assertTrue(CONFIG.is_file(), "nominal configuration is not extracted")
        return json.loads(CONFIG.read_text())

    def inspect(self, value):
        result = self.api().inspect_config(value)
        # This must succeed even when the inspected input contains NaN/Infinity.
        json.dumps(result, allow_nan=False)
        self.assertIs(type(result["ready"]), bool)
        self.assertIsInstance(result["missing"], list)
        self.assertIsInstance(result["errors"], list)
        self.assertIsInstance(result["known"], dict)
        self.assertIsInstance(result["limitations"], list)
        return result

    def assert_status(self, value, expected, path=None):
        result = self.inspect(value)
        self.assertEqual(result["status"], expected, result)
        self.assertEqual(result["ready"], expected == "CONFIG_READY")
        if path is not None:
            self.assertIn(path, result["missing"])
        return result

    def test_nominal_preserves_exact_original_numbers_and_explanations(self):
        nominal = self.nominal()
        original = json.loads(SOURCE.read_text())
        for key, value in original.items():
            if key != "schema":
                self.assertEqual(nominal[key], value, key)
        self.assertEqual(nominal["schema"], "tools.camera-geometry.v1")
        self.assertEqual(nominal["length_unit"], "m")
        self.assertEqual(nominal["quaternion_order"], "wxyz")
        self.assertEqual(nominal["projection_model"], "pinhole_no_distortion")
        self.assertEqual(nominal["validated_task_layouts"], {"stack_bowls": [0]})
        self.assertTrue(nominal["config_id"])
        provenance = nominal["provenance"]
        self.assertEqual(provenance["status"], "nominal_public_geometry")
        sources = {item["path"]: item["sha256"] for item in provenance["sources"]}
        self.assertEqual(sources[str(SOURCE.relative_to(ROOT))],
                         hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertIn("not physical calibration", provenance["note"])

    def test_nominal_is_ready_without_claiming_physical_success(self):
        result = self.assert_status(self.nominal(), "CONFIG_READY")
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["known"]["validated_task_layouts"], {"stack_bowls": [0]})
        self.assertTrue(any("calibration" in item.lower() for item in result["limitations"]))
        self.assertTrue(any("grasp" in item.lower() for item in result["limitations"]))

    def test_load_returns_validated_explicit_data_without_defaults(self):
        loaded = self.api().load_config(CONFIG)
        self.assertEqual(loaded, self.nominal())
        self.assertNotIn("plane_z", loaded)
        self.assertNotIn("object_plane", loaded)

    def test_template_reports_unknown_geometry_and_scope(self):
        self.assertTrue(TEMPLATE.is_file(), "unknown-parameter template is missing")
        value = json.loads(TEMPLATE.read_text())
        self.assertEqual(set(value), set(self.nominal()))
        self.assertEqual(value["validated_task_layouts"], {})
        self.assertTrue(all(x is None for row in value["head_to_environment_usd"] for x in row))
        self.assertTrue(all(x is None for row in value["eef_to_wrist_usd"] for x in row))
        for camera in value["cameras"].values():
            self.assertEqual(camera["resolution"], [None, None])
            self.assertTrue(all(x is None for row in camera["K"] for x in row))
        result = self.assert_status(value, "INCOMPLETE", "validated_task_layouts")
        self.assertIn("head_to_environment_usd[0][0]", result["missing"])
        self.assertIn("cameras.cam_high.K[0][0]", result["missing"])

    def test_missing_and_null_fields_report_paths(self):
        for key in self.nominal():
            for mutation in ("missing", "null"):
                with self.subTest(key=key, mutation=mutation):
                    value = self.nominal()
                    if mutation == "missing":
                        del value[key]
                    else:
                        value[key] = None
                    self.assert_status(value, "INCOMPLETE", key)
        value = self.nominal()
        del value["cameras"]["cam_high"]["K"]
        self.assert_status(value, "INCOMPLETE", "cameras.cam_high.K")
        value = self.nominal()
        value["cameras"]["cam_high"]["resolution"][1] = None
        self.assert_status(value, "INCOMPLETE", "cameras.cam_high.resolution[1]")

    def test_empty_required_text_and_provenance_are_incomplete(self):
        for key in ("config_id", "applicability", "intrinsics_convention"):
            value = self.nominal()
            value[key] = "  "
            self.assert_status(value, "INCOMPLETE", key)
        for field, replacement, path in (
            ("sources", [], "provenance.sources"),
            ("status", "", "provenance.status"),
            ("note", None, "provenance.note"),
        ):
            value = self.nominal()
            value["provenance"][field] = replacement
            self.assert_status(value, "INCOMPLETE", path)
        for field in ("path", "sha256"):
            value = self.nominal()
            value["provenance"]["sources"][0][field] = None
            self.assert_status(value, "INCOMPLETE", "provenance.sources[0]." + field)

    def test_incompatible_schema_profile_conventions_are_unsupported(self):
        for key, replacement in (
            ("schema", "future.v9"), ("robot_profile", "other"),
            ("eef_frame", "world"), ("eef_link", "tcp"),
            ("camera_convention", "+z forward"), ("length_unit", "mm"),
            ("quaternion_order", "xyzw"), ("projection_model", "fisheye"),
        ):
            with self.subTest(key=key):
                value = self.nominal()
                value[key] = replacement
                self.assert_status(value, "UNSUPPORTED")

    def test_intrinsics_description_accepts_another_explicit_public_source(self):
        value = self.nominal()
        value["intrinsics_convention"] = "Explicit pinhole intrinsics from versioned public camera specification"
        result = self.assert_status(value, "CONFIG_READY")
        self.assertEqual(result["known"]["intrinsics_convention"], value["intrinsics_convention"])

    def test_camera_set_is_explicit_and_exact(self):
        value = self.nominal()
        value["cameras"]["extra"] = copy.deepcopy(value["cameras"]["cam_high"])
        self.assert_status(value, "UNSUPPORTED")
        value = self.nominal()
        del value["cameras"]["cam_high"]
        self.assert_status(value, "INCOMPLETE", "cameras.cam_high")
        value = self.nominal()
        value["cameras"] = [value["cameras"]["cam_high"]] * 3
        self.assert_status(value, "INVALID")

    def test_positive_finite_focal_lengths_required(self):
        for axis in (0, 1):
            for replacement in (0, -1, True, "288", float("nan"), float("inf"), -float("inf")):
                with self.subTest(axis=axis, replacement=replacement):
                    value = self.nominal()
                    value["cameras"]["cam_high"]["K"][axis][axis] = replacement
                    self.assert_status(value, "INVALID")

    def test_intrinsics_shape_last_row_and_zero_skew(self):
        for matrix in ([[1, 0], [0, 1]], [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
                       [[1, 0, 0], [0.1, 1, 0], [0, 0, 1]]):
            value = self.nominal()
            value["cameras"]["cam_high"]["K"] = matrix
            self.assert_status(value, "INVALID")
        value = self.nominal()
        value["cameras"]["cam_high"]["K"][0][1] = 0.1
        self.assert_status(value, "UNSUPPORTED")

    def test_resolution_is_positive_integer_pair(self):
        for size in ([0, 480], [-1, 480], [True, 480], [640.0, 480],
                     [640, 480, 3], "640x480", {"w": 640, "h": 480}):
            value = self.nominal()
            value["cameras"]["cam_high"]["resolution"] = size
            self.assert_status(value, "INVALID")

    def test_rigid_transforms_reject_nonorthogonal_rotations_and_reflections(self):
        for key in ("head_to_environment_usd", "eef_to_wrist_usd"):
            for matrix in (
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                [[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            ):
                with self.subTest(key=key, matrix=matrix):
                    value = self.nominal()
                    value[key] = matrix
                    self.assert_status(value, "INVALID")

    def test_malformed_metadata_is_invalid_without_type_errors(self):
        for key, replacement in (
            ("schema", []), ("config_id", 42), ("provenance", []),
            ("validated_task_layouts", []), ("cameras", 42),
            ("head_to_environment_usd", "identity"), ("eef_to_wrist_usd", {}),
        ):
            value = self.nominal()
            value[key] = replacement
            self.assert_status(value, "INVALID")
        for sources in ([None], ["source"], [{"path": "x", "sha256": "invalid"}]):
            value = self.nominal()
            value["provenance"]["sources"] = sources
            self.assert_status(value, "INCOMPLETE" if sources == [None] else "INVALID")

    def test_scope_cannot_be_empty_duplicate_or_malformed(self):
        for scope, expected in (({}, "INCOMPLETE"), ({"stack_bowls": []}, "INCOMPLETE"),
                                ({"stack_bowls": [0, 0]}, "INVALID"),
                                ({"stack_bowls": [True]}, "INVALID"),
                                ({"stack_bowls": [-1]}, "INVALID"),
                                ({"stack_bowls": "0"}, "INVALID")):
            value = self.nominal()
            value["validated_task_layouts"] = scope
            self.assert_status(value, expected)

    def test_arbitrary_json_types_and_nonfinite_values_are_safe(self):
        for value in (None, [], [1], 0, 1.5, True, "config", float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assert_status(value, "INVALID")
        for value in ({}, {"cameras": {}}, {"cameras": {"cam_high": None}}):
            self.assert_status(value, "INCOMPLETE")
        for key, replacement in (("unexpected", float("nan")), ("config_id", float("inf"))):
            value = self.nominal()
            value[key] = replacement
            self.assert_status(value, "INVALID")

    def test_load_rejects_invalid_json_duplicates_and_missing_file_as_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for text in ('{"schema":', '{"schema":"a","schema":"b"}',
                         '{"cameras":{"cam_high":{},"cam_high":{}}}',
                         '{"x":NaN}', '{"x":Infinity}', 'null'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    self.api().load_config(path)
            path.write_text(json.dumps({"schema": "tools.camera-geometry.v1"}))
            with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
                self.api().load_config(path)
            with self.assertRaises(ValueError):
                self.api().load_config(Path(directory) / "absent.json")

    def test_parse_preserves_incomplete_config_for_structured_inspection(self):
        parse = getattr(self.api(), "parse_config", None)
        self.assertTrue(callable(parse), "shared strict JSON parser is not implemented")
        incomplete = {"schema": "tools.camera-geometry.v1", "cameras": {"cam_high": None}}
        for raw in (json.dumps(incomplete), json.dumps(incomplete).encode("utf-8")):
            value = parse(raw)
            self.assertEqual(value, incomplete)
            self.assert_status(value, "INCOMPLETE", "cameras.cam_high")

    def test_parse_rejects_duplicate_keys_including_nested_cameras(self):
        parse = getattr(self.api(), "parse_config", None)
        self.assertTrue(callable(parse), "shared strict JSON parser is not implemented")
        for raw in ('{"schema":"a","schema":"b"}',
                    b'{"cameras":{"cam_high":{},"cam_high":{}}}'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                parse(raw)

    def test_parse_reports_malformed_json_and_wrong_raw_types_as_value_error(self):
        parse = getattr(self.api(), "parse_config", None)
        self.assertTrue(callable(parse), "shared strict JSON parser is not implemented")
        for raw in ('{"schema":', b'\xff', None, [], {}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse(raw)


if __name__ == "__main__":
    unittest.main()
