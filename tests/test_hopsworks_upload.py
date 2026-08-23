"""Local regression tests for Hopsworks upload preparation and error handling."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    TARGET_COLUMN,
)
from src.features import pipeline


class HopsworksUploadTestCase(unittest.TestCase):
    """Verify upload behavior without importing the SDK or accessing a network."""

    def setUp(self) -> None:
        self.source = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01 03:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                ],
                "aqi": [150.0, 100.0, 120.0],
                "temperature": [23.0, 21.0, 22.0],
                "humidity": [53.0, 51.0, 52.0],
                "pressure": [1003.0, 1001.0, 1002.0],
                "modeled_pm10": [83.0, 81.0, 82.0],
                "modeled_pm25": [43.0, 41.0, 42.0],
                "ozone": [33.0, 31.0, 32.0],
                "no2": [13.0, 11.0, 12.0],
                "no": [3.0, 1.0, 2.0],
                "so2": [8.0, 6.0, 7.0],
                "co": [0.8, 0.6, 0.7],
                # Deliberately incorrect values must be recomputed from timestamp.
                "hour": [99, 99, 99],
                "day": [99, 99, 99],
                "day_of_week": [99, 99, 99],
                "week_of_year": [99, 99, 99],
                "month": [99, 99, 99],
                "aqi_change_rate": [99.0, 99.0, 99.0],
                "aqi_lag_1": [99.0, 99.0, 99.0],
            }
        )

    @staticmethod
    def _fake_hopsworks(insert_error: BaseException | None = None):
        feature_group = Mock(name="feature_group")
        feature_group.online_enabled = False
        if insert_error is not None:
            feature_group.insert.side_effect = insert_error

        feature_store = Mock(name="feature_store")
        feature_store.get_feature_group.side_effect = Exception("Not found")
        feature_store.get_or_create_feature_group.return_value = feature_group

        project = Mock(name="project")
        project.get_feature_store.return_value = feature_store

        login = Mock(name="login", return_value=project)
        module = types.SimpleNamespace(login=login)
        return module, login, project, feature_store, feature_group

    def test_prepare_dataframe_preserves_schema_precision_and_direct_aqi_features(self) -> None:
        original = self.source.copy(deep=True)

        prepared = pipeline._prepare_hopsworks_dataframe(self.source)

        self.assertEqual(prepared.columns.tolist(), ["timestamp", TARGET_COLUMN, *FEATURE_COLUMNS])
        self.assertEqual(len(prepared.columns), 19)
        self.assertEqual(str(prepared["timestamp"].dtype), "datetime64[us]")
        self.assertEqual(str(prepared[TARGET_COLUMN].dtype), "int64")
        self.assertEqual(str(prepared["week_of_year"].dtype), "int64")
        self.assertEqual(str(prepared["hour"].dtype), "int32")
        self.assertEqual(str(prepared["day"].dtype), "int32")
        self.assertEqual(str(prepared["day_of_week"].dtype), "int32")
        self.assertEqual(str(prepared["month"].dtype), "int32")
        self.assertEqual(prepared["timestamp"].dt.hour.tolist(), [1, 2, 3])
        self.assertEqual(prepared[TARGET_COLUMN].tolist(), [100, 120, 150])
        self.assertEqual(prepared["hour"].tolist(), [1, 2, 3])
        self.assertEqual(prepared["day"].tolist(), [1, 1, 1])
        self.assertEqual(prepared["day_of_week"].tolist(), [3, 3, 3])
        self.assertEqual(prepared["week_of_year"].tolist(), [1, 1, 1])
        self.assertEqual(prepared["month"].tolist(), [1, 1, 1])
        self.assertEqual(prepared["aqi_change_rate"].tolist(), [0.0, 0.0, 20.0])
        self.assertEqual(prepared["aqi_lag_1"].tolist(), [100.0, 100.0, 120.0])
        pd.testing.assert_frame_equal(self.source, original)

    def test_each_native_transport_signature_is_recognized_on_windows(self) -> None:
        with patch.object(pipeline.os, "name", "nt"):
            for signature in pipeline._NATIVE_HOPSWORKS_TRANSPORT_SIGNATURES:
                with self.subTest(signature=signature):
                    self.assertTrue(
                        pipeline._is_native_hopsworks_transport_error(
                            RuntimeError(f"Transport failed: {signature.upper()}")
                        )
                    )

    def test_native_transport_signature_is_found_through_cause(self) -> None:
        inner = OSError("Failed to libgssapi_krb5")
        outer = RuntimeError("outer failure")
        outer.__cause__ = inner

        with patch.object(pipeline.os, "name", "nt"):
            self.assertTrue(pipeline._is_native_hopsworks_transport_error(outer))

    def test_native_transport_signature_is_found_through_context(self) -> None:
        inner = OSError("RPC listener disconnected")
        outer = RuntimeError("outer failure")
        outer.__context__ = inner

        with patch.object(pipeline.os, "name", "nt"):
            self.assertTrue(pipeline._is_native_hopsworks_transport_error(outer))

    def test_exception_chain_cycle_terminates_safely(self) -> None:
        first = RuntimeError("first unrelated failure")
        second = RuntimeError("second unrelated failure")
        first.__cause__ = second
        second.__cause__ = first

        with patch.object(pipeline.os, "name", "nt"):
            self.assertFalse(pipeline._is_native_hopsworks_transport_error(first))

    def test_native_transport_error_is_rejected_on_non_windows(self) -> None:
        with patch.object(pipeline.os, "name", "posix"):
            self.assertFalse(
                pipeline._is_native_hopsworks_transport_error(
                    RuntimeError("Generic HdfsObjectStore error")
                )
            )

    def test_unrelated_error_is_rejected_on_windows(self) -> None:
        with patch.object(pipeline.os, "name", "nt"):
            self.assertFalse(
                pipeline._is_native_hopsworks_transport_error(
                    ValueError("Feature schema mismatch")
                )
            )

    def test_upload_configures_feature_group_and_inserts_prepared_dataframe(self) -> None:
        fake, login, project, feature_store, feature_group = self._fake_hopsworks()

        with patch.dict(sys.modules, {"hopsworks": fake}), patch.dict(
            os.environ, {"HOPSWORKS_API_KEY": "test-api-key"}, clear=False
        ), patch.object(pipeline, "_uses_windows_hopsworks_transport", return_value=False):
            pipeline._upload_to_hopsworks(self.source)

        login.assert_called_once_with(api_key_value="test-api-key")
        project.get_feature_store.assert_called_once_with()
        feature_store.get_or_create_feature_group.assert_called_once_with(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            description=(
                "Lahore AQI features — 17 clean weather/AQ features and "
                "Open-Meteo us_aqi target."
            ),
            event_time="timestamp",
            online_enabled=True,
        )
        feature_group.insert.assert_called_once()
        inserted = feature_group.insert.call_args.args[0]
        expected = pipeline._prepare_hopsworks_dataframe(self.source)
        pd.testing.assert_frame_equal(inserted, expected)

    def test_upload_requires_api_key_before_login(self) -> None:
        fake, login, _project, _feature_store, _feature_group = self._fake_hopsworks()

        with patch.dict(sys.modules, {"hopsworks": fake}), patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "HOPSWORKS_API_KEY"):
                pipeline._upload_to_hopsworks(self.source)

        login.assert_not_called()

    def test_upload_reraises_unrelated_insert_error_unchanged(self) -> None:
        insert_error = ValueError("Feature schema mismatch")
        fake, _login, _project, _feature_store, _feature_group = self._fake_hopsworks(
            insert_error
        )

        with patch.dict(sys.modules, {"hopsworks": fake}), patch.dict(
            os.environ, {"HOPSWORKS_API_KEY": "test-api-key"}, clear=False
        ), patch.object(pipeline, "_uses_windows_hopsworks_transport", return_value=False), patch.object(pipeline.os, "name", "nt"):
            with self.assertRaises(ValueError) as raised:
                pipeline._upload_to_hopsworks(self.source)

        self.assertIs(raised.exception, insert_error)

    def test_upload_wraps_native_windows_transport_error_with_original_cause(self) -> None:
        insert_error = OSError("IO error occurred while communicating with HDFS")
        fake, _login, _project, _feature_store, _feature_group = self._fake_hopsworks(
            insert_error
        )

        with patch.dict(sys.modules, {"hopsworks": fake}), patch.dict(
            os.environ, {"HOPSWORKS_API_KEY": "test-api-key"}, clear=False
        ), patch.object(pipeline, "_uses_windows_hopsworks_transport", return_value=False), patch.object(pipeline.os, "name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "Linux or WSL") as raised:
                pipeline._upload_to_hopsworks(self.source)

        self.assertIs(raised.exception.__cause__, insert_error)


if __name__ == "__main__":
    unittest.main()
