"""Tests for MLflowTracker. All MLflow calls are mocked."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from monitoring.mlflow_tracker import MLflowTracker


def _make_config(enabled=False, fail_silently=True):
    return {
        "mlflow": {
            "enabled": enabled,
            "tracking_uri": "./mlruns",
            "experiment_name": "test",
            "run_name_prefix": "test",
            "default_tags": {"project": "test"},
            "log_artifacts": {"plots": True, "results_csv": True, "config_snapshot": True},
            "model_registry": {
                "enabled": False,
                "register_best_model": False,
                "model_name": "test_model",
                "tags": {},
            },
            "connection": {"fail_silently": fail_silently},
        }
    }


class TestTrackerInit:

    def test_disabled_by_config(self):
        tracker = MLflowTracker(_make_config(enabled=False))
        assert tracker.enabled is False
        assert tracker._initialized is False

    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_enabled_calls_initialize(self, mock_uri, mock_exp):
        tracker = MLflowTracker(_make_config(enabled=True))
        mock_uri.assert_called_once_with("./mlruns")
        mock_exp.assert_called_once_with("test")
        assert tracker.enabled is True
        assert tracker._initialized is True

    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri", side_effect=Exception("fail"))
    def test_init_failure_silent(self, mock_uri):
        tracker = MLflowTracker(_make_config(enabled=True, fail_silently=True))
        assert tracker.enabled is False

    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri", side_effect=Exception("fail"))
    def test_init_failure_loud(self, mock_uri):
        with pytest.raises(Exception, match="fail"):
            MLflowTracker(_make_config(enabled=True, fail_silently=False))


class TestRunManagement:

    def test_pipeline_run_disabled_yields_none(self):
        tracker = MLflowTracker(_make_config(enabled=False))
        with tracker.start_pipeline_run() as run:
            assert run is None

    @patch("monitoring.mlflow_tracker.mlflow.end_run")
    @patch("monitoring.mlflow_tracker.mlflow.start_run")
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_pipeline_run_enabled(self, mock_uri, mock_exp, mock_start, mock_end):
        mock_start.return_value = MagicMock(info=MagicMock(run_id="abc123"))
        tracker = MLflowTracker(_make_config(enabled=True))

        with tracker.start_pipeline_run() as run:
            assert run is not None
        mock_start.assert_called()
        mock_end.assert_called()

    def test_model_run_disabled_yields_none(self):
        tracker = MLflowTracker(_make_config(enabled=False))
        with tracker.start_model_run("RF") as run:
            assert run is None

    @patch("monitoring.mlflow_tracker.mlflow.end_run")
    @patch("monitoring.mlflow_tracker.mlflow.start_run")
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_model_run_enabled_nested(self, mock_uri, mock_exp, mock_start, mock_end):
        mock_start.return_value = MagicMock()
        tracker = MLflowTracker(_make_config(enabled=True))

        with tracker.start_model_run("RF"):
            pass

        # The second call to start_run (first is pipeline_run if used, but here direct)
        call_kwargs = mock_start.call_args
        assert call_kwargs[1].get("nested") is True


class TestLogging:

    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def _make_enabled_tracker(self, mock_uri, mock_exp):
        return MLflowTracker(_make_config(enabled=True))

    def test_log_params_disabled_is_noop(self):
        tracker = MLflowTracker(_make_config(enabled=False))
        # Should not raise
        tracker.log_params({"a": 1, "b": 2})

    @patch("monitoring.mlflow_tracker.mlflow.log_param")
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_log_params_enabled(self, mock_uri, mock_exp, mock_log):
        tracker = MLflowTracker(_make_config(enabled=True))
        tracker.log_params({"a": 1, "b": 2})
        assert mock_log.call_count == 2

    @patch("monitoring.mlflow_tracker.mlflow.log_param")
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_log_params_flattens_nested(self, mock_uri, mock_exp, mock_log):
        tracker = MLflowTracker(_make_config(enabled=True))
        tracker.log_params({"outer": {"inner": "val"}})
        mock_log.assert_called_once_with("outer.inner", "val")

    def test_log_metrics_disabled_is_noop(self):
        tracker = MLflowTracker(_make_config(enabled=False))
        tracker.log_metrics({"acc": 0.9})

    @patch("monitoring.mlflow_tracker.mlflow.log_metric")
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_log_metrics_skips_non_numeric(self, mock_uri, mock_exp, mock_log):
        tracker = MLflowTracker(_make_config(enabled=True))
        tracker.log_metrics({"model": "RF", "acc": 0.9, "f1": 0.8})
        # Only acc and f1 should be logged (not "model")
        assert mock_log.call_count == 2

    @patch("monitoring.mlflow_tracker.mlflow.log_artifact")
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_log_artifact_missing_file_skipped(self, mock_uri, mock_exp, mock_log):
        tracker = MLflowTracker(_make_config(enabled=True))
        tracker.log_artifact("nonexistent_file_xyz.csv")
        mock_log.assert_not_called()


class TestSafeExecute:

    @patch("monitoring.mlflow_tracker.mlflow.log_param", side_effect=Exception("fail"))
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_catches_when_silent(self, mock_uri, mock_exp, mock_log):
        tracker = MLflowTracker(_make_config(enabled=True, fail_silently=True))
        result = tracker._safe_execute(mock_log, "key", "val")
        assert result is None

    @patch("monitoring.mlflow_tracker.mlflow.log_param", side_effect=Exception("fail"))
    @patch("monitoring.mlflow_tracker.mlflow.set_experiment")
    @patch("monitoring.mlflow_tracker.mlflow.set_tracking_uri")
    def test_raises_when_not_silent(self, mock_uri, mock_exp, mock_log):
        tracker = MLflowTracker(_make_config(enabled=True, fail_silently=False))
        with pytest.raises(Exception, match="fail"):
            tracker._safe_execute(mock_log, "key", "val")
