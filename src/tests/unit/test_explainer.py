"""Tests for serving.explainer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.explainer import ChurnExplainer


class TestChurnExplainer:
    def _make_scaled_df(self, feature_columns):
        """Create a 1-row scaled DataFrame with controlled values."""
        np.random.seed(99)
        values = np.random.randn(1, len(feature_columns))
        return pd.DataFrame(values, columns=feature_columns)

    def test_explain_returns_concerns(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        assert "concerns" in result
        assert isinstance(result["concerns"], list)

    def test_explain_returns_retention_plan(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        assert "retention_plan" in result
        assert isinstance(result["retention_plan"], list)

    def test_top_5_concerns(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        assert len(result["concerns"]) == 5

    def test_concerns_sorted_by_impact(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        scores = [c["impact_score"] for c in result["concerns"]]
        assert scores == sorted(scores, reverse=True)

    def test_concern_has_required_fields(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        for c in result["concerns"]:
            assert "feature" in c
            assert "description" in c
            assert "impact_score" in c

    def test_retention_action_has_required_fields(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        for a in result["retention_plan"]:
            assert "action" in a
            assert "reason" in a
            assert "priority" in a
            assert a["priority"] in ("high", "medium", "low")

    def test_priorities_ordered(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        priority_order = {"high": 0, "medium": 1, "low": 2}
        priorities = [priority_order[a["priority"]] for a in result["retention_plan"]]
        assert priorities == sorted(priorities)

    def test_no_duplicate_actions(self, mock_rf_model, feature_columns):
        explainer = ChurnExplainer(mock_rf_model, feature_columns)
        result = explainer.explain(self._make_scaled_df(feature_columns), {})
        actions = [a["action"] for a in result["retention_plan"]]
        assert len(actions) == len(set(actions))
