"""
Custom metrics and scoring utilities.
Provides business-focused metrics for model evaluation.
"""

import numpy as np
from sklearn.metrics import make_scorer
from typing import Callable


def business_cost_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0
) -> float:
    """
    Calculate business cost based on false positives and false negatives.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        fp_cost: Cost of false positive (wrongly predicting churn)
        fn_cost: Cost of false negative (missing a churner)

    Returns:
        Negative cost (higher is better)
    """
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total_cost = (fp * fp_cost) + (fn * fn_cost)
    return -total_cost  # Negative because we want to minimize cost


def create_business_cost_scorer(
    fp_cost: float = 1.0,
    fn_cost: float = 5.0
) -> Callable:
    """
    Create a scikit-learn scorer for business cost.

    Args:
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative

    Returns:
        Scorer function
    """
    return make_scorer(
        business_cost_score,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        greater_is_better=True
    )
