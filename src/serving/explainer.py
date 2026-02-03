"""
AI Explainability for churn predictions.
Uses Random Forest feature importances × scaled feature values
to identify per-customer risk factors and generate retention plans.
"""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

# Maps each of the 43 encoded feature names to a human-readable concern description.
FEATURE_DESCRIPTIONS = {
    # Raw / binary-encoded features
    "SeniorCitizen": "Senior citizen — demographic associated with higher churn",
    "Partner": "No partner — single customers are less sticky",
    "Dependents": "No dependents — fewer household ties to the service",
    "tenure": "Short tenure — customer hasn't built loyalty yet",
    "PhoneService": "Phone service status affects engagement level",
    "PaperlessBilling": "Paperless billing — less tangible connection to service",
    "MonthlyCharges": "High monthly charges relative to perceived value",
    "TotalCharges": "Total spend pattern indicates engagement level",

    # Engineered numeric features
    "charge_ratio": "High ratio of monthly to total charges — possible recent price increase",
    "avg_monthly_charge": "High average monthly spending over tenure",
    "high_monthly_charge": "Monthly charges above median — price-sensitive segment",
    "services_count": "Few active services — low platform engagement",

    # Engineered indicator features
    "has_internet": "Internet service status drives engagement",
    "has_phone": "Phone service status drives engagement",
    "has_family": "No family ties (partner/dependents) — less sticky",
    "contract_risk": "Month-to-month contract — no long-term commitment",
    "payment_risk": "Electronic check payment — linked to higher churn",
    "senior_alone": "Senior citizen without family — vulnerable segment",

    # One-hot: gender
    "gender_Male": "Gender demographic factor",

    # One-hot: MultipleLines
    "MultipleLines_No phone service": "No phone service — limited product usage",
    "MultipleLines_Yes": "Multiple lines — higher engagement",

    # One-hot: InternetService
    "InternetService_Fiber optic": "Fiber optic customers churn more often",
    "InternetService_No": "No internet service — limited product engagement",

    # One-hot: OnlineSecurity
    "OnlineSecurity_No internet service": "No internet — can't use security services",
    "OnlineSecurity_Yes": "Has online security — value-add service",

    # One-hot: OnlineBackup
    "OnlineBackup_No internet service": "No internet — can't use backup services",
    "OnlineBackup_Yes": "Has online backup — value-add service",

    # One-hot: DeviceProtection
    "DeviceProtection_No internet service": "No internet — can't use protection services",
    "DeviceProtection_Yes": "Has device protection — value-add service",

    # One-hot: TechSupport
    "TechSupport_No internet service": "No internet — can't use tech support",
    "TechSupport_Yes": "Has tech support — customer feels supported",

    # One-hot: StreamingTV
    "StreamingTV_No internet service": "No internet — can't use streaming",
    "StreamingTV_Yes": "Has streaming TV — entertainment engagement",

    # One-hot: StreamingMovies
    "StreamingMovies_No internet service": "No internet — can't use streaming",
    "StreamingMovies_Yes": "Has streaming movies — entertainment engagement",

    # One-hot: Contract
    "Contract_One year": "One-year contract — moderate commitment",
    "Contract_Two year": "Two-year contract — strong commitment",

    # One-hot: PaymentMethod
    "PaymentMethod_Credit card (automatic)": "Auto-pay via credit card — lower friction",
    "PaymentMethod_Electronic check": "Electronic check — manual payment, higher churn risk",
    "PaymentMethod_Mailed check": "Mailed check — traditional payment method",

    # One-hot: tenure_group
    "tenure_group_1-2 years": "1-2 year customer — still in early loyalty phase",
    "tenure_group_2-4 years": "2-4 year customer — established relationship",
    "tenure_group_4+ years": "Long-term customer (4+ years) — strong loyalty",
}

# Maps feature names to retention action + reason.
# Each entry: (action, reason)
RETENTION_ACTIONS = {
    "contract_risk": (
        "Offer 15-20% discount for upgrading to an annual contract",
        "Month-to-month flexibility is the top churn driver",
    ),
    "payment_risk": (
        "Incentivize auto-pay switch with a $5/month credit",
        "Electronic check users churn at higher rates",
    ),
    "PaymentMethod_Electronic check": (
        "Incentivize auto-pay switch with a $5/month credit",
        "Manual payment increases churn friction",
    ),
    "tenure": (
        "Enroll in new customer loyalty program with early perks",
        "Short-tenure customers need early engagement",
    ),
    "tenure_group_1-2 years": (
        "Offer loyalty milestone reward at the 2-year mark",
        "Customers in the 1-2 year window are still building commitment",
    ),
    "MonthlyCharges": (
        "Review pricing and offer a bundle discount",
        "High monthly charges without perceived value drive churn",
    ),
    "high_monthly_charge": (
        "Offer a bundle discount for adding services",
        "Above-median pricing triggers price sensitivity",
    ),
    "charge_ratio": (
        "Review recent billing changes and offer price-lock guarantee",
        "Rising charge ratio suggests a recent price increase",
    ),
    "InternetService_Fiber optic": (
        "Bundle online security or tech support free for 6 months",
        "Fiber optic customers churn more — add value to retain",
    ),
    "services_count": (
        "Offer free trial of 2 additional services (backup, security)",
        "Low service adoption means less platform stickiness",
    ),
    "TechSupport_Yes": (
        "Offer complimentary 3-month tech support trial",
        "Lack of tech support leaves customers unsupported",
    ),
    "OnlineSecurity_Yes": (
        "Bundle free online security for 6 months",
        "Security services increase perceived value",
    ),
    "senior_alone": (
        "Assign a dedicated account manager",
        "Senior citizens without family need personalized support",
    ),
    "SeniorCitizen": (
        "Provide senior-friendly support and simplified billing",
        "Senior customers benefit from dedicated assistance",
    ),
    "has_family": (
        "Offer a family plan or multi-line discount",
        "Customers without family ties are less sticky",
    ),
    "Partner": (
        "Offer a partner referral discount",
        "Single customers have fewer household ties to the service",
    ),
    "Dependents": (
        "Offer a family bundle with parental controls",
        "No dependents means fewer reasons to stay",
    ),
    "PaperlessBilling": (
        "Send monthly value summary showing savings and usage",
        "Paperless customers have less tangible connection to service",
    ),
}

# Fallback for features without a specific action
_DEFAULT_ACTION = (
    "Schedule a customer success check-in call",
    "Proactive outreach to understand customer needs",
)


class ChurnExplainer:
    """Explains churn predictions using feature importance × feature value."""

    def __init__(self, model, feature_columns: list):
        self.feature_columns = feature_columns
        self.importances = model.feature_importances_

    def explain(self, scaled_features_df: pd.DataFrame, raw_customer_data: dict) -> dict:
        """
        Generate concerns and retention plan for a single prediction.

        Args:
            scaled_features_df: Scaled feature values (1-row DataFrame, 43 columns)
            raw_customer_data: Original customer input dict

        Returns:
            {"concerns": [...], "retention_plan": [...]}
        """
        values = scaled_features_df.values[0]
        contributions = self.importances * np.abs(values)

        # Top 5 contributing features
        top_indices = np.argsort(contributions)[::-1][:5]

        concerns = []
        seen_actions = set()
        retention_plan = []

        for rank, idx in enumerate(top_indices, start=1):
            feature = self.feature_columns[idx]
            score = float(contributions[idx])

            description = FEATURE_DESCRIPTIONS.get(feature, feature)
            concerns.append({
                "feature": feature,
                "description": description,
                "impact_score": round(score, 4),
            })

            # Find matching retention action
            action_text, reason = RETENTION_ACTIONS.get(feature, _DEFAULT_ACTION)

            # Avoid duplicate actions
            if action_text not in seen_actions:
                seen_actions.add(action_text)
                if rank <= 2:
                    priority = "high"
                elif rank <= 4:
                    priority = "medium"
                else:
                    priority = "low"

                retention_plan.append({
                    "action": action_text,
                    "reason": reason,
                    "priority": priority,
                })

        return {"concerns": concerns, "retention_plan": retention_plan}
