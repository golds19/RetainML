"""
RetainML - Feature Engineering and Baseline Model
This script performs feature engineering and trains baseline models for customer churn prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class ChurnFeatureEngineering:
    """Feature engineering pipeline for customer churn prediction."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self, file_path):
        """Load and perform initial data cleaning."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Convert TotalCharges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Handle missing values
        missing_count = df['TotalCharges'].isnull().sum()
        print(f"Missing values in TotalCharges: {missing_count}")

        if missing_count > 0:
            # Fill missing TotalCharges with MonthlyCharges for customers with 0 tenure
            df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
            print(f"Missing values filled with MonthlyCharges")

        print(f"Final shape: {df.shape}")
        return df

    def create_features(self, df):
        """Create engineered features."""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)

        df = df.copy()

        # 1. Tenure groups
        print("\n1. Creating tenure groups...")
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
        )

        # 2. Monthly to total charge ratio
        print("2. Creating charge ratio...")
        df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)  # +1 to avoid division by zero

        # 3. Average monthly charge (TotalCharges / tenure)
        print("3. Creating average monthly charge...")
        df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero

        # 4. Service usage count
        print("4. Creating service usage indicators...")
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        # Count services used (excluding 'No' and 'No internet service' / 'No phone service')
        df['services_count'] = 0
        for col in service_cols:
            df['services_count'] += (~df[col].isin(['No', 'No internet service', 'No phone service'])).astype(int)

        # 5. Has internet service
        df['has_internet'] = (df['InternetService'] != 'No').astype(int)

        # 6. Has phone service
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)

        # 7. Family indicator (has partner or dependents)
        print("5. Creating family indicators...")
        df['has_family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)

        # 8. Contract risk (month-to-month is higher risk)
        print("6. Creating contract risk indicator...")
        df['contract_risk'] = (df['Contract'] == 'Month-to-month').astype(int)

        # 9. Payment risk (Electronic check has higher churn)
        print("7. Creating payment risk indicator...")
        df['payment_risk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

        # 10. Senior with no family (potentially vulnerable)
        df['senior_alone'] = ((df['SeniorCitizen'] == 1) & (df['has_family'] == 0)).astype(int)

        # 11. High charge indicator
        df['high_monthly_charge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

        print(f"\nNew features created. Total features: {df.shape[1]}")
        return df

    def encode_features(self, df):
        """Encode categorical features."""
        print("\n" + "="*80)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*80)

        df = df.copy()

        # Drop customerID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            print("Dropped customerID")

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target variable from categorical list
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')

        print(f"\nCategorical columns to encode: {len(categorical_cols)}")

        # Binary encoding for Yes/No columns
        binary_cols = []
        for col in categorical_cols:
            unique_vals = df[col].unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({'Yes', 'No'}):
                binary_cols.append(col)
                df[col] = (df[col] == 'Yes').astype(int)

        print(f"Binary encoded columns: {len(binary_cols)}")

        # One-hot encoding for multi-class categorical columns
        remaining_categorical = [col for col in categorical_cols if col not in binary_cols]

        if remaining_categorical:
            print(f"One-hot encoding columns: {remaining_categorical}")
            df = pd.get_dummies(df, columns=remaining_categorical, drop_first=True)

        print(f"\nFinal feature count: {df.shape[1]}")
        return df

    def prepare_data(self, df, target_col='Churn', test_size=0.2, random_state=42):
        """Prepare data for modeling."""
        print("\n" + "="*80)
        print("PREPARING DATA FOR MODELING")
        print("="*80)

        # Encode target variable
        df[target_col] = (df[target_col] == 'Yes').astype(int)

        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target distribution:")
        print(y.value_counts())
        print(f"Churn rate: {y.mean():.2%}")

        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_test_scaled, y_train, y_test

    def apply_smote(self, X_train, y_train, random_state=42):
        """Apply SMOTE to handle class imbalance."""
        print("\n" + "="*80)
        print("APPLYING SMOTE FOR CLASS BALANCING")
        print("="*80)

        print(f"Before SMOTE - Class distribution:")
        print(y_train.value_counts())

        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"\nAfter SMOTE - Class distribution:")
        print(pd.Series(y_train_balanced).value_counts())

        return X_train_balanced, y_train_balanced


class BaselineModels:
    """Train and evaluate baseline models."""

    def __init__(self):
        self.models = {}
        self.results = {}

    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple baseline models."""
        print("\n" + "="*80)
        print("TRAINING BASELINE MODELS")
        print("="*80)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            )
        }

        results = []

        for name, model in models.items():
            print(f"\n{'-'*80}")
            print(f"Training {name}...")

            # Train model
            model.fit(X_train, y_train)
            self.models[name] = model

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            }

            results.append(metrics)

            # Print metrics
            print(f"Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall:    {metrics['Recall']:.4f}")
            print(f"F1-Score:  {metrics['F1-Score']:.4f}")
            print(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")

            # Store predictions for later analysis
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': metrics
            }

        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('F1-Score', ascending=False)

        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(results_df.to_string(index=False))

        return results_df

    def evaluate_model(self, model_name, y_test):
        """Detailed evaluation of a specific model."""
        print("\n" + "="*80)
        print(f"DETAILED EVALUATION: {model_name}")
        print("="*80)

        y_pred = self.results[model_name]['y_pred']

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

        return cm

    def plot_results(self, y_test, save_path='../results'):
        """Create visualizations for model results."""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        import os
        os.makedirs(save_path, exist_ok=True)

        # 1. Model Comparison Bar Plot
        print("\n1. Creating model comparison plot...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        for idx, metric in enumerate(metrics):
            if idx < len(axes):
                metric_values = [self.results[model]['metrics'][metric] for model in self.models.keys()]
                model_names = list(self.models.keys())

                bars = axes[idx].bar(model_names, metric_values, color=['#3498db', '#e74c3c', '#2ecc71'])
                axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel(metric, fontsize=10)
                axes[idx].set_ylim([0, 1])
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(axis='y', alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{height:.3f}',
                                 ha='center', va='bottom', fontsize=9)

        # Remove extra subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}/model_comparison.png")
        plt.close()

        # 2. Confusion Matrices
        print("2. Creating confusion matrices...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = self.results[name]['y_pred']
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, square=True)
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
            axes[idx].set_xticklabels(['No Churn', 'Churn'])
            axes[idx].set_yticklabels(['No Churn', 'Churn'])

        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}/confusion_matrices.png")
        plt.close()

        # 3. ROC Curves
        print("3. Creating ROC curves...")
        plt.figure(figsize=(10, 8))

        for name in self.models.keys():
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}/roc_curves.png")
        plt.close()

        print("\nAll visualizations saved successfully!")

    def get_feature_importance(self, model_name, feature_names, top_n=20):
        """Get and plot feature importance for tree-based models."""
        if model_name not in ['Random Forest', 'Gradient Boosting']:
            print(f"Feature importance not available for {model_name}")
            return None

        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE: {model_name}")
        print(f"{'='*80}")

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            print(f"\nTop {top_n} Important Features:")
            print(feature_importance_df.head(top_n).to_string(index=False))

            # Plot
            plt.figure(figsize=(10, 8))
            top_features = feature_importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'../results/feature_importance_{model_name.replace(" ", "_").lower()}.png',
                       dpi=300, bbox_inches='tight')
            print(f"\nSaved feature importance plot")
            plt.close()

            return feature_importance_df

        return None


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("RETAINML - CUSTOMER CHURN PREDICTION")
    print("Feature Engineering & Baseline Model Development")
    print("="*80)

    # Initialize
    fe = ChurnFeatureEngineering()

    # Load data
    data_path = '../data/raw/Telco-Customer-Churn.csv'
    df = fe.load_data(data_path)

    # Feature engineering
    df_engineered = fe.create_features(df)

    # Encode features
    df_encoded = fe.encode_features(df_engineered)

    # Prepare data
    X_train, X_test, y_train, y_test = fe.prepare_data(df_encoded)

    # Apply SMOTE
    X_train_balanced, y_train_balanced = fe.apply_smote(X_train, y_train)

    # Train baseline models
    baseline = BaselineModels()
    results_df = baseline.train_models(X_train_balanced, y_train_balanced, X_test, y_test)

    # Detailed evaluation of best model
    best_model = results_df.iloc[0]['Model']
    baseline.evaluate_model(best_model, y_test)

    # Get feature importance
    baseline.get_feature_importance('Random Forest', X_train.columns, top_n=20)
    baseline.get_feature_importance('Gradient Boosting', X_train.columns, top_n=20)

    # Create visualizations
    baseline.plot_results(y_test)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    results_df.to_csv('../results/baseline_model_results.csv', index=False)
    print("Results saved to ../results/baseline_model_results.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nBest Model: {best_model}")
    print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
    print("\nNext steps:")
    print("1. Review feature importance and select top features")
    print("2. Perform hyperparameter tuning")
    print("3. Try ensemble methods")
    print("4. Implement cross-validation for robust evaluation")


if __name__ == "__main__":
    main()
