import pandas as pd
import shap

from src.pipeline import create_forecasting_features


def build_historical_features(history, feature_names):
    """Create historical rolling-window inputs for the forecasting model."""
    rows = []

    for end in range(12, len(history)):
        features = create_forecasting_features(history.iloc[:end])
        rows.append(features)

    if not rows:
        raise ValueError("Not enough historical data to calculate SHAP values.")

    X = pd.concat(rows, ignore_index=True)
    return X.reindex(columns=feature_names)


def calculate_shap(model, X, sample_size=120):
    """Calculate SHAP explanations for a sample of historical predictions."""
    X_sample = X.tail(sample_size).copy()

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)

    return X_sample, explanation
