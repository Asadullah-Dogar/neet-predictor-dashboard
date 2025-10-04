"""
Explainability Module for NEET Predictor

This module provides functions to:
- Compute SHAP values for model interpretation
- Calculate odds ratios with confidence intervals
- Generate feature importance plots
- Produce natural language explanations

Author: Data Science Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# SHAP for model interpretation
import shap

# Statsmodels for odds ratios
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from scipy import stats

# Sklearn imports
from sklearn.linear_model import LogisticRegression

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

RANDOM_SEED = 42


def compute_shap_values(model: Any,
                       X_sample: np.ndarray,
                       feature_names: List[str] = None,
                       sample_size: int = 100,
                       model_type: str = 'linear') -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretation.
    
    Parameters:
    -----------
    model : fitted model
        Trained model (logistic regression, tree-based, etc.)
    X_sample : array-like
        Sample of features to compute SHAP values for
    feature_names : list, optional
        Names of features
    sample_size : int
        Number of samples to use for SHAP computation (for efficiency)
    model_type : str
        Type of model: 'linear', 'tree', or 'kernel'
        
    Returns:
    --------
    shap_data : dict
        Dictionary containing:
        - shap_values: SHAP values array
        - expected_value: Base value
        - feature_names: Feature names
        - explainer: SHAP explainer object
    """
    print(f"\n{'='*60}")
    print("COMPUTING SHAP VALUES")
    print(f"{'='*60}")
    
    # Sample data if too large
    if len(X_sample) > sample_size:
        print(f"Sampling {sample_size} instances for SHAP computation...")
        idx = np.random.choice(len(X_sample), size=sample_size, replace=False)
        X_shap = X_sample[idx]
    else:
        X_shap = X_sample
    
    print(f"Computing SHAP values for {len(X_shap)} samples...")
    
    # Choose appropriate explainer
    if model_type == 'linear' or isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_shap)
        shap_values = explainer.shap_values(X_shap)
    elif model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Use KernelExplainer as fallback (slower)
        print("Using KernelExplainer (this may take longer)...")
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_shap, 50))
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    print("✓ SHAP computation complete!")
    
    return {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'feature_names': feature_names,
        'explainer': explainer,
        'X_sample': X_shap
    }


def plot_shap_summary(shap_data: Dict,
                     output_path: str = 'outputs/shap_summary.png',
                     top_k: int = 20):
    """
    Create SHAP summary plot showing feature importance.
    
    Parameters:
    -----------
    shap_data : dict
        SHAP data from compute_shap_values()
    output_path : str
        Path to save plot
    top_k : int
        Number of top features to display
    """
    print(f"\nCreating SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_data['shap_values'],
        shap_data['X_sample'],
        feature_names=shap_data['feature_names'],
        max_display=top_k,
        show=False
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ SHAP summary plot saved to {output_path}")


def plot_shap_waterfall(shap_data: Dict,
                       instance_idx: int = 0,
                       output_path: str = 'outputs/shap_waterfall.png'):
    """
    Create SHAP waterfall plot for a single prediction.
    
    Parameters:
    -----------
    shap_data : dict
        SHAP data from compute_shap_values()
    instance_idx : int
        Index of instance to explain
    output_path : str
        Path to save plot
    """
    print(f"\nCreating SHAP waterfall plot for instance {instance_idx}...")
    
    # Create explanation object for shap v0.40+
    shap_exp = shap.Explanation(
        values=shap_data['shap_values'][instance_idx],
        base_values=shap_data['expected_value'],
        data=shap_data['X_sample'][instance_idx],
        feature_names=shap_data['feature_names']
    )
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ SHAP waterfall plot saved to {output_path}")


def get_feature_importance_df(shap_data: Dict) -> pd.DataFrame:
    """
    Create feature importance dataframe from SHAP values.
    
    Parameters:
    -----------
    shap_data : dict
        SHAP data from compute_shap_values()
        
    Returns:
    --------
    importance_df : pd.DataFrame
        Sorted feature importance with mean absolute SHAP values
    """
    shap_values = shap_data['shap_values']
    feature_names = shap_data['feature_names']
    
    # Calculate mean absolute SHAP value for each feature
    importance = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names if feature_names else [f'Feature_{i}' for i in range(len(importance))],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def compute_odds_ratios(model: LogisticRegression,
                       feature_names: List[str],
                       confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute odds ratios with confidence intervals for logistic regression.
    
    Parameters:
    -----------
    model : LogisticRegression
        Fitted logistic regression model
    feature_names : list
        Names of features
    confidence : float
        Confidence level for intervals
        
    Returns:
    --------
    or_df : pd.DataFrame
        Dataframe with odds ratios and confidence intervals
    """
    print(f"\n{'='*60}")
    print("COMPUTING ODDS RATIOS")
    print(f"{'='*60}")
    
    # Extract coefficients
    coefs = model.coef_[0]
    
    # Compute odds ratios
    odds_ratios = np.exp(coefs)
    
    # For confidence intervals, we need standard errors
    # If using statsmodels Logit, we get these directly
    # For sklearn, we approximate using bootstrap or assume normal distribution
    
    # Simple approximation (for demonstration)
    # In production, use statsmodels Logit for proper inference
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Rough SE estimate (this is simplified - use statsmodels for proper CI)
    # se_coefs = np.sqrt(np.diag(np.linalg.inv(model._coef_hessian)))
    # For sklearn, we'll just show point estimates
    
    or_df = pd.DataFrame({
        'feature': feature_names if feature_names else [f'Feature_{i}' for i in range(len(coefs))],
        'coefficient': coefs,
        'odds_ratio': odds_ratios,
        'log_odds': coefs
    })
    
    # Add interpretation
    or_df['interpretation'] = or_df['odds_ratio'].apply(
        lambda x: f"{'Increases' if x > 1 else 'Decreases'} odds by {abs((x-1)*100):.1f}%"
    )
    
    # Sort by absolute coefficient
    or_df = or_df.iloc[np.argsort(-np.abs(coefs))]
    
    print("\nTop 10 Features by Odds Ratio:")
    print(or_df.head(10)[['feature', 'odds_ratio', 'interpretation']].to_string(index=False))
    
    return or_df


def fit_statsmodels_logit(X: np.ndarray,
                          y: np.ndarray,
                          feature_names: List[str] = None) -> Tuple[Any, pd.DataFrame]:
    """
    Fit logistic regression using statsmodels for proper statistical inference.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    feature_names : list
        Feature names
        
    Returns:
    --------
    result : statsmodels results object
    or_df : pd.DataFrame
        Odds ratios with confidence intervals
    """
    print(f"\n{'='*60}")
    print("FITTING STATSMODELS LOGIT")
    print(f"{'='*60}")
    
    # Add constant (intercept)
    X_with_const = sm.add_constant(X)
    
    # Fit model
    print("Fitting logistic regression with statsmodels...")
    logit_model = Logit(y, X_with_const)
    result = logit_model.fit(disp=0)
    
    # Extract results (exclude intercept)
    params = result.params[1:]
    conf_int = result.conf_int()[1:]  # Returns ndarray, exclude first row
    
    # Compute odds ratios
    odds_ratios = np.exp(params)
    or_ci_lower = np.exp(conf_int[:, 0])  # First column of confidence intervals
    or_ci_upper = np.exp(conf_int[:, 1])  # Second column of confidence intervals
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(params))]
    
    # Convert to arrays
    params_arr = params.values if hasattr(params, 'values') else np.array(params)
    odds_ratios_arr = odds_ratios.values if hasattr(odds_ratios, 'values') else np.array(odds_ratios)
    or_ci_lower_arr = or_ci_lower if isinstance(or_ci_lower, np.ndarray) else np.array(or_ci_lower)
    or_ci_upper_arr = or_ci_upper if isinstance(or_ci_upper, np.ndarray) else np.array(or_ci_upper)
    p_values_arr = result.pvalues[1:].values if hasattr(result.pvalues[1:], 'values') else np.array(result.pvalues[1:])
    
    or_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': params_arr,
        'odds_ratio': odds_ratios_arr,
        'or_ci_lower': or_ci_lower_arr,
        'or_ci_upper': or_ci_upper_arr,
        'p_value': p_values_arr
    })
    
    # Add significance stars
    or_df['significance'] = or_df['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # Sort by absolute coefficient
    or_df = or_df.iloc[np.argsort(-np.abs(params_arr))]
    
    print("\nModel summary:")
    print(f"  AIC: {result.aic:.2f}")
    print(f"  Pseudo R²: {result.prsquared:.4f}")
    
    print("\nTop 10 Features:")
    cols_to_show = ['feature', 'odds_ratio', 'or_ci_lower', 'or_ci_upper', 'p_value', 'significance']
    print(or_df.head(10)[cols_to_show].to_string(index=False))
    
    return result, or_df


def generate_natural_language_explanation(or_df: pd.DataFrame,
                                         top_k: int = 5) -> List[str]:
    """
    Generate natural language explanations for top features.
    
    Parameters:
    -----------
    or_df : pd.DataFrame
        Odds ratio dataframe
    top_k : int
        Number of top features to explain
        
    Returns:
    --------
    explanations : list of str
        Natural language explanations
    """
    explanations = []
    
    top_features = or_df.head(top_k)
    
    for _, row in top_features.iterrows():
        feature = row['feature']
        or_val = row['odds_ratio']
        
        # Clean feature name
        feature_clean = feature.replace('_', ' ').title()
        
        if or_val > 1:
            pct_change = (or_val - 1) * 100
            direction = "increases"
            explanation = (
                f"**{feature_clean}**: Having this characteristic {direction} "
                f"the odds of being NEET by {pct_change:.1f}%, "
                f"after controlling for other factors."
            )
        else:
            pct_change = (1 - or_val) * 100
            direction = "decreases"
            explanation = (
                f"**{feature_clean}**: Having this characteristic {direction} "
                f"the odds of being NEET by {pct_change:.1f}%, "
                f"after controlling for other factors."
            )
        
        # Add significance note if available
        if 'significance' in row and row['significance']:
            explanation += f" {row['significance']}"
        
        explanations.append(explanation)
    
    return explanations


def plot_odds_ratios(or_df: pd.DataFrame,
                    output_path: str = 'outputs/odds_ratios.png',
                    top_k: int = 15):
    """
    Create forest plot of odds ratios with confidence intervals.
    
    Parameters:
    -----------
    or_df : pd.DataFrame
        Odds ratio dataframe
    output_path : str
        Path to save plot
    top_k : int
        Number of top features to plot
    """
    print(f"\nCreating odds ratio plot...")
    
    # Select top features
    plot_df = or_df.head(top_k).copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot odds ratios
    y_pos = np.arange(len(plot_df))
    
    ax.scatter(plot_df['odds_ratio'], y_pos, s=100, zorder=3)
    
    # Add confidence intervals if available
    if 'or_ci_lower' in plot_df.columns:
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax.plot([row['or_ci_lower'], row['or_ci_upper']], 
                   [i, i], 'k-', linewidth=2, zorder=2)
    
    # Add reference line at OR=1
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    ax.set_xlabel('Odds Ratio', fontsize=12)
    ax.set_title(f'Top {top_k} Features: Odds Ratios with 95% CI', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Odds ratio plot saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("EXPLAINABILITY MODULE - UNIT TESTS")
    print("="*60)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=RANDOM_SEED
    )
    
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Train model
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)
    
    print(f"\nModel trained on {len(X_train)} samples")
    
    # Test SHAP
    shap_data = compute_shap_values(
        model, X_test, feature_names, sample_size=50, model_type='linear'
    )
    
    # Test feature importance
    importance_df = get_feature_importance_df(shap_data)
    print("\nFeature Importance (SHAP):")
    print(importance_df.head())
    
    # Test odds ratios
    or_df = compute_odds_ratios(model, feature_names)
    
    # Test statsmodels logit
    result, or_df_sm = fit_statsmodels_logit(X_train, y_train, feature_names)
    
    # Test natural language explanation
    explanations = generate_natural_language_explanation(or_df_sm, top_k=3)
    print("\nNatural Language Explanations:")
    for i, exp in enumerate(explanations, 1):
        print(f"{i}. {exp}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
