"""
Modeling Module for NEET Predictor

This module provides functions to:
- Train logistic regression and tree-based models
- Perform cross-validation with proper stratification
- Calibrate probability outputs
- Save and load trained models
- Evaluate model performance

Author: Data Science Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)

# For odds ratios and statistical inference
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def prepare_data_for_model(df: pd.DataFrame,
                           target_col: str = 'NEET',
                           numeric_features: List[str] = None,
                           categorical_features: List[str] = None,
                           weight_col: Optional[str] = None,
                           test_size: float = 0.2,
                           random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Prepare data for modeling with proper train/test split and preprocessing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_col : str
        Name of target column
    numeric_features : list
        List of numeric feature column names
    categorical_features : list
        List of categorical feature column names
    weight_col : str, optional
        Name of sample weight column
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing:
        - X_train, X_test: Feature matrices
        - y_train, y_test: Target vectors
        - sample_weight_train, sample_weight_test: Weight vectors (if applicable)
        - preprocessor: Fitted preprocessing pipeline
        - feature_names: List of final feature names after encoding
    """
    print("\n" + "="*60)
    print("PREPARING DATA FOR MODELING")
    print("="*60)
    
    # Auto-detect features if not provided
    if numeric_features is None and categorical_features is None:
        # Exclude target and special columns
        exclude_cols = [target_col, weight_col, 'id_hash', 'in_education', 
                       'employed', 'in_training']
        exclude_cols = [c for c in exclude_cols if c is not None]
        
        all_features = [c for c in df.columns if c not in exclude_cols]
        
        numeric_features = df[all_features].select_dtypes(
            include=['int64', 'float64', 'int32', 'float32']
        ).columns.tolist()
        
        categorical_features = df[all_features].select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
    
    print(f"\nFeatures:")
    print(f"  Numeric: {len(numeric_features)} - {numeric_features[:5]}{'...' if len(numeric_features) > 5 else ''}")
    print(f"  Categorical: {len(categorical_features)} - {categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}")
    
    # Extract X and y
    all_features = numeric_features + categorical_features
    X = df[all_features].copy()
    y = df[target_col].copy()
    
    # Extract sample weights if provided
    sample_weight = df[weight_col].values if weight_col and weight_col in df.columns else None
    
    # Stratified train-test split
    if sample_weight is not None:
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X, y, sample_weight,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        sw_train, sw_test = None, None
    
    print(f"\nTrain-test split:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  NEET prevalence - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Fit preprocessor on training data
    print("\n✓ Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after encoding
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                cat_features = transformer.named_steps['encoder'].get_feature_names_out(features)
                feature_names.extend(cat_features)
    
    print(f"✓ Final feature space: {X_train_processed.shape[1]} features")
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'sample_weight_train': sw_train,
        'sample_weight_test': sw_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }


def train_model(X_train: np.ndarray,
               y_train: np.ndarray,
               sample_weight: Optional[np.ndarray] = None,
               model_type: str = 'logistic',
               **model_params) -> Any:
    """
    Train a classification model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    sample_weight : array-like, optional
        Sample weights
    model_type : str
        Type of model: 'logistic', 'random_forest', 'gradient_boosting'
    **model_params : dict
        Additional parameters for the model
        
    Returns:
    --------
    model : fitted model object
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    if model_type == 'logistic':
        default_params = {
            'random_state': RANDOM_SEED,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'saga',
            'penalty': 'l2',
            'C': 1.0
        }
        default_params.update(model_params)
        model = LogisticRegression(**default_params)
        
    elif model_type == 'random_forest':
        default_params = {
            'random_state': RANDOM_SEED,
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'class_weight': 'balanced'
        }
        default_params.update(model_params)
        model = RandomForestClassifier(**default_params)
        
    elif model_type == 'gradient_boosting':
        default_params = {
            'random_state': RANDOM_SEED,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 50,
            'min_samples_leaf': 20
        }
        default_params.update(model_params)
        model = GradientBoostingClassifier(**default_params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print(f"Training {model_type} model (this may take a moment)...")
    
    fit_params = {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight
    
    model.fit(X_train, y_train, **fit_params)
    
    print(f"✓ Model training complete!")
    
    return model


def cross_validate_model(model: Any,
                        X: np.ndarray,
                        y: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None,
                        cv_folds: int = 5,
                        scoring: List[str] = None) -> pd.DataFrame:
    """
    Perform stratified k-fold cross-validation.
    
    Parameters:
    -----------
    model : sklearn model
        Model to cross-validate
    X : array-like
        Features
    y : array-like
        Target
    sample_weight : array-like, optional
        Sample weights
    cv_folds : int
        Number of CV folds
    scoring : list of str
        Scoring metrics
        
    Returns:
    --------
    cv_results : pd.DataFrame
        Cross-validation results
    """
    if scoring is None:
        scoring = ['roc_auc', 'precision', 'recall', 'f1']
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({cv_folds} folds)")
    print(f"{'='*60}")
    
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Prepare parameters for cross_validate (scikit-learn 1.4+ uses params instead of fit_params)
    params = {}
    if sample_weight is not None:
        params['params'] = {'sample_weight': sample_weight}
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        **params
    )
    
    # Format results
    results_df = pd.DataFrame(cv_results)
    
    print("\nCross-validation results:")
    for metric in scoring:
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        print(f"  {metric:15s}: Train = {train_scores.mean():.4f} (±{train_scores.std():.4f}), "
              f"Test = {test_scores.mean():.4f} (±{test_scores.std():.4f})")
    
    return results_df


def evaluate_model(model: Any,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  sample_weight: Optional[np.ndarray] = None,
                  threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Parameters:
    -----------
    model : fitted model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    sample_weight : array-like, optional
        Test sample weights
    threshold : float
        Classification threshold
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba, sample_weight=sample_weight),
        'brier_score': brier_score_loss(y_test, y_pred_proba, sample_weight=sample_weight),
        'precision': precision_score(y_test, y_pred, sample_weight=sample_weight),
        'recall': recall_score(y_test, y_pred, sample_weight=sample_weight),
        'f1_score': f1_score(y_test, y_pred, sample_weight=sample_weight)
    }
    
    print("\nTest Set Metrics:")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Brier Score:  {metrics['brier_score']:.4f} (lower is better)")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, sample_weight=sample_weight)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, sample_weight=sample_weight))
    
    return metrics


def calibrate_model(model: Any,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   sample_weight: Optional[np.ndarray] = None,
                   method: str = 'sigmoid',
                   cv: int = 5) -> CalibratedClassifierCV:
    """
    Calibrate model probability outputs.
    
    Parameters:
    -----------
    model : fitted model
        Model to calibrate
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    sample_weight : array-like, optional
        Sample weights
    method : str
        Calibration method: 'sigmoid' or 'isotonic'
    cv : int or 'prefit'
        Cross-validation strategy
        
    Returns:
    --------
    calibrated_model : CalibratedClassifierCV
        Calibrated model
    """
    print(f"\n{'='*60}")
    print(f"CALIBRATING MODEL (method={method})")
    print(f"{'='*60}")
    
    fit_params = {}
    if sample_weight is not None:
        fit_params = {'sample_weight': sample_weight}
    
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv
    )
    
    calibrated_model.fit(X_train, y_train, **fit_params)
    
    print("✓ Model calibration complete!")
    
    return calibrated_model


def save_model(model: Any,
              preprocessor: Any,
              filepath: str,
              metadata: Dict = None):
    """
    Save trained model and preprocessor to disk.
    
    Parameters:
    -----------
    model : fitted model
        Trained model
    preprocessor : fitted preprocessor
        Preprocessing pipeline
    filepath : str
        Path to save model (without extension)
    metadata : dict, optional
        Additional metadata to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'metadata': metadata or {}
    }
    
    full_path = f"{filepath}.pkl"
    joblib.dump(model_data, full_path)
    
    print(f"\n✓ Model saved to {full_path}")
    print(f"  File size: {Path(full_path).stat().st_size / 1024:.2f} KB")


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load trained model and preprocessor from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to model file
        
    Returns:
    --------
    model_data : dict
        Dictionary containing model, preprocessor, and metadata
    """
    if not filepath.endswith('.pkl'):
        filepath = f"{filepath}.pkl"
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_data = joblib.load(filepath)
    
    print(f"✓ Model loaded from {filepath}")
    
    return model_data


def compute_precision_at_k(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          k_percent: float = 10) -> float:
    """
    Compute precision at top k% of predictions.
    
    This metric measures model performance when targeting highest-risk individuals.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    k_percent : float
        Percentage of top predictions to consider
        
    Returns:
    --------
    precision_at_k : float
        Precision in top k% of predictions
    """
    n = len(y_true)
    k = int(n * k_percent / 100)
    
    # Get indices of top k predictions
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    
    # Calculate precision
    precision = y_true[top_k_idx].sum() / k
    
    return precision


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("MODELING MODULE - UNIT TESTS")
    print("="*60)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=RANDOM_SEED
    )
    
    # Create sample weights
    sample_weight = np.random.uniform(0.5, 2.0, size=len(y))
    
    # Train-test split
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weight,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )
    
    print(f"\nSynthetic data created:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  NEET rate: {y.mean():.2%}")
    
    # Train logistic regression
    model = train_model(X_train, y_train, sw_train, model_type='logistic')
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, sw_test)
    
    # Cross-validation
    cv_results = cross_validate_model(model, X, y, sample_weight, cv_folds=3)
    
    # Test precision@k
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    p_at_10 = compute_precision_at_k(y_test, y_pred_proba, k_percent=10)
    print(f"\n✓ Precision@10%: {p_at_10:.4f}")
    
    # Test save/load
    save_model(model, None, 'models/test_model', metadata={'test': True})
    loaded = load_model('models/test_model')
    print(f"✓ Model loaded successfully: {type(loaded['model'])}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
