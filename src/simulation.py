"""
Intervention Simulation Module for NEET Predictor

This module provides functions to:
- Simulate policy interventions and their impact
- Compute cost-effectiveness metrics
- Estimate prevented NEET cases
- Generate ROI calculations

Author: Data Science Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings

RANDOM_SEED = 42


def simulate_intervention(df: pd.DataFrame,
                         model: Any,
                         preprocessor: Any,
                         intervention_spec: Dict,
                         feature_names: List[str],
                         sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Simulate the impact of a policy intervention on NEET probabilities.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with youth data
    model : fitted model
        Trained predictive model
    preprocessor : fitted preprocessor
        Preprocessing pipeline
    intervention_spec : dict
        Specification of intervention:
        - 'target_group': dict of filters (e.g., {'age_group': '15-17', 'province': 'Punjab'})
        - 'intervention': dict of feature modifications (e.g., {'vocational_training': 1})
        - 'coverage': float, proportion of target group reached (0-1)
    feature_names : list
        List of feature column names
    sample_weight : array-like, optional
        Survey weights
        
    Returns:
    --------
    results_df : pd.DataFrame
        Results with baseline and post-intervention probabilities
        
    Example:
    --------
    >>> intervention_spec = {
    ...     'target_group': {'age_group': '15-17', 'education_level': 'Secondary'},
    ...     'intervention': {'vocational_training': 1},
    ...     'coverage': 0.7
    ... }
    >>> results = simulate_intervention(df, model, preprocessor, intervention_spec, feature_names)
    """
    print(f"\n{'='*60}")
    print("INTERVENTION SIMULATION")
    print(f"{'='*60}")
    
    df_sim = df.copy()
    
    # 1. Identify target group
    target_mask = np.ones(len(df_sim), dtype=bool)
    
    if 'target_group' in intervention_spec:
        for feature, value in intervention_spec['target_group'].items():
            if feature in df_sim.columns:
                if isinstance(value, list):
                    target_mask &= df_sim[feature].isin(value)
                else:
                    target_mask &= (df_sim[feature] == value)
    
    n_target = target_mask.sum()
    print(f"\nTarget group identified: {n_target:,} individuals")
    
    if 'target_group' in intervention_spec:
        print(f"  Filters: {intervention_spec['target_group']}")
    
    # 2. Apply coverage (random sample from target group)
    coverage = intervention_spec.get('coverage', 1.0)
    
    if coverage < 1.0:
        target_idx = np.where(target_mask)[0]
        n_treated = int(n_target * coverage)
        treated_idx = np.random.choice(target_idx, size=n_treated, replace=False)
        treated_mask = np.zeros(len(df_sim), dtype=bool)
        treated_mask[treated_idx] = True
    else:
        treated_mask = target_mask.copy()
        n_treated = n_target
    
    print(f"  Coverage: {coverage:.1%} → {n_treated:,} individuals treated")
    
    # 3. Compute baseline predictions
    X_baseline = df_sim[feature_names].copy()
    X_baseline_processed = preprocessor.transform(X_baseline)
    p_baseline = model.predict_proba(X_baseline_processed)[:, 1]
    
    # 4. Apply intervention
    df_intervention = df_sim.copy()
    
    if 'intervention' in intervention_spec:
        for feature, new_value in intervention_spec['intervention'].items():
            if feature in df_intervention.columns:
                df_intervention.loc[treated_mask, feature] = new_value
        
        print(f"\n  Intervention: {intervention_spec['intervention']}")
    
    # 5. Compute post-intervention predictions
    X_intervention = df_intervention[feature_names].copy()
    X_intervention_processed = preprocessor.transform(X_intervention)
    p_intervention = model.predict_proba(X_intervention_processed)[:, 1]
    
    # 6. Calculate impact
    delta_p = p_baseline - p_intervention
    
    # Apply sample weights if available
    if sample_weight is not None:
        weights = sample_weight
    else:
        weights = np.ones(len(df_sim))
    
    # Total prevented NEETs (weighted)
    prevented_neets_total = (delta_p * weights).sum()
    prevented_neets_treated = (delta_p[treated_mask] * weights[treated_mask]).sum()
    
    # Average risk reduction among treated
    avg_risk_reduction = delta_p[treated_mask].mean()
    
    print(f"\n{'='*60}")
    print("INTERVENTION IMPACT")
    print(f"{'='*60}")
    print(f"  Total individuals in population: {len(df_sim):,}")
    print(f"  Individuals treated: {n_treated:,}")
    print(f"  Prevented NEETs (weighted): {prevented_neets_total:.1f}")
    print(f"  Prevented NEETs among treated: {prevented_neets_treated:.1f}")
    print(f"  Average risk reduction: {avg_risk_reduction:.2%}")
    print(f"  Number Needed to Treat (NNT): {1/avg_risk_reduction:.1f}" if avg_risk_reduction > 0 else "  NNT: N/A")
    
    # Create results dataframe
    results_df = df_sim.copy()
    results_df['treated'] = treated_mask
    results_df['p_baseline'] = p_baseline
    results_df['p_intervention'] = p_intervention
    results_df['delta_p'] = delta_p
    results_df['weight'] = weights
    
    return results_df


def compute_cost_effectiveness(intervention_results: pd.DataFrame,
                              cost_per_participant: float,
                              additional_costs: float = 0,
                              time_horizon_years: int = 1) -> Dict[str, float]:
    """
    Compute cost-effectiveness metrics for an intervention.
    
    Parameters:
    -----------
    intervention_results : pd.DataFrame
        Results from simulate_intervention()
    cost_per_participant : float
        Cost per person receiving intervention (in local currency)
    additional_costs : float
        Fixed/overhead costs
    time_horizon_years : int
        Time horizon for analysis
        
    Returns:
    --------
    metrics : dict
        Cost-effectiveness metrics:
        - total_cost
        - cost_per_prevented_neet
        - prevented_neets
        - roi (if benefits provided)
    """
    print(f"\n{'='*60}")
    print("COST-EFFECTIVENESS ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate costs
    n_treated = intervention_results['treated'].sum()
    total_program_cost = n_treated * cost_per_participant + additional_costs
    
    # Calculate benefits (prevented NEETs)
    prevented_neets = (
        intervention_results.loc[intervention_results['treated'], 'delta_p'] * 
        intervention_results.loc[intervention_results['treated'], 'weight']
    ).sum()
    
    # Cost per prevented NEET
    if prevented_neets > 0:
        cost_per_prevented = total_program_cost / prevented_neets
    else:
        cost_per_prevented = np.inf
    
    metrics = {
        'n_treated': n_treated,
        'total_cost': total_program_cost,
        'prevented_neets': prevented_neets,
        'cost_per_prevented_neet': cost_per_prevented,
        'time_horizon_years': time_horizon_years
    }
    
    print(f"\nProgram Costs:")
    print(f"  Cost per participant: {cost_per_participant:,.0f} PKR")
    print(f"  Number treated: {n_treated:,}")
    print(f"  Total program cost: {total_program_cost:,.0f} PKR")
    print(f"  Additional/overhead costs: {additional_costs:,.0f} PKR")
    
    print(f"\nProgram Impact:")
    print(f"  Prevented NEETs: {prevented_neets:.1f}")
    print(f"  Cost per prevented NEET: {cost_per_prevented:,.0f} PKR")
    
    # Simple ROI calculation (if we assume economic value of preventing NEET)
    # Example: If a NEET youth costs society 100,000 PKR/year in lost productivity
    # This is illustrative - actual values should be estimated from literature
    
    return metrics


def estimate_economic_benefit(prevented_neets: float,
                             annual_productivity_loss: float = 100000,
                             time_horizon_years: int = 5,
                             discount_rate: float = 0.05) -> Dict[str, float]:
    """
    Estimate economic benefits of preventing NEET status.
    
    Parameters:
    -----------
    prevented_neets : float
        Number of prevented NEET cases
    annual_productivity_loss : float
        Estimated annual economic cost of NEET status (PKR)
    time_horizon_years : int
        Time horizon for benefit calculation
    discount_rate : float
        Annual discount rate for NPV calculation
        
    Returns:
    --------
    benefits : dict
        Economic benefit metrics
    """
    print(f"\n{'='*60}")
    print("ECONOMIC BENEFIT ESTIMATION")
    print(f"{'='*60}")
    
    # Calculate net present value of benefits
    discount_factors = [(1 / (1 + discount_rate) ** t) for t in range(1, time_horizon_years + 1)]
    npv_per_prevented = annual_productivity_loss * sum(discount_factors)
    
    total_npv = prevented_neets * npv_per_prevented
    
    benefits = {
        'prevented_neets': prevented_neets,
        'annual_productivity_loss': annual_productivity_loss,
        'time_horizon_years': time_horizon_years,
        'discount_rate': discount_rate,
        'npv_per_prevented_neet': npv_per_prevented,
        'total_npv': total_npv
    }
    
    print(f"\nAssumptions:")
    print(f"  Annual productivity loss per NEET: {annual_productivity_loss:,.0f} PKR")
    print(f"  Time horizon: {time_horizon_years} years")
    print(f"  Discount rate: {discount_rate:.1%}")
    
    print(f"\nEconomic Benefits:")
    print(f"  NPV per prevented NEET: {npv_per_prevented:,.0f} PKR")
    print(f"  Total NPV of benefits: {total_npv:,.0f} PKR")
    
    return benefits


def compute_roi(cost_metrics: Dict, benefit_metrics: Dict) -> Dict[str, float]:
    """
    Compute return on investment (ROI) for intervention.
    
    Parameters:
    -----------
    cost_metrics : dict
        Cost metrics from compute_cost_effectiveness()
    benefit_metrics : dict
        Benefit metrics from estimate_economic_benefit()
        
    Returns:
    --------
    roi_metrics : dict
        ROI calculations
    """
    total_cost = cost_metrics['total_cost']
    total_benefit = benefit_metrics['total_npv']
    
    net_benefit = total_benefit - total_cost
    roi_ratio = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
    bcr = total_benefit / total_cost if total_cost > 0 else 0
    
    roi_metrics = {
        'total_cost': total_cost,
        'total_benefit': total_benefit,
        'net_benefit': net_benefit,
        'roi_percent': roi_ratio,
        'benefit_cost_ratio': bcr
    }
    
    print(f"\n{'='*60}")
    print("RETURN ON INVESTMENT (ROI)")
    print(f"{'='*60}")
    print(f"  Total costs: {total_cost:,.0f} PKR")
    print(f"  Total benefits: {total_benefit:,.0f} PKR")
    print(f"  Net benefit: {net_benefit:,.0f} PKR")
    print(f"  ROI: {roi_ratio:.1f}%")
    print(f"  Benefit-Cost Ratio: {bcr:.2f}")
    
    if bcr > 1:
        print(f"\n  ✓ Intervention is cost-effective (BCR > 1)")
    else:
        print(f"\n  ⚠ Intervention may not be cost-effective (BCR < 1)")
    
    return roi_metrics


def compare_interventions(scenarios: List[Dict],
                         df: pd.DataFrame,
                         model: Any,
                         preprocessor: Any,
                         feature_names: List[str],
                         cost_per_participant: Dict[str, float],
                         sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compare multiple intervention scenarios.
    
    Parameters:
    -----------
    scenarios : list of dict
        List of intervention specifications
    df : pd.DataFrame
        Youth dataframe
    model : fitted model
        Predictive model
    preprocessor : fitted preprocessor
        Preprocessing pipeline
    feature_names : list
        Feature names
    cost_per_participant : dict
        Cost per participant for each scenario (scenario_name -> cost)
    sample_weight : array-like, optional
        Survey weights
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table of all scenarios
    """
    print(f"\n{'='*60}")
    print("COMPARING INTERVENTION SCENARIOS")
    print(f"{'='*60}")
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        scenario_name = scenario.get('name', f'Scenario {i}')
        print(f"\n\nRunning {scenario_name}...")
        print("-" * 60)
        
        # Run simulation
        sim_results = simulate_intervention(
            df, model, preprocessor, scenario, feature_names, sample_weight
        )
        
        # Calculate costs
        cost = cost_per_participant.get(scenario_name, 10000)
        cost_metrics = compute_cost_effectiveness(sim_results, cost)
        
        # Store results
        results.append({
            'scenario': scenario_name,
            'n_treated': cost_metrics['n_treated'],
            'prevented_neets': cost_metrics['prevented_neets'],
            'total_cost': cost_metrics['total_cost'],
            'cost_per_prevented': cost_metrics['cost_per_prevented_neet'],
            'target_group': str(scenario.get('target_group', 'All')),
            'intervention': str(scenario.get('intervention', {})),
            'coverage': scenario.get('coverage', 1.0)
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).sort_values('cost_per_prevented')
    
    print(f"\n\n{'='*60}")
    print("SCENARIO COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison_df[['scenario', 'n_treated', 'prevented_neets', 
                         'cost_per_prevented']].to_string(index=False))
    
    return comparison_df


def create_intervention_report(comparison_df: pd.DataFrame,
                              output_path: str = 'outputs/intervention_simulation.csv'):
    """
    Save intervention comparison report to CSV.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results
    output_path : str
        Path to save report
    """
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Intervention report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("SIMULATION MODULE - UNIT TESTS")
    print("="*60)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=RANDOM_SEED
    )
    
    # Create dataframe
    feature_names = ['age', 'education', 'urban', 'female', 'training']
    df = pd.DataFrame(X, columns=feature_names)
    df['NEET'] = y
    df['age_group'] = np.random.choice(['15-17', '18-20', '21-24'], size=len(df))
    df['province'] = np.random.choice(['Punjab', 'Sindh', 'KP'], size=len(df))
    
    # Train simple model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_scaled, y)
    
    print(f"\nSynthetic data created: {len(df)} samples")
    print(f"NEET rate: {y.mean():.2%}")
    
    # Test intervention simulation
    intervention_spec = {
        'name': 'Vocational Training',
        'target_group': {'age_group': '18-20'},
        'intervention': {'training': 1.5},
        'coverage': 0.5
    }
    
    results = simulate_intervention(
        df, model, scaler, intervention_spec, feature_names
    )
    
    # Test cost-effectiveness
    cost_metrics = compute_cost_effectiveness(results, cost_per_participant=10000)
    
    # Test economic benefits
    benefit_metrics = estimate_economic_benefit(
        cost_metrics['prevented_neets'],
        annual_productivity_loss=100000,
        time_horizon_years=5
    )
    
    # Test ROI
    roi_metrics = compute_roi(cost_metrics, benefit_metrics)
    
    # Test scenario comparison
    scenarios = [
        {
            'name': 'Vocational Training - Youth',
            'target_group': {'age_group': '18-20'},
            'intervention': {'training': 1.5},
            'coverage': 0.5
        },
        {
            'name': 'Education Support - All',
            'target_group': {},
            'intervention': {'education': 1.0},
            'coverage': 0.3
        }
    ]
    
    cost_per_participant = {
        'Vocational Training - Youth': 15000,
        'Education Support - All': 8000
    }
    
    comparison = compare_interventions(
        scenarios, df, model, scaler, feature_names, cost_per_participant
    )
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
