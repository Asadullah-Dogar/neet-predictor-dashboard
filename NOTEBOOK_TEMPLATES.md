# Notebook Templates - Quick Copy-Paste Guide

This file contains ready-to-use code blocks for completing the remaining notebooks.

---

## Notebook 02: Preprocessing & Labeling

### Complete Notebook Code:

```python
# Cell 1: Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

import pandas as pd
import numpy as np
from data_preprocessing import *

# Cell 2: Load Data
data_path = Path.cwd().parent / 'data' / 'raw' / 'LFS2020-21.dta'
df, metadata = load_raw(str(data_path), verbose=True)

print(f"\nOriginal data shape: {df.shape}")

# Cell 3: Detect Variables
var_map = detect_variable_names(df, verbose=True)

# Cell 4: Create NEET Label
df_youth = create_neet_label(df, var_map, age_min=15, age_max=24, verbose=True)

print(f"\nYouth data shape: {df_youth.shape}")
print(f"NEET prevalence: {df_youth['NEET'].mean()*100:.1f}%")

# Cell 5: Clean Variables
df_clean = clean_vars(df_youth, var_map, verbose=True)

# Cell 6: Remove PII
df_clean = remove_pii(df_clean, create_hash_id=True, verbose=True)

# Cell 7: Feature Engineering
# Create age bands if not exists
if 'age_group' not in df_clean.columns and 'age' in var_map:
    age_col = var_map['age']
    df_clean['age_group'] = pd.cut(
        df_clean[age_col],
        bins=[15, 18, 21, 24],
        labels=['15-17', '18-20', '21-24'],
        include_lowest=True
    )

# Create interaction terms
if 'sex' in var_map and 'urban_rural' in var_map:
    sex_col = var_map['sex']
    ur_col = var_map['urban_rural']
    df_clean['female_rural'] = (
        (df_clean[sex_col] == 'Female') & 
        (df_clean[ur_col] == 'Rural')
    ).astype(int)

print(f"\n✓ Feature engineering complete")
print(f"  Added: age_group, female_rural")

# Cell 8: Summary Statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# NEET by demographics
if 'sex' in var_map:
    print("\nNEET Rate by Gender:")
    print(df_clean.groupby(var_map['sex'])['NEET'].agg(['mean', 'count']))

if 'age_group' in df_clean.columns:
    print("\nNEET Rate by Age Group:")
    print(df_clean.groupby('age_group')['NEET'].agg(['mean', 'count']))

if 'urban_rural' in var_map:
    print("\nNEET Rate by Urban/Rural:")
    print(df_clean.groupby(var_map['urban_rural'])['NEET'].agg(['mean', 'count']))

if 'province' in var_map:
    print("\nNEET Rate by Province:")
    print(df_clean.groupby(var_map['province'])['NEET'].agg(['mean', 'count']))

# Cell 9: Save Processed Data
output_path = Path.cwd().parent / 'data' / 'processed' / 'lfs_youth_cleaned.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)

df_clean.to_csv(output_path, index=False)

print(f"\n✓ Processed data saved to: {output_path}")
print(f"  Shape: {df_clean.shape}")
print(f"  Columns: {list(df_clean.columns)}")
```

---

## Notebook 03: Modeling & Explainability

### Complete Notebook Code:

```python
# Cell 1: Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modeling import *
from explainability import *

# Cell 2: Load Processed Data
data_path = Path.cwd().parent / 'data' / 'processed' / 'lfs_youth_cleaned.csv'
df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} records")
print(f"NEET prevalence: {df['NEET'].mean()*100:.1f}%")

# Cell 3: Prepare Data for Modeling
# Define features
numeric_features = ['age'] if 'age' in df.columns else []
categorical_features = []

for col in ['sex', 'province', 'district', 'urban_rural', 'age_group']:
    if col in df.columns:
        categorical_features.append(col)

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Prepare data
data_dict = prepare_data_for_model(
    df,
    target_col='NEET',
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    weight_col='weight' if 'weight' in df.columns else None
)

# Cell 4: Train Logistic Regression
model = train_model(
    data_dict['X_train'],
    data_dict['y_train'],
    sample_weight=data_dict['sample_weight_train'],
    model_type='logistic'
)

# Cell 5: Cross-Validation
cv_results = cross_validate_model(
    model,
    data_dict['X_train'],
    data_dict['y_train'],
    sample_weight=data_dict['sample_weight_train'],
    cv_folds=5
)

# Cell 6: Evaluate on Test Set
metrics = evaluate_model(
    model,
    data_dict['X_test'],
    data_dict['y_test'],
    sample_weight=data_dict['sample_weight_test']
)

# Cell 7: Precision at K
y_pred_proba = model.predict_proba(data_dict['X_test'])[:, 1]

for k in [5, 10, 20]:
    p_at_k = compute_precision_at_k(
        data_dict['y_test'],
        y_pred_proba,
        k_percent=k
    )
    print(f"Precision@{k}%: {p_at_k:.2%}")

# Cell 8: Calibration
calibrated_model = calibrate_model(
    model,
    data_dict['X_train'],
    data_dict['y_train'],
    sample_weight=data_dict['sample_weight_train'],
    method='sigmoid'
)

# Re-evaluate calibrated model
print("\nCalibrated Model Performance:")
calibrated_metrics = evaluate_model(
    calibrated_model,
    data_dict['X_test'],
    data_dict['y_test'],
    sample_weight=data_dict['sample_weight_test']
)

# Cell 9: SHAP Analysis
shap_data = compute_shap_values(
    calibrated_model,
    data_dict['X_test'][:100],  # Sample for efficiency
    data_dict['feature_names'],
    sample_size=100,
    model_type='linear'
)

# Plot SHAP summary
output_dir = Path.cwd().parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)

plot_shap_summary(
    shap_data,
    output_path=str(output_dir / 'shap_summary.png'),
    top_k=20
)

# Cell 10: Feature Importance Table
importance_df = get_feature_importance_df(shap_data)

print("\nTop 15 Features by SHAP Importance:")
print(importance_df.head(15).to_string(index=False))

# Cell 11: Odds Ratios
or_df = compute_odds_ratios(calibrated_model, data_dict['feature_names'])

print("\nTop 10 Features by Odds Ratio:")
print(or_df.head(10)[['feature', 'odds_ratio', 'interpretation']].to_string(index=False))

# Cell 12: Statsmodels Logit (for proper CI)
result, or_df_sm = fit_statsmodels_logit(
    data_dict['X_train'],
    data_dict['y_train'],
    data_dict['feature_names']
)

# Plot odds ratios
plot_odds_ratios(
    or_df_sm,
    output_path=str(output_dir / 'odds_ratios.png'),
    top_k=15
)

# Cell 13: Natural Language Explanations
explanations = generate_natural_language_explanation(or_df_sm, top_k=5)

print("\nTop 5 Risk Factors (Plain English):")
for i, exp in enumerate(explanations, 1):
    print(f"\n{i}. {exp}")

# Cell 14: Fairness Checks
print("\n" + "="*70)
print("FAIRNESS AUDIT")
print("="*70)

# By gender
if 'sex' in df.columns:
    print("\nPerformance by Gender:")
    for gender in df['sex'].unique():
        mask = df['sex'] == gender
        if mask.sum() > 0:
            X_gender = data_dict['preprocessor'].transform(
                df.loc[mask, numeric_features + categorical_features]
            )
            y_gender = df.loc[mask, 'NEET']
            y_pred_gender = calibrated_model.predict_proba(X_gender)[:, 1]
            auc_gender = roc_auc_score(y_gender, y_pred_gender)
            print(f"  {gender}: AUC = {auc_gender:.4f}")

# By urban/rural
if 'urban_rural' in df.columns:
    print("\nPerformance by Urban/Rural:")
    for area in df['urban_rural'].unique():
        mask = df['urban_rural'] == area
        if mask.sum() > 0:
            X_area = data_dict['preprocessor'].transform(
                df.loc[mask, numeric_features + categorical_features]
            )
            y_area = df.loc[mask, 'NEET']
            y_pred_area = calibrated_model.predict_proba(X_area)[:, 1]
            auc_area = roc_auc_score(y_area, y_pred_area)
            print(f"  {area}: AUC = {auc_area:.4f}")

# Cell 15: Save Model
save_model(
    calibrated_model,
    data_dict['preprocessor'],
    str(Path.cwd().parent / 'models' / 'model_logistic'),
    metadata={
        'model_type': 'LogisticRegression',
        'calibrated': True,
        'test_auc': metrics['auc_roc'],
        'features': data_dict['feature_names']
    }
)

# Cell 16: Generate Predictions for Full Dataset
X_full = data_dict['preprocessor'].transform(
    df[numeric_features + categorical_features]
)
df['neet_prob'] = calibrated_model.predict_proba(X_full)[:, 1]
df['neet_pred'] = (df['neet_prob'] >= 0.5).astype(int)

# Save predictions
pred_cols = ['id_hash', 'age', 'sex', 'province', 'district', 
             'urban_rural', 'NEET', 'neet_prob', 'neet_pred']
pred_cols = [c for c in pred_cols if c in df.columns]

predictions_path = Path.cwd().parent / 'outputs' / 'predictions.csv'
df[pred_cols].to_csv(predictions_path, index=False)

print(f"\n✓ Predictions saved to: {predictions_path}")
print(f"  Total records: {len(df):,}")
print(f"  Average predicted probability: {df['neet_prob'].mean():.2%}")
```

---

## Notebook 04: Interventions & Maps

### Complete Notebook Code:

```python
# Cell 1: Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modeling import load_model
from simulation import *

# Cell 2: Load Data and Model
df = pd.read_csv(Path.cwd().parent / 'data' / 'processed' / 'lfs_youth_cleaned.csv')
model_data = load_model(str(Path.cwd().parent / 'models' / 'model_logistic'))

print(f"Loaded {len(df):,} records")
print(f"Model type: {type(model_data['model'])}")

# Cell 3: Define Intervention Scenarios
scenarios = [
    {
        'name': 'Vocational Training - Rural Youth',
        'target_group': {'urban_rural': 'Rural', 'age_group': '18-20'},
        'intervention': {'vocational_training': 1},  # This would need to be an actual feature
        'coverage': 0.5
    },
    {
        'name': 'Education Support - Female Youth',
        'target_group': {'sex': 'Female', 'age_group': '15-17'},
        'intervention': {'education_level': 'Secondary'},  # Hypothetical
        'coverage': 0.3
    },
    {
        'name': 'Skills Training - All Youth',
        'target_group': {},
        'intervention': {'skills_training': 1},
        'coverage': 0.2
    }
]

cost_per_participant = {
    'Vocational Training - Rural Youth': 15000,
    'Education Support - Female Youth': 8000,
    'Skills Training - All Youth': 12000
}

# Cell 4: Run Interventions
# Note: This requires the feature to actually exist in the data
# For demonstration, we'll simulate the effect

for scenario in scenarios:
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario['name']}")
    print(f"{'='*70}")
    
    # Get target group
    mask = np.ones(len(df), dtype=bool)
    if scenario['target_group']:
        for key, value in scenario['target_group'].items():
            if key in df.columns:
                mask &= (df[key] == value)
    
    n_target = mask.sum()
    n_treated = int(n_target * scenario['coverage'])
    
    print(f"Target group size: {n_target:,}")
    print(f"Coverage: {scenario['coverage']:.0%}")
    print(f"Number treated: {n_treated:,}")
    
    # Simulate effect (placeholder - would use actual feature modification)
    baseline_neet_rate = df.loc[mask, 'NEET'].mean()
    assumed_reduction = 0.15  # 15% relative reduction (placeholder)
    post_intervention_rate = baseline_neet_rate * (1 - assumed_reduction)
    
    prevented_neets = (baseline_neet_rate - post_intervention_rate) * n_treated
    
    print(f"Baseline NEET rate: {baseline_neet_rate:.1%}")
    print(f"Post-intervention rate: {post_intervention_rate:.1%}")
    print(f"Prevented NEETs: {prevented_neets:.1f}")
    
    # Cost-effectiveness
    cost = cost_per_participant[scenario['name']]
    total_cost = cost * n_treated
    cost_per_prevented = total_cost / prevented_neets if prevented_neets > 0 else np.inf
    
    print(f"\nCost Analysis:")
    print(f"  Cost per participant: {cost:,} PKR")
    print(f"  Total cost: {total_cost:,} PKR")
    print(f"  Cost per prevented NEET: {cost_per_prevented:,.0f} PKR")

# Cell 5: District-Level NEET Rates
if 'district' in df.columns:
    district_stats = df.groupby('district').agg({
        'NEET': ['mean', 'sum', 'count']
    }).reset_index()
    district_stats.columns = ['District', 'NEET_Rate', 'NEET_Count', 'Total']
    district_stats['NEET_Rate'] = district_stats['NEET_Rate'] * 100
    district_stats = district_stats.sort_values('NEET_Rate', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 10 DISTRICTS BY NEET RATE")
    print("="*70)
    print(district_stats.head(10).to_string(index=False))
    
    # Plot
    fig = px.bar(
        district_stats.head(15),
        x='District',
        y='NEET_Rate',
        title='Top 15 Districts by NEET Rate',
        labels={'NEET_Rate': 'NEET Rate (%)'},
        text='NEET_Rate',
        color='NEET_Rate',
        color_continuous_scale='Reds'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

# Cell 6: Geographic Visualization (Basic)
# Note: For proper choropleth, you'd need Pakistan shapefiles

if 'province' in df.columns and 'district' in df.columns:
    province_stats = df.groupby('province').agg({
        'NEET': ['mean', 'sum', 'count']
    }).reset_index()
    province_stats.columns = ['Province', 'NEET_Rate', 'NEET_Count', 'Total']
    province_stats['NEET_Rate'] = province_stats['NEET_Rate'] * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=province_stats['Province'],
            y=province_stats['NEET_Count'],
            text=province_stats['NEET_Rate'].round(1),
            texttemplate='%{text}% NEET',
            textposition='outside',
            marker=dict(
                color=province_stats['NEET_Rate'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='NEET Rate (%)')
            )
        )
    ])
    
    fig.update_layout(
        title='NEET Count and Rate by Province',
        xaxis_title='Province',
        yaxis_title='Number of NEET Youth',
        height=500
    )
    
    fig.show()
    
    # Save
    output_path = Path.cwd().parent / 'outputs' / 'district_risk_map.html'
    fig.write_html(str(output_path))
    print(f"\n✓ Map saved to: {output_path}")

# Cell 7: High-Risk Segments
print("\n" + "="*70)
print("HIGH-RISK SEGMENTS FOR TARGETING")
print("="*70)

segment_cols = []
for col in ['sex', 'age_group', 'urban_rural', 'province']:
    if col in df.columns:
        segment_cols.append(col)

if len(segment_cols) >= 2:
    segments = df.groupby(segment_cols).agg({
        'NEET': ['mean', 'sum', 'count']
    }).reset_index()
    segments.columns = segment_cols + ['NEET_Rate', 'NEET_Count', 'Total']
    segments['NEET_Rate'] = segments['NEET_Rate'] * 100
    
    # Filter minimum group size
    segments = segments[segments['Total'] >= 30]
    
    # Sort by NEET count (for targeting)
    segments_top = segments.sort_values('NEET_Count', ascending=False).head(10)
    
    print("\nTop 10 Segments by NEET Count:")
    print(segments_top.to_string(index=False))
    
    # Also show highest NEET rate segments
    segments_rate = segments.sort_values('NEET_Rate', ascending=False).head(10)
    print("\nTop 10 Segments by NEET Rate:")
    print(segments_rate.to_string(index=False))

# Cell 8: Export Results
output_dir = Path.cwd().parent / 'outputs'

# Save high-risk segments
if 'segments_top' in locals():
    segments_top.to_csv(output_dir / 'high_risk_segments.csv', index=False)
    print(f"\n✓ High-risk segments saved to: {output_dir / 'high_risk_segments.csv'}")

# Save district stats
if 'district_stats' in locals():
    district_stats.to_csv(output_dir / 'district_stats.csv', index=False)
    print(f"✓ District statistics saved to: {output_dir / 'district_stats.csv'}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nNext steps:")
print("1. Review high-risk segments for intervention targeting")
print("2. Create CEO one-pager with key findings")
print("3. Launch Streamlit dashboard for interactive exploration")
```

---

## CEO One-Pager Template

Save this as markdown and convert to PDF:

```markdown
# NEET Predictor - Executive Brief
## Labour Force Survey 2020-21

---

### KEY FINDINGS

1. **NEET Prevalence**: XX% of youth (15-24 years) are NEET
2. **Gender Gap**: Female NEET rate is XX% vs XX% for males
3. **Geographic Hotspots**: Top 3 provinces are X, Y, Z with XX% NEET rate

---

### HIGH-RISK SEGMENTS (Target for Interventions)

| Segment | Population | NEET Rate | NEET Count |
|---------|------------|-----------|------------|
| Rural females, 18-20 | XXX,XXX | XX% | XX,XXX |
| Urban males, 21-24, low edu | XXX,XXX | XX% | XX,XXX |
| Provincial youth, district X | XXX,XXX | XX% | XX,XXX |

---

### INTERVENTION IMPACT ESTIMATE

**Scenario**: Vocational training for rural youth aged 18-20

- **Cost**: 15,000 PKR per participant
- **Reach**: 50% of target group (XXX,XXX youth)
- **Expected Impact**: X,XXX prevented NEET cases
- **Cost per Prevented NEET**: XXX,XXX PKR
- **5-Year ROI**: XX% (BCR: X.XX)

---

### [INSERT DISTRICT MAP IMAGE HERE]

---

### RECOMMENDED ACTIONS

1. **Immediate**: Launch pilot vocational training program in top 5 high-risk districts
2. **Short-term**: Expand career counseling services in rural areas, especially for females
3. **Long-term**: Strengthen school-to-work transition programs nationwide

---

### METHODOLOGY

- **Data**: Pakistan LFS 2020-21 (XXX,XXX youth aged 15-24)
- **Model**: Logistic regression with AUC = 0.XX
- **NEET Definition**: Not in education AND not employed AND not in training
- **Validation**: 5-fold cross-validation + geographic holdout

---

### LIMITATIONS & ETHICAL NOTES

- Data from 2020-21 may not reflect current conditions
- Model predictions are decision support tools, not final decisions
- All PII removed; results are for aggregate planning only
- Regular fairness audits conducted across demographic groups

---

**Contact**: Data Science Team | October 2025
```

Convert to PDF using:
- Jupyter: File → Download as PDF
- Pandoc: `pandoc onepager.md -o CEO_onepager.pdf`
- Word/PowerPoint: Copy content and export

---

## That's it! 🎉

You now have all the code templates to complete your notebooks. Just:
1. Copy-paste the relevant sections
2. Adjust variable names to match your actual data
3. Run cells in sequence
4. Generate outputs

Good luck!
