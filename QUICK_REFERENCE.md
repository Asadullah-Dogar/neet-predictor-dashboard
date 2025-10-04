# NEET Predictor - Quick Reference Guide

## 🚀 Quick Start (5 minutes)

```powershell
# 1. Navigate to project
cd "f:\NEET Predictor"

# 2. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your data file
# Move LFS2020-21.dta to: data\raw\

# 5. Run the pipeline test
.\run_all.ps1

# 6. Start Jupyter
jupyter notebook
```

## 📓 Notebook Execution Order

1. **01_EDA.ipynb** → Exploratory data analysis
2. **02_Preprocessing_Labeling.ipynb** → Create NEET labels & clean data
3. **03_Modeling_and_Explainability.ipynb** → Train models & interpret
4. **04_Intervention_Simulations_and_Maps.ipynb** → Simulate interventions

## 🎯 Key Functions Quick Reference

### Data Loading & Preprocessing
```python
from src.data_preprocessing import *

# Load data
df, metadata = load_raw('data/raw/LFS2020-21.dta')

# Detect variables
var_map = detect_variable_names(df)

# Create NEET label
df_youth = create_neet_label(df, var_map, age_min=15, age_max=24)

# Clean variables
df_clean = clean_vars(df_youth, var_map)

# Remove PII
df_clean = remove_pii(df_clean)
```

### Modeling
```python
from src.modeling import *

# Prepare data
data_dict = prepare_data_for_model(
    df, 
    target_col='NEET',
    weight_col='sample_weight'  # if available
)

# Train model
model = train_model(
    data_dict['X_train'],
    data_dict['y_train'],
    sample_weight=data_dict['sample_weight_train'],
    model_type='logistic'
)

# Cross-validate
cv_results = cross_validate_model(
    model,
    data_dict['X_train'],
    data_dict['y_train']
)

# Evaluate
metrics = evaluate_model(
    model,
    data_dict['X_test'],
    data_dict['y_test']
)

# Calibrate
calibrated_model = calibrate_model(
    model,
    data_dict['X_train'],
    data_dict['y_train']
)

# Save
save_model(
    calibrated_model,
    data_dict['preprocessor'],
    'models/model_logistic'
)
```

### Explainability
```python
from src.explainability import *

# SHAP analysis
shap_data = compute_shap_values(
    model,
    data_dict['X_test'][:100],  # Sample for efficiency
    data_dict['feature_names'],
    model_type='linear'
)

# Plot SHAP summary
plot_shap_summary(shap_data, output_path='outputs/shap_summary.png')

# Get feature importance
importance_df = get_feature_importance_df(shap_data)

# Odds ratios (for logistic regression)
or_df = compute_odds_ratios(model, data_dict['feature_names'])

# Statsmodels for CI
result, or_df_sm = fit_statsmodels_logit(
    data_dict['X_train'],
    data_dict['y_train'],
    data_dict['feature_names']
)

# Natural language explanations
explanations = generate_natural_language_explanation(or_df_sm, top_k=5)
```

### Intervention Simulation
```python
from src.simulation import *

# Load model
model_data = load_model('models/model_logistic')

# Define intervention
intervention_spec = {
    'name': 'Vocational Training',
    'target_group': {
        'age_group': '18-20',
        'urban_rural': 'Rural'
    },
    'intervention': {
        'vocational_training': 1  # Set feature to 1
    },
    'coverage': 0.5  # Reach 50% of target group
}

# Simulate
results = simulate_intervention(
    df,
    model_data['model'],
    model_data['preprocessor'],
    intervention_spec,
    model_data['feature_names']
)

# Cost-effectiveness
cost_metrics = compute_cost_effectiveness(
    results,
    cost_per_participant=15000  # PKR
)

# Economic benefit
benefit_metrics = estimate_economic_benefit(
    cost_metrics['prevented_neets'],
    annual_productivity_loss=100000,  # PKR
    time_horizon_years=5
)

# ROI
roi_metrics = compute_roi(cost_metrics, benefit_metrics)

# Compare scenarios
scenarios = [
    {
        'name': 'Scenario 1',
        'target_group': {'age_group': '18-20'},
        'intervention': {'vocational_training': 1},
        'coverage': 0.5
    },
    {
        'name': 'Scenario 2',
        'target_group': {'sex': 'Female'},
        'intervention': {'education': 1.0},
        'coverage': 0.3
    }
]

cost_per_participant = {
    'Scenario 1': 15000,
    'Scenario 2': 8000
}

comparison = compare_interventions(
    scenarios,
    df,
    model_data['model'],
    model_data['preprocessor'],
    model_data['feature_names'],
    cost_per_participant
)
```

## 📊 Dashboard Commands

```powershell
# Launch dashboard
streamlit run streamlit_app\app.py

# Dashboard will open at: http://localhost:8501
```

**Dashboard Features:**
- Filter by province, district, age, gender, urban/rural
- View NEET rates by demographics
- Identify high-risk segments
- Download filtered predictions as CSV

## 🧪 Testing Commands

```powershell
# Run all tests
pytest tests\ -v

# Run with coverage
pytest tests\ -v --cov=src --cov-report=html

# Run specific test file
pytest tests\test_preprocessing.py -v

# Run specific test
pytest tests\test_preprocessing.py::TestNEETLabelCreation::test_neet_label_is_neet -v
```

## 📁 Key File Locations

### Inputs
- `data/raw/LFS2020-21.dta` - Original Stata file

### Outputs
- `data/processed/lfs_youth_cleaned.csv` - Cleaned youth data
- `data/raw/schema.txt` - Variable schema
- `outputs/eda_plots/*.png` - EDA visualizations
- `outputs/shap_summary.png` - Feature importance
- `outputs/predictions.csv` - Final predictions
- `outputs/intervention_simulation.csv` - Intervention scenarios
- `models/model_logistic.pkl` - Trained model
- `reports/CEO_onepager.pdf` - Executive brief

## 🐛 Common Issues & Solutions

### Issue: ModuleNotFoundError: No module named 'src'
```python
# Add to top of notebook:
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

### Issue: FileNotFoundError: LFS2020-21.dta
```powershell
# Move file to correct location:
Move-Item LFS2020-21.dta "data\raw\"
```

### Issue: SHAP computation too slow
```python
# Use smaller sample:
shap_data = compute_shap_values(model, X_test[:50], ...)
```

### Issue: Streamlit shows "Data file not found"
```powershell
# Run preprocessing notebook first:
jupyter notebook notebooks/02_Preprocessing_Labeling.ipynb
# Then run all cells to generate lfs_youth_cleaned.csv
```

### Issue: Memory error when loading large dataset
```python
# Load in chunks:
chunks = []
for chunk in pd.read_stata('data/raw/LFS2020-21.dta', chunksize=10000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

## 📖 Documentation Links

- **Project README**: `README.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Fairness Notes**: `reports/fairness_note.md`
- **Module Docstrings**: Check each `.py` file in `src/`

## 🎯 Typical Workflow

```
Day 1: Data Exploration
├── Run 01_EDA.ipynb
└── Review data quality and distributions

Day 2: Preprocessing & Labeling
├── Run 02_Preprocessing_Labeling.ipynb
├── Verify NEET label logic
└── Save cleaned data

Day 3: Modeling
├── Run 03_Modeling_and_Explainability.ipynb
├── Train and evaluate models
├── Compute SHAP and odds ratios
└── Save trained model

Day 4: Interventions & Reporting
├── Run 04_Intervention_Simulations_and_Maps.ipynb
├── Simulate intervention scenarios
├── Create district maps
├── Generate CEO one-pager
└── Launch dashboard for stakeholder demo
```

## 💡 Pro Tips

1. **Save frequently** - Notebooks can lose state
2. **Use small samples** during development (df.sample(1000))
3. **Check data types** - Stata categorical vs. string issues
4. **Apply survey weights** when reporting aggregate statistics
5. **Version control** - Commit after each major milestone
6. **Document assumptions** - Add markdown cells explaining choices
7. **Test edge cases** - What if district is missing? What if all male?
8. **Validate predictions** - Do predicted high-risk areas match domain knowledge?

## 📧 Getting Help

1. Check `PROJECT_SUMMARY.md` for detailed instructions
2. Review docstrings in source files (`src/*.py`)
3. Run unit tests to verify setup: `pytest tests\ -v`
4. Check error messages carefully - they're usually informative

## ✅ Pre-Flight Checklist

Before running the full pipeline:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Data file in `data/raw/LFS2020-21.dta`
- [ ] All src modules importable (test with `python src/data_preprocessing.py`)
- [ ] Unit tests passing (`pytest tests/ -v`)

## 🎉 Success Indicators

You've successfully completed the project when you have:

- [x] All 4 notebooks executed without errors
- [x] `lfs_youth_cleaned.csv` created
- [x] Model saved to `models/model_logistic.pkl`
- [x] SHAP plots in `outputs/`
- [x] Intervention simulation results
- [x] Streamlit dashboard running
- [x] CEO one-pager created
- [x] All tests passing

---

**Happy coding! 🚀**
