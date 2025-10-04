# NEET Predictor - LFS 2020-21

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, reproducible machine learning project for predicting NEET (Not in Education, Employment or Training) status among Pakistani youth (ages 15-24) using the Labour Force Survey 2020-21 data.

## 📋 Project Overview

This project aims to:
- **Identify** youth at highest risk of NEET status
- **Understand** key drivers of NEET (age, gender, education, geography, etc.)
- **Predict** NEET probability with calibrated, interpretable models
- **Simulate** policy interventions and estimate their cost-effectiveness
- **Deliver** actionable insights for policymakers and program managers

### Key Features
- ✅ Clean, documented codebase following data science best practices
- ✅ Reproducible pipeline with fixed random seeds
- ✅ Interpretable models (Logistic Regression with odds ratios + SHAP)
- ✅ Fairness checks across demographic subgroups
- ✅ Intervention simulation with ROI calculations
- ✅ Interactive Streamlit dashboard
- ✅ Comprehensive unit tests

---

## 📁 Project Structure

```
NEET-Predictor-LFS2020-21/
│
├── data/
│   ├── raw/
│   │   ├── LFS2020-21.dta          # Original Stata file (not in git)
│   │   └── schema.txt              # Variable schema and labels
│   └── processed/
│       └── lfs_youth_cleaned.csv   # Cleaned youth dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                          # Exploratory Data Analysis
│   ├── 02_Preprocessing_Labeling.ipynb       # Data cleaning & NEET labeling
│   ├── 03_Modeling_and_Explainability.ipynb  # Model training & interpretation
│   └── 04_Intervention_Simulations_and_Maps.ipynb  # Policy simulations
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading, cleaning, NEET label creation
│   ├── modeling.py              # Model training, CV, evaluation
│   ├── explainability.py        # SHAP, odds ratios, feature importance
│   └── simulation.py            # Intervention simulation, cost-effectiveness
│
├── streamlit_app/
│   └── app.py                   # Interactive dashboard
│
├── outputs/
│   ├── eda_plots/               # EDA visualizations
│   ├── predictions.csv          # Final predictions with probabilities
│   ├── eval_report.md           # Model evaluation report
│   ├── shap_summary.png         # SHAP feature importance
│   ├── district_risk_map.html   # Interactive district risk map
│   └── intervention_simulation.csv  # Intervention scenarios & ROI
│
├── reports/
│   ├── CEO_onepager.pdf         # Executive brief for C-suite
│   └── fairness_note.md         # Fairness & ethical considerations
│
├── models/
│   └── model_logistic.pkl       # Trained model + preprocessor
│
├── tests/
│   └── test_preprocessing.py    # Unit tests for data preprocessing
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository** (or navigate to the project folder):
   ```bash
   cd "f:\NEET Predictor"
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Place the data file**:
   - Ensure `LFS2020-21.dta` is in the `data/raw/` directory
   - If not already there, move it:
     ```powershell
     Move-Item LFS2020-21.dta data\raw\
     ```

---

## 📊 Usage

### Option 1: Run Notebooks (Interactive)

Explore the full analysis step-by-step:

```powershell
jupyter notebook
```

Then open and run notebooks in order:
1. `notebooks/01_EDA.ipynb` - Initial data exploration
2. `notebooks/02_Preprocessing_Labeling.ipynb` - Create NEET labels
3. `notebooks/03_Modeling_and_Explainability.ipynb` - Train and interpret models
4. `notebooks/04_Intervention_Simulations_and_Maps.ipynb` - Simulate interventions

### Option 2: Run Python Scripts

Execute the preprocessing and modeling pipeline:

```powershell
# Test preprocessing functions
python src\data_preprocessing.py

# Test modeling functions
python src\modeling.py

# Run unit tests
pytest tests\ -v
```

### Option 3: Launch Streamlit Dashboard

Interactive web application for exploring predictions:

```powershell
streamlit run streamlit_app\app.py
```

The dashboard will open at `http://localhost:8501` with:
- District-level risk maps
- High-risk segment identification
- Filterable predictions by demographics
- Download functionality for targeted outreach

---

## 🔬 Methodology

### 1. NEET Label Definition

A youth (15-24 years) is classified as **NEET** if they are:
- ❌ **NOT** currently in education (not attending school/college)
- ❌ **NOT** employed (not working or having a job)
- ❌ **NOT** in training (not in vocational/skills training)

**NEET = 1** if all three conditions are true, else **NEET = 0**

### 2. Feature Engineering

- **Demographics**: Age, gender, marital status
- **Education**: Education level (none/primary/secondary/tertiary), literacy
- **Geography**: Province, district, urban/rural
- **Household**: Household size, wealth quintile (if available)
- **Interactions**: Female × Rural, Age × Education, etc.

### 3. Modeling Approach

**Primary Model**: Logistic Regression
- **Why**: Interpretable coefficients (odds ratios), well-calibrated probabilities, fast training
- **Configuration**: L2 regularization, balanced class weights, survey weights applied

**Secondary Model** (optional): Gradient Boosting
- **Why**: Captures non-linear relationships, higher predictive power
- **Trade-off**: Less interpretable than logistic regression

### 4. Model Evaluation

- **Stratified 5-fold cross-validation** with survey weights
- **Geographic holdout**: Leave-one-province-out validation
- **Metrics**: AUC-ROC, Precision@K (top 10% risk), Recall, Brier score
- **Calibration**: Post-hoc calibration using isotonic regression
- **Fairness**: Subgroup analysis by gender, urban/rural, province

### 5. Explainability

- **SHAP values**: Model-agnostic feature importance
- **Odds ratios**: Effect sizes with 95% confidence intervals (statsmodels)
- **Natural language**: Plain-English explanations of top risk factors

### 6. Intervention Simulation

- **Target group**: Define by demographics (e.g., 18-20 year-olds in rural KP)
- **Intervention**: Modify features (e.g., set vocational_training=1)
- **Coverage**: Proportion of target group reached
- **Impact**: Estimated prevented NEETs, cost per prevented NEET, ROI

---

## 📈 Key Findings (Example)

> **Note**: Actual findings depend on the real LFS 2020-21 data. Below are illustrative examples.

### Top Risk Factors for NEET Status

1. **Female gender**: +85% higher odds (especially in rural areas)
2. **Low education**: Secondary education or less → +120% odds
3. **Rural residence**: +45% higher odds
4. **Province**: Balochistan and KP have 30-40% higher rates
5. **Age**: 21-24 age group at higher risk than 15-17

### High-Risk Segments

| Segment | N (000s) | NEET Rate | Target Priority |
|---------|----------|-----------|----------------|
| Rural females, 18-24, secondary edu | 250 | 65% | **High** |
| Urban males, 21-24, no degree | 180 | 42% | Medium |
| Rural youth, 18-20, primary edu | 320 | 58% | **High** |

### Intervention ROI (Example Scenario)

**Scenario**: Vocational training for rural females aged 18-24
- **Cost per participant**: 15,000 PKR
- **Coverage**: 50% (125,000 participants)
- **Total cost**: 1.875 billion PKR
- **Prevented NEETs**: 8,500
- **Cost per prevented NEET**: 220,588 PKR
- **5-year NPV of benefits**: 4.25 billion PKR
- **ROI**: 127% (Benefit-Cost Ratio: 2.27)

---

## ⚖️ Ethical Considerations & Limitations

### Data Privacy
- ✅ All PII removed (names, addresses, phone numbers, CNIC)
- ✅ Anonymous hash IDs generated for tracking
- ✅ Aggregate-level reporting only

### Fairness & Bias
- ⚠️ Model predictions should NOT be used for punitive measures
- ⚠️ Predictions reflect historical patterns—may perpetuate existing biases
- ✅ Regular fairness audits by subgroup (gender, geography)
- ✅ Disparate impact metrics calculated and reported

### Limitations
1. **Data vintage**: LFS 2020-21 may not reflect current conditions
2. **Causality**: Predictive model shows associations, not causal effects
3. **Missing variables**: Mental health, discrimination, local labor markets not captured
4. **Survey weights**: Should be applied when reporting population-level estimates
5. **Geographic**: District-level predictions require sufficient sample sizes

### Recommendations for Use
- ✅ Use for **targeting** and **resource allocation**, not individual exclusion
- ✅ Combine with qualitative insights and local knowledge
- ✅ Monitor for unintended consequences (e.g., stigmatization)
- ✅ Update model regularly as new data becomes available

---

## 📚 Data Source & Citation

**Data**: Pakistan Labour Force Survey 2020-21  
**Source**: Pakistan Bureau of Statistics (PBS)  
**URL**: [http://www.pbs.gov.pk](http://www.pbs.gov.pk)  

**Citation**:
```
Pakistan Bureau of Statistics. (2021). Labour Force Survey 2020-21. 
Government of Pakistan. Retrieved from http://www.pbs.gov.pk
```

---

## 🧪 Testing

Run all unit tests:

```powershell
pytest tests\ -v --cov=src --cov-report=html
```

This will:
- Test NEET label creation logic
- Test data cleaning functions
- Test variable detection
- Generate coverage report in `htmlcov/index.html`

---

## 🛠️ Development & Contribution

### Code Style
- Follow PEP 8 style guide
- Use type hints for function signatures
- Write docstrings for all public functions (Google style)

### Git Workflow
```powershell
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, test, commit
git add .
git commit -m "feat: Add feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

### Commit Message Convention
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests

---

## 📞 Contact & Support

**Project Lead**: Data Science Team  
**Email**: [your-email@example.com]  
**Organization**: [Your Organization Name]  

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email the project team
- Review documentation in `/reports/`

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Pakistan Bureau of Statistics for providing the LFS 2020-21 data
- Open-source community for scikit-learn, SHAP, pandas, and other libraries
- Policy stakeholders and domain experts for guidance

---

## 📅 Version History

- **v1.0.0** (October 2025): Initial release
  - Complete preprocessing pipeline
  - Logistic regression baseline model
  - SHAP explainability
  - Intervention simulations
  - Streamlit dashboard
  - Unit tests and documentation

---

## 🔜 Future Enhancements

- [ ] Add time-series analysis (if panel data available)
- [ ] Incorporate spatial econometrics for district spillovers
- [ ] Build ensemble models (stacked generalization)
- [ ] Add real-time prediction API
- [ ] Integrate with HRMIS for program monitoring
- [ ] Multi-language support (Urdu)

---

**Built with ❤️ for evidence-based policymaking**
