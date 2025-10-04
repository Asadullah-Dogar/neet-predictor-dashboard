# Run All - NEET Predictor Pipeline
# PowerShell script to execute the entire analysis pipeline

Write-Host "================================" -ForegroundColor Cyan
Write-Host "NEET Predictor - Full Pipeline" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠ Virtual environment not detected. Activating..." -ForegroundColor Yellow
    if (Test-Path ".\venv\Scripts\Activate.ps1") {
        .\venv\Scripts\Activate.ps1
        Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "❌ Virtual environment not found. Please run: python -m venv venv" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Step 1: Checking data file..." -ForegroundColor Cyan
if (-not (Test-Path "data\raw\LFS2020-21.dta")) {
    Write-Host "❌ Data file not found: data\raw\LFS2020-21.dta" -ForegroundColor Red
    Write-Host "   Please place your LFS data file in data\raw\" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "✓ Data file found" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 2: Running unit tests..." -ForegroundColor Cyan
pytest tests\test_preprocessing.py -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Some tests failed. Continue anyway? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -ne "Y") {
        exit 1
    }
}

Write-Host ""
Write-Host "Step 3: Testing data preprocessing module..." -ForegroundColor Cyan
python src\data_preprocessing.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Data preprocessing module test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 4: Testing modeling module..." -ForegroundColor Cyan
python src\modeling.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Modeling module test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 5: Testing explainability module..." -ForegroundColor Cyan
python src\explainability.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Explainability module test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 6: Testing simulation module..." -ForegroundColor Cyan
python src\simulation.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Simulation module test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "✓ All module tests passed!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run Jupyter notebooks in order:" -ForegroundColor White
Write-Host "   jupyter notebook" -ForegroundColor Yellow
Write-Host "   - Open and run: notebooks/01_EDA.ipynb" -ForegroundColor White
Write-Host "   - Open and run: notebooks/02_Preprocessing_Labeling.ipynb" -ForegroundColor White
Write-Host "   - Open and run: notebooks/03_Modeling_and_Explainability.ipynb" -ForegroundColor White
Write-Host "   - Open and run: notebooks/04_Intervention_Simulations_and_Maps.ipynb" -ForegroundColor White
Write-Host ""
Write-Host "2. Launch Streamlit dashboard:" -ForegroundColor White
Write-Host "   streamlit run streamlit_app\app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Generate CEO one-pager from notebook outputs" -ForegroundColor White
Write-Host ""
Write-Host "See PROJECT_SUMMARY.md for detailed instructions." -ForegroundColor White
Write-Host ""
