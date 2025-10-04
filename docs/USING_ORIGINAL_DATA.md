# Using Original LFS2020-21.dta File - Step-by-Step Guide

## Overview
This guide explains how to use the original Pakistan Labour Force Survey 2020-21 Stata file (.dta) in the NEET Predictor project.

---

## Option 1: Work Directly with .dta File (Recommended)

The project's `load_raw()` function now supports both .dta and .csv formats.

### Steps:

1. **Place the file in the correct location:**
   ```
   f:\NEET Predictor\data\raw\LFS2020-21.dta
   ```

2. **Run the preprocessing notebook:**
   - Open: `notebooks/02_Preprocessing_Labeling.ipynb`
   - The notebook will automatically:
     - Load the .dta file with all metadata preserved
     - Detect variable names
     - Create NEET labels
     - Clean and process the data
     - Save to `data/processed/lfs_youth_cleaned.csv`

3. **The notebook works with .dta directly** - no conversion needed!

---

## Option 2: Convert .dta to .csv First

If you prefer working with CSV format:

### Step 1: Convert to CSV

Run the conversion script:
```powershell
cd "f:\NEET Predictor"
python scripts\convert_dta_to_csv.py
```

This will:
- Read `data/raw/LFS2020-21.dta`
- Convert to `data/raw/LFS2020-21.csv`
- Preserve all data

### Step 2: Update the notebook

In `notebooks/02_Preprocessing_Labeling.ipynb`, change the path in cell 4:

**From:**
```python
raw_data_path = '../data/raw/LFS2020-21.dta'
```

**To:**
```python
raw_data_path = '../data/raw/LFS2020-21.csv'
```

### Step 3: Run the notebook

The `load_raw()` function will automatically detect it's a CSV and handle it appropriately.

---

## Key Advantages of Each Approach

### Using .dta directly:
✅ Preserves variable labels and value labels from Stata
✅ Maintains data types and categorical variables
✅ No conversion step needed
✅ Original file format

### Using .csv after conversion:
✅ Faster loading times for repeated analysis
✅ Compatible with more tools
✅ Smaller file size (sometimes)
✅ Easier to inspect in Excel/text editors

---

## File Structure

```
f:\NEET Predictor\
├── data/
│   ├── raw/
│   │   ├── LFS2020-21.dta          ← Place your original file here
│   │   └── LFS2020-21.csv          ← Optional: converted version
│   └── processed/
│       └── lfs_youth_cleaned.csv   ← Output from preprocessing
├── notebooks/
│   └── 02_Preprocessing_Labeling.ipynb  ← Main preprocessing notebook
└── scripts/
    └── convert_dta_to_csv.py       ← Conversion utility
```

---

## Important Notes

1. **File Size**: The original LFS file is typically 50-200 MB depending on the year
   
2. **Variable Detection**: The `detect_variable_names()` function will auto-detect:
   - Age column
   - Sex/Gender column
   - Education status
   - Employment status
   - Training status
   - Province, District
   - Urban/Rural

3. **NEET Label Creation**: Automatically identifies youth (15-24) and creates NEET label:
   - NEET = 1 if NOT in education AND NOT employed AND NOT in training
   - NEET = 0 otherwise

4. **Processing Time**: 
   - .dta: ~30-60 seconds for loading
   - .csv: ~10-20 seconds for loading
   - Full processing: 2-5 minutes

---

## Troubleshooting

### Error: "File not found"
- Ensure file is in `data/raw/` directory
- Check filename exactly matches: `LFS2020-21.dta`
- Verify file permissions

### Error: "Could not read Stata file"
- Your .dta file might be Stata 15+ format
- Try converting to CSV first using Option 2

### Error: "Module 'pyreadstat' not found"
- This is an alternative Stata reader
- Not required - pandas.read_stata works fine

### Conversion script fails
- Check pandas version: `pip install pandas>=1.5.0`
- Ensure enough disk space for CSV output

---

## Next Steps After Preprocessing

Once `02_Preprocessing_Labeling.ipynb` completes:

1. ✅ Cleaned data saved to `data/processed/lfs_youth_cleaned.csv`
2. 🔄 Run `03_Modeling_and_Explainability.ipynb` for ML models
3. 🔄 Run `04_Intervention_Simulations_and_Maps.ipynb` for policy analysis
4. 🚀 Dashboard at http://localhost:8503 will work automatically

---

## Questions?

Check these files for more help:
- `README.md` - Project overview
- `FINAL_GUIDE.md` - Complete workflow guide
- `QUICK_REFERENCE.md` - Command reference
- `PROJECT_SUMMARY.md` - Technical details

---

**Ready to start?** Place your `LFS2020-21.dta` file in `data/raw/` and run the preprocessing notebook!
