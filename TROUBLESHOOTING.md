# 🔧 Fixing Streamlit Cloud Deployment Errors

## Error: "Error installing requirements"

This usually happens due to package version conflicts or incompatibilities. Here are solutions:

---

## Solution 1: Update requirements.txt (Recommended)

I've updated your `requirements.txt` with specific versions that work on Streamlit Cloud.

**Push the changes:**

```powershell
cd "f:\NEET Predictor"
git add requirements.txt
git commit -m "Fix: Update requirements for Streamlit Cloud compatibility"
git push
```

Streamlit Cloud will automatically detect the push and redeploy (takes 2-3 minutes).

---

## Solution 2: Use Minimal Requirements (If Solution 1 fails)

If the full requirements still fail, use the minimal version:

```powershell
# Backup current requirements
Copy-Item requirements.txt requirements-full.txt

# Use minimal requirements
Copy-Item requirements-minimal.txt requirements.txt

# Push changes
git add requirements.txt
git commit -m "Fix: Use minimal requirements"
git push
```

This removes all optional packages and only installs what's absolutely necessary.

---

## Solution 3: Check Streamlit Cloud Logs

To see the exact error:

1. Go to your app on Streamlit Cloud
2. Click **"Manage app"** (bottom right)
3. Look at the **terminal/logs** section
4. Find the error message (usually shows which package failed)

Common errors and fixes:

### Error: "Could not find a version that satisfies the requirement pandas>=2.0.0"

**Fix:** Use specific versions instead of `>=`
```
pandas==2.2.0  # instead of pandas>=2.0.0
```

### Error: "ERROR: No matching distribution found for X"

**Fix:** The package doesn't exist or name is wrong. Check:
```powershell
# Test package name locally
pip search PACKAGE_NAME
```

### Error: "Building wheel for X failed"

**Fix:** Package needs compilation. Either:
- Use pre-compiled wheel version
- Remove the package if not essential
- Add to `packages.txt` for system-level install

---

## Solution 4: Python Version Issue

Check if your Python version is compatible:

1. In Streamlit Cloud app settings
2. Click **"Advanced settings"**
3. Set Python version to: **3.11** (recommended) or **3.9**
4. Save and redeploy

---

## Solution 5: Check File Size Limits

Streamlit Cloud has limits:

- **Repository size**: < 1GB
- **Individual file**: < 100MB
- **Total app memory**: 1GB (free tier)

**Check your data file:**

```powershell
cd "f:\NEET Predictor"
dir data\processed\lfs_youth_cleaned.csv
```

If > 100MB, reduce the sample size:

```python
import pandas as pd
df = pd.read_csv('data/processed/lfs_youth_cleaned.csv')
df_small = df.sample(n=5000, random_state=42)  # Take 5000 rows
df_small.to_csv('data/processed/lfs_youth_cleaned.csv', index=False)
```

Then push:
```powershell
git add data/processed/lfs_youth_cleaned.csv
git commit -m "Reduce data file size"
git push
```

---

## Solution 6: Clear Cache and Reboot

Sometimes Streamlit Cloud needs a fresh start:

1. Go to app management
2. Click **"⋮"** (three dots menu)
3. Click **"Reboot app"**
4. Wait 2-3 minutes

---

## Solution 7: Check .gitignore

Make sure required files aren't being ignored:

```powershell
# Check what's being tracked
git ls-files

# Should include:
# - requirements.txt
# - streamlit_app/app.py
# - data/processed/lfs_youth_cleaned.csv
```

If missing, remove from .gitignore and push:

```powershell
git add -f data/processed/lfs_youth_cleaned.csv
git commit -m "Force add data file"
git push
```

---

## Solution 8: Start Fresh (Last Resort)

If nothing works, redeploy from scratch:

```powershell
# 1. Delete the app on Streamlit Cloud
# 2. Use minimal requirements
Copy-Item requirements-minimal.txt requirements.txt

# 3. Commit and push
git add requirements.txt
git commit -m "Fresh start with minimal requirements"
git push

# 4. Redeploy on Streamlit Cloud with same settings
```

---

## Working Requirements.txt Template

If all else fails, use this proven template:

```txt
streamlit==1.31.0
pandas==2.2.0
numpy==1.26.3
plotly==5.18.0
```

This is the absolute minimum and WILL work.

---

## Getting Help

If still stuck:

1. **Copy the full error from Streamlit Cloud logs**
2. **Post in Streamlit Forum**: https://discuss.streamlit.io
3. **Include**:
   - Full error message
   - Your requirements.txt
   - Python version
   - Link to your GitHub repo (if public)

---

## Quick Checklist

- [ ] Updated requirements.txt with specific versions
- [ ] Committed and pushed changes
- [ ] Waited 2-3 minutes for auto-redeploy
- [ ] Checked logs in "Manage app"
- [ ] Verified data file < 100MB
- [ ] Set Python version to 3.11
- [ ] Rebooted app if needed

---

## Test Locally First

Before deploying, test locally:

```powershell
# Create fresh virtual environment
python -m venv test_env
.\test_env\Scripts\activate

# Install from requirements
pip install -r requirements.txt

# Run app
streamlit run streamlit_app\app.py

# If this works locally, it should work on cloud
```

---

## Contact Me

If you need more help, share:
1. Screenshot of the error from Streamlit Cloud logs
2. Your requirements.txt content
3. Data file size

And I'll help troubleshoot further!
