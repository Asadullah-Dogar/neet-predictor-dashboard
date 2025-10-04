# 🚀 Deploying NEET Predictor to Streamlit Cloud - Complete Guide

This guide walks you through deploying your NEET Predictor dashboard to Streamlit Cloud step by step.

---

## Prerequisites

✅ GitHub account (free)  
✅ Streamlit Cloud account (free - sign up at [share.streamlit.io](https://share.streamlit.io))  
✅ Git installed on your computer  
✅ Your NEET Predictor project ready

---

## Step 1: Prepare Your Project for Deployment

### 1.1 Verify Required Files Exist

Make sure you have these files in your project:

```
f:\NEET Predictor\
├── .streamlit/
│   └── config.toml          ✅ Created
├── streamlit_app/
│   └── app.py               ✅ Updated for cloud
├── data/
│   └── processed/
│       └── lfs_youth_cleaned.csv   ✅ Sample data
├── requirements-deploy.txt   ✅ Lightweight dependencies
├── packages.txt             ✅ System packages
└── .gitignore               ✅ Existing
```

### 1.2 Choose Your Requirements File

You have two options:

**Option A: Lightweight (Recommended for Cloud)**
```powershell
Copy-Item requirements-deploy.txt requirements.txt
```

**Option B: Full Features (Includes ML models)**
```powershell
# Keep existing requirements.txt (but deployment may be slower)
```

**💡 Recommendation:** Use Option A for faster deployment and lower resource usage.

---

## Step 2: Initialize Git Repository

Open PowerShell in your project directory:

```powershell
cd "f:\NEET Predictor"

# Initialize Git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: NEET Predictor Dashboard"
```

---

## Step 3: Create GitHub Repository

### 3.1 Via GitHub Website (Easier):

1. Go to [github.com](https://github.com) and login
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in:
   - **Repository name**: `neet-predictor-dashboard`
   - **Description**: "NEET Predictor Dashboard - Pakistan Youth Labour Force Analysis"
   - **Visibility**: Choose Public or Private
   - ⚠️ **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### 3.2 Link Your Local Repository to GitHub:

Copy the commands shown on GitHub (they'll look like this):

```powershell
git remote add origin https://github.com/YOUR_USERNAME/neet-predictor-dashboard.git
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME`** with your actual GitHub username!

---

## Step 4: Optimize for Deployment (Optional but Recommended)

### 4.1 Reduce File Sizes

If your CSV file is too large (>100MB), create a smaller sample:

```python
# Run this in Python or a notebook
import pandas as pd

df = pd.read_csv('data/processed/lfs_youth_cleaned.csv')

# Take a 10% sample
df_sample = df.sample(frac=0.1, random_state=42)

# Save smaller file
df_sample.to_csv('data/processed/lfs_youth_cleaned.csv', index=False)

print(f"Reduced from {len(df):,} to {len(df_sample):,} rows")
```

### 4.2 Update .gitignore (if needed)

Ensure large files are ignored:

```powershell
# Add to .gitignore if not already there
echo "*.dta" >> .gitignore
echo "*.pkl" >> .gitignore
echo ".venv/" >> .gitignore
git add .gitignore
git commit -m "Update gitignore"
git push
```

---

## Step 5: Deploy to Streamlit Cloud

### 5.1 Sign Up/Login to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up"** or **"Sign in with GitHub"**
3. Authorize Streamlit to access your GitHub repositories

### 5.2 Create New App

1. Click **"New app"** button
2. Fill in the form:

```
Repository: YOUR_USERNAME/neet-predictor-dashboard
Branch: main
Main file path: streamlit_app/app.py
```

3. **Advanced settings** (click to expand):
   - Python version: `3.9` or `3.11`
   - Secrets: Leave empty for now

4. Click **"Deploy!"**

### 5.3 Wait for Deployment

⏱️ First deployment takes 5-10 minutes:
- Installing Python packages
- Loading your data
- Starting the app

You'll see a progress log. Don't close the page!

---

## Step 6: Access Your Deployed Dashboard

Once deployment completes, you'll get a URL like:

```
https://YOUR_USERNAME-neet-predictor-dashboard-streamlit-app-app-XXXXX.streamlit.app
```

🎉 **Your dashboard is now live!**

---

## Step 7: Share and Manage

### Sharing Your Dashboard

Share the URL with:
- ✅ Colleagues and stakeholders
- ✅ Policy makers
- ✅ Research teams
- ✅ On your resume/portfolio

### Managing Your App

From Streamlit Cloud dashboard:
- **Settings** → Change app settings
- **Reboot** → Restart if needed
- **Delete** → Remove app
- **Logs** → View errors/debugging info

---

## Troubleshooting

### ❌ Problem: "ModuleNotFoundError"

**Solution:** Check `requirements.txt` has all needed packages

```powershell
git add requirements.txt
git commit -m "Fix dependencies"
git push
```

Streamlit will auto-redeploy when you push changes!

### ❌ Problem: "File not found: data/processed/lfs_youth_cleaned.csv"

**Solution:** Ensure the CSV is committed to Git:

```powershell
git add data/processed/lfs_youth_cleaned.csv
git commit -m "Add sample data"
git push
```

### ❌ Problem: App is slow or times out

**Solutions:**
1. Use smaller sample data (see Step 4.1)
2. Remove heavy packages like `shap` from requirements
3. Add caching with `@st.cache_data` (already done)

### ❌ Problem: Git push fails

**Solution:** If file too large:

```powershell
# Remove large files from Git history
git rm --cached data/raw/*.dta
git commit -m "Remove large files"
git push
```

---

## Making Updates

After initial deployment, updating is easy:

```powershell
# 1. Make your changes to code/data

# 2. Commit and push
git add .
git commit -m "Description of changes"
git push

# 3. Streamlit Cloud auto-redeploys! ✨
```

No need to manually redeploy - changes appear in 1-2 minutes!

---

## Advanced Configuration

### Adding Secrets (API Keys, etc.)

If you add external APIs later:

1. Go to Streamlit Cloud → Your app → Settings
2. Click **"Secrets"**
3. Add in TOML format:

```toml
[api]
key = "your-api-key-here"
```

4. Access in code:

```python
import streamlit as st
api_key = st.secrets["api"]["key"]
```

### Custom Domain (Paid Feature)

Streamlit Cloud paid plans allow custom domains like:
```
neet-predictor.yourdomain.com
```

---

## Cost

### Free Tier Includes:
- ✅ 1 private app + unlimited public apps
- ✅ 1 GB RAM per app
- ✅ Auto-redeployment on code changes
- ✅ Community support

### Paid Tiers ($20+/month):
- More resources
- More apps
- Custom domains
- Priority support

**For most use cases, FREE tier is sufficient!** 🎉

---

## Checklist

Before deployment:
- [ ] Git initialized and first commit made
- [ ] GitHub repository created
- [ ] All code pushed to GitHub
- [ ] Sample data file < 100MB
- [ ] `requirements.txt` or `requirements-deploy.txt` ready
- [ ] Streamlit Cloud account created
- [ ] App URL shared with stakeholders

---

## Next Steps

Once deployed:

1. **Monitor Usage**
   - Check Streamlit Cloud analytics
   - Review user feedback

2. **Iterate**
   - Add new features
   - Update data monthly/quarterly
   - Improve visualizations

3. **Scale**
   - Add more provinces/districts
   - Integrate real-time predictions
   - Add user authentication

---

## Support

Need help?
- 📖 [Streamlit Docs](https://docs.streamlit.io)
- 💬 [Streamlit Forum](https://discuss.streamlit.io)
- 📧 Email: support@streamlit.io

---

## Quick Reference Commands

```powershell
# Initial setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main

# Making updates
git add .
git commit -m "Update: description"
git push

# Check status
git status
git log --oneline

# Undo last commit (if needed)
git reset --soft HEAD~1
```

---

**🎉 Congratulations! Your NEET Predictor is now deployed to the world!**

Share your dashboard URL and make an impact with data-driven youth policy! 🚀
