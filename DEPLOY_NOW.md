# 🚀 Quick Deployment Checklist

Follow these steps to deploy your NEET Predictor to Streamlit Cloud in ~15 minutes.

## Step 1: Prepare Requirements (2 mins)

```powershell
cd "f:\NEET Predictor"

# Use lightweight requirements for deployment
Copy-Item requirements-deploy.txt requirements.txt -Force
```

## Step 2: Initialize Git (3 mins)

```powershell
git init
git add .
git commit -m "Initial commit: NEET Predictor Dashboard"
```

## Step 3: Create GitHub Repo (5 mins)

1. Go to https://github.com/new
2. Repository name: `neet-predictor-dashboard`
3. Description: "NEET Predictor Dashboard - Pakistan Youth Analysis"
4. Choose Public or Private
5. Click "Create repository"
6. Run these commands (replace YOUR_USERNAME):

```powershell
git remote add origin https://github.com/YOUR_USERNAME/neet-predictor-dashboard.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud (5 mins)

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Fill in:
   - Repository: `YOUR_USERNAME/neet-predictor-dashboard`
   - Branch: `main`
   - Main file: `streamlit_app/app.py`
5. Click "Deploy!"
6. Wait 5-10 minutes for first deployment

## Step 5: Done! 🎉

You'll get a URL like:
```
https://YOUR_USERNAME-neet-predictor-dashboard-xxxxx.streamlit.app
```

Share it with the world!

---

## Updating Your Dashboard

After making changes:

```powershell
git add .
git commit -m "Update: description of changes"
git push
```

Streamlit Cloud will automatically redeploy! ✨

---

## Troubleshooting

**Problem: "File not found"**
```powershell
git add data/processed/lfs_youth_cleaned.csv
git commit -m "Add data file"
git push
```

**Problem: "Module not found"**
```powershell
git add requirements.txt
git commit -m "Fix dependencies"
git push
```

**Problem: "File too large"**
```powershell
# Reduce data file size or use Git LFS
git rm --cached data/raw/*.dta
git commit -m "Remove large files"
git push
```

---

**📖 Full Guide:** See `docs/DEPLOYMENT_GUIDE.md` for detailed instructions

**🆘 Need Help?** Check [Streamlit Docs](https://docs.streamlit.io) or [Forum](https://discuss.streamlit.io)
