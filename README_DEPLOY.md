# Deploying PulseIQ on Railway (private)

This folder is a deployment copy of `pulseiq-backend-v3` + `pulseiq-frontend-v3`.
The backend serves the frontend too, so it is **one service, one URL**. The v3
folders are untouched.

**Why private:** `ml_assets/data/processed_test_unseen.npz` holds MIMIC-IV waveforms.
PhysioNet's data-use agreement does not allow sharing MIMIC data publicly. So:
- the GitHub repository must be **Private**, and
- the site asks for an **access password** (`SITE_PASSWORD`) before anything
  loads, on top of the app's own login. Give it only to your guide or examiners.

**Why Railway:** the app needs about 1 GB of memory (PyTorch + models). Render's
free and Starter plans have 512 MB, which is not enough; Render needs its Standard
plan (2 GB). Railway's Hobby plan runs it for roughly $5 a month plus usage.

---

## 1. Put the code on GitHub (private), about 5 min
1. Install Git for Windows if needed: https://git-scm.com/download/win
2. Go to https://github.com/new, name it `pulseiq`, choose **Private**, and
   click Create (don't add a README).
3. Open a terminal in this folder (`D:\VT\FINALLLLL\pulseiq-deploy`) and run:
   ```
   git init
   git add .
   git commit -m "PulseIQ deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/pulseiq.git
   git push -u origin main
   ```
   (Sign in to GitHub when the window pops up.)

## 2. Create the Railway service, about 10 min
1. Sign up at https://railway.com with your GitHub account and pick the
   **Hobby** plan.
2. Click **New Project**, then **Deploy from GitHub repo**, then choose `pulseiq`
   and allow Railway to access it. Railway finds the `Dockerfile` and
   `railway.json` automatically.
3. Open the service, go to **Variables**, and add:
   | Name | Value |
   |---|---|
   | `SITE_PASSWORD` | a password you choose, e.g. `Pulse-VT-2026` |
   | `PULSEIQ_DATA_DIR` | `/data` |
4. Right-click the service, choose **Attach Volume**, and set the mount path to
   `/data`. This keeps accounts, uploaded files and the login signing key across
   redeploys.
5. Go to **Settings**, then **Networking**, then **Generate Domain**. You get a
   link like `https://pulseiq-production.up.railway.app`.
6. Wait for the first build (5–10 min, it downloads PyTorch). It's ready when
   the deployment shows **Active** and the health check `/api/health` passes.

## 3. Use it
- Open the link, enter the `SITE_PASSWORD`, then use **Demo Access** or a demo
  account (password `Demo@123`, see `README_V3.md` in the backend-v3 folder).
- On first start the database is created with the clinic's 7 patients and
  Dr. Demo.

## Updating later
Make changes in this folder, then run:
```
git add .
git commit -m "update"
git push
```
Railway redeploys automatically.

## Notes
- **Chatbot:** it uses a local Ollama model on your computer, which doesn't
  exist on Railway, so the chat page will say the assistant is offline.
  Everything else works.
- **Running this folder locally:** run
  `python -m uvicorn main:app --port 8001`, then open http://localhost:8001.
  With no `SITE_PASSWORD` set, there is no gate.
- **Never make the repository public** while it contains
  `processed_test_unseen.npz`.
