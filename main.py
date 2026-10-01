"""
PulseIQ Backend API  (v2: validation audit applied - see REMOVED_RESULTS / validation_summary)
====================
Real FastAPI web service that loads the actual trained ML/DL models,
scaler, feature-selection lists and the held-out test dataset from
`ml_assets/` (copied directly from the user's project_2_complete_workspace),
and exposes them over HTTP so the PulseIQ frontend can display REAL
model predictions, REAL SHAP / Integrated-Gradients explanations, and
REAL clinically-validated metrics instead of hardcoded demo numbers.

Run with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8001

All classical-ML and single-modality-DL numbers below are computed LIVE,
on server startup, against ml_assets/data/processed_test_unseen.npz
(11-subject / 327-window held-out test set the models never trained on).

The calibrated-DL, 15-min-recalibrated-DL and multimodal (ECG+PPG+PTT)
rows in /api/model-comparison are NOT re-computed live (those pipelines
require baseline-anchoring / recalibration / ECG-PTT extraction steps
that are out of scope for this API) -- they are reported verbatim from
the project's own markdown evaluation reports, and are clearly flagged
with "source": "report" so the frontend never presents them as live.
The multimodal row is additionally flagged as research-only because its
PTT feature is derived from the true BP label during training (known
label-leakage) -- it must never be framed as a live/invertible endpoint.
"""

import os
import sys
import time
import json
import uuid
import sqlite3
import datetime
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import torch
import requests
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

import auth

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml_assets")
SRC_DIR = os.path.join(ML_DIR, "src")
MODELS_DIR = os.path.join(ML_DIR, "models")
DATA_DIR = os.path.join(ML_DIR, "data")
VAULT_DIR = os.path.join(os.environ.get("PULSEIQ_DATA_DIR", BASE_DIR), "vault_files")   # real uploaded documents

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ML_DIR)

from feature_engineering import extract_window_features  # noqa: E402
from config import Config  # noqa: E402
from dl_model import PPGResNetBiLSTM  # noqa: E402
from metrics import (  # noqa: E402
    compute_regression_metrics,
    evaluate_aami_sp10,
    evaluate_bhs_grade,
)

TARGETS = ["SBP", "DBP", "MAP"]

# ---------------------------------------------------------------------------
# Demo patients -> real subject IDs in the held-out test set.
# These are the fictional names already used across the frontend
# (icu-monitor.html, patient-dashboard.html, doctor-dashboard.html).
# Each is pinned to a REAL subject_id from processed_test_unseen.npz so
# every prediction/explanation is computed from real PPG/VPG/APG windows,
# not synthetic data.
# ---------------------------------------------------------------------------
# EVERY unique subject in the test set gets a patient entry -- the pool is a
# name supply, not a limit. If the dataset ever contains more subjects than
# names here, the extras fall back to "Subject <id>" rather than being dropped,
# so no real subject is ever silently hidden from the roster.
DEMO_PATIENT_POOL = [
    ("rohit-sharma",    "Rohit Sharma"),
    ("meena-patil",     "Meena Patil"),
    ("arvind-kumar",    "Arvind Kumar"),
    ("sunita-nair",     "Sunita Nair"),
    ("imran-qureshi",   "Imran Qureshi"),
    ("lakshmi-iyer",    "Lakshmi Iyer"),
    ("david-fernandes", "David Fernandes"),
    ("neha-bansal",     "Neha Bansal"),
    ("thomas-mathew",   "Thomas Mathew"),
    ("priya-desai",     "Priya Desai"),
    ("aakash-rao",      "Aakash Rao"),
    ("farah-siddiqui",  "Farah Siddiqui"),
]

STATE = {}

# ---------------------------------------------------------------------------
# Health chatbot -- backed by a LOCAL LLM via Ollama (free, no API key,
# runs entirely on your own machine at http://localhost:11434). Nothing here
# calls Claude, OpenAI, or any hosted API -- the model referenced below must
# be pulled locally first with `ollama pull <model>`. See README.md.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = 120

CHAT_SYSTEM_PROMPT = (
    "You are the PulseIQ Health Assistant, a general wellness chatbot inside a "
    "cuffless blood-pressure monitoring app. Rules you must always follow: "
    "1) Give general, evidence-based health/wellness information only -- never "
    "diagnose a condition, never tell the user to change or stop a medication "
    "dose, never claim to interpret their specific lab or clinical results as "
    "a diagnosis. "
    "2) If the user describes symptoms that could be an emergency (e.g. chest "
    "pain, severe shortness of breath, signs of stroke), tell them clearly to "
    "seek emergency care immediately. "
    "3) Keep answers concise and practical (roughly 2-5 sentences unless the "
    "user clearly wants more detail). "
    "4) Always keep in mind you are not a substitute for professional medical "
    "advice -- the app already shows a persistent disclaimer to the user, so "
    "you do not need to repeat it in every message, but do mention consulting "
    "a doctor when the topic is genuinely clinical (new symptoms, medication "
    "questions, abnormal readings). "
    "5) If BP/vitals context is provided below, you may reference it naturally, "
    "but do not overstate what a single reading means."
)


def _load_classical():
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler_StandardScaler.joblib"))
    feature_names = list(scaler.feature_names_in_)
    selected = {
        t: list(joblib.load(os.path.join(MODELS_DIR, f"selected_features_{t}.joblib")))
        for t in TARGETS
    }
    models = {
        t: joblib.load(os.path.join(MODELS_DIR, f"best_model_{t}.joblib"))
        for t in TARGETS
    }
    return scaler, feature_names, selected, models


def _load_dl():
    model = PPGResNetBiLSTM()
    state_dict = torch.load(
        os.path.join(MODELS_DIR, "best_resnet_bilstm_model.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _extract_features_row(ppg, vpg, apg):
    feats = extract_window_features(ppg, vpg, apg, Config.SAMPLING_RATE)
    return pd.DataFrame([feats])


def classical_predict_window(i: int):
    """Real AdaBoost/SVR predictions for test-set window i, per target."""
    X = STATE["X"]
    row = _extract_features_row(X[i, 0, :], X[i, 1, :], X[i, 2, :])
    row = row[STATE["feature_names"]]
    Xs = STATE["scaler"].transform(row)
    preds = {}
    for t in TARGETS:
        idx = [STATE["feature_names"].index(f) for f in STATE["selected"][t]]
        preds[t] = float(STATE["classical_models"][t].predict(Xs[:, idx])[0])
    return preds, Xs


def dl_predict_window(i: int):
    X = STATE["X"]
    x = torch.tensor(X[i : i + 1], dtype=torch.float32)
    with torch.no_grad():
        pred = STATE["dl_model"](x).numpy()[0]
    return {t: float(pred[j]) for j, t in enumerate(TARGETS)}


def surrogate_features(i: int) -> np.ndarray:
    """The 63 pulse measurements for window i, scaled exactly as the surrogate saw them."""
    X = STATE["X"]
    row = _extract_features_row(X[i, 0, :], X[i, 1, :], X[i, 2, :])
    row = row.reindex(columns=STATE["feature_names"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return STATE["scaler"].transform(row)


def surrogate_shap(feat: np.ndarray, target: str):
    """SHAP values (mmHg per measurement), starting point, and the surrogate's estimate."""
    sur = STATE["surrogate"]
    sv = sur["explainers"][target](feat)
    base = float(np.array(sv.base_values).ravel()[0])
    est = float(sur["models"][target].predict(feat)[0])
    return sv.values[0], base, est


def resolve_patient(patient_id: str) -> int:
    """Return the first test-set window index belonging to a demo patient."""
    if patient_id not in STATE["patient_index"]:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    return STATE["patient_index"][patient_id]["first_window"]


def lookup_patient(key: str) -> Optional[dict]:
    """
    Accept either a demo key ("rohit-sharma") or a raw dataset subject id
    ("p119"), so a real account linked to a recording can address it directly
    without going through the demo naming layer.
    """
    idx = STATE.get("patient_index", {})
    if key in idx:
        return idx[key]
    for pid, info in idx.items():
        if info["subject_id"] == key:
            return info
    return None


# ---------------------------------------------------------------------------
# v3 access control
# ---------------------------------------------------------------------------
def require_doctor(current_user: dict = Depends(auth.get_current_user)) -> dict:
    """Role is checked on the server, never trusted from the browser."""
    if current_user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only")
    return current_user


def authorize_patient(patient_id: str, current_user: dict, window: Optional[int] = None) -> dict:
    """Doctors may open any recording; a patient only the one assigned to them.
    A window index must belong to that same recording, so nobody can reach
    another patient's data through a borrowed window number."""
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    if window is not None and window not in info["window_indices"]:
        raise HTTPException(status_code=400, detail="window index does not belong to this patient")
    if current_user["role"] == "doctor":
        return info
    if current_user.get("subject_id") and info["subject_id"] == current_user["subject_id"]:
        return info
    raise HTTPException(status_code=403, detail="You can only view your own recording")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PulseIQ Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    t0 = time.time()
    print("[PulseIQ] Initializing auth database (pulseiq.db) ...")
    auth.init_db()
    auth.init_feature_tables()
    os.makedirs(VAULT_DIR, exist_ok=True)

    print("[PulseIQ] Loading dataset ...")
    npz = np.load(os.path.join(DATA_DIR, "processed_test_unseen.npz"))
    X, Y, subject_ids = npz["X"], npz["Y"], npz["subject_ids"]
    STATE["X"] = X
    STATE["Y"] = Y
    STATE["subject_ids"] = subject_ids

    print("[PulseIQ] Loading classical ML models ...")
    scaler, feature_names, selected, classical_models = _load_classical()
    STATE["scaler"] = scaler
    STATE["feature_names"] = feature_names
    STATE["selected"] = selected
    STATE["classical_models"] = classical_models

    print("[PulseIQ] Loading deep-learning model (PPGResNetBiLSTM) ...")
    STATE["dl_model"] = _load_dl()

    print("[PulseIQ] Mapping demo patients to real subject IDs ...")
    unique_subjects = list(dict.fromkeys(subject_ids.tolist()))
    patient_index = {}
    for i, subj in enumerate(unique_subjects):          # EVERY subject, not a fixed 4
        if i < len(DEMO_PATIENT_POOL):
            pid, real_name = DEMO_PATIENT_POOL[i]
        else:
            pid, real_name = f"subject-{subj}", f"Subject {subj}"
        window_idxs = np.where(subject_ids == subj)[0].tolist()
        patient_index[pid] = {
            "name": real_name,
            "subject_id": str(subj),
            "bed": f"{i + 1:02d}",
            "display_id": f"P2-2024-{i + 1:03d}",
            "first_window": int(window_idxs[0]),
            "window_indices": window_idxs,
            "num_windows": len(window_idxs),
        }
    STATE["patient_index"] = patient_index

    print("[PulseIQ] Computing LIVE metrics over full held-out test set "
          f"({X.shape[0]} windows, {len(unique_subjects)} subjects) ...")
    classical_preds = np.zeros_like(Y)
    dl_preds = np.zeros_like(Y)
    for i in range(X.shape[0]):
        c_pred, _ = classical_predict_window(i)
        d_pred = dl_predict_window(i)
        for j, t in enumerate(TARGETS):
            classical_preds[i, j] = c_pred[t]
            dl_preds[i, j] = d_pred[t]
    STATE["classical_test_preds"] = classical_preds
    STATE["dl_test_preds"] = dl_preds

    live_comparison = {}
    for j, t in enumerate(TARGETS):
        y_true = Y[:, j]
        c_metrics = compute_regression_metrics(y_true, classical_preds[:, j])
        c_aami = evaluate_aami_sp10(y_true, classical_preds[:, j], t)
        c_bhs = evaluate_bhs_grade(y_true, classical_preds[:, j], t)
        d_metrics = compute_regression_metrics(y_true, dl_preds[:, j])
        d_aami = evaluate_aami_sp10(y_true, dl_preds[:, j], t)
        d_bhs = evaluate_bhs_grade(y_true, dl_preds[:, j], t)
        base_pred = np.full_like(y_true, TRAIN_MEAN_BP[t])
        live_comparison[t] = {
            "classical": {"metrics": c_metrics, "aami": c_aami, "bhs": c_bhs},
            "dl": {"metrics": d_metrics, "aami": d_aami, "bhs": d_bhs},
            "baseline": {"metrics": compute_regression_metrics(y_true, base_pred),
                         "aami": evaluate_aami_sp10(y_true, base_pred, t),
                         "bhs": evaluate_bhs_grade(y_true, base_pred, t)},
        }
    STATE["live_comparison"] = live_comparison

    # v3 (Option B): the explanation now describes the DEEP model — the one that
    # produces the reading. A surrogate (gradient-boosted trees) was trained to copy
    # the deep model's outputs from the 63 named pulse measurements
    # (tools/build_surrogate.py). SHAP on the surrogate, against 100 training
    # windows, says which measurements pushed this reading up or down.
    print("[PulseIQ] Loading deep-model surrogate for explanations ...")
    import shap
    mdir = os.path.join(ML_DIR, "models")
    smeta = json.load(open(os.path.join(mdir, "surrogate_meta.json")))
    sbg = np.load(os.path.join(mdir, "surrogate_background.npy"))
    surrogates, sexplainers = {}, {}
    for t in TARGETS:
        m = joblib.load(os.path.join(mdir, f"surrogate_{t}.joblib"))
        surrogates[t] = m
        sexplainers[t] = shap.TreeExplainer(m, data=sbg, feature_perturbation="interventional")
    assert smeta["features"] == feature_names, "surrogate feature order must match the scaler"
    STATE["surrogate"] = {"models": surrogates, "explainers": sexplainers, "meta": smeta}

    auth.seed_accounts([{"name": v["name"], "subject_id": v["subject_id"]}
                        for v in STATE["patient_index"].values()])
    print(f"[PulseIQ] Startup complete in {time.time() - t0:.1f}s. "
          f"Demo patients: {list(patient_index.keys())}")


# ---------------------------------------------------------------------------
# Auth (real accounts, SQLite + hashed passwords + JWT sessions)
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str  # "patient" or "doctor"


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    try:
        user = auth.create_user(req.name, req.email, req.password, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = auth.create_access_token(user)
    return {"token": token, "user": user}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = auth.verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = auth.create_access_token(user)
    return {"token": token, "user": user}


class DemoRequest(BaseModel):
    role: str = "patient"


@app.post("/api/auth/demo")
def demo_login(req: DemoRequest):
    """Demo Access: quietly signs in as the seeded demo patient (Rohit Sharma,
    shared demo recording) or the seeded demo doctor. Same code path as a real
    login, so every page is protected the same way."""
    if req.role == "doctor":
        email = auth.DEMO_DOCTOR_EMAIL
    else:
        info = lookup_patient(auth.DEMO_SUBJECT)
        email = auth.seeded_patient_email(info["name"]) if info else None
    user = auth.get_user_by_email(email) if email else None
    if not user:
        raise HTTPException(status_code=503, detail="Demo accounts are not ready yet")
    token = auth.create_access_token(user)
    return {"token": token, "user": user}


@app.get("/api/auth/me")
def me(current_user: dict = Depends(auth.get_current_user)):
    return {"user": current_user}


# ---------------------------------------------------------------------------
# Linking an account to a real recording in the dataset
# ---------------------------------------------------------------------------
# A freshly signed-up account has no recording of its own. Rather than quietly
# showing someone else's waveform under their name, the account stays unlinked
# until a Patient ID is entered, and the UI says so.
class LinkSubjectRequest(BaseModel):
    subjectId: Optional[str] = None   # null unlinks


@app.get("/api/recordings")
def recordings(current_user: dict = Depends(require_doctor)):
    """Patient IDs available to link, so the picker can't offer a bad value."""
    out = []
    for pid, info in STATE.get("patient_index", {}).items():
        out.append({
            "subjectId": info["subject_id"],
            "demoKey": pid,
            "demoName": info["name"],
            "numWindows": info["num_windows"],
        })
    out.sort(key=lambda r: r["subjectId"])
    return {"recordings": out}


@app.post("/api/me/subject")
def link_subject(req: LinkSubjectRequest, current_user: dict = Depends(auth.get_current_user)):
    # v3: recordings are assigned by the clinic. Patients cannot pick one.
    raise HTTPException(status_code=403,
                        detail="Recordings are assigned by your clinic and can't be changed here.")
    if req.subjectId in (None, ""):
        auth.set_user_subject(current_user["id"], None)
        return {"subjectId": None, "linked": False}

    info = lookup_patient(req.subjectId)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"No recording found with Patient ID '{req.subjectId}'. "
                   f"Check the ID and try again.",
        )
    auth.set_user_subject(current_user["id"], info["subject_id"])
    return {
        "subjectId": info["subject_id"],
        "linked": True,
        "numWindows": info["num_windows"],
    }


# ---------------------------------------------------------------------------
# Medical Vault — real file storage, owned by the account
# ---------------------------------------------------------------------------
VAULT_CATEGORIES = ["reports", "prescriptions", "lab", "scans"]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Only formats a clinic would actually hand over. Blocking executables here
# matters because these files are served back out for download.
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".csv", ".doc", ".docx"}


@app.get("/api/vault")
def vault_list(current_user: dict = Depends(auth.get_current_user)):
    conn = auth.get_db()
    try:
        rows = conn.execute(
            """SELECT id, original_name, mime, size_bytes, category, uploaded_at
               FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC""",
            (current_user["id"],),
        ).fetchall()
        docs = [{
            "id": r["id"], "name": r["original_name"], "mime": r["mime"],
            "sizeBytes": r["size_bytes"], "category": r["category"],
            "uploadedAt": r["uploaded_at"],
        } for r in rows]
        return {"documents": docs, "categories": VAULT_CATEGORIES,
                "totalBytes": sum(d["sizeBytes"] for d in docs)}
    finally:
        conn.close()


@app.post("/api/vault/upload")
async def vault_upload(
    file: UploadFile = File(...),
    category: str = Form("reports"),
    current_user: dict = Depends(auth.get_current_user),
):
    if category not in VAULT_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"category must be one of {VAULT_CATEGORIES}")

    original = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"'{ext or 'this file type'}' isn't allowed. Accepted: {', '.join(sorted(ALLOWED_EXT))}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="That file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is larger than the 15 MB limit")

    user_dir = os.path.join(VAULT_DIR, str(current_user["id"]))
    os.makedirs(user_dir, exist_ok=True)
    stored = uuid.uuid4().hex + ext           # never trust the client's name on disk
    with open(os.path.join(user_dir, stored), "wb") as f:
        f.write(data)

    conn = auth.get_db()
    try:
        cur = conn.execute(
            """INSERT INTO documents (user_id, original_name, stored_name, mime, size_bytes, category, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (current_user["id"], original, stored, file.content_type, len(data), category, time.time()),
        )
        conn.commit()
        return {"id": cur.lastrowid, "name": original, "sizeBytes": len(data), "category": category}
    finally:
        conn.close()


def _owned_document(doc_id: int, user_id: int):
    conn = auth.get_db()
    try:
        r = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise HTTPException(status_code=404, detail="Document not found")
    # Ownership is checked server-side; a guessed id must not expose someone else's file.
    if r["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="That document belongs to another account")
    return r


@app.get("/api/vault/{doc_id}/download")
def vault_download(doc_id: int, current_user: dict = Depends(auth.get_current_user)):
    r = _owned_document(doc_id, current_user["id"])
    path = os.path.join(VAULT_DIR, str(r["user_id"]), r["stored_name"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing from storage")
    return FileResponse(path, filename=r["original_name"],
                        media_type=r["mime"] or "application/octet-stream")


@app.delete("/api/vault/{doc_id}")
def vault_delete(doc_id: int, current_user: dict = Depends(auth.get_current_user)):
    r = _owned_document(doc_id, current_user["id"])
    path = os.path.join(VAULT_DIR, str(r["user_id"]), r["stored_name"])
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass                                  # DB row still goes, file is orphaned at worst
    conn = auth.get_db()
    try:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
    return {"deleted": doc_id}


# ---------------------------------------------------------------------------
# Medicine reminders — real schedules and real adherence
# ---------------------------------------------------------------------------
class MedicineRequest(BaseModel):
    name: str
    dosage: Optional[str] = ""
    times: List[str]                          # ["08:00", "20:00"]
    notes: Optional[str] = ""


def _today():
    return datetime.date.today().isoformat()


def _valid_time(t: str) -> bool:
    try:
        datetime.datetime.strptime(t, "%H:%M")
        return True
    except ValueError:
        return False


@app.get("/api/medicines")
def medicines_list(days: int = 30, day: Optional[str] = None,
                   current_user: dict = Depends(auth.get_current_user)):
    days = max(1, min(120, days))
    # `day` lets the calendar show and edit any date, not just today.
    selected = day or _today()
    try:
        sel_date = datetime.date.fromisoformat(selected)
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
    # Future dates are viewable: the calendar shows what is scheduled ahead.
    # Logging a dose that has not happened yet is still refused, in
    # /api/medicines/{id}/dose, which is where that rule belongs.
    if sel_date > datetime.date.today() + datetime.timedelta(days=366):
        raise HTTPException(status_code=400, detail="day is too far in the future")

    conn = auth.get_db()
    try:
        meds = conn.execute(
            "SELECT * FROM medicines WHERE user_id = ? AND active = 1 ORDER BY created_at",
            (current_user["id"],),
        ).fetchall()
        taken_sel = {
            (r["medicine_id"], r["slot"])
            for r in conn.execute(
                "SELECT medicine_id, slot FROM medicine_logs WHERE user_id = ? AND day = ?",
                (current_user["id"], selected),
            ).fetchall()
        }

        out = []
        for m in meds:
            slots = json.loads(m["times"])
            created = datetime.date.fromtimestamp(m["created_at"])
            out.append({
                "id": m["id"], "name": m["name"], "dosage": m["dosage"] or "",
                "notes": m["notes"] or "", "times": slots,
                "takenOnDay": [s for s in slots if (m["id"], s) in taken_sel],
                # A medicine can't be logged before it was added.
                "scheduledOnDay": created <= sel_date,
                "createdAt": m["created_at"],
            })

        # Adherence over the window: doses actually logged vs doses scheduled
        # since each medicine was created (never counting days before it existed).
        start = (datetime.date.today() - datetime.timedelta(days=days - 1))
        expected = 0
        for m in meds:
            slots = json.loads(m["times"])
            created = datetime.date.fromtimestamp(m["created_at"])
            first = max(start, created)
            if first <= datetime.date.today():
                expected += ((datetime.date.today() - first).days + 1) * len(slots)
        logged = conn.execute(
            "SELECT COUNT(*) AS n FROM medicine_logs WHERE user_id = ? AND day >= ?",
            (current_user["id"], start.isoformat()),
        ).fetchone()["n"]

        by_day = {
            r["day"]: r["n"] for r in conn.execute(
                """SELECT day, COUNT(*) AS n FROM medicine_logs
                   WHERE user_id = ? AND day >= ? GROUP BY day""",
                (current_user["id"], start.isoformat()),
            ).fetchall()
        }
        history = []
        for i in range(days):
            d = (start + datetime.timedelta(days=i)).isoformat()
            history.append({"day": d, "taken": by_day.get(d, 0)})

        return {
            "medicines": out,
            "today": _today(),
            "selectedDay": selected,
            "adherence": {
                "windowDays": days,
                "expected": expected,
                "taken": int(logged),
                "percent": round(logged / expected * 100, 1) if expected else None,
            },
            "history": history,
        }
    finally:
        conn.close()


@app.post("/api/medicines")
def medicine_create(req: MedicineRequest, current_user: dict = Depends(auth.get_current_user)):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Medicine name is required")
    times = [t.strip() for t in req.times if t and t.strip()]
    if not times:
        raise HTTPException(status_code=400, detail="Add at least one reminder time")
    bad = [t for t in times if not _valid_time(t)]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid time format: {', '.join(bad)} (use HH:MM)")
    times = sorted(set(times))

    conn = auth.get_db()
    try:
        cur = conn.execute(
            """INSERT INTO medicines (user_id, name, dosage, times, notes, active, created_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (current_user["id"], name, (req.dosage or "").strip(),
             json.dumps(times), (req.notes or "").strip(), time.time()),
        )
        conn.commit()
        return {"id": cur.lastrowid, "name": name, "times": times}
    finally:
        conn.close()


def _owned_medicine(med_id: int, user_id: int):
    conn = auth.get_db()
    try:
        r = conn.execute("SELECT * FROM medicines WHERE id = ?", (med_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise HTTPException(status_code=404, detail="Medicine not found")
    if r["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="That medicine belongs to another account")
    return r


@app.delete("/api/medicines/{med_id}")
def medicine_delete(med_id: int, current_user: dict = Depends(auth.get_current_user)):
    _owned_medicine(med_id, current_user["id"])
    conn = auth.get_db()
    try:
        conn.execute("DELETE FROM medicine_logs WHERE medicine_id = ?", (med_id,))
        conn.execute("DELETE FROM medicines WHERE id = ?", (med_id,))
        conn.commit()
    finally:
        conn.close()
    return {"deleted": med_id}


class DoseRequest(BaseModel):
    slot: str
    day: Optional[str] = None
    taken: bool = True


@app.post("/api/medicines/{med_id}/dose")
def medicine_dose(med_id: int, req: DoseRequest, current_user: dict = Depends(auth.get_current_user)):
    m = _owned_medicine(med_id, current_user["id"])
    slots = json.loads(m["times"])
    if req.slot not in slots:
        raise HTTPException(status_code=400, detail=f"'{req.slot}' is not a scheduled time for this medicine")
    day = req.day or _today()
    try:
        day_date = datetime.date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
    # A dose can be logged late, but never in advance — that would record
    # something that has not happened.
    if day_date > datetime.date.today():
        raise HTTPException(status_code=400, detail="Cannot log a dose for a future date")

    conn = auth.get_db()
    try:
        if req.taken:
            try:
                conn.execute(
                    "INSERT INTO medicine_logs (medicine_id, user_id, day, slot, taken_at) VALUES (?, ?, ?, ?, ?)",
                    (med_id, current_user["id"], day, req.slot, time.time()),
                )
            except sqlite3.IntegrityError:
                pass                          # already marked; tapping twice is harmless
        else:
            conn.execute(
                "DELETE FROM medicine_logs WHERE medicine_id = ? AND day = ? AND slot = ?",
                (med_id, day, req.slot),
            )
        conn.commit()
    finally:
        conn.close()
    return {"medicineId": med_id, "day": day, "slot": req.slot, "taken": req.taken}


# ---------------------------------------------------------------------------
# Direct messaging — real accounts talking to each other
# ---------------------------------------------------------------------------
class SendMessageRequest(BaseModel):
    recipientId: int
    body: str
    replyTo: Optional[int] = None


class EditMessageRequest(BaseModel):
    body: str


@app.get("/api/contacts")
def contacts(current_user: dict = Depends(auth.get_current_user)):
    """Who this user can message: doctors see patients, patients see doctors."""
    want = "patient" if current_user["role"] == "doctor" else "doctor"
    people = auth.list_users_by_role(want, exclude_id=current_user["id"])
    out = []
    for p in people:
        last = auth.last_message_with(current_user["id"], p["id"])
        out.append({
            "id": p["id"],
            "name": p["name"],
            "role": p["role"],
            "lastMessage": last["body"] if last else None,
            "lastAt": last["createdAt"] if last else None,
            "unread": auth.unread_count_from(current_user["id"], p["id"]),
        })
    # Unread first, then most recent activity.
    out.sort(key=lambda c: (-(1 if c["unread"] else 0), c["lastAt"] is None, -(c["lastAt"] or 0)))
    return {"contacts": out, "you": current_user, "totalUnread": auth.total_unread(current_user["id"])}


@app.get("/api/unread")
def unread(current_user: dict = Depends(auth.get_current_user)):
    """Cheap poll for the sidebar badge, so any page can show a new-message dot."""
    return {"totalUnread": auth.total_unread(current_user["id"])}


@app.get("/api/notifications")
def notifications(current_user: dict = Depends(auth.get_current_user)):
    """
    Real events for the signed-in account only. Every item is derived from
    something that actually happened: a message that arrived, a dose that is
    scheduled and not yet logged, an account with no recording attached, or the
    band the model's current reading falls into. Nothing here is generated to
    fill the list -- an account with nothing going on gets an empty list.
    """
    uid = current_user["id"]
    items = []
    now = datetime.datetime.now()

    # --- 1. Unread messages, grouped by who sent them ---------------------
    conn = auth.get_db()
    try:
        rows = conn.execute(
            """SELECT m.sender_id AS sid, u.name AS sname, u.role AS srole,
                      COUNT(*) AS n, MAX(m.created_at) AS last_at,
                      (SELECT body FROM messages m2
                        WHERE m2.sender_id = m.sender_id AND m2.recipient_id = ?
                          AND m2.read_at IS NULL
                        ORDER BY m2.created_at DESC LIMIT 1) AS last_body
                 FROM messages m JOIN users u ON u.id = m.sender_id
                WHERE m.recipient_id = ? AND m.read_at IS NULL
             GROUP BY m.sender_id
             ORDER BY last_at DESC""",
            (uid, uid),
        ).fetchall()
    finally:
        conn.close()

    for r in rows:
        who = ("Dr. " if r["srole"] == "doctor" else "") + r["sname"]
        items.append({
            "id": f"msg-{r['sid']}",
            "kind": "message",
            "icon": "\U0001f4ac",
            "title": f"{r['n']} new message{'s' if r['n'] > 1 else ''} from {who}",
            "body": (r["last_body"] or "")[:110],
            "href": "chat.html",
            "at": r["last_at"],
            "unread": True,
        })

    # --- 2. Medicine doses: scheduled today and not logged ----------------
    if current_user["role"] == "patient":
        today = _today()
        conn = auth.get_db()
        try:
            meds = conn.execute(
                "SELECT * FROM medicines WHERE user_id = ? AND active = 1", (uid,)
            ).fetchall()
            taken = {
                (x["medicine_id"], x["slot"])
                for x in conn.execute(
                    "SELECT medicine_id, slot FROM medicine_logs WHERE user_id = ? AND day = ?",
                    (uid, today),
                ).fetchall()
            }
        finally:
            conn.close()

        overdue, upcoming = [], []
        for m in meds:
            for slot in json.loads(m["times"]):
                if (m["id"], slot) in taken:
                    continue
                try:
                    hh, mm = [int(x) for x in slot.split(":")[:2]]
                except ValueError:
                    continue
                when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                label = f"{m['name']}{(' ' + m['dosage']) if m['dosage'] else ''}"
                (overdue if when < now else upcoming).append((when, label, slot))

        overdue.sort(key=lambda x: x[0])
        upcoming.sort(key=lambda x: x[0])

        if overdue:
            names = ", ".join(sorted({o[1] for o in overdue}))[:110]
            items.append({
                "id": "med-overdue",
                "kind": "medicine",
                "icon": "\U0001f48a",
                "title": f"{len(overdue)} dose{'s' if len(overdue) > 1 else ''} not logged today",
                "body": names,
                "href": "medicine-reminders.html",
                "at": overdue[0][0].timestamp(),
                "unread": True,
            })
        if upcoming:
            when, label, slot = upcoming[0]
            items.append({
                "id": "med-next",
                "kind": "medicine",
                "icon": "\u23f0",
                "title": f"Next dose at {slot}",
                "body": label,
                "href": "medicine-reminders.html",
                "at": when.timestamp(),
                "unread": False,
            })

        # --- 3. Account with no recording attached ------------------------
        if not current_user.get("subject_id"):
            items.append({
                "id": "no-subject",
                "kind": "account",
                "icon": "\U0001f517",
                "title": "No recording linked to this account",
                "body": "Readings stay empty until a Patient ID is linked.",
                "href": "profile.html",
                "at": now.timestamp(),
                "unread": True,
            })
        else:
            # --- 4. What band the model's current reading falls into ------
            try:
                info = lookup_patient(current_user["subject_id"])
                if info is not None:
                    dl = dl_predict_window(info["first_window"])
                    cls = classify_bp(dl["SBP"], dl["DBP"])
                    if cls["band"] != "Normal":
                        items.append({
                            "id": "bp-band",
                            "kind": "reading",
                            "icon": "\u2764\ufe0f",
                            "title": f"Latest reading reads {cls['band']} "
                                     f"({dl['SBP']:.0f}/{dl['DBP']:.0f} mmHg)",
                            "body": cls["why"] + " - model estimate, not a diagnosis.",
                            "href": "xai-results.html",
                            "at": now.timestamp(),
                            "unread": True,
                        })
            except Exception:
                pass

    items.sort(key=lambda x: x.get("at") or 0, reverse=True)
    return {
        "items": items,
        "unreadCount": sum(1 for i in items if i.get("unread")),
        "generatedAt": now.timestamp(),
    }


@app.get("/api/messages/{other_id}")
def messages(other_id: int, current_user: dict = Depends(auth.get_current_user)):
    other = auth.get_user_by_id(other_id)
    if not other:
        raise HTTPException(status_code=404, detail="That user no longer exists")
    # Patients message doctors and vice versa — same-role chat isn't a thing here.
    if other["role"] == current_user["role"]:
        raise HTTPException(status_code=403, detail="You can only message the other role")
    # Opening the thread is what marks it read.
    marked = auth.mark_conversation_read(current_user["id"], other_id)
    return {
        "with": {"id": other["id"], "name": other["name"], "role": other["role"]},
        "messages": auth.get_conversation(current_user["id"], other_id),
        "markedRead": marked,
        "totalUnread": auth.total_unread(current_user["id"]),
    }


@app.post("/api/messages")
def send_message(req: SendMessageRequest, current_user: dict = Depends(auth.get_current_user)):
    other = auth.get_user_by_id(req.recipientId)
    if not other:
        raise HTTPException(status_code=404, detail="That user no longer exists")
    if other["role"] == current_user["role"]:
        raise HTTPException(status_code=403, detail="You can only message the other role")
    try:
        return auth.insert_message(current_user["id"], req.recipientId, req.body, req.replyTo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/api/messages/{message_id}")
def edit_message(message_id: int, req: EditMessageRequest,
                 current_user: dict = Depends(auth.get_current_user)):
    """Rewrite your own message. The thread marks it as edited."""
    try:
        return auth.edit_message(message_id, current_user["id"], req.body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/messages/{message_id}")
def delete_message(message_id: int, scope: str = "me",
                   current_user: dict = Depends(auth.get_current_user)):
    """scope=me hides it from you alone; scope=everyone unsends it for both."""
    if scope not in ("me", "everyone"):
        raise HTTPException(status_code=400, detail="scope must be 'me' or 'everyone'")
    try:
        if scope == "everyone":
            auth.delete_message_for_everyone(message_id, current_user["id"])
        else:
            auth.delete_message_for_me(message_id, current_user["id"])
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except TimeoutError:
        raise HTTPException(
            status_code=403,
            detail="This message is more than an hour old — you can only delete it for yourself now",
        )
    return {"id": message_id, "scope": scope, "deleted": True}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {
            "classical": list(STATE.get("classical_models", {}).keys()),
            "dl": STATE.get("dl_model") is not None,
        },
        "test_set_windows": int(STATE["X"].shape[0]) if "X" in STATE else 0,
        "test_set_subjects": len(set(STATE["subject_ids"].tolist())) if "subject_ids" in STATE else 0,
    }


@app.get("/api/patients")
def list_patients(current_user: dict = Depends(require_doctor)):
    out = []
    for pid, info in STATE["patient_index"].items():
        out.append({
            "id": pid,
            "name": info["name"],
            "subjectId": info["subject_id"],
            "bed": info["bed"],
            "displayId": info["display_id"],
            "numWindows": info["num_windows"],
        })
    return {"patients": out}


def prediction_reliability(classical_pred: dict, dl_pred: dict) -> dict:
    """
    Honest reliability figures for a single prediction. There is no calibrated
    per-sample confidence from this model, so nothing here is invented: every
    number is either the deep model's measured error on the held-out test set
    or a direct comparison of the two independent models on THIS window.
    """
    live = STATE.get("live_comparison") or {}
    out = {}
    for t in TARGETS:
        entry = live.get(t, {}).get("dl", {})
        m = entry.get("metrics", {}) or {}
        bhs = entry.get("bhs", {}) or {}
        aami = entry.get("aami", {}) or {}
        c = classical_pred.get(t)
        d = dl_pred.get(t)
        out[t] = {
            # Mean absolute error of this model across every held-out window.
            "maeMmHg": m.get("MAE"),
            "rmseMmHg": m.get("RMSE"),
            # BHS cumulative bands: share of test predictions within N mmHg.
            "pctWithin5": bhs.get("pct_le_5mmHg"),
            "pctWithin10": bhs.get("pct_le_10mmHg"),
            "pctWithin15": bhs.get("pct_le_15mmHg"),
            "bhsGrade": bhs.get("grade"),
            "aamiPass": aami.get("aami_compliant"),
            # How far the two independent models land apart on this window.
            "modelGapMmHg": (abs(c - d) if (c is not None and d is not None) else None),
        }
    out["note"] = (
        "maeMmHg / pctWithin* are measured on the held-out test set, not a "
        "per-sample confidence. modelGapMmHg is the distance between the "
        "classical and deep models on this window."
    )
    return out


@app.get("/api/predict/{patient_id}")
def predict(patient_id: str, window: Optional[int] = None,
            current_user: dict = Depends(auth.get_current_user)):
    authorize_patient(patient_id, current_user, window)   # v3: ownership check
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

    if window is not None:
        if window not in info["window_indices"]:
            raise HTTPException(status_code=400, detail="window index does not belong to this patient")
        i = window
    else:
        i = info["first_window"]

    classical_pred, _ = classical_predict_window(i)
    dl_pred = dl_predict_window(i)
    true_label = {t: float(STATE["Y"][i, j]) for j, t in enumerate(TARGETS)}

    return {
        "patientId": patient_id,
        "patientName": info["name"],
        "subjectId": info["subject_id"],
        "windowIndex": i,
        "numWindowsAvailable": info["num_windows"],
        "classical": classical_pred,
        "dl": dl_pred,
        "trueLabel": true_label,
        "reliability": prediction_reliability(classical_pred, dl_pred),
        "note": "classical = best per-target classical model (AdaBoost/SVR); "
                "dl = PPGResNetBiLSTM (single-modality). trueLabel is the "
                "ground-truth arterial-line BP for this held-out window.",
    }


@app.get("/api/waveform/{patient_id}")
def waveform(patient_id: str, window: Optional[int] = None, downsample: int = 3,
             current_user: dict = Depends(auth.get_current_user)):
    """
    The real recorded PPG / VPG / APG waveform for this patient's real test
    subject -- the exact three channels the deep model consumes as input.
    Lightweight: pure array slicing, no model inference, so the ICU monitor
    can poll it cheaply. Heart rate is derived from real PPG systolic peaks.
    """
    authorize_patient(patient_id, current_user, window)   # v3: ownership check
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

    i = window if window is not None else info["first_window"]
    if i not in info["window_indices"]:
        raise HTTPException(status_code=400, detail="window index does not belong to this patient")

    downsample = max(1, min(10, downsample))
    X = STATE["X"]
    ppg = X[i, 0, :]
    fs = Config.SAMPLING_RATE

    # Real heart rate from systolic peaks (>=0.4s apart -> max 150 bpm)
    hr_bpm = None
    try:
        from scipy.signal import find_peaks
        norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        peaks, _ = find_peaks(norm, distance=int(0.4 * fs), height=0.2)
        if len(peaks) >= 2:
            hr_bpm = float(60.0 * fs / np.mean(np.diff(peaks)))
    except Exception:
        pass

    dl_pred = dl_predict_window(i)

    return {
        "patientId": patient_id,
        "patientName": info["name"],
        "subjectId": info["subject_id"],
        "windowIndex": int(i),
        "numWindowsAvailable": info["num_windows"],
        "windowIndices": info["window_indices"],
        "samplingRateHz": fs,
        "downsample": downsample,
        "durationSec": float(X.shape[2] / fs),
        "channels": {
            "ppg": X[i, 0, ::downsample].tolist(),
            "vpg": X[i, 1, ::downsample].tolist(),
            "apg": X[i, 2, ::downsample].tolist(),
        },
        "heartRateBpm": hr_bpm,
        "dl": dl_pred,
        "trueLabel": {t: float(STATE["Y"][i, j]) for j, t in enumerate(TARGETS)},
        "note": "PPG/VPG/APG are the real recorded channels for this subject. "
                "Heart rate is computed from real PPG peaks. SpO2 and "
                "respiration are NOT in this dataset and are not returned.",
    }


@app.get("/api/subject/{patient_id}")
def subject_detail(patient_id: str, current_user: dict = Depends(auth.get_current_user)):
    """
    Full transparency on the REAL person behind a demo patient: every real
    window belonging to their real subject_id in processed_test_unseen.npz,
    with real ground-truth BP and real model predictions for each one (from
    the cached full-test-set evaluation computed at startup -- no extra
    inference cost here). This is what a demo name like "Rohit Sharma"
    actually maps to underneath: a real, anonymized ICU subject_id, not a
    real named person.
    """
    authorize_patient(patient_id, current_user)   # v3: ownership check
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

    Y = STATE["Y"]
    classical_preds = STATE["classical_test_preds"]
    dl_preds = STATE["dl_test_preds"]

    windows = []
    for i in info["window_indices"]:
        windows.append({
            "windowIndex": int(i),
            "trueLabel": {t: float(Y[i, j]) for j, t in enumerate(TARGETS)},
            "classical": {t: float(classical_preds[i, j]) for j, t in enumerate(TARGETS)},
            "dl": {t: float(dl_preds[i, j]) for j, t in enumerate(TARGETS)},
        })

    dl_errors = {
        t: float(np.mean(np.abs(dl_preds[info["window_indices"], j] - Y[info["window_indices"], j])))
        for j, t in enumerate(TARGETS)
    }

    return {
        "patientId": patient_id,
        "displayName": info["name"],
        "realSubjectId": info["subject_id"],
        "totalRealWindows": info["num_windows"],
        "datasetSource": "processed_test_unseen.npz (held-out test set, unseen during training)",
        "perWindowMeanAbsError_DL": dl_errors,
        "windows": windows,
        "disclosure": f"'{info['name']}' is a fictional display name PulseIQ assigns to real "
                       f"anonymized subject '{info['subject_id']}' from the held-out test set, "
                       f"purely so the demo has a human-readable identity. Every reading, "
                       f"waveform, and prediction shown for this patient is computed from that "
                       f"real subject's real recorded data -- none of it is synthetic.",
    }


# ---------------------------------------------------------------------------
# Clinical explanation layer
# ---------------------------------------------------------------------------
# Raw feature names ("vpg_spectral_centroid") are meaningless to a clinician or
# a patient, so each engineered feature is mapped to the physiological property
# it actually measures, and grouped into themes a human can reason about.
# Descriptions state what the signal property IS; they deliberately avoid
# asserting causation about the person's health.
FEATURE_GROUPS = {
    "rate": {
        "label": "Heart rate & rhythm",
        "what": "How fast the heart is beating and how regular the beat-to-beat timing is.",
        "features": [
            "ppg_hr_bpm", "ppg_pulse_interval_std",
            "ppg_lf_hf_ratio", "vpg_lf_hf_ratio", "apg_lf_hf_ratio",
            "ppg_dominant_frequency", "vpg_dominant_frequency",
        ],
    },
    "shape": {
        "label": "Pulse shape & timing",
        "what": "How wide each pulse is and how quickly it rises and falls — narrower, faster pulses tend to accompany higher pressure.",
        "features": [
            "ppg_pulse_width_25", "ppg_pulse_width_50", "ppg_pulse_width_75",
            "ppg_pulse_width_90", "ppg_stat_kurtosis", "vpg_stat_kurtosis",
            "ppg_rise_time", "ppg_decay_time", "ppg_dicrotic_notch_location",
            "vpg_stat_skew",
        ],
    },
    "stiffness": {
        "label": "Arterial stiffness indicators",
        "what": "Features of the pulse's acceleration wave that reflect how stiff or compliant the arteries appear.",
        "features": [
            "apg_dominant_frequency", "apg_spectral_centroid",
            "apg_spectral_entropy", "apg_spectral_rolloff",
            "ppg_stiffness_index", "ppg_reflection_index", "ppg_augmentation_index",
            "apg_aging_index", "apg_wave_b", "apg_wave_c", "apg_wave_d", "apg_wave_e",
            "apg_energy", "apg_stat_std", "apg_stat_skew", "apg_stat_kurtosis",
            "apg_stat_iqr", "apg_stat_cov", "apg_stat_p90", "apg_band_energy_hf",
        ],
    },
    "strength": {
        "label": "Pulse strength & perfusion",
        "what": "How strong the pulse signal is overall, reflecting blood volume reaching the sensor.",
        "features": [
            "ppg_signal_energy", "ppg_stat_p10", "ppg_stat_p75", "ppg_stat_p90",
            "ppg_stat_std", "ppg_stat_iqr", "ppg_stat_mad",
            "vpg_mean_amplitude", "vpg_stat_std", "vpg_stat_iqr", "vpg_stat_mad",
            "vpg_stat_p90", "vpg_stat_cov",
        ],
    },
    "complexity": {
        "label": "Waveform detail & regularity",
        "what": "How much fine structure and high-frequency detail the pulse waveform carries.",
        "features": [
            "ppg_spectral_centroid", "ppg_spectral_entropy", "ppg_spectral_rolloff",
            "ppg_band_energy_hf", "vpg_spectral_centroid", "vpg_spectral_entropy",
            "vpg_spectral_rolloff", "vpg_band_energy_hf", "ppg_hjorth_complexity",
            "vpg_hjorth_mobility", "vpg_hjorth_complexity",
            "apg_hjorth_mobility", "apg_hjorth_complexity",
        ],
    },
}

FEATURE_LABELS = {
    "ppg_hr_bpm": "Heart rate",
    "ppg_pulse_interval_std": "Beat-to-beat timing variation",
    "ppg_lf_hf_ratio": "Autonomic balance (pulse)",
    "vpg_lf_hf_ratio": "Autonomic balance (upstroke)",
    "apg_lf_hf_ratio": "Autonomic balance (acceleration)",
    "ppg_pulse_width_25": "Pulse width (near peak)",
    "ppg_pulse_width_50": "Pulse width (mid height)",
    "ppg_pulse_width_75": "Pulse width (lower third)",
    "ppg_pulse_width_90": "Pulse width (near base)",
    "ppg_stat_kurtosis": "Pulse peak sharpness",
    "vpg_stat_kurtosis": "Upstroke sharpness",
    "apg_dominant_frequency": "Dominant stiffness frequency",
    "apg_spectral_centroid": "Acceleration wave balance",
    "apg_spectral_entropy": "Acceleration wave regularity",
    "apg_spectral_rolloff": "Acceleration high-frequency edge",
    "ppg_signal_energy": "Overall pulse strength",
    "ppg_stat_p10": "Pulse trough level",
    "ppg_stat_p75": "Upper pulse level",
    "ppg_stat_p90": "Pulse peak level",
    "ppg_stat_std": "Pulse amplitude spread",
    "ppg_stat_iqr": "Pulse amplitude range",
    "ppg_stat_mad": "Pulse amplitude deviation",
    "ppg_spectral_centroid": "Waveform frequency balance",
    "ppg_spectral_entropy": "Waveform regularity",
    "ppg_spectral_rolloff": "Waveform high-frequency edge",
    "ppg_band_energy_hf": "High-frequency pulse energy",
    "vpg_spectral_centroid": "Upstroke frequency balance",
    "vpg_spectral_entropy": "Upstroke regularity",
    "vpg_spectral_rolloff": "Upstroke high-frequency edge",
    # v3: names for the remaining measurements, so the surrogate can use all 63
    "ppg_dominant_frequency": "Main pulse rhythm frequency",
    "vpg_dominant_frequency": "Main upstroke rhythm frequency",
    "ppg_rise_time": "Pulse rise time",
    "ppg_decay_time": "Pulse fall time",
    "ppg_dicrotic_notch_location": "Valve-closure notch timing",
    "vpg_stat_skew": "Upstroke asymmetry",
    "ppg_stiffness_index": "Stiffness index",
    "ppg_reflection_index": "Wave reflection index",
    "ppg_augmentation_index": "Augmentation index",
    "apg_aging_index": "Vascular ageing index",
    "apg_wave_b": "Acceleration wave, early dip (b)",
    "apg_wave_c": "Acceleration wave, mid rise (c)",
    "apg_wave_d": "Acceleration wave, late dip (d)",
    "apg_wave_e": "Acceleration wave, notch peak (e)",
    "apg_energy": "Acceleration wave strength",
    "apg_stat_std": "Acceleration wave spread",
    "apg_stat_skew": "Acceleration wave asymmetry",
    "apg_stat_kurtosis": "Acceleration wave sharpness",
    "apg_stat_iqr": "Acceleration wave range",
    "apg_stat_cov": "Acceleration wave variability",
    "apg_stat_p90": "Acceleration wave peak level",
    "apg_band_energy_hf": "Acceleration high-frequency energy",
    "vpg_mean_amplitude": "Upstroke strength",
    "vpg_stat_std": "Upstroke spread",
    "vpg_stat_iqr": "Upstroke range",
    "vpg_stat_mad": "Upstroke deviation",
    "vpg_stat_p90": "Upstroke peak level",
    "vpg_stat_cov": "Upstroke variability",
    "vpg_band_energy_hf": "Upstroke high-frequency energy",
    "ppg_hjorth_complexity": "Pulse waveform complexity",
    "vpg_hjorth_mobility": "Upstroke mobility",
    "vpg_hjorth_complexity": "Upstroke complexity",
    "apg_hjorth_mobility": "Acceleration wave mobility",
    "apg_hjorth_complexity": "Acceleration wave complexity",
}

# The surrogate's explanation is trusted for a reading only when its own
# estimate is within this many mmHg of the deep model's (AAMI's 5 mmHg bar).
AGREE_MMHG = 5.0

_GROUP_OF = {}
for _gk, _g in FEATURE_GROUPS.items():
    for _f in _g["features"]:
        _GROUP_OF[_f] = _gk


def classify_bp(sbp: float, dbp: float) -> dict:
    """Standard adult bands, graded on systolic; diastolic can only escalate.
    Describes the reading — explicitly not a diagnosis."""
    if sbp >= 140 or dbp >= 90:
        band, tone = "High", "high"
        why = f"systolic {sbp:.0f} is at or above 140, or diastolic {dbp:.0f} is at or above 90"
    elif sbp >= 130 or dbp >= 80:
        band, tone = "Elevated", "elevated"
        why = f"systolic {sbp:.0f} is at or above 130, or diastolic {dbp:.0f} is at or above 80"
    elif sbp >= 120:
        band, tone = "Borderline", "elevated"
        why = f"systolic {sbp:.0f} falls in the 120–129 range while diastolic stays under 80"
    elif sbp < 90:
        band, tone = "Low", "elevated"
        why = f"systolic {sbp:.0f} is below 90"
    else:
        band, tone = "Normal", "normal"
        why = f"systolic {sbp:.0f} is under 120 and diastolic {dbp:.0f} is under 80"
    return {
        "band": band,
        "tone": tone,
        "why": why,
        "bands": [
            {"name": "Low",        "range": "< 90",     "from": 60,  "to": 90},
            {"name": "Normal",     "range": "90–119",   "from": 90,  "to": 120},
            {"name": "Borderline", "range": "120–129",  "from": 120, "to": 130},
            {"name": "Elevated",   "range": "130–139",  "from": 130, "to": 140},
            {"name": "High",       "range": "≥ 140",    "from": 140, "to": 180},
        ],
    }


@app.get("/api/explain/summary/{patient_id}")
def explain_summary(patient_id: str, target: str = "SBP", window: Optional[int] = None,
                    current_user: dict = Depends(auth.get_current_user)):
    """
    Everything the 'why this reading' view needs, in clinical language:
    the prediction, how it's classified, and which physiological signal
    groups pushed the estimate up or down (real SHAP, grouped into themes).
    """
    authorize_patient(patient_id, current_user, window)   # v3: ownership check
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    i = window if window is not None else info["first_window"]

    # The dataset and model are fixed, so this result is deterministic —
    # computing once and caching turns a ~15s wait into an instant response.
    cache_key = (patient_id, target, int(i))
    cached = STATE.setdefault("summary_cache", {}).get(cache_key)
    if cached is not None:
        return cached

    feat = surrogate_features(i)
    dl_pred = dl_predict_window(i)
    sur = STATE["surrogate"]
    shap_vals, base_value, copy_est = surrogate_shap(feat, target)

    # Roll per-measurement SHAP up into the clinical themes.
    feature_names = STATE["feature_names"]
    group_totals = {k: 0.0 for k in FEATURE_GROUPS}
    group_feats = {k: [] for k in FEATURE_GROUPS}
    for f, v in zip(feature_names, shap_vals):
        gk = _GROUP_OF.get(f)
        if gk is None:
            continue
        group_totals[gk] += float(v)
        group_feats[gk].append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "shapValue": float(v),
        })

    groups = []
    for gk, g in FEATURE_GROUPS.items():
        if not group_feats[gk]:
            continue
        total = group_totals[gk]
        groups.append({
            "key": gk,
            "label": g["label"],
            "what": g["what"],
            "mmHg": round(total, 2),
            "direction": "raised" if total > 0 else ("lowered" if total < 0 else "neutral"),
            "topFeatures": sorted(group_feats[gk], key=lambda r: abs(r["shapValue"]), reverse=True)[:3],
        })
    groups.sort(key=lambda g: abs(g["mmHg"]), reverse=True)

    sbp, dbp = dl_pred["SBP"], dl_pred["DBP"]
    classification = classify_bp(sbp, dbp)
    fid = sur["meta"]["fidelity"][target]
    reading_t = dl_pred[target]

    result = {
        "patientId": patient_id,
        "patientName": info["name"],
        "windowIndex": int(i),
        "target": target,
        "reading": {
            "SBP": round(sbp, 1), "DBP": round(dbp, 1), "MAP": round(dl_pred["MAP"], 1),
        },
        "classification": classification,
        "magnitude": {
            "explainedModel": "Deep model (the one that produced this reading)",
            "method": "SHAP on a surrogate trained to copy the deep model",
            "explainedTarget": target,
            # Average of the deep model's readings over 100 training windows,
            # as copied by the surrogate: where the explanation starts from.
            "averageEstimate": round(base_value, 1),
            "copyEstimate": round(copy_est, 1),
            "reading": round(reading_t, 1),
            "netShift": round(copy_est - base_value, 1),
            "unexplained": round(reading_t - copy_est, 1),
            "groups": groups,
            # Per-reading check: does the readable copy land close to the deep
            # model on THIS window? If not, the factors are only a rough guide.
            "agreement": {
                "closeWithinMmHg": AGREE_MMHG,
                "gapMmHg": round(abs(reading_t - copy_est), 1),
                "close": bool(abs(reading_t - copy_est) <= AGREE_MMHG),
            },
            "fidelity": {
                "r2": fid["r2"], "maeMmHg": fid["mae_mmHg"],
                "within5Pct": fid.get("within_5_mmHg_pct"),
                "maeIfGuessingAverage": fid.get("mae_if_guessing_average"),
                "on": sur["meta"].get("fidelity_on"),
            },
        },
        "note": "The reading comes from the deep model. It reads the raw waveform, so a "
                "second, readable model was trained to copy it from 63 named pulse "
                "measurements; SHAP on that copy shows which measurements pushed this "
                "reading up or down. Descriptive only — not a diagnosis.",
    }
    STATE["summary_cache"][cache_key] = result
    return result


@app.get("/api/explain/shap/{patient_id}")
def explain_shap(patient_id: str, target: str = "SBP", window: Optional[int] = None,
                 current_user: dict = Depends(auth.get_current_user)):
    authorize_patient(patient_id, current_user, window)   # v3: ownership check
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    i = window if window is not None else info["first_window"]

    feat = surrogate_features(i)
    vals, base, est = surrogate_shap(feat, target)
    contributions = sorted(
        [{"feature": f, "label": FEATURE_LABELS.get(f, f), "shapValue": float(v)}
         for f, v in zip(STATE["feature_names"], vals)],
        key=lambda r: abs(r["shapValue"]),
        reverse=True,
    )
    return {
        "patientId": patient_id,
        "target": target,
        "windowIndex": i,
        "explainedModel": "Deep model (via surrogate)",
        "baseValue": base,
        "prediction": est,
        "deepModelReading": dl_predict_window(i)[target],
        "contributions": contributions,
    }



def beat_attention(ppg: np.ndarray, attr: np.ndarray, fs: int, n_pts: int = 120):
    """
    v3: fold the IG attribution onto one "average heartbeat".

    The BiLSTM gives the last seconds of every window more attribution than the
    first (a recency bias of the architecture, not of the patient). To remove it,
    each beat's attention is rescaled to sum to 1 before averaging, so every beat
    counts equally wherever it sits in the window.

    Each beat (foot to next foot) is split into three phases and each phase is
    resampled to a fixed length, so the phases line up across beats:
      rise   foot -> systolic peak
      peak   peak -> 40% of the way to the next foot (peak and dicrotic notch)
      fall   the rest (diastolic decay)
    ppg:  [T] raw PPG channel;  attr: [3, T] IG attribution (signed).
    """
    from beats import segment_beats
    beats = segment_beats(ppg, fs)
    if beats is None:
        return None
    med = float(np.median([b[3] - b[0] for b in beats]))
    fu = np.median([(b[1] - b[0]) / (b[3] - b[0]) for b in beats])
    fn = np.median([(b[2] - b[1]) / (b[3] - b[0]) for b in beats])
    nu = max(4, int(round(n_pts * fu))); nn = max(4, int(round(n_pts * fn))); nd = n_pts - nu - nn

    def piece(y, a, n):
        # resample the signal; resample attention so the piece keeps its total
        xs = np.linspace(0, len(y) - 1, n)
        yy = np.interp(xs, np.arange(len(y)), y)
        aa = np.interp(xs, np.arange(len(a)), a)
        aa = aa * (a.sum() / (aa.sum() or 1.0))
        return yy, aa

    absA = np.abs(attr)
    tot = absA.sum(0)
    shapes, atts, chs = [], [], []
    for f0, p0, c0, f1 in beats:
        ys, as_ = [], []
        for (a0, a1, n) in [(f0, p0, nu), (p0, c0, nn), (c0, f1, nd)]:
            yy, aa = piece(ppg[a0:a1 + 1], tot[a0:a1 + 1], n)
            ys.append(yy); as_.append(aa)
        y = np.concatenate(ys); a = np.concatenate(as_)
        y = (y - y.min()) / ((y.max() - y.min()) or 1.0)
        a = a / (a.sum() or 1.0)                      # every beat counts equally
        cs = absA[:, f0:f1].sum(1); cs = cs / (cs.sum() or 1.0)
        shapes.append(y); atts.append(a); chs.append(cs)
    att = np.mean(atts, 0)
    bounds = [0, nu, nu + nn, n_pts]
    phases = {}
    for k, (a0, a1) in zip(["upstroke", "notch", "decay"], zip(bounds[:-1], bounds[1:])):
        a_share = float(att[a0:a1].sum()); t_share = (a1 - a0) / n_pts
        phases[k] = {"attentionShare": round(a_share, 3), "timeShare": round(t_share, 3),
                     "density": round(a_share / t_share, 3)}
    ch = np.mean(chs, 0)
    return {
        "nBeats": len(beats),
        "points": n_pts,
        "phaseBounds": bounds,
        "shape": np.round(np.mean(shapes, 0), 4).tolist(),
        "beats": [np.round(y, 3).tolist() for y in shapes[:20]],
        "attention": np.round(att, 5).tolist(),
        "phases": phases,
        "channelShare": {"ppg": round(float(ch[0]), 3), "vpg": round(float(ch[1]), 3), "apg": round(float(ch[2]), 3)},
        "beatSeconds": round(float(med) / fs, 2),
    }


@app.get("/api/explain/saliency/{patient_id}")
def explain_saliency(patient_id: str, target: str = "SBP", window: Optional[int] = None,
                     current_user: dict = Depends(auth.get_current_user)):
    authorize_patient(patient_id, current_user, window)   # v3: ownership check
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    i = window if window is not None else info["first_window"]

    from captum.attr import IntegratedGradients

    target_idx = TARGETS.index(target)
    model = STATE["dl_model"]
    x = torch.tensor(STATE["X"][i : i + 1], dtype=torch.float32, requires_grad=True)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(x, target=target_idx, n_steps=32)
    attr = attributions.detach().numpy()[0]  # [3, 1500]

    with torch.no_grad():
        pred = model(x).numpy()[0]

    # Downsample to keep the payload light for the frontend chart (1500 -> 300 pts)
    step = 5
    return {
        "patientId": patient_id,
        "target": target,
        "windowIndex": i,
        "prediction": {t: float(pred[j]) for j, t in enumerate(TARGETS)},
        "samplingRateHz": Config.SAMPLING_RATE,
        "channels": {
            "ppg": STATE["X"][i, 0, ::step].tolist(),
            "vpg": STATE["X"][i, 1, ::step].tolist(),
            "apg": STATE["X"][i, 2, ::step].tolist(),
        },
        "attribution": {
            "ppg": attr[0, ::step].tolist(),
            "vpg": attr[1, ::step].tolist(),
            "apg": attr[2, ::step].tolist(),
        },
        # v3: attention folded onto one average heartbeat, each beat weighted
        # equally (removes the model's end-of-window bias). See beat_attention().
        "beat": beat_attention(STATE["X"][i, 0], attr, Config.SAMPLING_RATE),
        "note": "Integrated Gradients attribution (captum), 32 steps, computed "
                "live against the real PPGResNetBiLSTM checkpoint.",
    }


# ---------------------------------------------------------------------------
# v2 (validation audit): verified results only.
# Removed from the comparison because they were found to be invalid:
#   * "Multimodal ECG+PPG+PTT, 5-min recalib" (3.23 mmHg SBP, "Grade A"): the ECG was synthetic and the
#     PTT input was computed from the TRUE SBP label (label leakage); its recalibration also scored
#     windows whose own label was used for the offset.
#   * "15-min periodic recalibration" and "calibrated 1-point anchor" rows: the calibration offsets used
#     the true BP of windows that were then scored (the anchor windows), which inflates accuracy.
# Everything below is either computed live on the held-out test set or copied from a reproducible
# evaluation script, and is labelled with its source.
# ---------------------------------------------------------------------------
# Mean BP of the MIMIC-IV training set (processed_train_mimic4.npz, 1,089 windows): used as the
# "average-guess" baseline that ignores the signal. Every model must beat this to add value.
TRAIN_MEAN_BP = {"SBP": 126.75, "DBP": 58.25, "MAP": 83.59}

REMOVED_RESULTS = [
    {"label": "Multimodal ECG+PPG+PTT with 5-min recalibration (SBP 3.23 mmHg, BHS Grade A)",
     "reason": "Invalid: synthetic ECG and a PTT input computed from the true SBP label (label leakage); "
               "recalibration scored windows whose own label was used."},
    {"label": "PPGResNetBiLSTM with 15-min periodic recalibration (SBP 13.06 mmHg)",
     "reason": "Invalid: calibration anchor windows were scored with their own true BP."},
    {"label": "PPGResNetBiLSTM with 1-point calibration anchor (SBP 20.07 mmHg)",
     "reason": "Superseded by the leak-free calibration protocol (anchor window not scored)."},
]

EXTERNAL_VALIDATION = {
    "dataset": "VitalDB (Seoul National University Hospital, surgical patients)",
    "patients": 40, "windows": 11903,
    "protocol": "Frozen models trained only on MIMIC-IV; identical preprocessing; no calibration.",
    "source": "external_validation_vitaldb/external_results/RESULTS_EXTERNAL.md",
    "MAE": {
        "PPGResNetBiLSTM (served)":       {"SBP": 17.05, "DBP": 13.44, "MAP": 14.11},
        "Classical ML (AdaBoost/SVR, served)": {"SBP": 23.02, "DBP": 13.23, "MAP": 14.51},
        "Average-guess baseline (MIMIC training mean)": {"SBP": 20.72, "DBP": 11.05, "MAP": 13.65},
    },
    "finding": "SBP: deep model 3.7 mmHg better than the average-guess (95% CI 1.5-5.9), mostly because it "
               "predicts the right average level; DBP: worse than the average-guess; with any calibration the "
               "average-guess is as good or better.",
}

POST_TRAINING_EXTENSIONS = [
    {"name": "Real ECG + pulse arrival time (PAT)",
     "setup": "Dataset rebuilt from raw MIMIC-IV waveforms with real ECG lead II; 44 training / 10 test patients; "
              "PAT = ECG R-peak to PPG steepest upslope; 3 seeds per deep model.",
     "sbp_mae_no_calibration": {"MIMIC test (10 pts)": {"Average-guess": 15.50, "PPG-only DL": 16.45,
                                                        "PPG+ECG+PAT DL": 18.00, "PAT+HR regression": 14.43},
                                "VitalDB (40 pts)": {"Average-guess": 18.34, "PPG-only DL": 16.59,
                                                     "PPG+ECG+PAT DL": 16.33, "PAT+HR regression": 18.44}},
     "finding": "Adding ECG and PAT did not beat PPG alone; Integrated Gradients showed the multimodal network "
                "gave PAT/HR under 1% of the attribution (PPG ~70%, ECG ~27%). Not deployed.",
     "source": "option_b_real_ecg/results/RESULTS_OPTION_B.md"},
    {"name": "Calibration-aware tracking (predict the change since the last cuff reading)",
     "setup": "Cuff reading at start / every 15 min / every 5 min; baseline = hold the last cuff value; "
              "cuff windows never scored.",
     "sbp_mae_cuff_every_15_min": {"MIMIC test": {"Hold last cuff": 8.51, "Change model (PPG features)": 9.10},
                                   "VitalDB": {"Hold last cuff": 15.79, "Change model (PPG features)": 14.52}},
     "finding": "Small, consistent gain in surgical patients (VitalDB, +1.35 mmHg, 78% of patients improved) "
                "but none in ICU patients (MIMIC). Not consistent enough to deploy.",
     "source": "option_b_real_ecg/results_calibration_aware/RESULTS_CALIBRATION_AWARE.md"},
]


@app.get("/api/model-comparison")
def model_comparison():
    live = STATE["live_comparison"]
    rows = []

    for t in TARGETS:
        rows.append({
            "model": "Classical ML (best per-target: AdaBoost/SVR)",
            "target": t,
            "source": "live",
            "MAE": round(live[t]["classical"]["metrics"]["MAE"], 2),
            "RMSE": round(live[t]["classical"]["metrics"]["RMSE"], 2),
            "aami": "PASS" if live[t]["classical"]["aami"]["aami_compliant"] else "FAIL",
            "bhsGrade": live[t]["classical"]["bhs"]["bhs_grade"].replace("Grade ", ""),
        })
    for t in TARGETS:
        rows.append({
            "model": "PPGResNetBiLSTM (single-modality DL)",
            "target": t,
            "source": "live",
            "MAE": round(live[t]["dl"]["metrics"]["MAE"], 2),
            "RMSE": round(live[t]["dl"]["metrics"]["RMSE"], 2),
            "aami": "PASS" if live[t]["dl"]["aami"]["aami_compliant"] else "FAIL",
            "bhsGrade": live[t]["dl"]["bhs"]["bhs_grade"].replace("Grade ", ""),
        })
    for t in TARGETS:
        b = live[t]["baseline"]
        rows.append({
            "model": "Average-guess baseline (training-set mean, ignores the signal)",
            "target": t,
            "source": "live",
            "MAE": round(b["metrics"]["MAE"], 2),
            "RMSE": round(b["metrics"]["RMSE"], 2),
            "aami": "PASS" if b["aami"]["aami_compliant"] else "FAIL",
            "bhsGrade": b["bhs"]["bhs_grade"].replace("Grade ", ""),
        })

    return {
        "testSet": {
            "name": "processed_test_unseen.npz",
            "windows": int(STATE["X"].shape[0]),
            "subjects": len(set(STATE["subject_ids"].tolist())),
        },
        "rows": rows,
    }


@app.get("/api/validation-summary")
def validation_summary():
    """Every verified result behind PulseIQ in one place, each labelled with its source."""
    live = STATE["live_comparison"]
    internal = []
    for key, label in [("dl", "PPGResNetBiLSTM (served)"), ("classical", "Classical ML (AdaBoost/SVR, served)"),
                       ("baseline", "Average-guess baseline (training mean)")]:
        internal.append({"model": label, **{t: {"MAE": round(live[t][key]["metrics"]["MAE"], 2),
                                                "RMSE": round(live[t][key]["metrics"]["RMSE"], 2),
                                                "aami": "PASS" if live[t][key]["aami"]["aami_compliant"] else "FAIL",
                                                "bhsGrade": live[t][key]["bhs"]["bhs_grade"].replace("Grade ", "")}
                                            for t in TARGETS}})
    return {
        "internal": {
            "dataset": "MIMIC-IV held-out test set (processed_test_unseen.npz)",
            "windows": int(STATE["X"].shape[0]),
            "patients": len(set(STATE["subject_ids"].tolist())),
            "protocol": "Patient-level split (no patient in both train and test); no calibration; computed live.",
            "rows": internal,
        },
        "external": EXTERNAL_VALIDATION,
        "extensions": POST_TRAINING_EXTENSIONS,
        "removed": REMOVED_RESULTS,
        "note": "PulseIQ is an explainability demonstrator, not a medical device. No model here meets AAMI "
                "or BHS clinical standards without calibration.",
    }


# ---------------------------------------------------------------------------
# Health chatbot (local LLM via Ollama -- free, no API key, runs on your machine)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    patientId: Optional[str] = None  # optional: include recent BP as context


@app.get("/api/chat/health")
def chat_health():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", [])]
        return {
            "ollamaReachable": True,
            "configuredModel": OLLAMA_MODEL,
            "modelPulled": any(OLLAMA_MODEL in m for m in models),
            "installedModels": models,
        }
    except Exception as e:
        return {
            "ollamaReachable": False,
            "configuredModel": OLLAMA_MODEL,
            "error": str(e),
            "hint": "Install Ollama (https://ollama.com/download), run "
                    f"'ollama pull {OLLAMA_MODEL}', then make sure Ollama is "
                    "running (it usually starts automatically after install).",
        }


@app.post("/api/chat")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    system_prompt = CHAT_SYSTEM_PROMPT
    # v3: BP context only for a signed-in caller allowed to see that recording.
    allowed = False
    if req.patientId and authorization:
        try:
            authorize_patient(req.patientId, auth.get_current_user(authorization))
            allowed = True
        except HTTPException:
            allowed = False
    if allowed and req.patientId in STATE.get("patient_index", {}):
        try:
            info = STATE["patient_index"][req.patientId]
            i = info["first_window"]
            dl_pred = dl_predict_window(i)
            system_prompt += (
                f"\n\nContext: the user's most recent PulseIQ reading (from the "
                f"PPGResNetBiLSTM model) was approximately "
                f"{round(dl_pred['SBP'])}/{round(dl_pred['DBP'])} mmHg "
                f"(MAP {round(dl_pred['MAP'])}). Only mention this if it's "
                f"relevant to what they're asking."
            )
        except Exception:
            pass  # context is best-effort; never block the chat over it

    messages = [{"role": "system", "content": system_prompt}]
    for m in req.history[-10:]:  # keep recent context bounded
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content", "").strip()
        if not reply:
            raise ValueError("empty response from local model")
        return {"reply": reply, "model": OLLAMA_MODEL, "source": "local-llm"}
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Local AI (Ollama) is not reachable at {OLLAMA_BASE_URL}. "
                   f"Install it from https://ollama.com/download, run "
                   f"'ollama pull {OLLAMA_MODEL}', and make sure it's running.",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Local AI took too long to respond. Try a smaller model "
                   "(e.g. 'ollama pull llama3.2:1b') if your machine is slow.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local AI error: {e}")


# ---------------------------------------------------------------------------
# Deployment: private-site gate + serving the frontend from the same server
# ---------------------------------------------------------------------------
# The app ships MIMIC-IV derived waveforms, which PhysioNet's data-use agreement
# does not allow to be shared publicly. When SITE_PASSWORD is set, every page and
# API call needs a one-time site password first (on top of the app's own login),
# so only people you give the password to can reach it.
import hmac
import hashlib
from urllib.parse import quote
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

SITE_PASSWORD = os.environ.get("SITE_PASSWORD", "")
STATIC_DIR = os.path.join(BASE_DIR, "static")
_GATE_OPEN = {"/gate", "/api/health"}


def _gate_token() -> str:
    return hmac.new(auth.SECRET_KEY.encode(), SITE_PASSWORD.encode(), hashlib.sha256).hexdigest()


@app.middleware("http")
async def site_gate(request: Request, call_next):
    path = request.url.path
    if SITE_PASSWORD and path not in _GATE_OPEN:
        if not hmac.compare_digest(request.cookies.get("pulseiq_gate", ""), _gate_token()):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Site password required"}, status_code=401)
            return RedirectResponse("/gate?next=" + quote(path))
    return await call_next(request)


_GATE_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PulseIQ · Private access</title><style>
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f4f7fb;font-family:Inter,system-ui,sans-serif;color:#0d1622}
.c{background:#fff;border:1px solid #d9dfe7;border-radius:16px;padding:32px 30px;width:min(360px,90vw);box-shadow:0 10px 30px rgba(13,22,34,.08)}
h1{margin:0 0 4px;font-size:22px}h1 span{color:#e0245e}p{margin:0 0 18px;color:#5a6678;font-size:13.5px;line-height:1.5}
input{width:100%;box-sizing:border-box;padding:11px 12px;border:1px solid #c9d1dc;border-radius:10px;font-size:15px;margin-bottom:12px}
button{width:100%;padding:11px;border:0;border-radius:10px;background:#e0245e;color:#fff;font-weight:700;font-size:15px;cursor:pointer}
.e{color:#c62b45;font-size:13px;margin:-4px 0 10px}</style></head><body><form class="c" method="post" action="/gate">
<h1>Pulse<span>IQ</span></h1><p>This research demo is private. Enter the access password you were given.</p>
__ERR__<input type="password" name="password" placeholder="Access password" autofocus required>
<input type="hidden" name="next" value="__NEXT__"><button type="submit">Continue</button></form></body></html>"""


def _gate_page(next_path: str, error: bool = False) -> HTMLResponse:
    safe_next = next_path if next_path.startswith("/") and not next_path.startswith("//") else "/"
    html = _GATE_HTML.replace("__NEXT__", safe_next.replace('"', "")).replace(
        "__ERR__", '<div class="e">Wrong password, try again.</div>' if error else "")
    return HTMLResponse(html, status_code=401 if error else 200)


@app.get("/gate")
def gate_form(next: str = "/"):
    return _gate_page(next)


@app.post("/gate")
async def gate_submit(password: str = Form(...), next: str = Form("/")):
    if not SITE_PASSWORD or not hmac.compare_digest(password, SITE_PASSWORD):
        time.sleep(1.0)   # slow down guessing
        return _gate_page(next, error=True)
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(target, status_code=303)
    resp.set_cookie("pulseiq_gate", _gate_token(), max_age=30 * 24 * 3600, httponly=True,
                    secure=os.environ.get("COOKIE_SECURE", "1") == "1", samesite="lax")
    return resp


# Frontend (must be mounted last so /api/* routes win)
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8001)), reload=False)
