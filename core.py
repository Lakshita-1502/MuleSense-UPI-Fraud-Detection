"""MuleSense core: synthetic data, model, explanation and tamper-evident audit log.

Everything the demo needs lives in this one file so it is easy to read and explain.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42
MODEL_VERSION = "mulesense-xgb-v1"
BANKS = [f"Bank{i}" for i in range(1, 9)]
ACC_TYPES = ["savings", "current"]

NUMERIC = ["inflow_count_24h", "distinct_senders_24h", "distinct_sender_states",
           "new_sender_share", "avg_inflow_amount", "passthrough_ratio",
           "median_dwell_minutes", "night_share", "device_changes_7d",
           "activity_surge_ratio", "prior_complaints", "account_age_days"]
FEATURES = NUMERIC + [f"bank_{b}" for b in BANKS] + [f"type_{t}" for t in ACC_TYPES]  # 22

# State-wise complaint counts from the Gujarat case report (total 132)
CASE_STATES = {"Gujarat": 22, "Karnataka": 14, "Maharashtra": 13, "Kerala": 10,
               "Uttar Pradesh": 10, "Rajasthan": 9, "Tamil Nadu": 8, "West Bengal": 7,
               "Haryana": 7, "Andhra Pradesh": 6, "Telangana": 6, "Delhi": 5,
               "Punjab": 3, "Bihar": 3, "Odisha": 2, "Chhattisgarh": 2, "Assam": 2,
               "Himachal Pradesh": 1, "Jammu and Kashmir": 1, "Puducherry": 1}


# ---------------------------------------------------------------- data ----
def _account(rng, mule, stealthy=False, shop=False, busy_week=False):
    if mule and not stealthy:
        n = rng.poisson(11); p_dist = 0.95
        states = 1 + rng.poisson(3.5); new = rng.beta(6, 2)
        amt = rng.lognormal(8.4, 1.0); pt = rng.beta(6, 2.5)
        dwell = rng.lognormal(3.8, 1.0); night = rng.beta(3, 6)
        dev = rng.poisson(0.7); surge = rng.lognormal(1.2, 0.9)
        comp = rng.poisson(0.5); age = rng.gamma(2.0, 260)
    elif mule:  # stealthy: a real person's account that looks almost normal
        n = rng.poisson(3); p_dist = 0.85
        states = 1 + rng.poisson(0.8); new = rng.beta(3, 4.5)
        amt = rng.lognormal(6.9, 1.0); pt = rng.beta(2.6, 4.2)
        dwell = rng.lognormal(5.3, 1.0); night = rng.beta(2, 8.5)
        dev = rng.poisson(0.3); surge = rng.lognormal(0.4, 0.55)
        comp = rng.poisson(0.2); age = rng.gamma(2.0, 260)
    else:
        n = rng.poisson(25 if shop else 2); p_dist = 0.75
        states = 1 + rng.poisson(0.6 if shop else 0.25); new = rng.beta(2, 6)
        amt = rng.lognormal(6.0, 1.0); pt = rng.beta(1.2, 10)
        dwell = rng.lognormal(6.3, 1.2); night = rng.beta(1.5, 12)
        dev = rng.poisson(0.1)
        surge = rng.lognormal(1.0, 0.5) if busy_week else rng.lognormal(0, 0.4)
        comp = rng.poisson(0.01); age = rng.gamma(2.5, 300)
    dist = int(rng.binomial(n, p_dist)) if n > 0 else 0
    states = int(min(states, max(dist, 1)))
    row = dict(inflow_count_24h=n, distinct_senders_24h=dist, distinct_sender_states=states,
               new_sender_share=new, avg_inflow_amount=amt, passthrough_ratio=pt,
               median_dwell_minutes=dwell, night_share=night, device_changes_7d=dev,
               activity_surge_ratio=surge, prior_complaints=comp, account_age_days=age)
    b, t = rng.integers(len(BANKS)), rng.integers(len(ACC_TYPES))  # pure noise columns
    for i, name in enumerate(BANKS):
        row[f"bank_{name}"] = int(i == b)
    for i, name in enumerate(ACC_TYPES):
        row[f"type_{name}"] = int(i == t)
    return row


def generate_accounts(n=30000, mule_rate=0.015, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        mule = rng.random() < mule_rate
        stealthy = mule and rng.random() < 0.40
        shop = (not mule) and rng.random() < 0.08
        busy = (not mule) and (not shop) and rng.random() < 0.06
        r = _account(rng, mule, stealthy, shop, busy)
        r["is_mule"] = int(mule)
        rows.append(r)
    return pd.DataFrame(rows)


def case_replay_accounts(n=60, seed=7):
    """n mule-style accounts (40% stealthy) whose payer states follow Fig. 1 of the case."""
    rng = np.random.default_rng(seed)
    names = list(CASE_STATES)
    probs = np.array(list(CASE_STATES.values()), float)
    probs /= probs.sum()
    rows = []
    for _ in range(n):
        r = _account(rng, True, rng.random() < 0.40)
        payers = max(r["distinct_senders_24h"], 1)
        r["distinct_sender_states"] = len(set(rng.choice(names, size=payers, p=probs)))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- model ----
def build_model():
    """Train XGBoost, pick the threshold on validation data, evaluate once on test data."""
    df = generate_accounts()
    X, y = df[FEATURES], df["is_mule"]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=SEED)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
                              eval_metric="aucpr", random_state=SEED, n_jobs=2)
    model.fit(X_tr, y_tr)
    p, r, t = precision_recall_curve(y_va, model.predict_proba(X_va)[:, 1])
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    thr = float(t[int(np.argmax(f1))])
    s_te = model.predict_proba(X_te)[:, 1]
    pred = (s_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    pc, rc, _ = precision_recall_curve(y_te, s_te)
    metrics = dict(threshold=thr, precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                   pr_auc=average_precision_score(y_te, s_te), roc_auc=roc_auc_score(y_te, s_te),
                   fpr=fp / (fp + tn), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                   accuracy=(tp + tn) / len(y_te), n_test=len(y_te), n_mule_test=int(y_te.sum()),
                   all_genuine_accuracy=1 - y_te.mean())
    return dict(model=model, threshold=thr, metrics=metrics, pr_curve=(pc, rc),
                importance=pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False))


def score(bundle, df):
    return bundle["model"].predict_proba(df[FEATURES])[:, 1]


def explain(bundle, df):
    """Per-account feature contributions (log-odds) from XGBoost's built-in TreeSHAP."""
    booster = bundle["model"].get_booster()
    contrib = booster.predict(xgb.DMatrix(df[FEATURES]), pred_contribs=True)[:, :-1]
    return pd.DataFrame(contrib, columns=FEATURES, index=df.index)


def action_for(s, thr):
    """Graded response instead of a straight block (see report, Section 5.10)."""
    if s >= max(0.90, thr):
        return "HOLD outgoing transfers + analyst review"
    if s >= thr:
        return "REVIEW by analyst / temporary limit"
    if s >= 0.5 * thr:
        return "WATCH (monitor closely)"
    return "NO ACTION"


# ----------------------------------------------- tamper-evident audit log ----
GENESIS = "0" * 64


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def append_log(path, account_id, features, s, thr, action):
    prev, idx = GENESIS, 0
    if os.path.exists(path):
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            prev, idx = lines[-1]["hash"], lines[-1]["index"] + 1
    body = dict(index=idx, timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                account_id=account_id, features={k: round(float(features[k]), 4) for k in NUMERIC},
                model_version=MODEL_VERSION, score=round(float(s), 4), threshold=round(float(thr), 4),
                action=action, prev_hash=prev)
    body["hash"] = _digest(body)
    with open(path, "a") as f:
        f.write(json.dumps(body) + "\n")
    return body


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def verify_log(records):
    """Recompute the hash chain. Returns (ok, index_of_first_bad_record or None)."""
    prev = GENESIS
    for rec in records:
        body = {k: v for k, v in rec.items() if k != "hash"}
        if rec["prev_hash"] != prev or _digest(body) != rec["hash"]:
            return False, rec["index"]
        prev = rec["hash"]
    return True, None
