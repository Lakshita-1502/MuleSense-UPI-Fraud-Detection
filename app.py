"""MuleSense - demo dashboard.  Run:  streamlit run app.py"""
import copy
import os

import pandas as pd
import streamlit as st

import core

LOG = "audit_log.jsonl"
st.set_page_config(page_title="MuleSense demo", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Training model (about 2 seconds)...")
def get_bundle():
    return core.build_model()


B = get_bundle()
THR = B["threshold"]

PRESETS = {
    "Typical genuine account": dict(inflow_count_24h=2, distinct_senders_24h=2, distinct_sender_states=1,
        new_sender_share=0.2, avg_inflow_amount=400, passthrough_ratio=0.1, median_dwell_minutes=550,
        night_share=0.1, device_changes_7d=0, activity_surge_ratio=1.0, prior_complaints=0, account_age_days=700),
    "Busy shop (genuine, looks busy)": dict(inflow_count_24h=25, distinct_senders_24h=19, distinct_sender_states=2,
        new_sender_share=0.3, avg_inflow_amount=350, passthrough_ratio=0.15, median_dwell_minutes=400,
        night_share=0.1, device_changes_7d=0, activity_surge_ratio=1.2, prior_complaints=0, account_age_days=900),
    "Obvious mule account": dict(inflow_count_24h=12, distinct_senders_24h=11, distinct_sender_states=6,
        new_sender_share=0.8, avg_inflow_amount=4500, passthrough_ratio=0.8, median_dwell_minutes=40,
        night_share=0.35, device_changes_7d=1, activity_surge_ratio=3.0, prior_complaints=1, account_age_days=300),
    "Stealthy mule (rented real account)": dict(inflow_count_24h=3, distinct_senders_24h=3, distinct_sender_states=2,
        new_sender_share=0.4, avg_inflow_amount=900, passthrough_ratio=0.35, median_dwell_minutes=250,
        night_share=0.12, device_changes_7d=0, activity_surge_ratio=1.3, prior_complaints=0, account_age_days=500),
}

st.title("🛡️ MuleSense: mule-account early-warning demo")
st.caption("Case: Gujarat interstate mule-account network (May 2026). All data here is synthetic.")
page = st.sidebar.radio("Page", ["1. Score an account", "2. Case replay (60 accounts)",
                                 "3. Model report", "4. Forensic audit log"])
st.sidebar.markdown(f"**Model:** `{core.MODEL_VERSION}`  \n**Alert threshold:** `{THR:.3f}`")

# ------------------------------------------------------------------ page 1
if page.startswith("1"):
    choice = st.selectbox("Start from a preset", list(PRESETS))
    p = PRESETS[choice]
    c1, c2, c3 = st.columns(3)
    f = {}
    with c1:
        f["inflow_count_24h"] = st.number_input("Payments received in 24 h", 0, 200, int(p["inflow_count_24h"]), key=choice + "a")
        f["distinct_senders_24h"] = st.number_input("Distinct payers in 24 h", 0, 200, int(p["distinct_senders_24h"]), key=choice + "b")
        f["distinct_sender_states"] = st.number_input("States the payers are in", 1, 20, int(p["distinct_sender_states"]), key=choice + "c")
        f["new_sender_share"] = st.slider("Share of payers new to this account", 0.0, 1.0, float(p["new_sender_share"]), key=choice + "d")
    with c2:
        f["avg_inflow_amount"] = st.number_input("Average amount received (INR)", 10, 100000, int(p["avg_inflow_amount"]), key=choice + "e")
        f["passthrough_ratio"] = st.slider("Share of inflow sent out within 1 hour", 0.0, 1.0, float(p["passthrough_ratio"]), key=choice + "f")
        f["median_dwell_minutes"] = st.number_input("Median minutes money stays", 1, 5000, int(p["median_dwell_minutes"]), key=choice + "g")
        f["night_share"] = st.slider("Share of activity 11 pm to 5 am", 0.0, 1.0, float(p["night_share"]), key=choice + "h")
    with c3:
        f["device_changes_7d"] = st.number_input("Device changes in 7 days", 0, 20, int(p["device_changes_7d"]), key=choice + "i")
        f["activity_surge_ratio"] = st.slider("This week vs usual weekly volume (x)", 0.0, 6.0, float(p["activity_surge_ratio"]), key=choice + "j")
        f["prior_complaints"] = st.number_input("Earlier complaints", 0, 20, int(p["prior_complaints"]), key=choice + "k")
        f["account_age_days"] = st.number_input("Account age (days)", 1, 5000, int(p["account_age_days"]), key=choice + "l")

    row = {k: float(v) for k, v in f.items()}
    row.update({c: 0 for c in core.FEATURES if c not in row})
    X = pd.DataFrame([row])
    s = float(core.score(B, X)[0])
    act = core.action_for(s, THR)
    m1, m2 = st.columns(2)
    m1.metric("Risk score (0 to 1)", f"{s:.3f}", delta=f"threshold {THR:.3f}", delta_color="off")
    m2.metric("Suggested action", act)
    st.progress(min(s, 1.0))

    contrib = core.explain(B, X).iloc[0][core.NUMERIC].sort_values()
    st.subheader("Why this score? (contribution of each feature, in log-odds)")
    st.bar_chart(contrib)
    top = contrib.sort_values(ascending=False).head(3)
    st.write("Top reasons pushing the score up: " + ", ".join(f"**{k}**" for k, v in top.items() if v > 0) or "none")
    if st.button("Log this decision to the audit trail"):
        rec = core.append_log(LOG, f"DEMO-{pd.Timestamp.now().strftime('%H%M%S')}", row, s, THR, act)
        st.success(f"Logged record #{rec['index']}  |  hash {rec['hash'][:16]}...")

# ------------------------------------------------------------------ page 2
elif page.startswith("2"):
    st.write("60 mule-style accounts (one per account the report says was used), with payer states drawn "
             "from the state-wise complaint counts of the case. Not used for training.")
    seed = st.number_input("Random seed", 1, 9999, 7)
    rep = core.case_replay_accounts(60, int(seed))
    rep["score"] = core.score(B, rep)
    rep["flagged"] = rep["score"] >= THR
    rep["action"] = [core.action_for(x, THR) for x in rep["score"]]
    a, b, c = st.columns(3)
    a.metric("Accounts flagged", f"{int(rep.flagged.sum())} of 60")
    b.metric("Replay recall", f"{rep.flagged.mean():.0%}")
    c.metric("Mean sender states", f"{rep.distinct_sender_states.mean():.1f}")
    st.dataframe(rep[["inflow_count_24h", "distinct_sender_states", "new_sender_share", "passthrough_ratio",
                      "score", "flagged", "action"]].sort_values("score", ascending=False))
    st.caption("Missed accounts are mostly the 'stealthy' ones, which look like ordinary customers.")
    if st.button("Log all flagged accounts to the audit trail"):
        n = 0
        for i, r in rep[rep.flagged].iterrows():
            core.append_log(LOG, f"REPLAY-{i:03d}", r, r["score"], THR, r["action"]); n += 1
        st.success(f"{n} records written.")

# ------------------------------------------------------------------ page 3
elif page.startswith("3"):
    m = B["metrics"]
    st.write(f"Held-out test set: {m['n_test']:,} accounts, {m['n_mule_test']} of them mules (about 1.5%).")
    t = pd.DataFrame({"Metric": ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC", "False positive rate"],
                      "Value": [f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['pr_auc']:.3f}",
                                f"{m['roc_auc']:.3f}", f"{m['fpr']:.2%}"]})
    c1, c2 = st.columns(2)
    c1.table(t)
    c2.table(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["Actual genuine", "Actual mule"],
                          columns=["Pred genuine", "Pred mule"]))
    st.info(f"Accuracy trap: calling every account genuine scores {m['all_genuine_accuracy']:.1%} accuracy "
            f"and catches 0 mules. This is why precision, recall and PR-AUC are reported.")
    pc, rc = B["pr_curve"]
    st.subheader("Precision-recall curve")
    st.line_chart(pd.DataFrame({"recall": rc, "precision": pc}).set_index("recall"))
    st.subheader("Feature importance (gain)")
    st.bar_chart(B["importance"].head(10))

# ------------------------------------------------------------------ page 4
else:
    recs = core.read_log(LOG)
    st.write("Every logged decision stores the account, time, features, model version, score, threshold and action, "
             "and is chained to the previous record with SHA-256. Changing any old record breaks the chain.")
    if not recs:
        st.warning("The log is empty. Log a decision on page 1 or 2 first.")
    else:
        st.dataframe(pd.DataFrame([{k: r[k] for k in ("index", "timestamp", "account_id", "score", "action")} |
                                   {"hash": r["hash"][:16] + "...", "prev_hash": r["prev_hash"][:16] + "..."} for r in recs]),
                     use_container_width=True)
        ok, bad = core.verify_log(recs)
        st.success(f"Chain verified: all {len(recs)} records intact.") if ok else st.error(f"Tampering detected at record #{bad}.")
        st.subheader("Tamper test")
        st.write("Edits one record in a copy held in memory (your file is not changed) and verifies again.")
        if st.button("Simulate an attacker changing a score"):
            fake = copy.deepcopy(recs)
            fake[len(fake) // 2]["score"] = 0.0
            ok2, bad2 = core.verify_log(fake)
            st.error(f"Verification FAILED at record #{bad2}: the edited record no longer matches its hash.") if not ok2 else st.write("No change detected.")
        if st.button("Clear log"):
            os.remove(LOG); st.rerun()
