# retrain_with_feedback.py
import os, sys, argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from pymongo import MongoClient, UpdateOne
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# ---- Config ----
MONGODB_URI   = os.getenv("MONGODB_URI")
DB_NAME       = os.getenv("DB_NAME", "leadgenerate")
FEEDBACK_COLL = os.getenv("FEEDBACK_COLL", "leads_feedback")
TRAIN_COLL    = os.getenv("TRAIN_COLL", "leads_training")
PRED_COLL     = os.getenv("PRED_COLL", "ranked_results")
META_COLL     = os.getenv("META_COLL", "model_meta")
MODEL_NAME    = os.getenv("MODEL_NAME", "lead_ranker_v1")

MODEL_PATH    = os.getenv("MODEL_PATH", "lead_ranker_lgb.txt")
ENCODER_PATH  = os.getenv("ENCODER_PATH", "ohe_encoder.joblib")
NEW_DATA_THRESHOLD = float(os.getenv("NEW_DATA_THRESHOLD", "0.05"))  # 5%

DROP_ALWAYS   = {"_id", "lead_id", "group_id", "state", "label", "source"}

def connect():
    cli = MongoClient(MONGODB_URI, uuidRepresentation="standard", tls=True)
    cli.admin.command("ping")
    return cli

def get_meta(db):
    return db[META_COLL].find_one({"model_name": MODEL_NAME}) or {}

def set_meta(db, **fields):
    fields["model_name"] = MODEL_NAME
    fields["updated_at"] = datetime.utcnow().isoformat()
    db[META_COLL].update_one({"model_name": MODEL_NAME}, {"$set": fields}, upsert=True)

def consolidate(db):
    """Copy leads_feedback → leads_training (upsert by lead_id)."""
    rows = list(db[FEEDBACK_COLL].find({}, {"_id":0}))
    if not rows:
        return 0
    ops = []
    for r in rows:
        st = str(r.get("state","")).lower()
        if st not in {"converted","rejected"}: 
            continue
        r["label"] = 1 if st == "converted" else 0
        r["labeled_at"] = r.get("labeled_at") or datetime.utcnow().isoformat()
        if not r.get("lead_id"): continue
        ops.append(UpdateOne({"lead_id": r["lead_id"]}, {"$set": r}, upsert=True))
    if ops:
        db[TRAIN_COLL].bulk_write(ops)
    return len(ops)

def load_training(db) -> pd.DataFrame:
    docs = list(db[TRAIN_COLL].find({}, {"_id":0}))
    if not docs: return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["state"] = df["state"].astype(str).str.lower()
    df["label"] = df["state"].map({"converted":1,"rejected":0})
    df = df[df["label"].isin([0,1])]
    df = df.drop_duplicates(subset=["lead_id"], keep="last")
    # ensure group_id present (all same product)
    if "group_id" not in df.columns:
        df["group_id"] = "1"
    return df

def split_features(df):
    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
    num_cols, cat_cols = [], []
    for c in feature_cols:
        z = pd.to_numeric(df[c], errors="coerce")
        if z.notna().sum() > 0:
            df[c] = z
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return df, num_cols, cat_cols

def fit_ohe(cat_df):
    if cat_df.shape[1] == 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(pd.DataFrame({"__dummy__":[0,1]}))
    else:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(cat_df.astype(str).fillna("__nan__"))
    joblib.dump(enc, ENCODER_PATH)
    return enc

def apply_ohe(enc, cat_df):
    if cat_df.shape[1] == 0:
        Z = enc.transform(pd.DataFrame({"__dummy__":[0]*len(cat_df)}))
        return pd.DataFrame(Z, index=cat_df.index)
    Z = enc.transform(cat_df.astype(str).fillna("__nan__"))
    return pd.DataFrame(Z, index=cat_df.index)

def train_binary(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
    }
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dte = lgb.Dataset(Xte, label=yte, reference=dtr, free_raw_data=False)
    model = lgb.train(params, dtr, valid_sets=[dtr,dte], valid_names=["train","valid"],
                      num_boost_round=1000, early_stopping_rounds=100)
    yhat = model.predict(Xte, num_iteration=model.best_iteration)
    try: auc = roc_auc_score(yte, yhat)
    except: auc = float('nan')
    model.save_model(MODEL_PATH)
    print(f"[retrain] saved {MODEL_PATH} best_iter={model.best_iteration} AUC(valid)={auc:.4f}")
    return model

def main(push: bool, force: bool):
    cli = connect()
    db = cli[DB_NAME]

    # 1) consolidate from feedback
    consolidate(db)

    # 2) check gate
    labeled_now = db[FEEDBACK_COLL].count_documents({})
    meta = get_meta(db)
    last = int(meta.get("last_trained_count", 0))
    grew = labeled_now >= max(1, last) * (1 + NEW_DATA_THRESHOLD)

    if not (force or last == 0 or grew):
        pct = 0 if last == 0 else (labeled_now - last) / max(1,last)
        print(f"[retrain] skip: labeled={labeled_now}, last={last}, growth={pct:.2%}, thresh={NEW_DATA_THRESHOLD:.2%}")
        return 0

    # 3) build frame
    df = load_training(db)
    if df.empty:
        print("[retrain] no labeled rows. exit.")
        return 0

    df, num_cols, cat_cols = split_features(df)
    y = df["label"].astype(int).values
    X_num = df[num_cols].fillna(0.0) if num_cols else pd.DataFrame(index=df.index)
    X_cat = df[cat_cols] if cat_cols else pd.DataFrame(index=df.index)
    enc = fit_ohe(X_cat)
    X_cat_ohe = apply_ohe(enc, X_cat)
    X = pd.concat([X_num.reset_index(drop=True), X_cat_ohe.reset_index(drop=True)], axis=1)

    # 4) train (binary—single product)
    model = train_binary(X, y)

    # 5) score all and export
    scores = model.predict(X, num_iteration=model.best_iteration)
    out = pd.DataFrame({
        "lead_id": df["lead_id"].astype(str).values,
        "group_id": df["group_id"].astype(str).values,
        "score": scores,
        "label": df["label"].values,
        "state": df["state"].values
    })
    out["rank"] = out["score"].rank(ascending=False, method="first")
    out = out.sort_values("rank")
    out.to_csv("ranked_predictions_all.csv", index=False)
    print(f"[retrain] wrote ranked_predictions_all.csv ({len(out)} rows)")

    # 6)  push back to mongo
    if push:
        ops = [UpdateOne({"lead_id": r["lead_id"]}, {"$set": r, "$setOnInsert": {"scored_at": datetime.utcnow().isoformat()}}, upsert=True)
               for r in out.to_dict(orient="records")]
        if ops:
            db[PRED_COLL].bulk_write(ops)
            print(f"[retrain] upserted {len(ops)} docs to {PRED_COLL}")

    # 7) update meta
    set_meta(db, last_trained_count=labeled_now, last_trained_at=datetime.utcnow().isoformat(),
             model_path=MODEL_PATH, encoder_path=ENCODER_PATH)
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true', help='push predictions to Mongo after training')
    ap.add_argument('--force', action='store_true', help='ignore 5% gate')
    args = ap.parse_args()
    sys.exit(main(push=args.push, force=args.force))
