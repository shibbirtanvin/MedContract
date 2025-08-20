# real_mimic_loader.py
# Build a "wide" cohort table from the MIMIC-IV *demo* CSVs.
# Requires the demo to be unzipped, with folders like: {root}/icu/icustays.csv.gz, {root}/hosp/patients.csv.gz, ...
import os
import numpy as np
import pandas as pd

def _read(root, rel):
    path = os.path.join(root, rel)
    return pd.read_csv(path, compression="infer")

demo_root = '/Users/shibbir/PycharmProjects/PyContractGen/mimic'
def load_mimic_demo_wide(demo_root: str, index_offset_hours: int = 6) -> pd.DataFrame:
    # --- Minimal core tables ---
    icu = _read(demo_root, os.path.join("icu","icustays.csv.gz"))[
        ["subject_id","hadm_id","stay_id","intime","outtime","first_careunit"]
    ].copy()
    pts = _read(demo_root, os.path.join("hosp","patients.csv.gz"))[
        ["subject_id","anchor_age"]  # anchor_age ~ age proxy
    ].copy().rename(columns={"anchor_age":"age"})
    icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])

    # join age
    df = icu.merge(pts, on="subject_id", how="left")

    # index time policy: predict at 6h after ICU admit (you can change this)
    df["index_time"] = df["intime"] + pd.to_timedelta(index_offset_hours, unit="h")

    # --- Lactate from hosp.labevents using d_labitems labels ---
    dlab = _read(demo_root, os.path.join("hosp","d_labitems.csv.gz"))[["itemid","label"]]
    lact_ids = dlab[dlab["label"].str.contains("lactate", case=True, na=False)]["itemid"].unique()
    try:
        labs = _read(demo_root, os.path.join("hosp","labevents.csv.gz"))[
            ["subject_id","hadm_id","itemid","charttime","valuenum","valueuom"]
        ]
        labs["charttime"] = pd.to_datetime(labs["charttime"])
        lact = labs[labs["itemid"].isin(lact_ids)].copy()
        # Align lactate to stays and keep last <= index_time; we also keep max as a severity proxy within window
        tmp = lact.merge(df[["subject_id","hadm_id","stay_id","index_time"]], on=["subject_id","hadm_id"], how="inner")
        tmp = tmp[tmp["charttime"] <= tmp["index_time"]]
        # Keep the max value (more conservative for severity) and its latest time
        agg = (tmp.sort_values(["stay_id","charttime"])
                  .groupby("stay_id")
                  .agg(lactate=("valuenum","max"), lactate_time=("charttime","max"))
                  .reset_index())
        df = df.merge(agg, on="stay_id", how="left")
    except FileNotFoundError:
        # demo may lack labevents in some distributions; fall back to NaN
        df["lactate"] = np.nan; df["lactate_time"] = pd.NaT

    # --- Feature timestamps for contracts ---
    # age is effectively static; give it a timestamp <= index_time
    df["age_time"] = df["intime"] + pd.to_timedelta(1, "h")
    # if lactate_time missing, set to index_time (conservative: measured at or before index)
    if "lactate_time" not in df or df["lactate_time"].isna().all():
        df["lactate_time"] = df["index_time"]

    # --- Label & flags ---
    df["los_hours"] = np.clip((df["outtime"] - df["intime"]).dt.total_seconds()/3600.0, 0, None)
    df["censored"] = df["outtime"].isna().astype(int)
    df["los_ge_48h"] = (df["los_hours"] >= 48).astype(int)

    # --- Simple numeric proxies from careunit ---
    cu = df["first_careunit"].astype(str).str.lower()
    df["is_micu"] = cu.str.contains("micu").astype(int)
    df["is_sicu"] = cu.str.contains("sicu").astype(int)

    # --- Finalize minimal feature set ---
    # We will use: age, lactate (imputed), is_micu, is_sicu
    # Impute lactate (demo can be sparse)
    if "lactate" not in df or df["lactate"].isna().all():
        df["lactate"] = np.nan
    med = float(df["lactate"].median()) if df["lactate"].notna().any() else 2.0
    df["lactate"] = df["lactate"].fillna(med)

    # Keep only required + extras for contracts
    keep = [
        "subject_id","hadm_id","stay_id","intime","outtime","index_time",
        "age","lactate","is_micu","is_sicu","age_time","lactate_time",
        "los_hours","censored","los_ge_48h","first_careunit"
    ]
    return df[keep]
