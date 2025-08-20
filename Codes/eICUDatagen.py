# generate_eicu_synth.py


import os, numpy as np, pandas as pd

OUT_DIR = "./eICU"
N_STAYS = 1200
INDEX_HOURS = 6
SEED = 7

UNITS = ["MICU", "SICU", "CCU", "TSICU", "CVICU", "Neuro ICU"]

def make_patient(n_stays: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_subjects = int(n_stays * 0.7)
    subjects = np.arange(100000, 100000 + n_subjects)
    subject_ids = rng.choice(subjects, size=n_stays, replace=True)

    stay_ids = np.arange(200000, 200000 + n_stays)

    los_hours = np.clip(rng.lognormal(mean=2.5, sigma=0.6, size=n_stays), 6, 240)
    unitadmitoffset = np.zeros(n_stays, dtype=float)
    unitdischargeoffset = (los_hours * 60.0).astype(float)

    age = np.clip(rng.normal(62, 15, size=n_stays), 18, 95)
    unittype = rng.choice(UNITS, size=n_stays, replace=True)

    df = pd.DataFrame({
        "patientunitstayid": stay_ids,
        "uniquepid": subject_ids,
        "unitadmitoffset": unitadmitoffset,
        "unitdischargeoffset": unitdischargeoffset,
        "unittype": unittype,
        "age": np.round(age, 1),
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def make_lab(patient_df: pd.DataFrame, index_hours: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _, r in patient_df.iterrows():
        stay_id = int(r["patientunitstayid"])
        k = rng.integers(0, 5)  # number of lactate measurements
        base = rng.lognormal(mean=0.4, sigma=0.4, size=k) + rng.uniform(0, 0.5, size=k)
        offs = rng.uniform(0, index_hours * 60.0, size=k)
        for v, o in zip(base, offs):
            rows.append({
                "patientunitstayid": stay_id,
                "labname": "Lactate",
                "labresult": round(float(v), 2),
                "labresultoffset": float(o)
            })
    return pd.DataFrame(rows, columns=["patientunitstayid", "labname", "labresult", "labresultoffset"])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patient = make_patient(N_STAYS, seed=SEED)
    lab = make_lab(patient, index_hours=INDEX_HOURS, seed=SEED)

    patient.to_csv(os.path.join(OUT_DIR, "patient.csv"), index=False)
    lab.to_csv(os.path.join(OUT_DIR, "lab.csv"), index=False)

    with open(os.path.join(OUT_DIR, "README_SYNTH.txt"), "w") as f:
        f.write(
            "Synthetic eICU-like folder for PyHealth eICUDataset.\n"
            f"Stays: {N_STAYS}, Seed: {SEED}, Index window: {INDEX_HOURS}h.\n"
            "Tables: patient.csv, lab.csv.\n"
        )

    print(f"Wrote synthetic eICU folder to: {OUT_DIR}")
    print(f"patient.csv: {len(patient):,} rows | lab.csv: {len(lab):,} rows")

if __name__ == "__main__":
    main()
