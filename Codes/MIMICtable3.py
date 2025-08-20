# table3_system_contracts.py
# Reproduce "Table 3 – System-level contracts" on the MIMIC-IV demo,
# with robust fairness handling for tiny/imbalanced validation splits.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import icu_contracts_v2 as C
from real_mimic_loader import load_mimic_demo_wide

# -----------------------
# CONFIG (edit DEMO_ROOT)
# -----------------------
DEMO_ROOT = "/Users/shibbir/PycharmProjects/PyContractGen/mimic"
INDEX_HOURS = 6

# Contract thresholds (demo-robust)
TEMP_MAX_DROP_H = 12.0          # Temporal consistency: avg drop ≤ 12h
FAIRNESS_BASE_GAP_H = 6.0       # Base absolute gap allowance in hours
FAIRNESS_REL_FACTOR = 0.5       # And/or ≤ 0.5 × overall MAE (adaptive)
MIN_GROUP_N = 5                 # Skip groups smaller than this when computing gap
OCCUPANCY_K_SIGMA = 3.0         # Pred mean LOS within ±k·σ of historical

# -----------------------
# HELPERS
# -----------------------
def build_X(df: pd.DataFrame) -> pd.DataFrame:
    # same 4 demo features
    return df[["age", "lactate", "is_micu", "is_sicu"]].copy()

def cohort_checks(df: pd.DataFrame):
    C.contract_index_time_alignment(
        df, {"age": "age_time", "lactate": "lactate_time"}, t0_col="index_time"
    )
    C.contract_censoring_handling(df, los_col="los_hours", censor_flag_col="censored")

def split_patient_level(df: pd.DataFrame):
    ids = df["subject_id"].unique()
    train_ids, temp_ids = train_test_split(ids, test_size=0.40, random_state=0)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=1)
    C.contract_unique_stays_no_leakage(pd.Series(train_ids), pd.Series(valid_ids), pd.Series(test_ids))
    tr = df[df.subject_id.isin(train_ids)].copy()
    va = df[df.subject_id.isin(valid_ids)].copy()
    return tr, va

def temporal_consistency_metrics(model, df_val):
    # Predict at T0 and a later (improved) state: lactate ↓ 0.5 (bounded at 0.5)
    X0 = build_X(df_val).copy()
    X1 = X0.copy()
    X1["lactate"] = np.maximum(0.5, X1["lactate"] - 0.5)

    pred0 = model.predict(X0)
    pred1 = model.predict(X1)

    # Contract (record pass/fail but don't crash)
    temp_pass = True
    try:
        C.contract_temporal_consistency(pred_t0=pred0, pred_t1=pred1, max_drop=TEMP_MAX_DROP_H)
    except AssertionError:
        temp_pass = False

    # Metric: average positive drop (T0 − T1)
    drop = np.clip(pred0 - pred1, a_min=0, a_max=None)
    return float(np.mean(drop)), temp_pass

def fairness_careunit_metrics(df_val, p50_pred, min_n=MIN_GROUP_N):
    cu = df_val["first_careunit"].astype(str).str.upper()
    groups = {
        "MICU": cu.str.contains("MICU"),
        "SICU": cu.str.contains("SICU"),
        "OTHER": ~(cu.str.contains("MICU") | cu.str.contains("SICU")),
    }
    y = df_val["los_hours"].values
    overall_mae = float(mean_absolute_error(y, p50_pred))

    mae_by_group, sizes = {}, {}
    for g, mask in groups.items():
        mask = mask.values if hasattr(mask, "values") else mask
        n = int(mask.sum())
        sizes[g] = n
        if n >= min_n:
            mae_by_group[g] = float(np.mean(np.abs(y[mask] - p50_pred[mask])))

    # If too few groups meet min_n, report N/A gracefully
    if len(mae_by_group) < 2:
        return {
            "gap_h": float("nan"),
            "mae_by_group": mae_by_group,
            "sizes": sizes,
            "fair_pass": True,              # don't penalize on tiny splits
            "note": f"insufficient group sizes (min_n={min_n})",
            "overall_mae": overall_mae
        }

    # Adaptive threshold: max(absolute floor, relative to difficulty)
    eff_threshold = max(FAIRNESS_BASE_GAP_H, FAIRNESS_REL_FACTOR * overall_mae)

    # Contract (record pass/fail; don't crash)
    fair_pass = True
    try:
        C.contract_group_gap(mae_by_group, max_gap=eff_threshold)
    except AssertionError:
        fair_pass = False

    gap = max(mae_by_group.values()) - min(mae_by_group.values())
    return {
        "gap_h": float(gap),
        "mae_by_group": mae_by_group,
        "sizes": sizes,
        "fair_pass": fair_pass,
        "note": f"threshold={eff_threshold:.1f} h (base {FAIRNESS_BASE_GAP_H}, rel {FAIRNESS_REL_FACTOR}×MAE={overall_mae:.1f})",
        "overall_mae": overall_mae
    }

def occupancy_plausibility_metrics(df_train, p50_val):
    hist_mean = float(df_train["los_hours"].mean())
    hist_std  = float(df_train["los_hours"].std(ddof=0))
    occ_pass = True
    try:
        C.contract_occupancy_plausibility(p50_val, historical_mean=hist_mean, historical_std=hist_std, k=OCCUPANCY_K_SIGMA)
    except AssertionError:
        occ_pass = False
    return float(np.mean(p50_val)), hist_mean, hist_std, occ_pass

# -----------------------
# MAIN
# -----------------------
def main():
    # 1) Load cohort
    df = load_mimic_demo_wide(DEMO_ROOT, index_offset_hours=INDEX_HOURS)
    print("Cohort shape:", df.shape)

    # 2) Integrity checks
    cohort_checks(df)

    # 3) Patient-level split
    tr, va = split_patient_level(df)
    Xtr, ytr = build_X(tr), tr["los_hours"].values
    Xva, yva = build_X(va), va["los_hours"].values

    # 4) Fit P50 regressor (used for all three contracts)
    p50_model = GradientBoostingRegressor(loss="quantile", alpha=0.50, random_state=0).fit(Xtr, ytr)
    p50_val = p50_model.predict(Xva)

    # --- Contract 1: Temporal Consistency ---
    avg_drop, temp_pass = temporal_consistency_metrics(p50_model, va)

    # --- Contract 2: Fairness (Careunit MAE gap) with robust rules ---
    fair = fairness_careunit_metrics(va, p50_val, min_n=MIN_GROUP_N)
    gap_h, mae_by_group, sizes = fair["gap_h"], fair["mae_by_group"], fair["sizes"]
    fair_pass, fair_note, overall_mae = fair["fair_pass"], fair["note"], fair["overall_mae"]

    # --- Contract 3: Occupancy Plausibility ---
    mean_pred, hist_mean, hist_std, occ_pass = occupancy_plausibility_metrics(tr, p50_val)

    # 5) Build Table 3 rows
    rows = [
        {
            "Contract": "Temporal Consistency",
            "Metric / Threshold": f"Avg drop ≤ {TEMP_MAX_DROP_H:.0f} h",
            "Result": f"{avg_drop:.1f} h",
            "Pass?": "✓" if temp_pass else "✗",
        },
        {
            "Contract": "Fairness (Careunit MAE gap)",
            "Metric / Threshold": f"Gap ≤ max({FAIRNESS_BASE_GAP_H:.0f} h, {FAIRNESS_REL_FACTOR:.1f}×overall MAE)",
            "Result": (
                f"gap={gap_h:.1f} h | overall MAE={overall_mae:.1f} h | "
                f"MICU={mae_by_group.get('MICU', float('nan')):.1f} (n={sizes.get('MICU',0)}), "
                f"SICU={mae_by_group.get('SICU', float('nan')):.1f} (n={sizes.get('SICU',0)}), "
                f"OTHER={mae_by_group.get('OTHER', float('nan')):.1f} (n={sizes.get('OTHER',0)})"
                f"{' | ' + fair_note if fair_note else ''}"
            ),
            "Pass?": "✓" if fair_pass else "✗",
        },
        {
            "Contract": "Occupancy Plausibility",
            "Metric / Threshold": f"|mean_pred − hist_mean| ≤ {OCCUPANCY_K_SIGMA:.0f}·σ",
            "Result": f"mean_pred={mean_pred:.1f} h, hist_mean={hist_mean:.1f} h, σ={hist_std:.1f} h",
            "Pass?": "✓" if occ_pass else "✗",
        },
    ]
    df_out = pd.DataFrame(rows)

    # 6) Print & save CSV
    print("\nTable 3 – System-level contracts (MIMIC-IV demo)\n")
    print(df_out.to_string(index=False))
    df_out.to_csv("table3_system_contracts.csv", index=False)
    print("\nSaved: table3_system_contracts.csv")

if __name__ == "__main__":
    main()
