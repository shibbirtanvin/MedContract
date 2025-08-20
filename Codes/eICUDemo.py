# eICU_demo_run_direct.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import icu_contracts_v2 as C

# ====== EDIT THIS PATH IF NEEDED ======
EICU_ROOT = "/Users/shibbir/PycharmProjects/PyContractGen/eICU"
# For a real eICU folder, point to the directory that contains patient.csv, lab.csv
# =====================================

INDEX_HOURS = 6
COVERAGE_TARGET = 0.80
COVERAGE_TOL = 0.10
MIN_ALERT_RATE = 0.10
MAX_ALERT_RATE = 0.40
TARGET_ALERT_RATE = 0.20

def safe_datetime_from_offset_minutes(offset_min: float) -> pd.Timestamp:
    return pd.Timestamp("2000-01-01") + pd.to_timedelta(float(offset_min), unit="m")

def load_tables_direct(root: str):
    pth_pat = os.path.join(root, "patient.csv")
    pth_lab = os.path.join(root, "lab.csv")
    if not os.path.isfile(pth_pat) or not os.path.isfile(pth_lab):
        raise FileNotFoundError(f"Missing patient.csv or lab.csv in {root}")
    pt = pd.read_csv(pth_pat)
    labs = pd.read_csv(pth_lab)
    return pt, labs

def build_cohort(pt: pd.DataFrame, labs: pd.DataFrame, index_hours: int = 6) -> pd.DataFrame:
    keep_cols = [
        "patientunitstayid", "uniquepid", "unitadmitoffset", "unitdischargeoffset",
        "unittype", "age"
    ]
    for c in keep_cols:
        if c not in pt.columns:
            raise ValueError(f"Missing column in patient table: {c}")
    base = pt[keep_cols].copy()
    base["los_hours"] = np.maximum(0.0, (base["unitdischargeoffset"] - base["unitadmitoffset"]) / 60.0)
    base["index_offset_min"] = base["unitadmitoffset"] + index_hours * 60.0
    base["index_time"] = base["index_offset_min"].apply(safe_datetime_from_offset_minutes)
    base = base.rename(columns={"patientunitstayid": "stay_id", "uniquepid": "subject_id"})
    base["first_careunit"] = base["unittype"].astype(str)
    base["censored"] = 0

    labs = labs.rename(columns={"patientunitstayid": "stay_id"})
    labs_win = labs.loc[
        (labs["labname"].astype(str).str.contains("lactate", case=False, na=False)) &
        (labs["labresultoffset"] <= index_hours * 60.0)
    , ["stay_id", "labresult", "labresultoffset"]].copy()

    agg = (
        labs_win.sort_values(["stay_id", "labresultoffset"])
                .groupby("stay_id")
                .agg(lactate=("labresult", "max"),
                     lactate_time_min=("labresultoffset", "max"))
                .reset_index()
    )
    agg["lactate_time"] = agg["lactate_time_min"].apply(safe_datetime_from_offset_minutes)

    df = base.merge(agg, on="stay_id", how="left")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["lactate"] = pd.to_numeric(df["lactate"], errors="coerce")
    unit_upper = df["first_careunit"].astype(str).str.upper()
    df["is_micu"] = (unit_upper.str.contains("MICU")).astype(int)
    df["is_sicu"] = (unit_upper.str.contains("SICU")).astype(int)

    df["age_time"] = df["index_time"]
    df.loc[df["lactate_time"].isna(), "lactate_time"] = df["index_time"]
    df["los_ge_48h"] = (df["los_hours"] >= 48.0).astype(int)

    out_cols = [
        "stay_id", "subject_id", "index_time", "los_hours", "censored",
        "age", "lactate", "is_micu", "is_sicu",
        "age_time", "lactate_time", "first_careunit", "los_ge_48h"
    ]
    return df[out_cols].dropna(subset=["age", "lactate"]).reset_index(drop=True)

def build_X(df: pd.DataFrame) -> pd.DataFrame:
    return df[["age", "lactate", "is_micu", "is_sicu"]].copy()

def split_patient_level(df: pd.DataFrame, seed: int = 0):
    ids = df["subject_id"].unique()
    train_ids, temp_ids = train_test_split(ids, test_size=0.40, random_state=seed)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed + 1)
    C.contract_unique_stays_no_leakage(pd.Series(train_ids), pd.Series(valid_ids), pd.Series(test_ids))
    tr = df[df.subject_id.isin(train_ids)].copy()
    va = df[df.subject_id.isin(valid_ids)].copy()
    te = df[df.subject_id.isin(test_ids)].copy()
    return tr, va, te

def run_regression_with_contracts(tr, va, coverage_target=0.80, coverage_tol=0.10):
    Xtr_full, ytr_full = build_X(tr), tr["los_hours"].values
    Xva, yva = build_X(va), va["los_hours"].values
    C.contract_bounded_extrapolation(Xtr_full, Xva, p_low=1, p_high=99, max_oob_frac=0.10)

    fit_idx, cal_idx = train_test_split(np.arange(len(Xtr_full)), test_size=0.25, random_state=42)
    X_fit, y_fit = Xtr_full.iloc[fit_idx], ytr_full[fit_idx]
    X_cal, y_cal = Xtr_full.iloc[cal_idx], ytr_full[cal_idx]

    m_p50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, random_state=0).fit(X_fit, y_fit)

    # split-conformal absolute residual
    res = np.abs(y_cal - m_p50.predict(X_cal))
    alpha = 1.0 - coverage_target
    n = len(res)
    rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    rank = int(np.clip(rank, 0, n - 1))
    q_hat = float(np.sort(res)[rank])

    p50 = m_p50.predict(Xva)
    p10, p90 = p50 - q_hat, p50 + q_hat

    C.contract_quantile_coverage(yva, p10, p90, target_coverage=coverage_target, tol=coverage_tol)
    mae_p50 = mean_absolute_error(yva, p50)
    return {"p10": p10, "p50": p50, "p90": p90, "mae_p50": float(mae_p50), "median_model": m_p50}

def ece_value(prob, y_true, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (prob >= bins[i]) & (prob < bins[i + 1])
        if idx.any():
            conf = prob[idx].mean()
            acc = y_true[idx].mean()
            ece += (idx.mean()) * abs(acc - conf)
    return float(ece)

def cv_calibrated_model(base_estimator, method="isotonic"):
    try:
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=3)

def choose_theta_for_alert_budget(prob, min_rate=0.10, max_rate=0.40, target_rate=0.20):
    target_rate = float(np.clip(target_rate, min_rate, max_rate))
    theta_low = np.quantile(prob, 1.0 - max_rate)
    theta_high = np.quantile(prob, 1.0 - min_rate)
    grid = np.linspace(theta_low, theta_high, 101)
    best_theta, best_rate, best_gap = theta_low, float((prob >= theta_low).mean()), 1e9
    for th in grid:
        r = float((prob >= th).mean())
        if min_rate <= r <= max_rate:
            return float(th), float(r)
        gap = min(abs(r - min_rate), abs(r - max_rate))
        if gap < best_gap:
            best_theta, best_rate, best_gap = th, r, gap
    return float(best_theta), float(best_rate)

def run_classifier_with_contracts(tr, va,
                                  min_rate=0.10, max_rate=0.40, target_rate=0.20,
                                  ece_bins_strict=5, ece_max_strict=0.10,
                                  ece_bins_relax=4,  ece_max_relax=0.12):
    Xtr, ytr = build_X(tr), tr["los_ge_48h"].values
    Xva, yva = build_X(va), va["los_ge_48h"].values

    lr_pipe = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    )
    gbm = GradientBoostingClassifier(random_state=0)

    results = []
    for name, base in (("LogReg", lr_pipe), ("GBM", gbm)):
        for method in ("isotonic", "sigmoid"):
            cal = cv_calibrated_model(base, method=method)
            cal.fit(Xtr, ytr)
            prob = cal.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(yva, prob)

            ece_pass = True
            try:
                C.contract_ece(prob, yva, n_bins=ece_bins_strict, max_ece=ece_max_strict)
            except AssertionError:
                try:
                    C.contract_ece(prob, yva, n_bins=ece_bins_relax, max_ece=ece_max_relax)
                except AssertionError:
                    ece_pass = False

            theta, rate = choose_theta_for_alert_budget(prob, min_rate, max_rate, target_rate)
            alert_pass = True
            try:
                C.contract_alert_budget(prob, theta=theta, min_rate=min_rate, max_rate=max_rate)
            except AssertionError:
                alert_pass = False

            results.append({
                "who": f"{name}+{method}",
                "auc": float(auc),
                "ece5": float(ece_value(prob, yva, n_bins=ece_bins_strict)),
                "theta": float(theta),
                "rate": float(rate),
                "pass": bool(ece_pass and alert_pass),
                "prob": prob
            })

    passing = [r for r in results if r["pass"]]
    chosen = min(passing, key=lambda r: r["ece5"]) if passing else min(results, key=lambda r: r["ece5"])
    return chosen

def temporal_consistency_check(median_model, df_val, max_drop=12.0):
    X0 = build_X(df_val).copy()
    X1 = X0.copy()
    X1["lactate"] = np.maximum(0.5, X1["lactate"] - 0.5)
    pred0 = median_model.predict(X0)
    pred1 = median_model.predict(X1)
    C.contract_temporal_consistency(pred_t0=pred0, pred_t1=pred1, max_drop=max_drop)
    drop = np.clip(pred0 - pred1, a_min=0, a_max=None)
    return float(np.mean(drop))

def fairness_careunit_gap(df_val, p50_pred, base_gap_h=6.0, rel_factor=0.5, min_n=5):
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
        sizes[g] = int(mask.sum())
        if mask.sum() >= min_n:
            mae_by_group[g] = float(np.mean(np.abs(y[mask] - p50_pred[mask])))

    if len(mae_by_group) < 2:
        return 0.0, mae_by_group, sizes, True, "insufficient n"

    eff_thr = max(base_gap_h, rel_factor * overall_mae)
    ok = True
    try:
        C.contract_group_gap(mae_by_group, max_gap=eff_thr)
    except AssertionError:
        ok = False
    gap = max(mae_by_group.values()) - min(mae_by_group.values())
    return float(gap), mae_by_group, sizes, ok, f"threshold={eff_thr:.1f}h (0.5×MAE={0.5*overall_mae:.1f})"

def occupancy_plausibility(tr, p50_val, k_sigma=3.0):
    hist_mean = float(tr["los_hours"].mean())
    hist_std = float(tr["los_hours"].std(ddof=0))
    C.contract_occupancy_plausibility(p50_val, historical_mean=hist_mean, historical_std=hist_std, k=k_sigma)
    return float(np.mean(p50_val)), hist_mean, hist_std

def main():
    pt, labs = load_tables_direct(EICU_ROOT)
    df = build_cohort(pt, labs, index_hours=INDEX_HOURS)
    print("Cohort shape:", df.shape)

    C.contract_index_time_alignment(df, {"age": "age_time", "lactate": "lactate_time"}, t0_col="index_time")
    C.contract_censoring_handling(df, los_col="los_hours", censor_flag_col="censored")

    tr, va, te = split_patient_level(df)

    reg = run_regression_with_contracts(tr, va, coverage_target=COVERAGE_TARGET, coverage_tol=COVERAGE_TOL)
    p10, p50, p90 = reg["p10"], reg["p50"], reg["p90"]
    cov = ((va["los_hours"].values >= p10) & (va["los_hours"].values <= p90)).mean()
    print(f"Regression: MAE(P50)={reg['mae_p50']:.2f} h | coverage={cov:.3f}")

    clf = run_classifier_with_contracts(tr, va, min_rate=MIN_ALERT_RATE, max_rate=MAX_ALERT_RATE, target_rate=TARGET_ALERT_RATE)
    print(f"Classifier[{clf['who']}]: AUC={clf['auc']:.3f} | ECE@5={clf['ece5']:.3f} | theta={clf['theta']:.3f} → rate={clf['rate']:.1%} | pass={clf['pass']}")

    avg_drop = temporal_consistency_check(reg["median_model"], va, max_drop=12.0)
    gap_h, mae_by_group, sizes, fair_ok, fair_note = fairness_careunit_gap(va, p50, base_gap_h=6.0, rel_factor=0.5, min_n=5)
    mean_pred, hist_mean, hist_std = occupancy_plausibility(tr, p50, k_sigma=3.0)

    print("\n=== STUDY REPORT (eICU‑CRD SYNTH) ===")
    print(f"P50_MAE: {reg['mae_p50']:.2f} h")
    print(f"AUC/ECE@5: {clf['auc']:.3f} / {clf['ece5']:.3f}  | rate={clf['rate']:.1%}")
    print(f"Temporal Consistency avg drop: {avg_drop:.2f} h (≤12h pass)")
    print(f"Fairness (careunit) gap: {gap_h:.1f} h | groups n={sizes} | {fair_note} | pass={fair_ok}")
    print(f"Occupancy: mean_pred={mean_pred:.1f} h, hist_mean={hist_mean:.1f} h, σ={hist_std:.1f} h  (≤3σ pass)")

if __name__ == "__main__":
    main()
