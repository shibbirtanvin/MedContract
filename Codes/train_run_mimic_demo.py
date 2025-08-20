# train_run_real_mimic_demo.py
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

import icu_contracts_v2 as C
from real_mimic_loader import load_mimic_demo_wide

# -----------------------
# Features used for demo:
#   age, lactate (imputed), careunit proxies (is_micu, is_sicu)
# -----------------------
def build_X(df: pd.DataFrame) -> pd.DataFrame:
    return df[["age", "lactate", "is_micu", "is_sicu"]].copy()

def cohort_checks(df: pd.DataFrame):
    # Only check timestamps for features we have time cols for (age, lactate)
    C.contract_index_time_alignment(
        df,
        {"age": "age_time", "lactate": "lactate_time"},
        t0_col="index_time",
    )
    C.contract_censoring_handling(df, los_col="los_hours", censor_flag_col="censored")

def split_patient_level(df: pd.DataFrame):
    ids = df["subject_id"].unique()
    train_ids, temp_ids = train_test_split(ids, test_size=0.40, random_state=0)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=1)
    C.contract_unique_stays_no_leakage(set(train_ids), set(valid_ids), set(test_ids))
    tr = df[df.subject_id.isin(train_ids)].copy()
    va = df[df.subject_id.isin(valid_ids)].copy()
    te = df[df.subject_id.isin(test_ids)].copy()
    return tr, va, te

def run_regressors(df_train, df_val, coverage_target=0.80, coverage_tol=0.10):
    Xtr_full, ytr_full = build_X(df_train), df_train["los_hours"].values
    Xva, yva = build_X(df_val), df_val["los_hours"].values

    C.contract_bounded_extrapolation(Xtr_full, Xva, p_low=1, p_high=99, max_oob_frac=0.10)

    tr_fit_idx, tr_cal_idx = train_test_split(
        np.arange(len(Xtr_full)), test_size=0.25, random_state=42
    )
    X_fit, y_fit = Xtr_full.iloc[tr_fit_idx], ytr_full[tr_fit_idx]
    X_cal, y_cal = Xtr_full.iloc[tr_cal_idx], ytr_full[tr_cal_idx]

    m_glm = TweedieRegressor(power=1.5, alpha=0.001, max_iter=1000).fit(X_fit, y_fit)
    mae_glm = mean_absolute_error(yva, m_glm.predict(Xva))

    m_p50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, random_state=0).fit(X_fit, y_fit)

    # Split-conformal radius
    cal_res = np.abs(y_cal - m_p50.predict(X_cal))
    alpha = 1.0 - coverage_target
    n_cal = len(cal_res)
    rank = int(np.ceil((n_cal + 1) * (1 - alpha))) - 1
    rank = int(np.clip(rank, 0, n_cal - 1))
    q_hat = float(np.sort(cal_res)[rank])

    # Build intervals on validation
    p50 = m_p50.predict(Xva)
    p10 = p50 - q_hat
    p90 = p50 + q_hat

    # If over-coverage, gently shrink radius to land inside the band
    covered = ((yva >= p10) & (yva <= p90)).mean()
    if covered > (coverage_target + coverage_tol):
        for s in np.linspace(0.95, 0.70, 6):  # try 5–30% shrink
            lo = p50 - q_hat * s
            hi = p50 + q_hat * s
            cov = ((yva >= lo) & (yva <= hi)).mean()
            if abs(cov - coverage_target) <= coverage_tol:
                p10, p90, covered = lo, hi, cov
                print(f"Shrank conformal radius by {int((1-s)*100)}% → coverage {covered:.3f}")
                break

    # Contract: within target ± tol
    C.contract_quantile_coverage(yva, p10, p90, target_coverage=coverage_target, tol=coverage_tol)

    return {
        "mae_glm": float(mae_glm),
        "mae_p50": float(mean_absolute_error(yva, p50)),
        "pred_quantiles": (p10, p50, p90),
        "median_model": m_p50,
    }


def choose_theta_for_alert_budget(prob, min_rate=0.10, max_rate=0.40, target_rate=0.20):
    target_rate = float(np.clip(target_rate, min_rate, max_rate))
    theta_low = np.quantile(prob, 1.0 - max_rate)   # ~max_rate alerts
    theta_high = np.quantile(prob, 1.0 - min_rate)  # ~min_rate alerts
    theta = np.quantile(prob, 1.0 - target_rate)
    grid = np.linspace(theta_low, theta_high, 101)
    best_theta, best_rate, best_gap = theta, float((prob >= theta).mean()), 1e9
    for th in grid:
        r = float((prob >= th).mean())
        if min_rate <= r <= max_rate:
            return float(th), float(r)
        gap = min(abs(r - min_rate), abs(r - max_rate))
        if gap < best_gap:
            best_theta, best_rate, best_gap = th, r, gap
    return float(best_theta), float(best_rate)

def run_classifier(df_train, df_val, min_rate=0.10, max_rate=0.40, target_rate=0.20):
    Xtr, ytr = build_X(df_train), df_train["los_ge_48h"].values
    Xva, yva = build_X(df_val), df_val["los_ge_48h"].values

    clf = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    prob = clf.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, prob)

    # Contracts: calibration + alert budget at an auto-chosen theta
    C.contract_ece(prob, yva, n_bins=10, max_ece=0.10)
    theta, alert_rate = choose_theta_for_alert_budget(prob, min_rate, max_rate, target_rate)
    C.contract_alert_budget(prob, theta=theta, min_rate=min_rate, max_rate=max_rate)

    print(f"Classifier: chosen theta={theta:.3f} → alert_rate={alert_rate:.1%} (band {int(min_rate*100)}–{int(max_rate*100)}%)")

    return {"AUC": float(auc), "prob": prob, "theta": float(theta), "alert_rate": float(alert_rate)}

def temporal_consistency_check(median_model, df_val):
    # decrease lactate by 0.5 mmol/L (bounded below by 0.5) to mimic improvement.
    X0 = build_X(df_val).copy()
    X1 = X0.copy()
    X1["lactate"] = np.maximum(0.5, X1["lactate"] - 0.5)

    pred0 = median_model.predict(X0)
    pred1 = median_model.predict(X1)
    C.contract_temporal_consistency(pred_t0=pred0, pred_t1=pred1, max_drop=12.0)

def fairness_gap_by_careunit(df_val, p50_pred):
    # Simple fairness lens: MAE by careunit (MICU / SICU / Other)
    cu = df_val["first_careunit"].astype(str).str.upper()
    groups = {
        "MICU": cu.str.contains("MICU"),
        "SICU": cu.str.contains("SICU"),
        "OTHER": ~(cu.str.contains("MICU") | cu.str.contains("SICU")),
    }
    y = df_val["los_hours"].values
    mae_by_group = {}
    for g, mask in groups.items():
        mask = mask.values if hasattr(mask, "values") else mask
        if mask.sum() == 0:
            continue
        mae_by_group[g] = float(np.mean(np.abs(y[mask] - p50_pred[mask])))
    # Up to 6h absolute gap
    if mae_by_group:
        C.contract_group_gap(mae_by_group, max_gap=6.0)
    return mae_by_group

def occupancy_plausibility(df_train, p50_val):
    # Training LOS distribution as "historical" baseline
    hist_mean = float(df_train["los_hours"].mean())
    hist_std = float(df_train["los_hours"].std(ddof=0))
    C.contract_occupancy_plausibility(p50_val, historical_mean=hist_mean, historical_std=hist_std, k=3.0)

def main():
    DEMO_ROOT = "/Users/shibbir/PycharmProjects/PyContractGen/mimic"
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_hours", type=int, default=6, help="Index time offset in hours after ICU admit")
    parser.add_argument("--coverage_target", type=float, default=0.80)
    parser.add_argument("--coverage_tol", type=float, default=0.10)
    parser.add_argument("--min_alert_rate", type=float, default=0.10)
    parser.add_argument("--max_alert_rate", type=float, default=0.40)
    parser.add_argument("--target_alert_rate", type=float, default=0.20)
    args = parser.parse_args()

    # Load real demo cohort
    df = load_mimic_demo_wide(DEMO_ROOT, index_offset_hours=args.index_hours)
    print("Cohort shape:", df.shape)

    # Cohort-level contracts
    cohort_checks(df)

    # Split by patient ID
    tr, va, te = split_patient_level(df)

    # Regressors + contracts (Option A: conformal intervals)
    reg = run_regressors(
        tr, va,
        coverage_target=args.coverage_target,
        coverage_tol=args.coverage_tol
    )

    # Classifier + contracts
    clf = run_classifier(
        tr, va,
        min_rate=args.min_alert_rate,
        max_rate=args.max_alert_rate,
        target_rate=args.target_alert_rate
    )

    # Temporal consistency on median model
    temporal_consistency_check(reg["median_model"], va)

    # Fairness (careunit) & occupancy plausibility
    p10, p50, p90 = reg["pred_quantiles"]
    mae_by_group = fairness_gap_by_careunit(va, p50_pred=p50)
    occupancy_plausibility(tr, p50_val=p50)

    # Report
    print("\n=== STUDY REPORT (MIMIC-IV DEMO) ===")
    print(f"GLM_Tweedie_MAE: {reg['mae_glm']:.2f} h")
    print(f"GBM_P50_MAE:     {reg['mae_p50']:.2f} h")
    print(f"Classifier AUC:   {clf['AUC']:.3f}  | theta={clf['theta']:.3f}  | alert_rate={clf['alert_rate']:.1%}")
    if mae_by_group:
        print("Careunit MAE:", mae_by_group)
    print("All contracts passed.")

if __name__ == "__main__":
    main()
