# eicu_make_tables_csv.py
# Build 3 tables for the eICU demo cohort (Regression+Coverage, Classifier Calibration, Fairness).
# Saves them as CSV files instead of LaTeX.

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

# ========= EDIT THIS PATH IF NEEDED =========
EICU_ROOT = "/Users/shibbir/PycharmProjects/PyContractGen/eICU"
# ===========================================

INDEX_HOURS = 6
COVERAGE_TARGET = 0.80
COVERAGE_TOL = 0.10
MIN_ALERT_RATE = 0.10
MAX_ALERT_RATE = 0.40
TARGET_ALERT_RATE = 0.20
FAIR_BASE_GAP_H = 6.0
FAIR_REL_FACTOR = 0.5
FAIR_MIN_N = 5

# ---------- Helpers ----------
def load_tables_direct(root: str):
    pt = pd.read_csv(os.path.join(root, "patient.csv"))
    labs = pd.read_csv(os.path.join(root, "lab.csv"))
    return pt, labs

def build_cohort(pt, labs, index_hours=6):
    base = pt[["patientunitstayid","uniquepid","unitadmitoffset","unitdischargeoffset","unittype","age"]].copy()
    base["los_hours"] = np.maximum(0.0,(base["unitdischargeoffset"]-base["unitadmitoffset"])/60.0)
    base = base.rename(columns={"patientunitstayid":"stay_id","uniquepid":"subject_id","unittype":"first_careunit"})
    base["censored"]=0
    base["index_time"]=pd.Timestamp("2000-01-01")

    labs = labs.rename(columns={"patientunitstayid":"stay_id"})
    lact = labs[labs["labname"].astype(str).str.contains("lactate",case=False,na=False)]
    agg=lact.groupby("stay_id").agg(lactate=("labresult","max")).reset_index()
    df=base.merge(agg,on="stay_id",how="left")

    df["is_micu"]=(df["first_careunit"].str.upper().str.contains("MICU")).astype(int)
    df["is_sicu"]=(df["first_careunit"].str.upper().str.contains("SICU")).astype(int)
    df["los_ge_48h"]=(df["los_hours"]>=48).astype(int)
    return df.dropna(subset=["age","lactate"]).reset_index(drop=True)

def build_X(df): return df[["age","lactate","is_micu","is_sicu"]]

def split(df,seed=0):
    ids=df["subject_id"].unique()
    tr_ids,temp=train_test_split(ids,test_size=0.4,random_state=seed)
    va_ids,te_ids=train_test_split(temp,test_size=0.5,random_state=seed+1)
    return df[df.subject_id.isin(tr_ids)], df[df.subject_id.isin(va_ids)], df[df.subject_id.isin(te_ids)]

# ---------- Table 1 ----------
def run_regression(tr,va):
    Xtr,ytr=build_X(tr),tr["los_hours"].values
    Xva,yva=build_X(va),va["los_hours"].values
    fit_idx,cal_idx=train_test_split(np.arange(len(Xtr)),test_size=0.25,random_state=42)
    m=GradientBoostingRegressor(loss="quantile",alpha=0.5,random_state=0).fit(Xtr.iloc[fit_idx],ytr[fit_idx])
    res=np.abs(ytr[cal_idx]-m.predict(Xtr.iloc[cal_idx]))
    q_hat=np.sort(res)[int(np.ceil((len(res)+1)*(1-COVERAGE_TARGET))-1)]
    p50=m.predict(Xva); p10,p90=p50-q_hat,p50+q_hat
    coverage=((yva>=p10)&(yva<=p90)).mean()
    mae=mean_absolute_error(yva,p50)
    return pd.DataFrame([{
        "Split":"Validation",
        "MAE(P50) [h]":f"{mae:.2f}",
        "Coverage@80%":f"{coverage:.3f}",
        "Conformal radius q̂ [h]":f"{q_hat:.2f}",
        "Pass (coverage)":"Yes" if abs(coverage-COVERAGE_TARGET)<=COVERAGE_TOL else "No"
    }]), p50

# ---------- Table 2 ----------
def ece(prob,y_true,bins=10):
    edges=np.linspace(0,1,bins+1); ece=0
    for i in range(bins):
        idx=(prob>=edges[i])&(prob<edges[i+1])
        if idx.any(): ece+=idx.mean()*abs(prob[idx].mean()-y_true[idx].mean())
    return float(ece)

def classifier_table(tr,va):
    Xtr,ytr=build_X(tr),tr["los_ge_48h"].values
    Xva,yva=build_X(va),va["los_ge_48h"].values
    rows=[]
    for name,base in (("LogReg",make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=2000,class_weight="balanced"))),
                      ("GBM",GradientBoostingClassifier(random_state=0))):
        for method in ("isotonic","sigmoid"):
            cal=CalibratedClassifierCV(estimator=base,method=method,cv=3)
            cal.fit(Xtr,ytr)
            prob=cal.predict_proba(Xva)[:,1]
            auc=roc_auc_score(yva,prob); ece5=ece(prob,yva,5)
            theta=np.quantile(prob,1-MAX_ALERT_RATE)
            rate=(prob>=theta).mean()
            rows.append({"Model":f"{name}+{method}","AUC":f"{auc:.3f}","ECE@5":f"{ece5:.3f}","θ":f"{theta:.3f}","Alert rate":f"{rate:.1%}","Pass":"Yes" if ece5<=0.12 and MIN_ALERT_RATE<=rate<=MAX_ALERT_RATE else "No"})
    return pd.DataFrame(rows)

# ---------- Table 3 ----------
def fairness_table(va,p50_pred):
    cu=va["first_careunit"].str.upper()
    groups={"MICU":cu.str.contains("MICU"),"SICU":cu.str.contains("SICU"),"OTHER":~(cu.str.contains("MICU")|cu.str.contains("SICU"))}
    y=va["los_hours"].values; overall=mean_absolute_error(y,p50_pred)
    maes={}; sizes={}
    for g,mask in groups.items(): sizes[g]=int(mask.sum());
    for g,mask in groups.items():
        if mask.sum()>=FAIR_MIN_N: maes[g]=float(np.mean(np.abs(y[mask]-p50_pred[mask])))
    gap=max(maes.values())-min(maes.values()) if len(maes)>=2 else np.nan
    thr=max(FAIR_BASE_GAP_H,FAIR_REL_FACTOR*overall)
    return pd.DataFrame([{
        "MICU MAE [h]":f"{maes.get('MICU',np.nan):.2f}",
        "SICU MAE [h]":f"{maes.get('SICU',np.nan):.2f}",
        "OTHER MAE [h]":f"{maes.get('OTHER',np.nan):.2f}",
        "Overall MAE [h]":f"{overall:.2f}",
        "Gap [h]":f"{gap:.2f}" if not np.isnan(gap) else "NA",
        "Threshold [h]":f"{thr:.2f}",
        "Pass":"Yes" if not np.isnan(gap) and gap<=thr else "No",
        "N_MICU/N_SICU/N_OTHER":f"{sizes['MICU']}/{sizes['SICU']}/{sizes['OTHER']}"
    }])

# ---------- Main ----------
def main():
    pt,labs=load_tables_direct(EICU_ROOT)
    df=build_cohort(pt,labs,INDEX_HOURS)
    tr,va,te=split(df)

    tab1,p50=run_regression(tr,va)
    tab2=classifier_table(tr,va)
    tab3=fairness_table(va,p50)

    tab1.to_csv("table1_eicu.csv",index=False)
    tab2.to_csv("table2_eicu.csv",index=False)
    tab3.to_csv("table3_eicu.csv",index=False)

    print("Saved: table1_eicu.csv, table2_eicu.csv, table3_eicu.csv")

    

if __name__=="__main__":
    main()
