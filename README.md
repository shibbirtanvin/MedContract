# MedContract
The replication packages for MedContract
Reproducibility Guide – MedContract Case Study
==============================================

This package reproduces the contract-based evaluation and tables for two ICU cohorts:
- MIMIC-IV demo (single-center)
- eICU synthetic or real-style (multi-center)

Directory (codes/)
------------------
eICU_tables.py            # Generate eICU Table 1/2/3 CSV/TeX
eICUDatagen.py            # Create synthetic eICU CSVs (patient.csv, lab.csv)
eICUDemo.py               # Run eICU pipeline; shows contract pass/fail
MIMICtable1.py            # Generate MIMIC-IV Table 1 CSV/TeX
MIMICtable2.py            # Generate MIMIC-IV Table 2 CSV/TeX
MIMICtable3.py            # Generate MIMIC-IV Table 3 CSV/TeX
real_mimic_loader.py      # Build MIMIC-IV cohort (demo or full) to a common schema
train_run_mimic_demo.py   # Train/evaluate on MIMIC-IV demo (contracts + metrics)

Additional modules used by scripts (repo root or codes/):
icu_contracts_v2.py       # Contract checks (data, robustness, deployment)

Results directory (final artifacts)
------------------------------------
All paper tables are consolidated under a top-level Results/ folder for easy review.

Python & OS
-----------
- Python 3.8 (tested)
- macOS/Linux/Windows (CPU only; no GPU required)

Quick Start
-----------
1) Create and activate a fresh environment:
   python3 -m venv .venv
   source .venv/bin/activate        # (Windows: .venv\Scripts\activate)

2) Install dependencies:
   pip install -r requirements.txt

3) Prepare datasets (see DATASET_DOWNLOADS.txt for details and path layout).

Data Layout (expected)
----------------------
data/
  mimic-iv-demo/                 # unzip of "mimic-iv-clinical-database-demo-2.2.zip"
    core/
    hosp/
    icu/
  eicu-synth/                    # created by eICUDatagen.py  (patient.csv, lab.csv)
  eicu-crd/                      # (optional) real eICU CSV export, same table names

SEEDS AND DETERMINISM
---------------------
- All scripts set numpy/sklearn seeds (random_state) for repeatability.
- Small numeric differences may occur across OS/BLAS; they won’t change pass/fail.

===================================
A) MIMIC-IV DEMO – RUNS AND TABLES
===================================

(1) Build cohort (optional; some scripts can read from raw demo directly):
    python codes/real_mimic_loader.py


(2) End-to-end contracts + metrics:
    python codes/train_run_mimic_demo.py 


    Outputs:
      - prints Study Report (MAE/AUC/ECE, contract pass/fail)
      - saves intermediate CSVs under outputs/mimic/

(3) Reproduce Tables 1–3 (MIMIC-IV):
    mkdir -p outputs/mimic/tables

    # Table 1 – Regression MAE + 80% interval coverage
    python codes/MIMICtable1.py


    # Table 2 – Classifier calibration & alert budget
    python codes/MIMICtable2.py



    # Table 3 – System-level contracts (fairness, temporal consistency, occupancy)
    python codes/MIMICtable3.py

      


===========================================
B) eICU – SYNTHETIC DATA + RUNS AND TABLES
===========================================

(1) Generate synthetic eICU CSVs (no credentials needed):
    python codes/eICUDatagen.py --out_dir data/eicu-synth --n_stays 1200 --seed 7

    This creates data/eicu-synth/patient.csv and lab.csv with realistic ranges.

(2) End-to-end demo with contracts:
    python codes/eICUDemo.py 

    Outputs:
      - prints Study Report (MAE/AUC/ECE, contract pass/fail)
      - saves intermediate CSVs under outputs/eicu/

(3) Reproduce Tables 1–3 (eICU):
    mkdir -p outputs/eicu/tables

    python codes/eICU_tables.py 

    Outputs:
      - outputs/eicu/tables/table1_eicu.csv  
      - outputs/eicu/tables/table2_eicu.csv  
      - outputs/eicu/tables/table3_eicu.csv  

OPTIONAL: Run eICU on a real export
-----------------------------------
If you have eICU-CRD 2.0 CSVs, place them under data/eicu-crd/ with the same table names
used in the scripts (e.g., patient.csv, lab.csv). Then run:
    python codes/eICUDemo.py --eicu_root data/eicu-crd --out_dir outputs/eicu_real
    python codes/eICU_tables.py --eicu_root data/eicu-crd --out_csv_dir outputs/eicu_real/tables --emit_tex


Troubleshooting
---------------
- Missing columns: confirm your CSV export matches expected table/column names.
- Very small cohorts: coverage/ECE contracts may be sensitive; scripts will print
  the chosen target/tolerance and any relaxed settings if applied.
- Contract violations are expected in “buggy”/demo modes (that’s the point!);
  “fixed” modes and table scripts enforce or report compliance.

Contact / Citation
------------------
Repository: https://github.com/shibbirtanvin/MedContract  (cite as \cite{medcontract})
