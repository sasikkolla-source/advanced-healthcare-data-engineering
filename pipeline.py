"""
Advanced Healthcare Data Engineering - Base Project
===================================================
End-to-end ETL pipeline using the medallion pattern:

    generate -> RAW (csv) -> CLEAN (validated, de-identified) -> WAREHOUSE (SQLite star schema)
                                                                      |
                                                              ANALYTICS reports

Run:  pip install pandas numpy
      python pipeline.py
"""
import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = Path(__file__).parent
RAW_DIR = BASE / "data" / "raw"
CLEAN_DIR = BASE / "data" / "clean"
DB_PATH = BASE / "data" / "warehouse.db"
REPORT_DIR = BASE / "reports"
SALT = "change-me-in-production"  # keep in a secrets manager in real systems
SEED = 42
N_PATIENTS, N_ENCOUNTERS = 500, 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("healthcare-etl")

for d in (RAW_DIR, CLEAN_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. EXTRACT: generate synthetic, intentionally messy source data
# ----------------------------------------------------------------------------
def generate_raw_data() -> None:
    rng = np.random.default_rng(SEED)

    patients = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(1, N_PATIENTS + 1)],
        "full_name": [f"Patient {i}" for i in range(1, N_PATIENTS + 1)],
        "dob": pd.to_datetime("1940-01-01")
        + pd.to_timedelta(rng.integers(0, 365 * 75, N_PATIENTS), unit="D"),
        "gender": rng.choice(["M", "F", "male", "female", "Other", None], N_PATIENTS,
                             p=[.3, .3, .15, .15, .05, .05]),
        "phone": [f"+91-9{rng.integers(100000000, 999999999)}" for _ in range(N_PATIENTS)],
        "city": rng.choice(["Hyderabad", "Mumbai", "Delhi", "Chennai", "Pune"], N_PATIENTS),
        "height_cm": rng.normal(165, 10, N_PATIENTS).round(1),
        "weight_kg": rng.normal(70, 15, N_PATIENTS).round(1),
    })
    # inject dirt: duplicates + impossible values
    patients = pd.concat([patients, patients.sample(10, random_state=SEED)])
    patients.loc[patients.sample(5, random_state=1).index, "weight_kg"] = -5

    diagnoses = {
        "E11": "Type 2 Diabetes", "I10": "Hypertension", "J45": "Asthma",
        "J18": "Pneumonia", "N18": "Chronic Kidney Disease", "Z00": "General Checkup",
    }
    admit = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, N_ENCOUNTERS), unit="D")
    los = rng.integers(0, 15, N_ENCOUNTERS)
    encounters = pd.DataFrame({
        "encounter_id": [f"E{i:06d}" for i in range(1, N_ENCOUNTERS + 1)],
        "patient_id": rng.choice(patients["patient_id"].unique(), N_ENCOUNTERS),
        "admit_date": admit,
        "discharge_date": admit + pd.to_timedelta(los, unit="D"),
        "encounter_type": rng.choice(["Inpatient", "Outpatient", "Emergency"], N_ENCOUNTERS, p=[.3, .5, .2]),
        "diagnosis_code": rng.choice(list(diagnoses), N_ENCOUNTERS),
        "total_cost": rng.gamma(2.0, 1500, N_ENCOUNTERS).round(2),
        "readmitted_30d": rng.choice([0, 1], N_ENCOUNTERS, p=[.88, .12]),
    })
    encounters["diagnosis_desc"] = encounters["diagnosis_code"].map(diagnoses)
    encounters.loc[encounters.sample(8, random_state=2).index, "discharge_date"] = (
        encounters["admit_date"] - pd.Timedelta(days=2)  # discharge before admit = bad
    )

    labs = pd.DataFrame({
        "lab_id": range(1, 3001),
        "encounter_id": rng.choice(encounters["encounter_id"], 3000),
        "test_name": rng.choice(["HbA1c", "Glucose", "Creatinine", "Systolic BP"], 3000),
    })
    ranges = {"HbA1c": (6.0, 1.5), "Glucose": (110, 35), "Creatinine": (1.0, 0.4), "Systolic BP": (128, 18)}
    labs["result_value"] = [round(rng.normal(*ranges[t]), 2) for t in labs["test_name"]]

    patients.to_csv(RAW_DIR / "patients.csv", index=False)
    encounters.to_csv(RAW_DIR / "encounters.csv", index=False)
    labs.to_csv(RAW_DIR / "labs.csv", index=False)
    log.info("RAW layer written: %d patients, %d encounters, %d labs",
             len(patients), len(encounters), len(labs))


# ----------------------------------------------------------------------------
# 2. TRANSFORM: clean, validate, de-identify, enrich
# ----------------------------------------------------------------------------
def pseudonymize(value: str) -> str:
    return hashlib.sha256(f"{SALT}{value}".encode()).hexdigest()[:16]


def transform_patients(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset="patient_id").copy()
    df["gender"] = (df["gender"].str.strip().str.lower()
                    .map({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
                    .fillna("Other/Unknown"))
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((pd.Timestamp("2024-12-31") - df["dob"]).dt.days // 365).astype(int)
    df["age_group"] = pd.cut(df["age"], [-1, 17, 39, 59, 79, 200],
                             labels=["0-17", "18-39", "40-59", "60-79", "80+"])
    # invalid physiological values -> NaN rather than silently keep
    df.loc[~df["weight_kg"].between(2, 300), "weight_kg"] = np.nan
    df.loc[~df["height_cm"].between(40, 250), "height_cm"] = np.nan
    df["bmi"] = (df["weight_kg"] / (df["height_cm"] / 100) ** 2).round(1)
    # PII handling (HIPAA-style de-identification)
    df["patient_key"] = df["patient_id"].map(pseudonymize)
    df = df.drop(columns=["full_name", "phone", "dob", "patient_id"]).assign(
        patient_id_src=df["patient_id"])
    log.info("patients: removed %d duplicates, PII dropped/hashed", n0 - len(df))
    return df


def transform_encounters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["admit_date"] = pd.to_datetime(df["admit_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    bad = df["discharge_date"] < df["admit_date"]
    log.info("encounters: quarantined %d rows with discharge < admit", bad.sum())
    df[bad].to_csv(CLEAN_DIR / "quarantine_encounters.csv", index=False)
    df = df[~bad].copy()
    df["length_of_stay"] = (df["discharge_date"] - df["admit_date"]).dt.days
    df["patient_key"] = df["patient_id"].map(pseudonymize)
    return df.drop(columns=["patient_id"])


def transform_labs(df: pd.DataFrame) -> pd.DataFrame:
    ref = {  # (low, high) reference ranges
        "HbA1c": (4.0, 5.7), "Glucose": (70, 100),
        "Creatinine": (0.6, 1.3), "Systolic BP": (90, 120),
    }
    df = df.dropna(subset=["result_value"]).copy()
    lo = df["test_name"].map(lambda t: ref[t][0])
    hi = df["test_name"].map(lambda t: ref[t][1])
    df["flag"] = np.select([df["result_value"] < lo, df["result_value"] > hi],
                           ["LOW", "HIGH"], default="NORMAL")
    return df


# ----------------------------------------------------------------------------
# 3. DATA QUALITY CHECKS
# ----------------------------------------------------------------------------
def run_quality_checks(p: pd.DataFrame, e: pd.DataFrame, l: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("patients unique key", p["patient_key"].is_unique),
        ("patients no null gender", p["gender"].notna().all()),
        ("age in 0-120", p["age"].between(0, 120).all()),
        ("encounters unique id", e["encounter_id"].is_unique),
        ("LOS non-negative", (e["length_of_stay"] >= 0).all()),
        ("cost positive", (e["total_cost"] > 0).all()),
        ("encounter->patient FK", e["patient_key"].isin(p["patient_key"]).all()),
        ("lab->encounter FK", l["encounter_id"].isin(e["encounter_id"]).all()),
    ]
    result = pd.DataFrame(checks, columns=["check", "passed"])
    for _, r in result.iterrows():
        (log.info if r.passed else log.error)("DQ %-26s %s", r.check, "PASS" if r.passed else "FAIL")
    if not result["passed"].all():
        raise ValueError("Data quality checks failed - aborting load")
    return result


# ----------------------------------------------------------------------------
# 4. LOAD: star schema in SQLite
# ----------------------------------------------------------------------------
DDL = """
DROP TABLE IF EXISTS fact_lab;   DROP TABLE IF EXISTS fact_encounter;
DROP TABLE IF EXISTS dim_patient; DROP TABLE IF EXISTS dim_date;
CREATE TABLE dim_patient (
    patient_key TEXT PRIMARY KEY, gender TEXT, age INTEGER, age_group TEXT,
    city TEXT, bmi REAL);
CREATE TABLE dim_date (date_key TEXT PRIMARY KEY, year INT, quarter INT, month INT, month_name TEXT);
CREATE TABLE fact_encounter (
    encounter_id TEXT PRIMARY KEY,
    patient_key TEXT REFERENCES dim_patient(patient_key),
    admit_date_key TEXT REFERENCES dim_date(date_key),
    encounter_type TEXT, diagnosis_code TEXT, diagnosis_desc TEXT,
    length_of_stay INT, total_cost REAL, readmitted_30d INT);
CREATE TABLE fact_lab (
    lab_id INTEGER PRIMARY KEY,
    encounter_id TEXT REFERENCES fact_encounter(encounter_id),
    test_name TEXT, result_value REAL, flag TEXT);
CREATE INDEX idx_enc_patient ON fact_encounter(patient_key);
CREATE INDEX idx_enc_dx ON fact_encounter(diagnosis_code);
"""


def load_warehouse(p: pd.DataFrame, e: pd.DataFrame, l: pd.DataFrame) -> None:
    dates = pd.DataFrame({"d": pd.date_range("2024-01-01", "2025-12-31")})
    dim_date = pd.DataFrame({
        "date_key": dates.d.dt.strftime("%Y-%m-%d"), "year": dates.d.dt.year,
        "quarter": dates.d.dt.quarter, "month": dates.d.dt.month,
        "month_name": dates.d.dt.month_name()})

    fact_enc = e.assign(admit_date_key=e["admit_date"].dt.strftime("%Y-%m-%d"))[
        ["encounter_id", "patient_key", "admit_date_key", "encounter_type", "diagnosis_code",
         "diagnosis_desc", "length_of_stay", "total_cost", "readmitted_30d"]]
    dim_pat = p[["patient_key", "gender", "age", "age_group", "city", "bmi"]].astype({"age_group": str})

    with sqlite3.connect(DB_PATH) as con:
        con.executescript(DDL)
        dim_pat.to_sql("dim_patient", con, if_exists="append", index=False)
        dim_date.to_sql("dim_date", con, if_exists="append", index=False)
        fact_enc.to_sql("fact_encounter", con, if_exists="append", index=False)
        l[["lab_id", "encounter_id", "test_name", "result_value", "flag"]].to_sql(
            "fact_lab", con, if_exists="append", index=False)
    log.info("Warehouse loaded -> %s", DB_PATH)


# ----------------------------------------------------------------------------
# 5. ANALYTICS
# ----------------------------------------------------------------------------
QUERIES = {
    "monthly_volume_and_cost": """
        SELECT d.year, d.month, COUNT(*) AS encounters,
               ROUND(SUM(f.total_cost),2) AS total_cost, ROUND(AVG(f.length_of_stay),2) AS avg_los
        FROM fact_encounter f JOIN dim_date d ON d.date_key = f.admit_date_key
        GROUP BY d.year, d.month ORDER BY d.year, d.month""",
    "readmission_by_diagnosis": """
        SELECT diagnosis_desc, COUNT(*) AS encounters,
               ROUND(100.0*AVG(readmitted_30d),1) AS readmission_rate_pct,
               ROUND(AVG(total_cost),2) AS avg_cost
        FROM fact_encounter GROUP BY diagnosis_desc ORDER BY readmission_rate_pct DESC""",
    "cost_by_age_group_gender": """
        SELECT p.age_group, p.gender, COUNT(*) AS encounters, ROUND(AVG(f.total_cost),2) AS avg_cost
        FROM fact_encounter f JOIN dim_patient p USING(patient_key)
        GROUP BY p.age_group, p.gender ORDER BY p.age_group, p.gender""",
    "abnormal_lab_rate": """
        SELECT test_name, COUNT(*) AS tests,
               ROUND(100.0*SUM(flag!='NORMAL')/COUNT(*),1) AS abnormal_pct
        FROM fact_lab GROUP BY test_name ORDER BY abnormal_pct DESC""",
    "high_utilizers_top10": """
        SELECT patient_key, COUNT(*) AS visits, ROUND(SUM(total_cost),2) AS total_cost
        FROM fact_encounter GROUP BY patient_key ORDER BY visits DESC, total_cost DESC LIMIT 10""",
}


def run_analytics() -> None:
    with sqlite3.connect(DB_PATH) as con:
        for name, sql in QUERIES.items():
            df = pd.read_sql(sql, con)
            df.to_csv(REPORT_DIR / f"{name}.csv", index=False)
            print(f"\n=== {name} ===")
            print(df.head(8).to_string(index=False))


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def main() -> None:
    log.info("Pipeline started")
    generate_raw_data()                                           # Extract

    p = transform_patients(pd.read_csv(RAW_DIR / "patients.csv"))  # Transform
    e = transform_encounters(pd.read_csv(RAW_DIR / "encounters.csv"))
    l = transform_labs(pd.read_csv(RAW_DIR / "labs.csv"))
    # keep referential integrity after quarantining
    l = l[l["encounter_id"].isin(e["encounter_id"])]
    p.to_csv(CLEAN_DIR / "patients_clean.csv", index=False)
    e.to_csv(CLEAN_DIR / "encounters_clean.csv", index=False)
    l.to_csv(CLEAN_DIR / "labs_clean.csv", index=False)

    run_quality_checks(p, e, l).to_csv(REPORT_DIR / "dq_report.csv", index=False)
    load_warehouse(p, e, l)                                        # Load
    run_analytics()                                                # Serve
    log.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
