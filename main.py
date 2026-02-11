import re
from pathlib import Path

import numpy as np
import pandas as pd

# ====== CONFIG ======
INPUT_PATH = "data/SRM_assignment_survey_responses.csv"  
OUTPUT_DIR = Path("cleaned_data")  
# ====================

# Ensure output directory exists before any writes
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_pathway(x: str) -> str:
    """Map messy pathway labels to 'JC' or 'Poly'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()

    # common variants
    if "jc" in s or "junior" in s:
        return "JC"
    if "poly" in s or "polytechnic" in s:
        return "Poly"
    return np.nan

def parse_number(x):
    """
    Parse numeric inputs safely.
    Accepts integers/decimals; rejects ranges like '6-7' by returning NaN.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # reject ranges like "6-7"
    if re.search(r"\d+\s*-\s*\d+", s):
        return np.nan

    # keep first numeric token (handles "3 hours" etc)
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    return float(m.group(1))

def pick_column(df, candidates):
    """Pick the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ====== LOAD ======
df_raw = pd.read_csv(INPUT_PATH)

# ====== COLUMN MAPPING (adjust only if your headers differ) ======
col_pathway = pick_column(df_raw, [
    "Are you from JC or Poly?",
    "Are you from JC or Poly",
    "JC or Poly",
    "Pathway",
])

col_daily_normal = pick_column(df_raw, [
    "On Average, how many hours do you study per day outside of school (number only)",
    "On Average, how many hours do you study per day outside of school",
    "Study hours per day outside school (normal week)",
    "StudyHours_Normal",
])

col_daily_exam = pick_column(df_raw, [
    "On Average, how many hours do you study per day outside of school during exam week (number only)",
    "On Average, how many hours do you study per day outside of school during exam week",
    "Study hours per day outside school (exam week)",
    "StudyHours_Exam",
])

col_stress = pick_column(df_raw, [
    "On a scale of 1-10, how stressed are you?",
    "On a scale of 1-10, how stressed are you",
    "Stress level",
    "Stress",
])

col_stress_reason = pick_column(df_raw, [
    "Why did you choose that stress level?",
    "Why did you choose that stress level",
    "Stress reason",
    "Reason",
])

missing = [("pathway", col_pathway), ("daily_normal", col_daily_normal)]
if any(v is None for _, v in missing):
    raise ValueError(
        "Could not find required columns. "
        f"Found columns: {list(df_raw.columns)}\n"
        f"Missing mappings: {[(k,v) for k,v in missing if v is None]}"
    )

# ====== BUILD CLEAN DF ======
df = df_raw.copy()

# Keep a simple respondent id (row index + 1)
df["RespondentID"] = np.arange(1, len(df) + 1)

# Normalize pathway
df["Pathway"] = df[col_pathway].apply(normalize_pathway)

# Parse daily normal study hours
df["StudyHours_Daily_Normal"] = df[col_daily_normal].apply(parse_number)

# Optional: parse exam-week daily hours (not enough responses in your description, but keep it)
if col_daily_exam is not None:
    df["StudyHours_Daily_Exam"] = df[col_daily_exam].apply(parse_number)
else:
    df["StudyHours_Daily_Exam"] = np.nan

# Stress (1–10)
if col_stress is not None:
    df["StressLevel"] = df[col_stress].apply(parse_number)
else:
    df["StressLevel"] = np.nan

# Stress reason (text)
if col_stress_reason is not None:
    df["StressReason_Raw"] = df[col_stress_reason].astype(str)
else:
    df["StressReason_Raw"] = ""

# Convert daily->weekly for the main variable used in tests
df["StudyHours_Weekly_Normal"] = df["StudyHours_Daily_Normal"] * 7

# ====== BASIC DATA QUALITY FLAGS ======
# You can tune these thresholds.
df["Flag_MissingPathway"] = df["Pathway"].isna()
df["Flag_MissingDaily"] = df["StudyHours_Daily_Normal"].isna()

# plausible range (outside school): 0 to 12 hours/day
df["Flag_OutlierDaily"] = (df["StudyHours_Daily_Normal"] < 0) | (df["StudyHours_Daily_Normal"] > 12)

# optional: mark very high but plausible values (e.g., >8)
df["Flag_VeryHighDaily"] = df["StudyHours_Daily_Normal"] > 8

# ====== FILTER TO ANALYSIS-READY ROWS ======
# For hypothesis testing, we usually require:
# - Pathway in {JC, Poly}
# - Daily normal is not missing and not out-of-range
df_analysis = df[
    df["Pathway"].isin(["JC", "Poly"])
    & df["StudyHours_Daily_Normal"].notna()
    & ~df["Flag_OutlierDaily"]
].copy()

# Split datasets for tests
df_poly = df_analysis[df_analysis["Pathway"] == "Poly"].copy()
df_jc = df_analysis[df_analysis["Pathway"] == "JC"].copy()

# ====== QUICK CHECK OUTPUTS ======
summary_lines = [
    "=== Row counts ===",
    f"Raw rows: {len(df_raw)}",
    f"Analysis-ready rows (JC/Poly + valid daily normal): {len(df_analysis)}",
    f"Poly n: {len(df_poly)}",
    f"JC n: {len(df_jc)}",
    "",
    "=== Descriptive stats (weekly normal) ===",
    f"Poly weekly mean: {df_poly['StudyHours_Weekly_Normal'].mean()}",
    f"Poly weekly sd  : {df_poly['StudyHours_Weekly_Normal'].std(ddof=1)}",
    f"JC weekly mean  : {df_jc['StudyHours_Weekly_Normal'].mean()}",
    f"JC weekly sd    : {df_jc['StudyHours_Weekly_Normal'].std(ddof=1)}",
]
summary_text = "\n".join(summary_lines)

print(summary_text)

# ====== EXPORT CLEAN FILES ======
cols_export = [
    "RespondentID",
    "Pathway",
    "StudyHours_Daily_Normal",
    "StudyHours_Weekly_Normal",
    "StudyHours_Daily_Exam",
    "StressLevel",
    "StressReason_Raw",
    "Flag_MissingPathway",
    "Flag_MissingDaily",
    "Flag_OutlierDaily",
    "Flag_VeryHighDaily",
]

clean_all_path = OUTPUT_DIR / "clean_all_analysis_ready.csv"
clean_poly_path = OUTPUT_DIR / "clean_poly_one_sample.csv"
clean_two_sample_path = OUTPUT_DIR / "clean_jc_vs_poly_two_sample.csv"
summary_path = OUTPUT_DIR / "summary_stats.txt"

df_analysis[cols_export].to_csv(clean_all_path, index=False)
df_poly[cols_export].to_csv(clean_poly_path, index=False)
df_analysis[df_analysis["Pathway"].isin(["JC", "Poly"])][cols_export].to_csv(clean_two_sample_path, index=False)
summary_path.write_text(summary_text + "\n", encoding="utf-8")

print("\nSaved:")
print("-", str(clean_all_path))
print("-", str(clean_poly_path))
print("-", str(clean_two_sample_path))
print("-", str(summary_path))
