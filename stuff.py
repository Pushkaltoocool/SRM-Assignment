import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ======================
# 1) LOAD
# ======================
BASE_DIR = Path(__file__).parent
INPUT_CANDIDATES = [
    Path("data/SRM_assignment_survey_responses_final.csv"),
    Path("data/SRM_assignment_survey_responses.csv"),
]
INPUT_PATH = next((p for p in INPUT_CANDIDATES if (BASE_DIR / p).exists()), None)
if INPUT_PATH is None:
    raise FileNotFoundError(f"None of the expected input files found: {INPUT_CANDIDATES}")

df_raw = pd.read_csv(BASE_DIR / INPUT_PATH)

# ======================
# 2) CLEAN / STANDARDISE
# ======================
def normalize_pathway(x):
    if pd.isna(x): 
        return np.nan
    s = str(x).strip().lower()
    if "jc" in s:
        return "JC"
    if "poly" in s:
        return "Poly"
    return np.nan

def parse_num(x):
    """Parse numeric values; keeps outliers; returns NaN if no number."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # reject ranges like "6-7" (not expected here, but safe)
    if re.search(r"\d+\s*-\s*\d+", s):
        return np.nan
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

df = df_raw.copy()
df["RespondentID"] = np.arange(1, len(df) + 1)

# columns (edit here if your Google Form headers change)
COL_PATHWAY = "Are you from JC or Poly?"
COL_DAILY = "On Average, how many hours do you study per day outside of school (number only)"
COL_STRESS = "On a scale of 1-10, how stressed are you?"
COL_REASON = "Why did you choose that stress level?"

df["Pathway"] = df[COL_PATHWAY].apply(normalize_pathway)
df["StudyHours_Daily_Normal"] = df[COL_DAILY].apply(parse_num)
df["StudyHours_Weekly_Normal"] = 7 * df["StudyHours_Daily_Normal"]
df["StressLevel"] = df[COL_STRESS].apply(parse_num)
df["StressReason"] = df[COL_REASON].astype(str)

# Keep outliers (do NOT remove). Optionally flag extremes for discussion only.
df["Flag_VeryHighDaily"] = df["StudyHours_Daily_Normal"] > 8

# Drop unnecessary columns and keep a clean, report-ready dataset
df_clean = df[[
    "RespondentID",
    "Pathway",
    "StudyHours_Daily_Normal",
    "StudyHours_Weekly_Normal",
    "StressLevel",
    "StressReason",
    "Flag_VeryHighDaily"
]].copy()

# Filter usable rows for analysis (must have pathway + normal-week study hours)
df_analysis = df_clean[
    df_clean["Pathway"].isin(["JC", "Poly"]) &
    df_clean["StudyHours_Daily_Normal"].notna()
].copy()

df_poly = df_analysis[df_analysis["Pathway"] == "Poly"].copy()
df_jc = df_analysis[df_analysis["Pathway"] == "JC"].copy()
df_two_sample = df_analysis[df_analysis["Pathway"].isin(["JC", "Poly"])].copy()

# ======================
# 3) SAVE CLEAN FILES
# ======================
OUTPUT_DIR = BASE_DIR / "cleaned_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUTPUT_DIR / "clean_presentable_all.csv"
OUT_POLY = OUTPUT_DIR / "clean_presentable_poly_only.csv"
OUT_JC_POLY = OUTPUT_DIR / "clean_presentable_jc_vs_poly.csv"

df_analysis.to_csv(OUT_ALL, index=False)
df_poly.to_csv(OUT_POLY, index=False)
df_two_sample.to_csv(OUT_JC_POLY, index=False)

print("Saved:")
print("-", str(OUT_ALL))
print("-", str(OUT_POLY))
print("-", str(OUT_JC_POLY))

# ======================
# 4) OPTIONAL: CHARTS FOR REPORT (PNG)
# ======================
# (A) Pathway distribution (bar)
counts = df_analysis["Pathway"].value_counts().reindex(["JC","Poly"])
plt.figure()
plt.bar(counts.index, counts.values)
plt.title("Respondents by Pathway (JC vs Poly)")
plt.xlabel("Pathway")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_pathway_bar.png", dpi=200)
plt.close()

# (B) Histogram of DAILY study hours (overall)
plt.figure()
plt.hist(df_analysis["StudyHours_Daily_Normal"], bins=range(0, 12), edgecolor="black", align="left")
plt.title("Daily Study Hours Outside School (Normal Week) — Overall")
plt.xlabel("Hours/day")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_hist_daily_overall.png", dpi=200)
plt.close()

# (C) Boxplot of WEEKLY study hours by group
plt.figure()
data = [
    df_jc["StudyHours_Weekly_Normal"].values,
    df_poly["StudyHours_Weekly_Normal"].values
]
plt.boxplot(data, labels=["JC","Poly"], showmeans=True)
plt.title("Weekly Study Hours Outside School (Normal Week) — JC vs Poly")
plt.ylabel("Hours/week")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_box_weekly_jc_vs_poly.png", dpi=200)
plt.close()

print("Saved charts:")
print("-", str(OUTPUT_DIR / "fig_pathway_bar.png"))
print("-", str(OUTPUT_DIR / "fig_hist_daily_overall.png"))
print("-", str(OUTPUT_DIR / "fig_box_weekly_jc_vs_poly.png"))

# ======================
# 5) OPTIONAL: RUN THE TWO TESTS (for Section 6)
# ======================
theta0 = 15.5  # benchmark mean (hours/week)

poly_weekly = df_poly["StudyHours_Weekly_Normal"]
jc_weekly = df_jc["StudyHours_Weekly_Normal"]

# One-sample t-test (two-tailed): Poly vs benchmark
t1 = (poly_weekly.mean() - theta0) / (poly_weekly.std(ddof=1) / np.sqrt(len(poly_weekly)))
p1 = 2 * (1 - stats.t.cdf(abs(t1), df=len(poly_weekly)-1))

# Welch two-sample t-test (one-tailed): JC > Poly
xj, xp = jc_weekly.mean(), poly_weekly.mean()
sj, sp = jc_weekly.std(ddof=1), poly_weekly.std(ddof=1)
nj, np_ = len(jc_weekly), len(poly_weekly)

se = np.sqrt((sj**2)/nj + (sp**2)/np_)
t2 = (xj - xp) / se
df_welch = ((sj**2)/nj + (sp**2)/np_)**2 / (((sj**2)/nj)**2/(nj-1) + ((sp**2)/np_)**2/(np_-1))
p2_one_tailed = 1 - stats.t.cdf(t2, df=df_welch)  # upper-tail

print("\n=== TEST OUTPUTS ===")
print(f"One-sample (Poly vs benchmark): t={t1:.3f}, df={len(poly_weekly)-1}, p(two-tailed)={p1:.4f}")
print(f"Welch two-sample (JC > Poly): t={t2:.3f}, df={df_welch:.2f}, p(one-tailed)={p2_one_tailed:.4f}")
