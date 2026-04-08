"""
Tier 3 — Statistical Simulation and Power Analysis

This script performs:
1. Bootstrap confidence intervals for mean GPA by internship status
2. Comparison with parametric t-test confidence intervals
3. Power analysis to determine sample size for 80% power
4. Simulation of false positive rate with synthetic data
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

# ---------------------------
# Step 1: Load dataset
# ---------------------------
df = pd.read_csv("data/student_performance.csv")

# Separate groups by internship status
intern_gpa = df[df['has_internship']=='Yes']['gpa'].values
no_intern_gpa = df[df['has_internship']=='No']['gpa'].values

# ---------------------------
# Step 2: Bootstrap 95% CI
# ---------------------------
def bootstrap_ci(data, n_bootstrap=10000, alpha=0.05):
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper

ci_intern = bootstrap_ci(intern_gpa)
ci_no_intern = bootstrap_ci(no_intern_gpa)
print(f"Bootstrap 95% CI GPA interns: {ci_intern}")
print(f"Bootstrap 95% CI GPA non-interns: {ci_no_intern}")

# ---------------------------
# Step 3: Parametric t-test CI
# ---------------------------
def ttest_ci(data, alpha=0.05):
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf(1 - alpha/2, df=len(data)-1)
    return mean - margin, mean + margin

ci_intern_t = ttest_ci(intern_gpa)
ci_no_intern_t = ttest_ci(no_intern_gpa)
print(f"T-test 95% CI GPA interns: {ci_intern_t}")
print(f"T-test 95% CI GPA non-interns: {ci_no_intern_t}")

# ---------------------------
# Step 4: Power analysis
# ---------------------------
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sdx, sdy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx-1)*sdx**2 + (ny-1)*sdy**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled

effect_size = cohens_d(intern_gpa, no_intern_gpa)
analysis = TTestIndPower()
sample_size_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f"Sample size needed for 80% power: {int(sample_size_needed)} per group")

# ---------------------------
# Step 5: Simulation of false positive rate
# ---------------------------
n_sim = 1000
alpha = 0.05
false_positives = 0

for _ in range(n_sim):
    # Generate synthetic data with no real difference
    group1 = np.random.normal(loc=3.0, scale=0.5, size=50)
    group2 = np.random.normal(loc=3.0, scale=0.5, size=50)
    
    t_stat, p_val = stats.ttest_ind(group1, group2)
    if p_val < alpha:
        false_positives += 1

fpr = false_positives / n_sim
print(f"False positive rate: {fpr:.3f} (should be close to {alpha})")