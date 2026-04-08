"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    df = pd.read_csv(filepath)
    os.makedirs("output", exist_ok=True)
    with open("output/data_profile.txt", "w") as f:

        # Shape
        rows, cols = df.shape
        f.write(f"Shape: {rows} rows, {cols} columns\n\n")

        # Data Types
        f.write("Data Types:\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].dtype}\n")
        f.write("\n")

        # Missing Values
        f.write("Missing Values:\n")
        total_rows = len(df)
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            f.write(f"{col}: {missing_count} missing ({missing_pct:.2f}%)\n")
        f.write("\n")

        # Descriptive Statistics
        f.write("Descriptive Statistics:\n")
        f.write(str(df.describe()))
        f.write("\n")

    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    os.makedirs("output", exist_ok=True)
    numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct"]
    for col in numeric_cols:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col.replace('_',' ').title()}")
        plt.xlabel(col.replace('_',' ').title())
        plt.ylabel("Frequency")
        plt.savefig(f"output/{col}_distribution.png")
        plt.close()
    
    plt.figure(figsize=(8,5))
    sns.boxplot(x="department", y="gpa", data=df)
    plt.title("GPA by Department")
    plt.xlabel("Department")
    plt.ylabel("GPA")
    plt.savefig("output/gpa_by_department.png")
    plt.close()

    plt.figure(figsize=(10,5))
    sns.countplot(x="scholarship", data=df)
    plt.title("Scholarship Distribution")
    plt.xlabel("Scholarship Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.savefig("output/scholarship_distribution.png")
    plt.close()



def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """
    os.makedirs("output", exist_ok=True)
    numeric_cols = ["course_load", "study_hours_weekly", "gpa", "attendance_pct", "commute_minutes"]
    corr_matrix = df[numeric_cols].corr(method="pearson")
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Numeric Variables")
    plt.savefig("output/correlation_heatmap.png")
    plt.close()
    
    corr_matrix_values = corr_matrix.copy().to_numpy().copy()
    np.fill_diagonal(corr_matrix_values, 0)
    corr_matrix_values = pd.DataFrame(corr_matrix_values, index=corr_matrix.index, columns=corr_matrix.columns)

    max_corr = corr_matrix_values.abs().unstack().sort_values(ascending=False).drop_duplicates()
    top_pair = max_corr.index[0]
    x_col, y_col = top_pair

    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f"Scatter plot: {x_col} vs {y_col}")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.savefig(f"output/scatter_{x_col}_vs_{y_col}.png")
    plt.close()




def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_anova',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests to consider:
        - t-test: Does GPA differ between students with and without internships?
        - ANOVA: Does GPA differ across departments?
        - Correlation test: Is the correlation between study hours and GPA significant?
    """
    results = {}

    interns_gpa = df[df['has_internship'] == 'Yes']['gpa']
    no_interns_gpa = df[df['has_internship'] == 'No']['gpa']

    t_stat, p_value = stats.ttest_ind(interns_gpa, no_interns_gpa, equal_var=False)
    n1, n2 = len(interns_gpa), len(no_interns_gpa)
    s1, s2 = interns_gpa.std(), no_interns_gpa.std()
    pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    cohens_d = (interns_gpa.mean() - no_interns_gpa.mean()) / pooled_sd

    results['internship_ttest'] = {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }

    print("Hypothesis 1: Internship vs GPA")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}, Cohen's d = {cohens_d:.3f}")
    if p_value < 0.05:
        print("Interpretation: Significant GPA difference between interns and non-interns.\n")
    else:
        print("Interpretation: No significant GPA difference.\n")

    contingency_table = pd.crosstab(df['scholarship'], df['department'])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    results['scholarship_chi2'] = {
        'chi2_stat': chi2_stat,
        'p_value': p_val,
        'dof': dof
    }

    print("Hypothesis 2: Scholarship vs Department")
    print(f"Chi2 = {chi2_stat:.3f}, p-value = {p_val:.4f}, dof = {dof}")
    if p_val < 0.05:
        print("Interpretation: Scholarship status is associated with department.\n")
    else:
        print("Interpretation: No significant association.\n")

    return results


def advanced_anova_analysis(df):
    """Perform ANOVA across departments and create violin plot."""
    import itertools
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # ANOVA test
    dept_groups = [df[df['department'] == dept]['gpa'] for dept in df['department'].unique()]
    f_stat, p_val = stats.f_oneway(*dept_groups)

    print("ANOVA Test: GPA across Departments")
    print(f"F-statistic = {f_stat:.3f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("Interpretation: Significant differences in GPA between departments.\n")
    else:
        print("Interpretation: No significant differences.\n")

    # Post-hoc pairwise t-tests if ANOVA significant
    if p_val < 0.05:
        depts = df['department'].unique()
        print("Post-hoc pairwise t-tests with Bonferroni correction:")
        pairs = list(itertools.combinations(depts, 2))
        for a, b in pairs:
            group_a = df[df['department'] == a]['gpa']
            group_b = df[df['department'] == b]['gpa']
            t_stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
            # Bonferroni correction: multiply p-value by number of comparisons
            p_adj = min(p * len(pairs), 1.0)
            print(f"{a} vs {b}: t = {t_stat:.3f}, raw p = {p:.4f}, adjusted p = {p_adj:.4f}")

    # Violin plot
    plt.figure(figsize=(10,6))
    sns.violinplot(x='department', y='gpa', data=df, inner='quartile', palette='Pastel1')
    plt.title("Violin Plot: GPA by Department")
    plt.xlabel("Department")
    plt.ylabel("GPA")
    plt.savefig("output/gpa_violin_by_department.png")
    plt.close()
    print("Violin plot saved: output/gpa_violin_by_department.png\n")






def write_findings(df, results):
    """Generate FINDINGS.md automatically from EDA results and hypothesis tests."""
    with open("output/FINDINGS.md", "w") as f:
        # 1. Dataset description
        rows, cols = df.shape
        f.write("# FINDINGS — Student Performance EDA\n\n")
        f.write("## 1. Dataset Description\n")
        f.write(f"- **Rows and Columns:** {rows} × {cols}\n")
        f.write(f"- **Columns:** {', '.join(df.columns)}\n")
        
        missing_info = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info.append(f"{col}: {missing_count} missing")
        if missing_info:
            f.write(f"- **Data Quality Issues:** {', '.join(missing_info)}\n")
        else:
            f.write("- **Data Quality Issues:** None\n")
        
        f.write("\n")

        # 2. Key Distribution Findings
        f.write("## 2. Key Distribution Findings\n")
        numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct"]
        for col in numeric_cols:
            skew = df[col].skew()
            f.write(f"- **{col.replace('_',' ').title()}:** skew = {skew:.2f}, see `output/{col}_distribution.png`\n")
        f.write("- **GPA by Department:** see `output/gpa_by_department.png`\n")
        f.write("- **Scholarship Distribution:** see `output/scholarship_distribution.png`\n\n")

        # 3. Notable Correlations
        f.write("## 3. Notable Correlations\n")
        numeric_cols_corr = ["course_load", "study_hours_weekly", "gpa", "attendance_pct", "commute_minutes"]
        corr_matrix = df[numeric_cols_corr].corr()
        corr_matrix_values = corr_matrix.copy().to_numpy().copy()
        np.fill_diagonal(corr_matrix_values, 0)
        corr_matrix_values = pd.DataFrame(corr_matrix_values, index=corr_matrix.index, columns=corr_matrix.columns)
        max_corr_pair = corr_matrix_values.abs().unstack().sort_values(ascending=False).drop_duplicates().index[0]
        r_value = corr_matrix.loc[max_corr_pair[0], max_corr_pair[1]]
        f.write(f"- Most correlated pair: {max_corr_pair[0]} vs {max_corr_pair[1]}, r = {r_value:.2f}, see `output/scatter_{max_corr_pair[0]}_vs_{max_corr_pair[1]}.png`\n")
        f.write("- Note: Correlation does not imply causation.\n\n")

        # 4. Hypothesis Test Results
        f.write("## 4. Hypothesis Test Results\n")
        # Hypothesis 1: internship
        t = results['internship_ttest']['t_stat']
        p = results['internship_ttest']['p_value']
        d = results['internship_ttest']['cohens_d']
        f.write("### Hypothesis 1: Internship vs GPA\n")
        f.write("- Test: Independent t-test\n")
        f.write(f"- t-statistic: {t:.3f}, p-value: {p:.4f}, Cohen's d: {d:.3f}\n")
        interp = "Significant difference" if p < 0.05 else "No significant difference"
        f.write(f"- Interpretation: {interp}\n\n")

        # Hypothesis 2: scholarship vs department
        chi2 = results['scholarship_chi2']['chi2_stat']
        p_val = results['scholarship_chi2']['p_value']
        dof = results['scholarship_chi2']['dof']
        f.write("### Hypothesis 2: Scholarship vs Department\n")
        f.write("- Test: Chi-square test\n")
        f.write(f"- Chi2 = {chi2:.3f}, p-value = {p_val:.4f}, dof = {dof}\n")
        interp2 = "Associated" if p_val < 0.05 else "No significant association"
        f.write(f"- Interpretation: Scholarship status is {interp2} with department.\n\n")

        # 5. Recommendations
        f.write("## 5. Recommendations\n")
        f.write("1. Encourage internships to improve student GPA.\n")
        f.write("2. Ensure fair scholarship distribution across departments.\n")
        f.write("3. Provide additional study support for students with low weekly study hours.\n\n")

        # References to charts
        f.write("## References to Saved Charts\n")
        f.write("- GPA by Department: `output/gpa_by_department.png`\n")
        f.write("- GPA Violin Plot by Department: `output/gpa_violin_by_department.png`\n")
        for col in numeric_cols:
            f.write(f"- {col.replace('_',' ').title()} Distribution: `output/{col}_distribution.png`\n")
        f.write(f"- Correlation Heatmap: `output/correlation_heatmap.png`\n")
        f.write(f"- Scatter Plot (Top Correlation): `output/scatter_{max_corr_pair[0]}_vs_{max_corr_pair[1]}.png`\n")


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)
    # Load and profile the dataset
    df = load_and_profile("data/student_performance.csv")
    plot_distributions(df)
    plot_correlations(df)
    results = run_hypothesis_tests(df)
    advanced_anova_analysis(df)
    write_findings(df, results)


if __name__ == "__main__":
    main()
