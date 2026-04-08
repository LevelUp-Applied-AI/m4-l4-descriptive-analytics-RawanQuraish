# FINDINGS — Student Performance EDA

## 1. Dataset Description
- **Rows and Columns:** 2000 × 10
- **Columns:** student_id, department, semester, course_load, study_hours_weekly, gpa, attendance_pct, has_internship, commute_minutes, scholarship
- **Data Quality Issues:** commute_minutes: 181 missing, scholarship: 389 missing

## 2. Key Distribution Findings
- **Gpa:** skew = -0.08, see `output/gpa_distribution.png`
- **Study Hours Weekly:** skew = -0.01, see `output/study_hours_weekly_distribution.png`
- **Attendance Pct:** skew = -0.15, see `output/attendance_pct_distribution.png`
- **GPA by Department:** see `output/gpa_by_department.png`
- **Scholarship Distribution:** see `output/scholarship_distribution.png`

## 3. Notable Correlations
- Most correlated pair: study_hours_weekly vs gpa, r = 0.64, see `output/scatter_study_hours_weekly_vs_gpa.png`
- Note: Correlation does not imply causation.

## 4. Hypothesis Test Results
### Hypothesis 1: Internship vs GPA
- Test: Independent t-test
- t-statistic: 14.229, p-value: 0.0000, Cohen's d: 0.690
- Interpretation: Significant difference

### Hypothesis 2: Scholarship vs Department
- Test: Chi-square test
- Chi2 = 13.949, p-value = 0.3040, dof = 12
- Interpretation: Scholarship status is No significant association with department.

## 5. Recommendations
1. Encourage internships to improve student GPA.
2. Ensure fair scholarship distribution across departments.
3. Provide additional study support for students with low weekly study hours.

## References to Saved Charts
- GPA by Department: `output/gpa_by_department.png`
- Gpa Distribution: `output/gpa_distribution.png`
- Study Hours Weekly Distribution: `output/study_hours_weekly_distribution.png`
- Attendance Pct Distribution: `output/attendance_pct_distribution.png`
- Correlation Heatmap: `output/correlation_heatmap.png`
- Scatter Plot (Top Correlation): `output/scatter_study_hours_weekly_vs_gpa.png`
