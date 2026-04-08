import pandas as pd
from eda_report import EDAReport
df = pd.read_csv("data/student_performance.csv")

report = EDAReport(df, output_dir="eda_output", plot_style="default")

report.generate_report()