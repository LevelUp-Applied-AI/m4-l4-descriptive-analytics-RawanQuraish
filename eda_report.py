# eda_report.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDAReport:
    def __init__(self, df, output_dir="output", numeric_cols=None, plot_style="default"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.numeric_cols = numeric_cols or df.select_dtypes(include="number").columns.tolist()
        plt.style.use("default")

    def data_profile(self):
        profile = {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "missing": self.df.isnull().sum().to_dict()
        }
        with open(os.path.join(self.output_dir, "data_profile.txt"), "w") as f:
            f.write(f"Shape: {profile['shape']}\n\n")
            f.write("Data Types:\n")
            for col, dtype in profile['dtypes'].items():
                f.write(f"{col}: {dtype}\n")
            f.write("\nMissing Values:\n")
            for col, miss in profile['missing'].items():
                f.write(f"{col}: {miss}\n")
        return profile

    def plot_distributions(self):
        for col in self.numeric_cols:
            plt.figure(figsize=(8,5))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(self.output_dir, f"{col}_distribution.png"))
            plt.close()

    def plot_correlations(self):
        corr = self.df[self.numeric_cols].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()
        return corr

    def plot_missing(self):
        plt.figure(figsize=(10,6))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False)
        plt.title("Missing Data Heatmap")
        plt.savefig(os.path.join(self.output_dir, "missing_data.png"))
        plt.close()

    def summarize_outliers(self):
        outliers = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers[col] = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        with open(os.path.join(self.output_dir, "outlier_summary.txt"), "w") as f:
            for col, count in outliers.items():
                f.write(f"{col}: {count} outliers\n")
        return outliers

    def generate_report(self):
        self.data_profile()
        self.plot_distributions()
        self.plot_correlations()
        self.plot_missing()
        self.summarize_outliers()
        print(f"EDA report generated in {self.output_dir}")