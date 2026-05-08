import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from PyALE import ale
import numpy as np
import optuna
import argparse

plt.rcParams["font.family"] = "Times New Roman"

def run_ale_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Loading data
    df = pd.read_excel(args.data_path)
    X = df.drop(columns=["id", "Scores"])
    y = df["Scores"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 2.  Train the model (load the optimal parameters from step 4)
    print("Training final model for ALE analysis...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. ALE 1D
    for i, feat in enumerate(X_train.columns, 1):
        print(f"Calculating ALE for {i}/{len(X_train.columns)}: {feat}")

        ale_df = ale(
            X_train,
            model,
            feature=[feat],
            include_CI=True,
            C=0.95,
            plot=False 
        )

        plt.figure(figsize=(6.5, 4.5))

        plt.plot(
            ale_df.index,
            ale_df["eff"],
            linewidth=2,
            color="#2E86C1",
            label="ALE effect"
        )

        if "lowerCI_95%" in ale_df.columns:
            plt.fill_between(
                ale_df.index,
                ale_df["lowerCI_95%"],
                ale_df["upperCI_95%"],
                color="#AED6F1",
                alpha=0.4,
                label="95% CI"
            )
        
        y_min = plt.ylim()[0]
        plt.plot(
            X_train[feat],
            np.full_like(X_train[feat], y_min),
            '|',
            color='black',
            alpha=0.2,
            markersize=10
        )

        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel(feat, fontsize=12, fontweight='bold')
        plt.ylabel("ALE Value", fontsize=12, fontweight='bold')
        plt.title(f"ALE 1D: {feat}", fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(frameon=False)
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"ALE1D_{feat}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/final_analysis.xlsx')
    parser.add_argument('--output_dir', default='./results/ALE_1D')
    args = parser.parse_args()
    run_ale_analysis(args)
