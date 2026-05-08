import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import optuna
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import argparse

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2

def run_shap_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

  # 1. Model training
    df = pd.read_excel(args.data_path)
    X = df.drop(columns=["id", "Scores"])
    y = df["Scores"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4 has already obtained the optimal parameters via Optuna; for the sake of simplicity, we will proceed directly to training here.
    # In practice, you can load the best model saved in step 4
    print("Training the finalized Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train)

    # 2. Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # 3. Beeswarm Plot
    print("Generating Beeswarm plot...")
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("Global Feature Importance (Beeswarm)", fontsize=14, weight='bold')
    plt.savefig(os.path.join(args.output_dir, "shap_beeswarm_global.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Dependence Plot
    LOWESS_FRAC = 0.30
    for fname in X_train.columns:
        print(f"Processing Dependence Plot for: {fname}")
        x = X_train[fname].values
        y_shap = shap_values[:, fname].values
        order = np.argsort(x)
        x_sorted, y_sorted = x[order], y_shap[order]

        lowess_fit = lowess(y_sorted, x_sorted, frac=LOWESS_FRAC, return_sorted=True)
        X_lin = sm.add_constant(x_sorted)
        lin_model = sm.OLS(y_sorted, X_lin).fit()
        ci = lin_model.get_prediction(X_lin).conf_int(alpha=0.05)
        
        x_low, y_low = lowess_fit[:, 0], lowess_fit[:, 1]
        cross_idx = np.where(np.sign(y_low[:-1]) * np.sign(y_low[1:]) < 0)[0]
        x_intersections = [x_low[i] - y_low[i] * (x_low[i+1] - x_low[i]) / (y_low[i+1] - y_low[i]) for i in cross_idx]

        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(x, y_shap, s=15, color="#1f4e79", alpha=0.6, label="SHAP values")
        ax.plot(x_sorted, lin_model.predict(X_lin), color="#C44E52", label="Linear fit")
        ax.plot(x_low, y_low, color="#4CAF50", linestyle="--", label="LOWESS smooth")
        
        for xi in x_intersections:
            ax.axvline(x=xi, color="#1f4e79", linestyle=":", alpha=0.7)
            ax.text(xi, ax.get_ylim()[0]*0.9, f"{xi:.2f}", ha='center', bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel(fname, fontsize=12)
        ax.set_ylabel("SHAP Value (Impact on PLVQ)", fontsize=12)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"dependence_{fname}.png"), dpi=300)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/final_analysis.xlsx')
    parser.add_argument('--output_dir', default='./results/shap_analysis')
    args = parser.parse_args()
    run_shap_analysis(args)
