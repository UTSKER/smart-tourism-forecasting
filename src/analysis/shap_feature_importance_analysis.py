# ============================================================
# SHAP FEATURE IMPORTANCE ANALYSIS
# For Tourism Forecasting Dataset
# ============================================================

# ---------------- IMPORT LIBRARIES ----------------
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Tirumala_Tirupati_Devasthanam.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "shap"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=["date"])

    target_col = "darshans"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.select_dtypes(include=["number"])

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_feature_importance_bar.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_feature_importance_summary.png", bbox_inches="tight")
    plt.close()

    print(f"SHAP feature importance analysis completed. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
