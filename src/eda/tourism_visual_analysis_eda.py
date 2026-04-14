# ============================================================
# TOURISM TIME SERIES ANALYSIS - SAVE ALL GRAPHS TO NEW FOLDER
# ============================================================

# ---------------- IMPORT LIBRARIES ----------------
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Tirumala_Tirupati_Devasthanam.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "time_series_analysis"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    df = df.asfreq("D")

    target_col = "darshans"
    df[target_col] = df[target_col].interpolate(method="linear")

    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (14, 6)

    plt.figure()
    plt.plot(df.index, df[target_col])
    plt.title("GRAPH 1: Time Series Plot - Darshans Over Time")
    plt.xlabel("Date")
    plt.ylabel("Darshans")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "graph_1_time_series_plot.png")
    plt.close()

    decomp = seasonal_decompose(
        df[target_col],
        model="additive",
        period=365,
        extrapolate_trend="freq",
    )

    windows = [
        (2014, 2017),
        (2018, 2021),
        (2022, 2025),
    ]

    for idx, (start_year, end_year) in enumerate(windows, start=1):
        mask = (
            (decomp.observed.index.year >= start_year)
            & (decomp.observed.index.year <= end_year)
        )

        obs = decomp.observed[mask]
        trend = decomp.trend[mask]
        seasonal = decomp.seasonal[mask]
        resid = decomp.resid[mask]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"GRAPH 2: Seasonal Decomposition ({start_year}-{end_year})",
            fontsize=16,
            fontweight="bold",
        )

        axes[0].plot(obs, linewidth=1.2)
        axes[0].set_title("Observed (Darshans)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(trend, linewidth=1.5)
        axes[1].set_title("Trend")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(seasonal, linewidth=1.2)
        axes[2].set_title("Seasonal")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(resid, linewidth=0.8, marker="o", markersize=2, alpha=0.7)
        axes[3].axhline(0, linestyle="--", linewidth=1)
        axes[3].set_title("Residual")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(
            OUTPUT_DIR
            / f"graph_2_seasonal_decomposition_window_{idx}_{start_year}_{end_year}.png"
        )
        plt.close()

    fig, ax = plt.subplots()
    plot_acf(df[target_col], lags=30, ax=ax)
    plt.title("GRAPH 3: Autocorrelation Function (ACF)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "graph_3_acf_plot.png")
    plt.close()

    fig, ax = plt.subplots()
    plot_pacf(df[target_col], lags=30, ax=ax)
    plt.title("GRAPH 4: Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "graph_4_pacf_plot.png")
    plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("GRAPH 5: Correlation Heatmap of All Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "graph_5_correlation_heatmap.png")
    plt.close()

    features = [col for col in df.columns if col != target_col]
    for feature in features:
        plt.figure()
        plt.scatter(df[feature], df[target_col])
        plt.title(f"GRAPH 6: Scatter Plot - {feature} vs {target_col}")
        plt.xlabel(feature)
        plt.ylabel(target_col)

        safe_name = feature.replace(" ", "_").lower()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"graph_6_scatter_{safe_name}.png")
        plt.close()

    categorical_features = [
        "weekend",
        "weekday",
        "summer",
        "is_public_holiday",
        "is_festival",
        "is_brahmostavam",
    ]

    for feature in categorical_features:
        if feature in df.columns:
            plt.figure()
            sns.boxplot(x=df[feature], y=df[target_col])
            plt.title(f"GRAPH 7: Boxplot - {feature} vs {target_col}")
            plt.xlabel(feature)
            plt.ylabel(target_col)

            safe_name = feature.replace(" ", "_").lower()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"graph_7_boxplot_{safe_name}.png")
            plt.close()

    df.hist(figsize=(16, 14), bins=20)
    plt.suptitle("GRAPH 8: Histogram Distribution of All Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "graph_8_histograms.png")
    plt.close()

    print(f"All graphs saved successfully in folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
