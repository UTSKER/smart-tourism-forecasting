# ============================================================
# STATIONARITY TESTING FOR COMPLETE TIME SERIES
# ADF + KPSS TEST
# ============================================================

# ---------------- IMPORT LIBRARIES ----------------
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

# ============================================================
# PROJECT PATH SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "Tirumala_Tirupati_Devasthanam.csv"

# ============================================================
# MAIN FUNCTION
# ============================================================
def main() -> None:
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(DATA_FILE)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date and set index
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    # ---------------- TARGET SERIES ----------------
    series = df["darshans"].dropna()

    # ============================================================
    # ADF TEST
    # ============================================================
    print("=" * 60)
    print("ADF TEST RESULTS")
    print("=" * 60)

    adf_result = adfuller(series)

    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print(f"Number of Lags Used: {adf_result[2]}")
    print(f"Number of Observations Used: {adf_result[3]}")

    print("\nCritical Values:")
    for key, value in adf_result[4].items():
        print(f"{key}: {value}")

    if adf_result[1] < 0.05:
        print("\nResult: The series is STATIONARY.")
    else:
        print("\nResult: The series is NON-STATIONARY.")

    # ============================================================
    # KPSS TEST
    # ============================================================
    print("\n" + "=" * 60)
    print("KPSS TEST RESULTS")
    print("=" * 60)

    kpss_result = kpss(series, regression="c")

    print(f"KPSS Statistic: {kpss_result[0]}")
    print(f"p-value: {kpss_result[1]}")
    print(f"Number of Lags Used: {kpss_result[2]}")

    print("\nCritical Values:")
    for key, value in kpss_result[3].items():
        print(f"{key}: {value}")

    if kpss_result[1] < 0.05:
        print("\nResult: Series is NON-STATIONARY.")
    else:
        print("\nResult: Series is STATIONARY.")

    # ============================================================
    # FINAL CONCLUSION
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL CONCLUSION")
    print("=" * 60)

    if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
        print("Strong evidence that the full time series is STATIONARY.")
    else:
        print("The full time series may be NON-STATIONARY.")

# ============================================================
# RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    main()