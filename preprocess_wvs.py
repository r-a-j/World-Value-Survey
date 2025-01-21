import os
import pandas as pd

def main():
    # 1. Paths to your input/output
    wave7_csv = "data/wave_7.csv"
    timeseries_csv = "data/time_series_1981_2022.csv"
    rename_xlsx = "data/time_series_list_of_variables_and_equivalences_1981_2022.xlsx"
    
    # Output files
    out_wave7_csv = "data/preprocessed/filtered_wave_7.csv"
    out_timeseries_csv = "data/preprocessed/filtered_time_series_1981_2022.csv"

    # 2. Read the rename Excel file => create a dictionary { original_name: new_name }
    #    Assumes the Excel has columns like [ "OriginalName", "NewName", ... ]
    rename_df = pd.read_excel(rename_xlsx)
    # Create a dictionary: rename_dict["E069_07"] = "Confidence_Parliament" etc.
    rename_dict = pd.Series(
        rename_df["Title"].values,
        index=rename_df["WVS7"]
    ).to_dict()

    # 3. Load wave_7 and time-series CSV data
    wave7 = pd.read_csv(wave7_csv, low_memory=False)
    ts_data = pd.read_csv(timeseries_csv, low_memory=False)

    # 4. For each dataset, we pick columns of interest (the code below is just an example).
    #    Adjust to whichever columns you prefer to keep.
    #    We'll do wave_7 first.
    keep_cols_wave7 = [
        "B_COUNTRY_ALPHA",  # country code
        "A_YEAR",           # survey year
        "E069_07",          # confidence in parliament
        "E069_11",          # confidence in government
        "E069_12",          # confidence in political parties
        "SACSECVAL",        # Welzel Secular Values
        "RESEMAVAL",        # Welzel Emancipative Values
        "I_AUTHORITY",      # sub-index if needed
        "I_DEVOUT"          # sub-index if needed
    ]
    
    # Filter wave7 dataset to those columns that actually exist:
    # (In case wave_7.csv has a slightly different naming, you can refine logic as needed.)
    wave7_filtered = wave7.loc[:, [c for c in keep_cols_wave7 if c in wave7.columns]]

    # 5. Rename columns using your rename_dict. 
    #    This will only rename columns where the old name is in rename_dict.
    wave7_filtered.rename(columns=rename_dict, inplace=True)

    # 6. Recode special WVS negative codes (e.g., -1, -2, -4, -5, etc.) as NaN
    #    We'll demonstrate for all numeric columns except country/year.
    for col in wave7_filtered.select_dtypes(include=["number"]).columns:
        # Example: turn negative values or impossible large values into NaN
        wave7_filtered.loc[wave7_filtered[col] < 0, col] = float("nan")
        # If your dataset has codes like 999, 777 => also add conditions:
        # wave7_filtered.loc[wave7_filtered[col] > 100, col] = float("nan")  # or some suitable rule

    # Save
    os.makedirs("preprocessed", exist_ok=True)
    wave7_filtered.to_csv(out_wave7_csv, index=False, encoding="utf-8")
    print(f"Saved filtered Wave 7 dataset => {out_wave7_csv}")

    # 7. Repeat for the time-series data (1981–2022)
    keep_cols_ts = [
        "B_COUNTRY_ALPHA",
        "A_YEAR",
        "E069_07",
        "E069_11",
        "E069_12",
        "SACSECVAL",
        "RESEMAVAL"
    ]

    ts_filtered = ts_data.loc[:, [c for c in keep_cols_ts if c in ts_data.columns]]
    
    # Rename columns
    ts_filtered.rename(columns=rename_dict, inplace=True)

    # Clean negative codes => NaN in numeric columns
    for col in ts_filtered.select_dtypes(include=["number"]).columns:
        ts_filtered.loc[ts_filtered[col] < 0, col] = float("nan")
        # again, adapt if you have other invalid codes to recode to NaN

    # Save
    ts_filtered.to_csv(out_timeseries_csv, index=False, encoding="utf-8")
    print(f"Saved filtered time-series dataset => {out_timeseries_csv}")


if __name__ == "__main__":
    main()
