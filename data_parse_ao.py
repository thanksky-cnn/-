"""
Parse NOAA AO monthly index text file into CSV (year, month, ao).

Input: data/NOAA .CPC. AO.monthly.txt
Output: data/ao_monthly.csv

The text file format:
  - Line 1: header with month abbreviations
  - Lines 2+: year followed by 12 space-separated monthly values (Jan–Dec)
  - Missing months (partial final year) are skipped
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "NOAA .CPC. AO.monthly.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "ao_monthly.csv")


def parse_ao_file(input_path, output_path):
    records = []
    with open(input_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 13:  # partial final year
            continue
        year = int(parts[0])
        for month_idx in range(1, 13):
            ao_value = float(parts[month_idx])
            records.append({"year": year, "month": month_idx, "ao": ao_value})

    df = pd.DataFrame(records)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"AO data parsed: {len(df)} records ({df['year'].min()}-{df['year'].max()})")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parse_ao_file(INPUT_FILE, OUTPUT_FILE)
