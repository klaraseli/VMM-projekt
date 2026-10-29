import pandas as pd
from clean_data import *
from load_data import *

data_path = "C://Users//Acer Aspire//Desktop//7. semestr//VMM//projekt//student_performance_Final.csv"
data = load_data(data_path)
data_clean = clean_data(data)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)

def report(df, name):
    print("=" * 80)
    print(f"{name}")
    print("=" * 80)
    print(f"Rozměry (řádky, sloupce): {df.shape}")
    print(f"Duplicitní řádky: {df.duplicated().sum()}")
    print("\nDatové typy:")
    print(df.dtypes)

    print("\nPočet a podíl NaN hodnot:")
    nan_info = pd.DataFrame({
        "nan_count": df.isna().sum(),
        "nan_pct": (df.isna().mean() * 100).round(2),
    })
    print(nan_info.sort_values("nan_count", ascending=False))

    print("\nStatistiky číselných příznaků:")
    print(df.describe().T)

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        print("\nStatistiky kategorických příznaků:")
        print(df[cat_cols].describe().T)
        print("\nPočet unikátních hodnot:")
        print(df[cat_cols].nunique())
        print("\nČetnosti hodnot (top 10):")
        for col in cat_cols:
            print(f"\n--- {col} ---")
            print(df[col].value_counts(dropna=False).head(10))

# Report pro data před a po čištění
report(data, "PŮVODNÍ DATA")
report(data_clean, "VYČIŠTĚNÁ DATA")

# Porovnání před/po
print("=" * 80)
print("POROVNÁNÍ PŘED / PO")
print("=" * 80)
print(f"Řádky: {data.shape[0]} -> {data_clean.shape[0]} (odstraněno {data.shape[0] - data_clean.shape[0]})")
print(f"Sloupce: {data.shape[1]} -> {data_clean.shape[1]}")
print(f"Celkem NaN: {data.isna().sum().sum()} -> {data_clean.isna().sum().sum()}")

removed_cols = set(data.columns) - set(data_clean.columns)
added_cols = set(data_clean.columns) - set(data.columns)
print(f"Odstraněné sloupce: {sorted(removed_cols)}")
print(f"Nově přidané sloupce: {sorted(added_cols)}")

common = [c for c in data.columns if c in data_clean.columns]
cmp = pd.DataFrame({
    "nan_před": data[common].isna().sum(),
    "nan_po": data_clean[common].isna().sum(),
    "dtype_před": data[common].dtypes.astype(str),
    "dtype_po": data_clean[common].dtypes.astype(str),
})
print("\nZměny po sloupcích:")
print(cmp)

