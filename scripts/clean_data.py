"""
clean_data.py - Čištění a předzpracování datasetu student_performance_Final.csv.
================================================================================
Postup:
1. Neplatné hodnoty mimo povolené rozsahy (RANGES) -> NaN (např. absences == -1).
2. Binární kategorie -> 0 / 1.
3. Vyloučení cílových proměnných (G1, G2, G3) z imputace, aby nedocházelo k úniku
   cílové proměnné (target data leakage) do vstupních příznaků.
4. Imputace chybějících hodnot pomocí KNN (pouze pro skutečně chybějící buňky,
   původní platná data zůstávají beze změny).
5. Vícehodnotové kategorie -> One-Hot encoding (dummy proměnné).

Použití:
    from clean_data import clean_data
    df_clean = clean_data(df_raw)
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# ------------------------------ nastavení ------------------------------
# Přípustné rozsahy numerických sloupců podle zadání (mimo -> nesmysl -> NaN)
# absences: podle zadání 0 až 93. V surových datech se vyskytuje -1 (missing kód),
# který se tímto pravidlem korektně převede na NaN.
RANGES = {
    "age": (15, 22),
    "absences": (0, 93),
    "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
}

# Absences mají přirozeně těžký ocas (maximum je 32, což je legitimní hodnota
# odpovídající chronické absenci studenta). Odstranění pomocí IQR by smazalo reálná
# data a stáhlo je k mediánu, proto se IQR na legitimní data neaplikuje.
IQR_COLS = []
IQR_K = 1.5

# Binární kategorie -> 0/1
BINARY_MAPS = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
    **{c: {"no": 0, "yes": 1} for c in
       ["schoolsup", "famsup", "paid", "activities", "nursery",
        "higher", "internet", "romantic"]},
}

# Vícehodnotové kategorie (půjdou do one-hot kódování)
MULTI_CATEGORIES = {
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["father", "mother", "other"],
}

# Sloupce cíle a zkouškových známek, které NESMÍ být použity pro imputaci příznaků
TARGET_COLUMNS = ["G1", "G2", "G3"]

KNN_NEIGHBORS = 5


# ------------------------------ funkce ---------------------------------
def replace_nonsense_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Nečíselné hodnoty a hodnoty mimo RANGES -> NaN (včetně absences == -1)."""
    df = df.copy()
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s.where(s.between(lo, hi))
    return df


def replace_outliers_iqr(df: pd.DataFrame, cols=None, k: float = IQR_K) -> pd.DataFrame:
    """Hodnoty mimo Q1 - k*IQR .. Q3 + k*IQR -> NaN (pouze pokud jsou specifikovány sloupce)."""
    if cols is None:
        cols = IQR_COLS
    if not cols:
        return df

    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].where(df[col].between(lower, upper))
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorie -> čísla. Neznámé hodnoty (nesmysly) se stanou NaN.
    Binární: 0/1. Vícehodnotové: dočasné pořadové kódy 0..n-1.
    """
    df = df.copy()
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().map(mapping).astype("float")
    for col, cats in MULTI_CATEGORIES.items():
        if col in df.columns:
            mapping = {c: float(i) for i, c in enumerate(cats)}
            df[col] = df[col].astype("string").str.strip().map(mapping).astype("float")
    return df


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = KNN_NEIGHBORS,
    exclude_targets: bool = True
) -> pd.DataFrame:
    """
    Doplní chybějící hodnoty (NaN) pomocí KNNImputeru.
    DŮLEŽITÉ: Pokud exclude_targets=True, cílové známky G1, G2, G3 se před výpočtem
    vzdáleností vyjmou, aby nedocházelo k target leakage do prediktorů!
    Původní platné (nechybějící) hodnoty zůstávají beze změny.
    """
    df = df.copy()

    # 1. Dočasné vyjmutí cílových sloupců
    targets = {}
    if exclude_targets:
        for t_col in TARGET_COLUMNS:
            if t_col in df.columns:
                targets[t_col] = df[t_col].copy()
                df = df.drop(columns=[t_col])

    # 2. Zaznamenání pozic původních NaN
    was_nan = df.isna()

    # 3. Škálování na 0-1 pro férový výpočet euklidovské vzdálenosti
    mins = df.min()
    maxs = df.max()
    span = (maxs - mins).replace(0, 1)
    scaled = (df - mins) / span

    # 4. KNN Imputace
    filled = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
    filled_df = pd.DataFrame(filled, columns=df.columns, index=df.index)
    imputed = filled_df * span + mins
    imputed = imputed.round().astype(int)

    # 5. Aktualizace POUZE původně chybějících hodnot (nechybějící data zůstávají netknutá)
    for col in df.columns:
        df.loc[was_nan[col], col] = imputed.loc[was_nan[col], col]

    # 6. Bezpečnostní oříznutí do mezí
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi).astype(int)
    for col in BINARY_MAPS:
        if col in df.columns:
            df[col] = df[col].clip(0, 1).astype(int)
    for col, cats in MULTI_CATEGORIES.items():
        if col in df.columns:
            df[col] = df[col].clip(0, len(cats) - 1).astype(int)

    # 7. Vrácení cílových sloupců zpět
    for t_col, t_series in targets.items():
        df[t_col] = t_series

    return df


def one_hot_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Kódy vícehodnotových kategorií -> sloupce 0/1 (např. Mjob_teacher)."""
    df = df.copy()
    for col, cats in MULTI_CATEGORIES.items():
        if col not in df.columns:
            continue
        labels = df[col].map(dict(enumerate(cats)))
        dummies = pd.get_dummies(labels, prefix=col, dtype=int)
        dummies = dummies.reindex(columns=[f"{col}_{c}" for c in cats], fill_value=0)
        df = pd.concat([df.drop(columns=col), dummies], axis=1)
    return df


def clean_data(df: pd.DataFrame, exclude_targets_from_imputation: bool = True) -> pd.DataFrame:
    """
    Kompletní čistící funkce:
    1. Nahrazení nesmyslných hodnot (včetně absences == -1) -> NaN
    2. Kódování kategorií do numerických kódů
    3. KNN imputace chybějících hodnot (s vyloučením targetů pro prevenci leakage)
    4. One-Hot encoding pro vícehodnotové kategorie
    """
    df = replace_nonsense_numeric(df)
    df = replace_outliers_iqr(df)
    df = encode_categorical(df)
    df = impute_knn(df, exclude_targets=exclude_targets_from_imputation)
    df = one_hot_multi(df)
    return df


if __name__ == "__main__":
    import os
    raw_path = "../data/raw/student_performance_Final.csv"
    if not os.path.exists(raw_path):
        raw_path = "data/raw/student_performance_Final.csv"

    print(f"[*] Spouštím čištění dat z: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    clean_df = clean_data(raw_df)
    print(f"[+] Hotovo! Vyčištěný tvar: {clean_df.shape}")
    print(f"[+] Absences: min={clean_df['absences'].min()}, max={clean_df['absences'].max()}, NaN={clean_df['absences'].isna().sum()}")
    print(f"[+] G3: min={clean_df['G3'].min()}, max={clean_df['G3'].max()}, NaN={clean_df['G3'].isna().sum()}")

    out_path = "data/processed/student_performance_cleaned.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clean_df.to_csv(out_path, index=False)
    print(f"[+] Uloženo do: {out_path}")
