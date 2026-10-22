"""
Čištění student_performance_Final.csv.

Postup: nesmyslné hodnoty -> NaN, odlehlé hodnoty (IQR) -> NaN,
kategorie -> čísla (0/1), NaN doplnit pomocí KNN, vícehodnotové kategorie -> one-hot.

Použití z main skriptu:

    from data_cleaning import clean_data
    df = load_data("student_performance_Final.csv")   # vaše načítací funkce
    df_clean = clean_data(df)
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# ------------------------------ nastavení ------------------------------
# Přípustné rozsahy numerických sloupců (mimo -> nesmysl -> NaN)
RANGES = {
    "age": (15, 22),
    "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
}

# Sloupce, kde hledáme odlehlé hodnoty pomocí IQR (spojité / široký rozsah).
# Na ordinálních škálách 1-5 nebo "failures" (skoro samé nuly) by IQR smazal
# úplně normální hodnoty, proto je tam nepoužíváme.
IQR_COLS = ["absences"]
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

# Vícehodnotové kategorie (po imputaci půjdou do one-hot)
MULTI_CATEGORIES = {
    "Mjob": ["at_home", "health", "other", "services", "teacher"],
    "Fjob": ["at_home", "health", "other", "services", "teacher"],
    "reason": ["course", "home", "other", "reputation"],
    "guardian": ["father", "mother", "other"],
}

KNN_NEIGHBORS = 5


# ------------------------------ funkce ---------------------------------
def replace_nonsense_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Nečíselné hodnoty a hodnoty mimo RANGES -> NaN."""
    df = df.copy()
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s.where(s.between(lo, hi))
    return df


def replace_outliers_iqr(df: pd.DataFrame, cols=IQR_COLS, k: float = IQR_K) -> pd.DataFrame:
    """Hodnoty mimo Q1 - k*IQR .. Q3 + k*IQR -> NaN."""
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
    Binární: 0/1. Vícehodnotové: pořadové kódy 0..n-1 (jen dočasně pro KNN).
    """
    df = df.copy()
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().map(mapping).astype("float")
    for col, cats in MULTI_CATEGORIES.items():
        if col in df.columns:
            mapping = {c: i for i, c in enumerate(cats)}
            df[col] = df[col].astype("string").str.strip().map(mapping).astype("float")
    return df


def impute_knn(df: pd.DataFrame, n_neighbors: int = KNN_NEIGHBORS) -> pd.DataFrame:
    """
    Doplní NaN nejbližšími sousedy. Sloupce se před výpočtem škálují na 0-1,
    aby 'absences' (0-93) nepřebily ostatní. Výsledky se zaokrouhlí na celá
    čísla, takže binární sloupce zůstanou 0/1 a kategorie platné kódy.
    """
    df = df.copy()
    mins, maxs = df.min(), df.max()
    span = (maxs - mins).replace(0, 1)
    scaled = (df - mins) / span

    filled = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
    filled = pd.DataFrame(filled, columns=df.columns, index=df.index)

    df = filled * span + mins
    df = df.round().astype(int)

    # pojistka: hodnoty zpět do povolených mezí
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    for col in BINARY_MAPS:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    for col, cats in MULTI_CATEGORIES.items():
        if col in df.columns:
            df[col] = df[col].clip(0, len(cats) - 1)
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Finální funkce: volá se z main skriptu."""
    df = replace_nonsense_numeric(df)
    df = replace_outliers_iqr(df)
    df = encode_categorical(df)
    df = impute_knn(df)
    df = one_hot_multi(df)
    return df
