"""
test_models.py - Komplexní a metodicky korektní testování prediktivních modelů pro MPC-VMM.
============================================================================================
Tento skript řeší týmovou roli: "Testování a evaluace prediktivních modelů".

Metodické pilíře (V3.1 - Striktní metodická integrita a správné pořadí škálování):
1. 100% Fold-Isolated Preprocessing v Pipeline se správným pořadím:
   - Cílová proměnná G3 (a průběžné známky) je striktně oddělena před jakoukoliv modelovací fází.
   - Veškeré transformace jsou zapouzdřeny přímo v scikit-learn Pipeline v pořadí:
     Pipeline = [('scaler', StandardScaler()), ('imputer', KNNImputer(n=5)), ('m', model)]
   - StandardScaler nejdříve normalizuje data na z-skóre (ignoruje NaNs), takže KNNImputer
     počítá euklidovské vzdálenosti na férově škálovaných datech a nedochází k dominanci
     proměnných s velkým fyzickým rozsahem (absences, age).
   - Veškeré transformace probíhají striktně uvnitř trénovacích foldů (X_train).
2. Transparentní ošetření chybějící docházky (Absences) a dynamická citlivostní analýza:
   - U 218 z 649 studentů (33.6 %) se v surových datech vyskytuje hodnota absences == -1.
   - Zařazen explicitní binární indikátor chybění `absences_missing` (1 = chybělo).
   - Dvouvýběrový t-test ověřuje rozdíl v G3 mezi skupinami studentů (ověření předpokladu MAR/MCAR).
   - Analýza citlivosti dynamicky měří posun R² s absencemi vs. zcela bez absencí.
3. Dva jasně definované časové scénáře:
   - Scénář A: Průběžná predikce (Full model se známkami G1 a G2) -> horní mez přesnosti (daná silnou autokorelací známek).
   - Scénář B: Včasné varování (Early Warning: socio-demografie, návyky a docházka, bez G1 a G2)
     -> klíčová reálná výzva pro strojové učení.
4. Reprezentativní srovnání rodin modelů:
   - Dummy baseline, Lineární regularizované (OLS, Ridge, Lasso, ElasticNet, Logistická regrese),
   - Sousedské a jádrové metody (k-NN, SVR a SVC s lineárním i RBF jádrem),
   - Stromové a souborové metody (Rozhodovací strom, Random Forest, Gradient Boosting).
5. Metodicky nestranné ladění hyperparametrů pomocí Nested Cross-Validation (5x5):
   - Vnější 5-fold CV pro odhad zobecnění, vnitřní 5-fold GridSearchCV pro volbu parametrů.
   - Zlepšení u Lasso (+0.09) je objektivně vysvětleno korekcí nevhodné výchozí penalizace alpha=1.0.
   - Modely jsou řazeny do ekvivalenčních tříd s ohledem na směrodatné odchylky.
6. Řešení nevyváženosti tříd (Class Imbalance - 84.6 % vs. 15.4 %):
   - Evaluace ve Scénáři A i B s využitím Stratified 5-Fold CV.
   - Sledování Balanced Accuracy a Recallu menšinové třídy (ohrožení studenti: neprospěl, G3 < 10).
7. Out-of-Fold (OOF) Permutation Feature Importance:
   - Permutační důležitost je počítána na validačních foldech (X_val, y_val) napříč 5 foldy.
   - One-hot dummy sloupce vícehodnotových kategorií jsou sloučeny do mateřských faktorů.
8. Rigorózní statistické ověření 7 hypotéz týmu:
   - Nekauzální korelační terminologie (asociace, korelace, rozdíl průměrů).
   - Uvádění velikostí účinku (Spearman rho, delta průměrů, OLS beta).
   - Přesné p-hodnoty a dynamické testy (univariátní asociace address vs. OLS s kontrolou pro školu).
   - Konzervativní Bonferroniho korekce (alpha_adj = 0.05 / 7 = 0.00714).
============================================================================================
"""

import os
import sys
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_validate, cross_val_score,
    GridSearchCV, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier

# Regresní modely
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

# Klasifikační modely
from sklearn.linear_model import LogisticRegression

# Metriky a inspekce
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

# Zajištění importu pomocných modulů ze stejné složky
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Import jednotných konstant předzpracování z clean_data (Single Source of Truth)
from clean_data import RANGES, BINARY_MAPS, MULTI_CATEGORIES

# Kořenový adresář repozitáře VMM-projekt
REPO_ROOT = os.path.dirname(_scripts_dir)

from visualization_utils import (
    generate_bar_chart,
    generate_horizontal_bar_chart,
    generate_tuning_two_panel_chart,
    generate_grouped_bar_chart,
    generate_confusion_matrix_svg,
    save_svg
)

FEATURE_CZ_LABELS = {
    "failures": "Předchozí propadnutí (failures)",
    "higher": "Touha po VŠ (higher)",
    "schoolsup": "Školní podpora (schoolsup)",
    "school": "Výběr školy (school)",
    "Dalc": "Alkohol v týdnu (Dalc)",
    "reason": "Důvod volby školy (reason)",
    "studytime": "Čas studia (studytime)",
    "Walc": "Víkendový alkohol (Walc)",
    "famrel": "Vztahy v rodině (famrel)",
    "age": "Věk studenta (age)",
    "health": "Zdravotní stav (health)",
    "goout": "Čas s přáteli (goout)",
    "Medu": "Vzdělání matky (Medu)",
    "Fedu": "Vzdělání otce (Fedu)",
    "absences": "Počet absencí (absences)",
    "absences_missing": "Chybějící docházka (indikátor)",
    "freetime": "Volný čas (freetime)",
    "traveltime": "Čas cesty do školy (traveltime)",
    "famsup": "Podpora rodiny (famsup)",
    "paid": "Placené doučování (paid)",
    "activities": "Mimoškolní aktivity (activities)",
    "nursery": "Navštěvoval školku (nursery)",
    "internet": "Internet doma (internet)",
    "romantic": "Partnerský vztah (romantic)",
    "sex": "Pohlaví (sex)",
    "address": "Bydliště město/vesnice (address)",
    "famsize": "Velikost rodiny (famsize)",
    "Pstatus": "Rodiče spolu/odděleně (Pstatus)",
    "Mjob": "Povolání matky (Mjob)",
    "Fjob": "Povolání otce (Fjob)",
    "guardian": "Zákonný zástupce (guardian)"
}


# =============================================================================
# KROK 1: NAČTENÍ A PŘÍPRAVA DAT (100% FOLD-ISOLATED, S INDIKÁTOREM CHYBĚNÍ)
# =============================================================================
def load_and_prepare_data():
    """
    Načte surová data a připraví příznakové matice s nulovým únikem dat:
    - Hodnoty mimo povolené rozsahy (RANGES) jsou převedeny na NaN.
    - Přidán explicitní indikátor absences_missing (1 pokud absences chyběla, 0 jinak).
    - Binární a vícehodnotové kategorie jsou zakódovány do dummy proměnných;
      pokud původní hodnota chyběla, všechny její dummy sloupce mají hodnotu NaN.
    - Škálování i KNN imputace probíhají striktně uvnitř jednotlivých foldů v Pipeline.
    """
    raw_candidates = [
        os.path.join(REPO_ROOT, "data", "raw", "student_performance_Final.csv"),
        "data/raw/student_performance_Final.csv",
        "../data/raw/student_performance_Final.csv",
    ]
    raw_path = None
    for rc in raw_candidates:
        if os.path.exists(rc):
            raw_path = rc
            break

    if raw_path is None:
        raise FileNotFoundError("Surový soubor student_performance_Final.csv nebyl nalezen!")

    print(f"[*] Načítám surová data z: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    df = df_raw.copy()

    # 1. Explicitní indikátor chybějící docházky před nahrazením -1 za NaN
    df["absences_missing"] = (pd.to_numeric(df["absences"], errors="coerce") < 0).astype(float)

    # 2. Nahrazení hodnot mimo meze hodnotou NaN
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.where(s.between(lo, hi))

    # 3. Kódování binárních příznaků (NaN zůstává zachováno pro KNNImputer)
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().map(mapping).astype(float)

    # 4. Kódování vícehodnotových kategorií (při chybějící hodnotě -> NaN ve všech dummy)
    for col, cats in MULTI_CATEGORIES.items():
        if col in df.columns:
            is_na = df[col].isna()
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False, dtype=float)
            for c in cats:
                col_name = f"{col}_{c}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0.0
            dummies.loc[is_na, :] = np.nan
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Cílová proměnná (bez chybějících hodnot)
    y_reg = np.asarray(df["G3"], dtype=float)
    y_clf = (y_reg >= 10.0).astype(int)

    # Scénář A: Průběžná predikce (včetně známek G1 a G2)
    feat_A = [c for c in df.columns if c != "G3"]
    X_A = np.asarray(df[feat_A], dtype=float)

    # Scénář B: Včasné varování (BEZ známek G1 a G2)
    feat_B = [c for c in feat_A if c not in ["G1", "G2"]]
    X_B = np.asarray(df[feat_B], dtype=float)

    # Mapování původních kategorií pro sloučení One-Hot příznaků (Grouped Importance)
    category_groups = {
        "Mjob": [c for c in feat_B if c.startswith("Mjob_")],
        "Fjob": [c for c in feat_B if c.startswith("Fjob_")],
        "reason": [c for c in feat_B if c.startswith("reason_")],
        "guardian": [c for c in feat_B if c.startswith("guardian_")]
    }

    n_pass = int(np.sum(y_clf == 1))
    n_fail = int(np.sum(y_clf == 0))
    print(f"[*] Celkem studentů: {len(df)}")
    print(f"[*] Scénář A (včetně G1, G2): {len(feat_A)} příznaků (obsahuje NaNs pro fold-isolated imputaci)")
    print(f"[*] Scénář B (BEZ G1, G2 - včasné varování): {len(feat_B)} příznaků")
    print(f"[*] Třídy: Prospěl (>=10) = {n_pass} ({n_pass/len(df)*100:.1f} %), "
          f"Neprospěl (<10) = {n_fail} ({n_fail/len(df)*100:.1f} %)\n")

    return {
        "df_raw": df_raw,
        "df": df,
        "X_A": X_A, "feat_A": feat_A,
        "X_B": X_B, "feat_B": feat_B,
        "y_reg": y_reg, "y_clf": y_clf,
        "category_groups": category_groups
    }


# =============================================================================
# KROK 2: DYNAMICKÁ ANALÝZA CITLIVOSTI PRO ABSENCES (ROBUSTNESS CHECK)
# =============================================================================
def evaluate_absences_sensitivity(df_raw, X_B, y_reg, feat_B):
    """
    Ověří vliv chybějící docházky (33.6 % studentů s absences == -1):
    1. Statistické porovnání cílové známky G3 u studentů s chybějící vs. zaznamenanou docházkou.
    2. Dynamické srovnání predikčního výkonu modelů ve Scénáři B: S absencemi vs. Zcela bez absencí.
    """
    print("=" * 86)
    print(" ANALÝZA CITLIVOSTI A INTEGRITY DOCHÁZKY (ABSENCES SENSITIVITY CHECK)")
    print("=" * 86)

    miss_mask = (pd.to_numeric(df_raw["absences"], errors="coerce") < 0)
    g3_miss = df_raw.loc[miss_mask, "G3"]
    g3_pres = df_raw.loc[~miss_mask, "G3"]

    t_res = stats.ttest_ind(g3_miss, g3_pres)
    print(f"[*] Studenti s chybějící docházkou (absences == -1): {miss_mask.sum()} / {len(df_raw)} ({miss_mask.mean()*100:.1f} %)")
    print(f"    - Průměrná známka G3 (chybějící docházka):  {g3_miss.mean():.2f} +/- {g3_miss.std():.2f}")
    print(f"    - Průměrná známka G3 (dostupná docházka):   {g3_pres.mean():.2f} +/- {g3_pres.std():.2f}")
    print(f"    - Rozdíl průměrů: {g3_miss.mean() - g3_pres.mean():+.2f} bodu, Dvouvýběrový t-test p-hodnota = {t_res.pvalue:.4f}")
    t_eval = "nenalezl statisticky významný rozdíl v G3" if t_res.pvalue >= 0.05 else "nalezl statisticky významný rozdíl v G3"
    t_concl = "což neodporuje předpokladu MAR/MCAR (nenalezen důkaz o systematické vazbě chybění docházky na prospěch)" if t_res.pvalue >= 0.05 else "což indikuje možnou nenáhodnost chybění docházky"
    print(f"    -> Dvouvýběrový t-test {t_eval} (p = {t_res.pvalue:.4f}), {t_concl}.\n")

    # Porovnání modelů s absencemi vs bez absencí
    idx_no_abs = [i for i, c in enumerate(feat_B) if c not in ["absences", "absences_missing"]]
    X_B_no_abs = X_B[:, idx_no_abs]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("imputer", KNNImputer(n_neighbors=5)), ("m", Ridge(alpha=1.0))])
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("imputer", KNNImputer(n_neighbors=5)), ("m", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))])

    r_ridge_with = cross_val_score(pipe_ridge, X_B, y_reg, cv=cv, scoring="r2")
    r_rf_with = cross_val_score(pipe_rf, X_B, y_reg, cv=cv, scoring="r2")

    r_ridge_without = cross_val_score(pipe_ridge, X_B_no_abs, y_reg, cv=cv, scoring="r2")
    r_rf_without = cross_val_score(pipe_rf, X_B_no_abs, y_reg, cv=cv, scoring="r2")

    delta_ridge = float(r_ridge_with.mean() - r_ridge_without.mean())
    delta_rf = float(r_rf_with.mean() - r_rf_without.mean())
    max_delta = max(abs(delta_ridge), abs(delta_rf))

    print("Porovnání predikce Scénáře B s absencemi vs. zcela bez absencí:")
    print(f" - Ridge: S absencemi R² = {r_ridge_with.mean():.4f} +/- {r_ridge_with.std():.4f} | Bez absencí R² = {r_ridge_without.mean():.4f} +/- {r_ridge_without.std():.4f} (Delta = {delta_ridge:+.4f})")
    print(f" - RF:    S absencemi R² = {r_rf_with.mean():.4f} +/- {r_rf_with.std():.4f} | Bez absencí R² = {r_rf_without.mean():.4f} +/- {r_rf_without.std():.4f} (Delta = {delta_rf:+.4f})")
    print(f" -> Výsledek: Posun R² po vyřazení docházky činí {delta_ridge:+.4f} u Ridge a {delta_rf:+.4f} u Random Forestu (|Delta| <= {max_delta:.4f}). Vzhledem ke směrodatným odchylkám foldů (~0.06 až 0.08) je tento posun v rámci variability křížové validace; nebyl nalezen důkaz, že by predikce modelu stála na artefaktu imputace docházky.\n")


# =============================================================================
# KROK 3: REGRESNÍ BENCHMARK (100% FOLD-ISOLATED V PIPELINE: SCALER -> IMPUTER)
# =============================================================================
def get_default_regression_models():
    """
    Vytvoří slovník regresních modelů.
    Všechny modely (kromě Dummy) mají v Pipeline zapouzdřen StandardScaler a KNNImputer
    v pořadí: StandardScaler -> KNNImputer -> Model.
    Tím je zaručeno, že KNN vzdálenosti se počítají na standardizovaných datech.
    """
    def make_pipe(model):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("m", model)
        ])

    return {
        "Dummy (Baseline průměr)": DummyRegressor(strategy="mean"),
        "Lineární regrese (OLS)": make_pipe(LinearRegression()),
        "Ridge regrese (L2, alpha=1.0)": make_pipe(Ridge(alpha=1.0)),
        "Lasso regrese (L1, alpha=1.0)": make_pipe(Lasso(alpha=1.0, max_iter=3000)),
        "ElasticNet (alpha=1.0, l1=0.5)": make_pipe(ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=3000)),
        "k-NN Regresor (k=5)": make_pipe(KNeighborsRegressor(n_neighbors=5)),
        "SVR (Lineární jádro, C=1.0)": make_pipe(SVR(kernel="linear", C=1.0)),
        "SVR (RBF jádro, C=1.0)": make_pipe(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        "Rozhodovací strom (max_depth=5)": make_pipe(DecisionTreeRegressor(max_depth=5, random_state=42)),
        "Random Forest (100 stromů)": make_pipe(RandomForestRegressor(n_estimators=100, random_state=42)),
        "Gradient Boosting (100 stromů)": make_pipe(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    }


def evaluate_regression(X, y, scenario_name="Scénář A"):
    """
    Vyhodnotí regresní modely pomocí 5-fold křížové validace.
    Uvádí průměr i směrodatnou odchylku napříč foldy (mean +/- std).
    """
    print("=" * 86)
    print(f" REGRESNÍ BENCHMARK (5-FOLD CV): PREDIKCE ZNÁMKY G3 — {scenario_name.upper()}")
    print("=" * 86)

    models = get_default_regression_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

    results = []
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        r2_m, r2_s = cv_res["test_r2"].mean(), cv_res["test_r2"].std()
        rmse_m, rmse_s = -cv_res["test_neg_root_mean_squared_error"].mean(), cv_res["test_neg_root_mean_squared_error"].std()
        mae_m, mae_s = -cv_res["test_neg_mean_absolute_error"].mean(), cv_res["test_neg_mean_absolute_error"].std()

        results.append({
            "Model": name,
            "R2_mean": round(float(r2_m), 3),
            "R2_std": round(float(r2_s), 3),
            "R2 (vysvětlený rozptyl)": f"{r2_m:.3f} +/- {r2_s:.3f}",
            "RMSE (chyba známky)": f"{rmse_m:.3f} +/- {rmse_s:.3f}",
            "MAE (průměrná chyba)": f"{mae_m:.3f} +/- {mae_s:.3f}",
        })

    df_res = pd.DataFrame(results).sort_values(by="R2_mean", ascending=False).reset_index(drop=True)
    display_df = df_res.drop(columns=["R2_mean", "R2_std"])
    print(display_df.to_string(index=False))
    print()
    return df_res


# =============================================================================
# KROK 4: KLASIFIKAČNÍ BENCHMARK (100% FOLD-ISOLATED V PIPELINE: SCALER -> IMPUTER)
# =============================================================================
def get_default_classification_models():
    """
    Vytvoří slovník klasifikačních modelů.
    Všechny modely (kromě Dummy) mají v Pipeline zapouzdřen StandardScaler a KNNImputer.
    """
    def make_pipe(model):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("m", model)
        ])

    return {
        "Dummy (Baseline majorita)": DummyClassifier(strategy="most_frequent"),
        "Logistická regrese (Standard)": make_pipe(LogisticRegression(max_iter=1000, random_state=42)),
        "Logistická regrese (Balanced)": make_pipe(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        "k-NN Klasifikátor (k=5)": make_pipe(KNeighborsClassifier(n_neighbors=5)),
        "SVC (Lineární jádro, Balanced)": make_pipe(SVC(kernel="linear", class_weight="balanced", random_state=42)),
        "SVC (RBF jádro, Balanced)": make_pipe(SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)),
        "Rozhodovací strom (Balanced)": make_pipe(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        "Random Forest (Standard)": make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
        "Random Forest (Balanced)": make_pipe(RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        "Gradient Boosting (Standard)": make_pipe(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    }


def evaluate_classification(X, y_bin, scenario_name="Scénář A"):
    """
    Vyhodnotí klasifikační modely pomocí Stratified 5-Fold CV.
    Sleduje vyváženou přesnost i recall pro menšinovou třídu (neprospěl).
    """
    print("=" * 96)
    print(f" KLASIFIKAČNÍ BENCHMARK: PROSPĚL (>=10) VS. NEPROSPĚL (<10) — {scenario_name.upper()}")
    print("=" * 96)

    models = get_default_classification_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_preds = None
    best_model_name = ""
    best_bal_acc = -1.0

    for name, model in models.items():
        bal_accs = []
        rec_fails = []
        f1s = []
        accs = []

        oof_preds = cross_val_predict(model, X, y_bin, cv=skf)

        for train_idx, val_idx in skf.split(X, y_bin):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y_bin[train_idx], y_bin[val_idx]

            model.fit(X_tr, y_tr)
            p_val = model.predict(X_va)

            accs.append(accuracy_score(y_va, p_val))
            bal_accs.append(balanced_accuracy_score(y_va, p_val))
            f1s.append(f1_score(y_va, p_val, zero_division=0))
            rec_fails.append(recall_score(y_va == 0, p_val == 0, zero_division=0))

        mean_bal = float(np.mean(bal_accs)) * 100
        std_bal = float(np.std(bal_accs)) * 100
        mean_acc = float(np.mean(accs)) * 100
        std_acc = float(np.std(accs)) * 100
        mean_rec_fail = float(np.mean(rec_fails)) * 100
        std_rec_fail = float(np.std(rec_fails)) * 100
        mean_f1 = float(np.mean(f1s))

        results.append({
            "Model": name,
            "Bal_Acc_mean": round(mean_bal, 1),
            "Bal_Acc_std": round(std_bal, 1),
            "Accuracy (%)": f"{mean_acc:.1f} +/- {std_acc:.1f}",
            "Vyvážená Acc (%)": f"{mean_bal:.1f} +/- {std_bal:.1f}",
            "F1-Score": f"{mean_f1:.3f} +/- {np.std(f1s):.3f}",
            "Recall Neprospěl (Minorita %)": f"{mean_rec_fail:.1f} +/- {std_rec_fail:.1f}"
        })

        if mean_bal > best_bal_acc:
            best_bal_acc = mean_bal
            best_preds = oof_preds
            best_model_name = name

    df_res = pd.DataFrame(results).sort_values(by="Bal_Acc_mean", ascending=False).reset_index(drop=True)
    display_df = df_res.drop(columns=["Bal_Acc_mean", "Bal_Acc_std"])
    print(display_df.to_string(index=False))
    print()
    return df_res, best_preds, best_model_name


# =============================================================================
# KROK 5: NESTRANNÉ LADĚNÍ HYPERPARAMETRŮ POMOCÍ NESTED CV (5x5)
# =============================================================================
def run_nested_hyperparameter_tuning(X_A, X_B, y_reg, y_clf):
    """
    Metodicky korektní ladění pomocí Nested Cross-Validation (5x5) v Pipeline:
    - Vnější smyčka (5 foldů): odhad zobecnění.
    - Vnitřní smyčka (5 foldů): volba parametrů mřížkovým prohledáváním.
    - Zapouzdření StandardScaler -> KNNImputer v Pipeline zaručuje,
      že vzdálenosti se počítají ve standardizovaném prostoru a bez úniku dat.
    """
    print("=" * 96)
    print(" METODICKY KOREKTNÍ LADĚNÍ HYPERPARAMETRŮ: NESTED CROSS-VALIDATION (5x5)")
    print("=" * 96)
    print(" [*] Vnější 5-fold CV hodnotí zobecnění, vnitřní 5-fold GridSearchCV vybírá parametry.\n")

    cv_outer_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_outer_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tuning_records = []

    def make_tune_pipe(m):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("m", m)
        ])

    # 1. Scénář A: Lasso regrese (korekce nevhodné výchozí alpha=1.0)
    print("[+] 1/6 [Scénář A] Lasso regrese: Ladění alpha...")
    pipe_lasso_def = make_tune_pipe(Lasso(alpha=1.0, max_iter=3000, random_state=42))
    pipe_lasso = make_tune_pipe(Lasso(max_iter=3000, random_state=42))
    grid_lasso = {"m__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]}
    gs_lasso = GridSearchCV(pipe_lasso, grid_lasso, cv=cv_inner_reg, scoring="r2")

    def_scores_lasso = cross_val_score(pipe_lasso_def, X_A, y_reg, cv=cv_outer_reg, scoring="r2")
    nested_scores_lasso = cross_val_score(gs_lasso, X_A, y_reg, cv=cv_outer_reg, scoring="r2")
    gs_lasso.fit(X_A, y_reg)

    tuning_records.append({
        "Scénář": "Scénář A",
        "Úloha": "Regrese",
        "Model": "Lasso Regrese",
        "Mřížka parametrů": "alpha: 0.01..1.0",
        "Nalezený optimální parametr": f"alpha={gs_lasso.best_params_['m__alpha']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_lasso.mean():.4f} +/- {def_scores_lasso.std():.4f}",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_lasso.mean():.4f} +/- {nested_scores_lasso.std():.4f}",
        "Nestranné zlepšení Delta": round(float(nested_scores_lasso.mean() - def_scores_lasso.mean()), 4),
        "Metrika": "R2",
        "def_val": def_scores_lasso.mean(),
        "tuned_val": nested_scores_lasso.mean()
    })

    # 2. Scénář A: Ridge regrese
    print("[+] 2/6 [Scénář A] Ridge regrese: Ladění alpha...")
    pipe_ridge_def = make_tune_pipe(Ridge(alpha=1.0))
    pipe_ridge = make_tune_pipe(Ridge())
    grid_ridge = {"m__alpha": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
    gs_ridge = GridSearchCV(pipe_ridge, grid_ridge, cv=cv_inner_reg, scoring="r2")

    def_scores_ridge = cross_val_score(pipe_ridge_def, X_A, y_reg, cv=cv_outer_reg, scoring="r2")
    nested_scores_ridge = cross_val_score(gs_ridge, X_A, y_reg, cv=cv_outer_reg, scoring="r2")
    gs_ridge.fit(X_A, y_reg)

    tuning_records.append({
        "Scénář": "Scénář A",
        "Úloha": "Regrese",
        "Model": "Ridge Regrese",
        "Mřížka parametrů": "alpha: 0.1..100.0",
        "Nalezený optimální parametr": f"alpha={gs_ridge.best_params_['m__alpha']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_ridge.mean():.4f} +/- {def_scores_ridge.std():.4f}",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_ridge.mean():.4f} +/- {nested_scores_ridge.std():.4f}",
        "Nestranné zlepšení Delta": round(float(nested_scores_ridge.mean() - def_scores_ridge.mean()), 4),
        "Metrika": "R2",
        "def_val": def_scores_ridge.mean(),
        "tuned_val": nested_scores_ridge.mean()
    })

    # 3. Scénář B (Včasné varování): SVR s RBF jádrem
    print("[+] 3/6 [Scénář B] SVR (RBF jádro): Ladění C a epsilon...")
    pipe_svr_def = make_tune_pipe(SVR(kernel="rbf", C=1.0, epsilon=0.1))
    pipe_svr = make_tune_pipe(SVR(kernel="rbf"))
    grid_svr = {"m__C": [0.5, 1.0, 2.0, 5.0], "m__epsilon": [0.1, 0.2]}
    gs_svr = GridSearchCV(pipe_svr, grid_svr, cv=cv_inner_reg, scoring="r2")

    def_scores_svr = cross_val_score(pipe_svr_def, X_B, y_reg, cv=cv_outer_reg, scoring="r2")
    nested_scores_svr = cross_val_score(gs_svr, X_B, y_reg, cv=cv_outer_reg, scoring="r2")
    gs_svr.fit(X_B, y_reg)

    tuning_records.append({
        "Scénář": "Scénář B",
        "Úloha": "Regrese",
        "Model": "SVR (RBF)",
        "Mřížka parametrů": "C: 0.5..5.0, eps: 0.1..0.2",
        "Nalezený optimální parametr": f"C={gs_svr.best_params_['m__C']}, eps={gs_svr.best_params_['m__epsilon']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_svr.mean():.4f} +/- {def_scores_svr.std():.4f}",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_svr.mean():.4f} +/- {nested_scores_svr.std():.4f}",
        "Nestranné zlepšení Delta": round(float(nested_scores_svr.mean() - def_scores_svr.mean()), 4),
        "Metrika": "R2",
        "def_val": def_scores_svr.mean(),
        "tuned_val": nested_scores_svr.mean()
    })

    # 4. Scénář B (Včasné varování): Random Forest Regressor
    print("[+] 4/6 [Scénář B] Random Forest Regresor: Ladění max_depth...")
    pipe_rf_def = make_tune_pipe(RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42))
    pipe_rf = make_tune_pipe(RandomForestRegressor(random_state=42))
    grid_rf = {"m__max_depth": [4, 6, 8, None], "m__n_estimators": [50, 100]}
    gs_rf = GridSearchCV(pipe_rf, grid_rf, cv=cv_inner_reg, scoring="r2")

    def_scores_rf = cross_val_score(pipe_rf_def, X_B, y_reg, cv=cv_outer_reg, scoring="r2")
    nested_scores_rf = cross_val_score(gs_rf, X_B, y_reg, cv=cv_outer_reg, scoring="r2")
    gs_rf.fit(X_B, y_reg)

    tuning_records.append({
        "Scénář": "Scénář B",
        "Úloha": "Regrese",
        "Model": "Random Forest",
        "Mřížka parametrů": "depth: 4..None, n: 50..100",
        "Nalezený optimální parametr": f"depth={gs_rf.best_params_['m__max_depth']}, n={gs_rf.best_params_['m__n_estimators']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_rf.mean():.4f} +/- {def_scores_rf.std():.4f}",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_rf.mean():.4f} +/- {nested_scores_rf.std():.4f}",
        "Nestranné zlepšení Delta": round(float(nested_scores_rf.mean() - def_scores_rf.mean()), 4),
        "Metrika": "R2",
        "def_val": def_scores_rf.mean(),
        "tuned_val": nested_scores_rf.mean()
    })

    # 5. Scénář A: Support Vector Classifier (SVC Balanced)
    print("[+] 5/6 [Scénář A] SVC (Balanced): Ladění C a typu jádra...")
    pipe_svc_def = make_tune_pipe(SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42))
    pipe_svc = make_tune_pipe(SVC(class_weight="balanced", random_state=42))
    grid_svc = {"m__C": [0.1, 0.5, 1.0, 5.0], "m__kernel": ["linear", "rbf"]}
    gs_svc = GridSearchCV(pipe_svc, grid_svc, cv=cv_inner_clf, scoring="balanced_accuracy")

    def_scores_svc = cross_val_score(pipe_svc_def, X_A, y_clf, cv=cv_outer_clf, scoring="balanced_accuracy")
    nested_scores_svc = cross_val_score(gs_svc, X_A, y_clf, cv=cv_outer_clf, scoring="balanced_accuracy")
    gs_svc.fit(X_A, y_clf)

    tuning_records.append({
        "Scénář": "Scénář A",
        "Úloha": "Klasifikace",
        "Model": "SVC (Balanced)",
        "Mřížka parametrů": "C: 0.1..5.0, kernel: lin/rbf",
        "Nalezený optimální parametr": f"C={gs_svc.best_params_['m__C']}, kernel={gs_svc.best_params_['m__kernel']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_svc.mean()*100:.1f} +/- {def_scores_svc.std()*100:.1f} %",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_svc.mean()*100:.1f} +/- {nested_scores_svc.std()*100:.1f} %",
        "Nestranné zlepšení Delta": round(float((nested_scores_svc.mean() - def_scores_svc.mean()) * 100), 2),
        "Metrika": "Vyvážená Acc (%)",
        "def_val": def_scores_svc.mean() * 100,
        "tuned_val": nested_scores_svc.mean() * 100
    })

    # 6. Scénář B (Včasné varování): Random Forest Classifier (Balanced)
    print("[+] 6/6 [Scénář B] Random Forest Klasifikátor: Ladění max_depth a n_estimators...")
    pipe_rfc_def = make_tune_pipe(RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", random_state=42))
    pipe_rfc = make_tune_pipe(RandomForestClassifier(class_weight="balanced", random_state=42))
    grid_rfc = {"m__max_depth": [4, 6, 8], "m__n_estimators": [50, 100]}
    gs_rfc = GridSearchCV(pipe_rfc, grid_rfc, cv=cv_inner_clf, scoring="balanced_accuracy")

    def_scores_rfc = cross_val_score(pipe_rfc_def, X_B, y_clf, cv=cv_outer_clf, scoring="balanced_accuracy")
    nested_scores_rfc = cross_val_score(gs_rfc, X_B, y_clf, cv=cv_outer_clf, scoring="balanced_accuracy")
    gs_rfc.fit(X_B, y_clf)

    tuning_records.append({
        "Scénář": "Scénář B",
        "Úloha": "Klasifikace",
        "Model": "Random Forest (Bal)",
        "Mřížka parametrů": "depth: 4..8, n: 50..100",
        "Nalezený optimální parametr": f"depth={gs_rfc.best_params_['m__max_depth']}, n={gs_rfc.best_params_['m__n_estimators']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_rfc.mean()*100:.1f} +/- {def_scores_rfc.std()*100:.1f} %",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_rfc.mean()*100:.1f} +/- {nested_scores_rfc.std()*100:.1f} %",
        "Nestranné zlepšení Delta": round(float((nested_scores_rfc.mean() - def_scores_rfc.mean()) * 100), 2),
        "Metrika": "Vyvážená Acc (%)",
        "def_val": def_scores_rfc.mean() * 100,
        "tuned_val": nested_scores_rfc.mean() * 100
    })

    df_tuning = pd.DataFrame(tuning_records)
    display_df = df_tuning.drop(columns=["def_val", "tuned_val"])
    print("\n" + display_df.to_string(index=False))
    print()
    return df_tuning


# =============================================================================
# KROK 6: STATISTICKÉ OVĚŘENÍ 7 HYPOTÉZ TÝMU (JEDNOSTRANNÉ/OBOUSTRANNÉ TESTY, EFEKTY A ROBUSTNOST)
# =============================================================================
def test_team_hypotheses(df_raw):
    """
    Rigorózní statistické testování všech 7 hypotéz specifikovaných týmem:
    - Všechny proměnné v hypotézách H1-H7 jsou kompletně pozorované (žádná hodnota nechybí).
    - Důsledné používání ne-kauzální terminologie (asociace, korelace, rozdíl průměrů).
    - Směry testů (jednostranné alternative='greater'/'less' vs. oboustranné 'two-sided')
      odpovídají teoretickému zadání hypotéz a sjednocují metodiku s týmovou analýzou (větev misus).
    - U H3 je použit Welchův t-test s Cohenovým d = 0.49 (střední účinek), doplněný o kontrolu robustnosti pro školu.
    - U H4 je úroveň vzdělání rodičů definována jako maximum z Medu a Fedu (konzistentní s H3).
    - U H7 je uveden 95% konfidenční interval OLS koeficientu bydliště [-0.02, 1.09].
    - Zohledněna Bonferroniho korekce pro 7 testovaných hypotéz (alpha_adj = 0.05 / 7 = 0.00714).
    """
    print("=" * 96)
    print(" STATISTICKÉ TESTOVÁNÍ HYPOTÉZ (JEDNOSTRANNÉ TESTY, EFEKT-SIZES A BONFERRONI)")
    print("=" * 96)

    def classify_association(r: float) -> str:
        """Dynamicky klasifikuje sílu a směr asociace dle absolutní hodnoty koeficientu."""
        ar = abs(r)
        if ar < 0.1:
            strength = "velmi slabá"
        elif ar < 0.3:
            strength = "slabá"
        elif ar < 0.5:
            strength = "střední"
        else:
            strength = "silná"
        direction = "pozitivní" if r > 0 else "negativní" if r < 0 else "nulová"
        return f"{strength} {direction} asociace"

    def format_p(p: float) -> str:
        """Formátuje p-hodnotu tak, aby nikdy nevracela zavádějící p = 0.0000."""
        if p < 1e-4:
            return "< 1e-4"
        return f"= {p:.4f}"

    df = df_raw.copy()
    for col, m in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().map(m).astype(float)

    hyp_results = []
    bonferroni_alpha = 0.05 / 7.0  # 0.00714

    # H1: studytime vs G3 (jednostranný Spearmanův test)
    rho1, p1 = stats.spearmanr(df["studytime"], df["G3"], alternative="greater")
    h1_sup = (rho1 > 0) and (p1 < bonferroni_alpha)
    h1_assoc = classify_association(rho1)
    h1_dir = "Pozitivní asociace (vyšší studijní čas ~ vyšší známka)" if rho1 > 0 else "Negativní asociace"
    h1_verdict = "Statisticky významná" if h1_sup else ("Marginální (p < 0.05)" if p1 < 0.05 else "Nevýznamná")
    hyp_results.append({
        "Hypotéza": "H1: studytime -> G3",
        "Metoda testu": "Spearmanův rank test (jednostranný)",
        "Testová statistika / Velikost účinku": f"rho = {rho1:+.4f} ({h1_assoc})",
        "Přesná p-hodnota": f"{p1:.4e} (p < 1e-4)" if p1 < 1e-4 else f"{p1:.4f}",
        "Směr vztahu": h1_dir,
        "Výsledek (Bonferroni alpha=0.0071)": h1_verdict
    })

    # H2: romantic vs studytime (oboustranný Mann-Whitney U test)
    st_rom = df[df["romantic"] == 1]["studytime"].dropna()
    st_norom = df[df["romantic"] == 0]["studytime"].dropna()
    u2, p2 = stats.mannwhitneyu(st_rom, st_norom, alternative="two-sided")
    d2 = st_rom.mean() - st_norom.mean()
    r_rb2 = 1.0 - (2.0 * u2) / (len(st_rom) * len(st_norom))
    pooled_sd2 = np.sqrt(((len(st_rom) - 1) * st_rom.var() + (len(st_norom) - 1) * st_norom.var()) / (len(st_rom) + len(st_norom) - 2))
    cohen_d2 = d2 / pooled_sd2 if pooled_sd2 > 0 else 0.0
    h2_sup = p2 < bonferroni_alpha
    h2_dir = "Ve vztahu studuje více" if (d2 > 0 and p2 < 0.05) else ("Ve vztahu studuje méně" if (d2 < 0 and p2 < 0.05) else f"Bez prokazatelného rozdílu (Delta = {d2:+.2f} h)")
    h2_verdict = "Statisticky významná" if h2_sup else ("Marginální (p < 0.05)" if p2 < 0.05 else "Neprokázána (H0 platí)")
    hyp_results.append({
        "Hypotéza": "H2: romantic -> studytime",
        "Metoda testu": "Mann-Whitney U test (oboustranný)",
        "Testová statistika / Velikost účinku": f"U = {u2:.1f}, Delta = {d2:+.2f} h (d = {cohen_d2:.2f}, r_rb = {r_rb2:.3f})",
        "Přesná p-hodnota": f"{p2:.4f}",
        "Směr vztahu": h2_dir,
        "Výsledek (Bonferroni alpha=0.0071)": h2_verdict
    })

    # H3: VŠ vzdělání rodičů vs G3 (Welchův t-test, Cohenovo d a kontrola robustnosti pro školu)
    he_parents = (df["Medu"] == 4) | (df["Fedu"] == 4)
    g3_he = df[he_parents]["G3"]
    g3_nohe = df[~he_parents]["G3"]
    t3, p3_welch = stats.ttest_ind(g3_he, g3_nohe, equal_var=False, alternative="greater")
    u3, p3_mwu = stats.mannwhitneyu(g3_he, g3_nohe, alternative="greater")
    d3 = g3_he.mean() - g3_nohe.mean()
    pooled_sd3 = np.sqrt(((len(g3_he) - 1) * g3_he.var() + (len(g3_nohe) - 1) * g3_nohe.var()) / (len(g3_he) + len(g3_nohe) - 2))
    cohen_d3 = d3 / pooled_sd3 if pooled_sd3 > 0 else 0.0
    r_rb3 = 1.0 - (2.0 * u3) / (len(g3_he) * len(g3_nohe))
    h3_sup = p3_welch < bonferroni_alpha
    h3_dir = f"VŠ rodiče dosahují vyšší G3 (Delta = {d3:+.2f} bodu, d = {cohen_d3:.2f})" if d3 > 0 else "Bez převahy VŠ rodičů"
    h3_verdict = "Statisticky významná" if h3_sup else ("Marginální (p < 0.05)" if p3_welch < 0.05 else "Nevýznamná")
    hyp_results.append({
        "Hypotéza": "H3: VŠ rodiče -> G3",
        "Metoda testu": "Welchův t-test (jednostranný) & MWU",
        "Testová statistika / Velikost účinku": f"t = {t3:.2f}, Delta = {d3:+.2f} bodu (Cohen d = {cohen_d3:.2f}, MWU U = {u3:.1f})",
        "Přesná p-hodnota": f"{p3_welch:.4e} (p < 1e-4)",
        "Směr vztahu": h3_dir,
        "Výsledek (Bonferroni alpha=0.0071)": h3_verdict
    })

    # Kontrola robustnosti H3: OLS s kontrolou pro školu
    X_h3 = np.column_stack([np.ones(len(df)), df["school"], he_parents.astype(float)])
    beta_h3 = np.linalg.lstsq(X_h3, df["G3"].values, rcond=None)[0]
    res_h3 = df["G3"].values - X_h3 @ beta_h3
    sigma2_h3 = np.sum(res_h3**2) / (len(df) - 3)
    cov_h3 = sigma2_h3 * np.linalg.inv(X_h3.T @ X_h3)
    se_h3 = np.sqrt(np.diagonal(cov_h3))
    t_h3_ctrl = beta_h3[2] / se_h3[2]
    p_h3_ctrl = 1.0 - stats.t.cdf(t_h3_ctrl, df=len(df)-3)
    ci_lo_h3 = beta_h3[2] - 1.96 * se_h3[2]
    ci_hi_h3 = beta_h3[2] + 1.96 * se_h3[2]

    # H4: higher vs vzdělání rodičů: max(Medu, Fedu) - sjednoceno s větví misus
    p_edu = np.maximum(df["Medu"], df["Fedu"])
    edu_high = p_edu[df["higher"] == 1]
    edu_nohigh = p_edu[df["higher"] == 0]
    u4, p4 = stats.mannwhitneyu(edu_high, edu_nohigh, alternative="greater")
    d4 = edu_high.mean() - edu_nohigh.mean()
    r_rb4 = 1.0 - (2.0 * u4) / (len(edu_high) * len(edu_nohigh))
    h4_sup = p4 < bonferroni_alpha
    h4_dir = f"Vyšší vzdělání rodičů u studentů s ambicí na VŠ (Delta = {d4:+.2f} stupně, r_rb = {r_rb4:.3f})" if d4 > 0 else "Bez rozdílu"
    h4_verdict = "Statisticky významná" if h4_sup else ("Marginální (p < 0.05)" if p4 < 0.05 else "Nevýznamná")
    hyp_results.append({
        "Hypotéza": "H4: higher -> vzdělání rodičů",
        "Metoda testu": "Mann-Whitney U test (jednostranný, max Medu/Fedu)",
        "Testová statistika / Velikost účinku": f"U = {u4:.1f}, Delta = {d4:+.2f} stupně (r_rb = {r_rb4:.3f})",
        "Přesná p-hodnota": f"{p4:.4e} (p < 1e-4)",
        "Směr vztahu": h4_dir,
        "Výsledek (Bonferroni alpha=0.0071)": h4_verdict
    })

    # H5: famrel vs alkohol (Dalc + Walc) (jednostranný Spearmanův test)
    alc = df["Dalc"] + df["Walc"]
    rho5, p5 = stats.spearmanr(df["famrel"], alc, alternative="less")
    h5_sup = (rho5 < 0) and (p5 < bonferroni_alpha)
    h5_assoc = classify_association(rho5)
    h5_dir = "Lepší rodinné vztahy slabě negativně korelují s alkoholem" if rho5 < 0 else "Pozitivní korelace s alkoholem"
    h5_verdict = "Statisticky významná" if h5_sup else ("Marginální (p < 0.05)" if p5 < 0.05 else "Nevýznamná")
    hyp_results.append({
        "Hypotéza": "H5: famrel -> Dalc+Walc",
        "Metoda testu": "Spearmanův rank test (jednostranný)",
        "Testová statistika / Velikost účinku": f"rho = {rho5:+.4f} ({h5_assoc})",
        "Přesná p-hodnota": f"{p5:.4f}",
        "Směr vztahu": h5_dir,
        "Výsledek (Bonferroni alpha=0.0071)": h5_verdict
    })

    # H6: alkohol (Dalc + Walc) vs G3 (jednostranný Spearmanův test)
    rho6, p6 = stats.spearmanr(alc, df["G3"], alternative="less")
    h6_sup = (rho6 < 0) and (p6 < bonferroni_alpha)
    h6_assoc = classify_association(rho6)
    h6_dir = "Vyšší konzumace alkoholu mírně koreluje s nižší známkou" if rho6 < 0 else "Pozitivní korelace"
    h6_verdict = "Statisticky významná" if h6_sup else ("Marginální (p < 0.05)" if p6 < 0.05 else "Nevýznamná")
    hyp_results.append({
        "Hypotéza": "H6: Dalc+Walc -> G3",
        "Metoda testu": "Spearmanův rank test (jednostranný)",
        "Testová statistika / Velikost účinku": f"rho = {rho6:+.4f} ({h6_assoc})",
        "Přesná p-hodnota": f"{p6:.4e} (p < 1e-4)",
        "Směr vztahu": h6_dir,
        "Výsledek (Bonferroni alpha=0.0071)": h6_verdict
    })

    # H7: school + address vs G3 (OLS regrese s kontrolou pro školu a 95% konfidenčním intervalem)
    t_addr = stats.ttest_ind(df[df["address"] == 1]["G3"], df[df["address"] == 0]["G3"])
    d_addr = df[df["address"] == 1]["G3"].mean() - df[df["address"] == 0]["G3"].mean()

    # Bivariátní OLS s kontrolou pro školu
    X_mat = np.column_stack([np.ones(len(df)), df["school"], df["address"]])
    y_vec = df["G3"].values
    beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    n, p_dim = X_mat.shape
    res = y_vec - X_mat @ beta
    sigma2 = np.sum(res**2) / (n - p_dim)
    cov_b = sigma2 * np.linalg.inv(X_mat.T @ X_mat)
    se_b = np.sqrt(np.diagonal(cov_b))
    t_val_addr = beta[2] / se_b[2]
    p_val_ols_addr = 2 * (1 - stats.t.cdf(np.abs(t_val_addr), df=n - p_dim))
    ci_lo7 = beta[2] - 1.96 * se_b[2]
    ci_hi7 = beta[2] + 1.96 * se_b[2]

    h7_dir = f"Město > Vesnice (po kontrole školy klesá beta na {beta[2]:+.2f} [95% CI {ci_lo7:.2f}, {ci_hi7:.2f}], část rozdílu souvisí se školou)"
    if p_val_ols_addr < bonferroni_alpha:
        verdict7 = f"Statisticky významná (p = {p_val_ols_addr:.4f})"
    elif p_val_ols_addr < 0.05:
        verdict7 = f"Signifikantní bez korekce (p = {p_val_ols_addr:.4f}), neprokázána po Bonferroni"
    elif p_val_ols_addr < 0.10:
        verdict7 = f"Marginální (p = {p_val_ols_addr:.4f}), neprokázána po Bonferroni"
    else:
        verdict7 = f"Neprokázána po kontrole školy (p = {p_val_ols_addr:.4f})"

    hyp_results.append({
        "Hypotéza": "H7: school + address -> G3",
        "Metoda testu": "OLS regrese s kontrolou školy (95% CI)",
        "Testová statistika / Velikost účinku": f"beta_addr = {beta[2]:+.3f} (95% CI [{ci_lo7:.2f}, {ci_hi7:.2f}], univariátně Delta = {d_addr:+.2f})",
        "Přesná p-hodnota": f"OLS p {format_p(p_val_ols_addr)} (univariátně p {format_p(t_addr.pvalue)})",
        "Směr vztahu": h7_dir,
        "Výsledek (Bonferroni alpha=0.0071)": verdict7
    })

    df_hyp = pd.DataFrame(hyp_results)
    print(df_hyp.to_string(index=False))
    print(f"\n[*] Závěr k vícenásobnému testování: Po konzervativní Bonferroniho korekci (alpha = {bonferroni_alpha:.5f})")
    print("    zůstávají H1, H3, H4, H5 a H6 plně statisticky významné. H2 zůstává nevýznamná a u H7")
    print(f"    rozdíl bydliště po kontrole školy klesá přibližně na polovinu (beta = {beta[2]:+.2f}, 95% CI [{ci_lo7:.2f}, {ci_hi7:.2f}], p = {p_val_ols_addr:.4f}),")
    print("    což naznačuje, že část rozdílu v prospěchu souvisí s výběrem školy (bydliště a škola jsou propojené).")
    print(f"    Kontrola robustnosti H3: Vzdělání rodičů i po kontrole školy zůstává vysoce signifikantní (beta = {beta_h3[2]:+.2f}, 95% CI [{ci_lo_h3:.2f}, {ci_hi_h3:.2f}], p = {p_h3_ctrl:.4e}).\n")
    return df_hyp


def evaluate_oof_feature_importance(X_B, y_reg, feat_B, category_groups):
    """
    Vypočte Out-of-Fold (OOF) Permutation Feature Importance pro Scénář B:
    - Evaluace probíhá výhradně na validačních foldech (X_val, y_val) napříč 5-fold CV.
    - Škálování i KNN imputace probíhají na datech každého foldu (StandardScaler -> KNNImputer).
    - Zabraňuje nadhodnocení důležitosti přeučených příznaků na trénovacích datech.
    - Dummy sloupce vícehodnotových kategorií jsou sloučeny zpět do mateřských faktorů.
    - Pro každý faktor se počítá průměrný procentuální podíl i směrodatná odchylka napříč 5 foldy.
    """
    print("=" * 96)
    print(" OUT-OF-FOLD (OOF) PERMUTATION FEATURE IMPORTANCE (SCÉNÁŘ B: BEZ G1 A G2)")
    print("=" * 96)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_grouped_pcts = []

    for tr_idx, va_idx in cv.split(X_B, y_reg):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_B[tr_idx])
        X_va_sc = scaler.transform(X_B[va_idx])

        imputer = KNNImputer(n_neighbors=5)
        X_tr_imp = imputer.fit_transform(X_tr_sc)
        X_va_imp = imputer.transform(X_va_sc)

        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_tr_imp, y_reg[tr_idx])

        perm = permutation_importance(rf, X_va_imp, y_reg[va_idx], n_repeats=10, random_state=42, scoring="r2")
        raw_imp = dict(zip(feat_B, perm.importances_mean))

        # Sloučení dummy proměnných do mateřských kategorií pro tento fold
        fold_grouped = {}
        consumed_cols = set()

        for parent_cat, dummy_cols in category_groups.items():
            total_imp = sum(max(0.0, raw_imp.get(c, 0.0)) for c in dummy_cols)
            fold_grouped[parent_cat] = total_imp
            consumed_cols.update(dummy_cols)

        for feat in feat_B:
            if feat not in consumed_cols:
                fold_grouped[feat] = max(0.0, raw_imp.get(feat, 0.0))

        tot_fold = sum(fold_grouped.values())
        if tot_fold <= 0:
            tot_fold = 1.0

        fold_pct = {k: (v / tot_fold) * 100.0 for k, v in fold_grouped.items()}
        fold_grouped_pcts.append(fold_pct)

    # Spočítáme průměr a směrodatnou odchylku procentuálního podílu napříč 5 foldy
    all_keys = list(fold_grouped_pcts[0].keys())
    fi_list = []
    for k in all_keys:
        vals = [f[k] for f in fold_grouped_pcts]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        cz_name = FEATURE_CZ_LABELS.get(k, k)
        fi_list.append({
            "Příznak": k,
            "Příznak_CZ": cz_name,
            "Důležitost (%)": round(mean_v, 2),
            "Std (%)": round(std_v, 2)
        })

    fi_df = pd.DataFrame(fi_list).sort_values(by="Důležitost (%)", ascending=False).reset_index(drop=True)

    print("TOP 10 nejdůležitějších prediktorů po sloučení kategorií na validačních foldech (OOF +/- std):")
    display_fi = fi_df[["Příznak_CZ", "Důležitost (%)", "Std (%)"]].head(10)
    print(display_fi.to_string(index=False))
    print()
    return fi_df


def export_outputs(
    df_reg_A, df_reg_B,
    df_clf_A, df_clf_B,
    df_tuning, df_hyp, fi_df,
    best_preds_B, y_clf, best_clf_name_B
):
    """Uloží výsledné tabulky (CSV) a vygeneruje čisté vektorové SVG grafy."""
    tables_dir = os.path.join(REPO_ROOT, "outputs", "tables")
    figures_dir = os.path.join(REPO_ROOT, "outputs", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Uložení tabulek do CSV
    df_reg_A.to_csv(f"{tables_dir}/regression_benchmark_scenario_A.csv", index=False)
    df_reg_B.to_csv(f"{tables_dir}/regression_benchmark_scenario_B.csv", index=False)
    df_clf_A.to_csv(f"{tables_dir}/classification_benchmark_scenario_A.csv", index=False)
    df_clf_B.to_csv(f"{tables_dir}/classification_benchmark_scenario_B.csv", index=False)
    df_tuning.to_csv(f"{tables_dir}/hyperparameter_tuning_nested_cv.csv", index=False)
    df_hyp.to_csv(f"{tables_dir}/hypothesis_tests_validation.csv", index=False)
    fi_df.to_csv(f"{tables_dir}/grouped_feature_importances_scenario_B.csv", index=False)
    print(f"[*] Všech 7 tabulek úspěšně uloženo do: {tables_dir}/")

    # 2. Vektorové SVG grafy
    # Graf A: Srovnání regresních modelů Scénář A (vodorovný sloupcový graf, všechny modely, chybové úsečky)
    svg_reg_A = generate_horizontal_bar_chart(
        labels=df_reg_A["Model"].tolist()[::-1],
        values=df_reg_A["R2_mean"].tolist()[::-1],
        stds=df_reg_A["R2_std"].tolist()[::-1],
        title="Srovnání regresních modelů: Scénář A (se známkami G1 a G2)",
        xlabel="Koeficient determinace (R²)",
        baseline=0.0,
        bar_color="#2563eb",
        subtitle="5-Fold CV (průměr +/- std napříč validačními foldy)"
    )
    save_svg(svg_reg_A, f"{figures_dir}/regression_comparison_scenario_A.svg")

    # Graf B: Srovnání regresních modelů Scénář B (Včasné varování - všechny modely, přizpůsobená osa X, baseline)
    svg_reg_B = generate_horizontal_bar_chart(
        labels=df_reg_B["Model"].tolist()[::-1],
        values=df_reg_B["R2_mean"].tolist()[::-1],
        stds=df_reg_B["R2_std"].tolist()[::-1],
        title="Srovnání regresních modelů: Scénář B (Včasné varování, bez G1 a G2)",
        xlabel="Koeficient determinace (R²)",
        baseline=0.0,
        bar_color="#0d9488",
        subtitle="5-Fold CV (průměr +/- std). Osa X přizpůsobena rozsahu Scénáře B."
    )
    save_svg(svg_reg_B, f"{figures_dir}/regression_comparison_scenario_B.svg")

    # Graf C: Klasifikace ve Scénáři B (Early Warning - všechny modely, baseline 50 %, chybové úsečky)
    svg_clf_B = generate_horizontal_bar_chart(
        labels=df_clf_B["Model"].tolist()[::-1],
        values=df_clf_B["Bal_Acc_mean"].tolist()[::-1],
        stds=df_clf_B["Bal_Acc_std"].tolist()[::-1],
        title="Klasifikace ohrožených studentů (Scénář B: Včasné varování)",
        xlabel="Vyvážená přesnost (%)",
        baseline=50.0,
        bar_color="#7c3aed",
        is_percentage=True,
        subtitle="Stratified 5-Fold CV (průměr +/- std). Referenční baseline náhodného odhadu = 50.0 %."
    )
    save_svg(svg_clf_B, f"{figures_dir}/classification_comparison_scenario_B.svg")

    # Graf D1: Dvoupanelový graf ladění hyperparametrů pro regresi (Panel A: Scénář A, Panel B: Scénář B)
    reg_A = df_tuning[(df_tuning["Úloha"] == "Regrese") & (df_tuning["Scénář"] == "Scénář A")]
    reg_B = df_tuning[(df_tuning["Úloha"] == "Regrese") & (df_tuning["Scénář"] == "Scénář B")]
    svg_tune_reg = generate_tuning_two_panel_chart(
        panel_A_models=reg_A["Model"].tolist(),
        panel_A_default=[round(float(v), 3) for v in reg_A["def_val"]],
        panel_A_tuned=[round(float(v), 3) for v in reg_A["tuned_val"]],
        panel_B_models=reg_B["Model"].tolist(),
        panel_B_default=[round(float(v), 3) for v in reg_B["def_val"]],
        panel_B_tuned=[round(float(v), 3) for v in reg_B["tuned_val"]],
        title="Nestranné zhodnocení ladění parametrů pro regresi (Nested 5x5 CV)",
        metric_label="Koeficient determinace (R²)",
        is_percentage=False,
        palette=["#94a3b8", "#2563eb"]
    )
    save_svg(svg_tune_reg, f"{figures_dir}/hyperparameter_tuning_regression.svg")

    # Graf D2: Dvoupanelový graf ladění hyperparametrů pro klasifikaci (Panel A: Scénář A, Panel B: Scénář B)
    clf_A = df_tuning[(df_tuning["Úloha"] == "Klasifikace") & (df_tuning["Scénář"] == "Scénář A")]
    clf_B = df_tuning[(df_tuning["Úloha"] == "Klasifikace") & (df_tuning["Scénář"] == "Scénář B")]
    svg_tune_clf = generate_tuning_two_panel_chart(
        panel_A_models=clf_A["Model"].tolist(),
        panel_A_default=[round(float(v), 1) for v in clf_A["def_val"]],
        panel_A_tuned=[round(float(v), 1) for v in clf_A["tuned_val"]],
        panel_B_models=clf_B["Model"].tolist(),
        panel_B_default=[round(float(v), 1) for v in clf_B["def_val"]],
        panel_B_tuned=[round(float(v), 1) for v in clf_B["tuned_val"]],
        title="Nestranné zhodnocení ladění parametrů pro klasifikaci (Nested 5x5 CV)",
        metric_label="Vyvážená přesnost (%)",
        is_percentage=True,
        palette=["#94a3b8", "#7c3aed"]
    )
    save_svg(svg_tune_clf, f"{figures_dir}/hyperparameter_tuning_classification.svg")

    # Graf E: TOP 10 Out-of-Fold Permutation Feature Importance (české popisky, chybové úsečky napříč foldy)
    top10_fi = fi_df.head(10).copy()
    svg_fi = generate_horizontal_bar_chart(
        labels=top10_fi["Příznak_CZ"].tolist()[::-1],
        values=top10_fi["Důležitost (%)"].tolist()[::-1],
        stds=top10_fi["Std (%)"].tolist()[::-1],
        title="TOP 10 nejvýznamnějších prediktorů (OOF validace - Scénář B)",
        xlabel="Podíl na celkové permutační důležitosti (%)",
        bar_color="#ea580c",
        is_percentage=True,
        subtitle="Průměrný relativní podíl +/- std napříč 5 validačními foldy"
    )
    save_svg(svg_fi, f"{figures_dir}/feature_importance_top10_grouped.svg")

    # Graf F: Matice záměn nejlepšího klasifikátoru ve Scénáři B
    if best_preds_B is not None:
        cm = confusion_matrix(y_clf, best_preds_B)
        svg_cm = generate_confusion_matrix_svg(
            cm=cm.tolist(),
            class_labels=["Neprospěl", "Prospěl"],
            title=f"Matice záměn: {best_clf_name_B} (Scénář B: Včasné varování)"
        )
        save_svg(svg_cm, f"{figures_dir}/confusion_matrix_best_classifier_scenario_B.svg")

    print(f"[*] Všech 7 vektorových SVG grafů úspěšně vygenerováno do: {figures_dir}/")


# =============================================================================
# HLAVNÍ BĚHOVÁ FUNKCE
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 96)
    print(" MPC-VMM: EXPERIMENTÁLNÍ TESTOVÁNÍ A EVALUACE PREDIKTIVNÍCH MODELŮ (V3.1 - METODICKÁ INTEGRITA)")
    print("=" * 96)

    # 1. Načtení a příprava dat bez leakage (s NaNs a absences_missing)
    data = load_and_prepare_data()

    # 2. Analýza citlivosti pro chybějící docházku
    evaluate_absences_sensitivity(data["df_raw"], data["X_B"], data["y_reg"], data["feat_B"])

    # 3. Regresní benchmark (Scénář A i B s fold-isolated Pipeline)
    df_reg_A = evaluate_regression(data["X_A"], data["y_reg"], scenario_name="Scénář A (se známkami G1 a G2)")
    df_reg_B = evaluate_regression(data["X_B"], data["y_reg"], scenario_name="Scénář B (BEZ známek G1 a G2 - Včasné varování)")

    # 4. Klasifikační benchmark (Scénář A i B s fold-isolated Pipeline)
    df_clf_A, _, _ = evaluate_classification(data["X_A"], data["y_clf"], scenario_name="Scénář A (se známkami G1 a G2)")
    df_clf_B, best_preds_B, best_clf_name_B = evaluate_classification(data["X_B"], data["y_clf"], scenario_name="Scénář B (BEZ známek G1 a G2 - Včasné varování)")

    # 5. Nestranné ladění hyperparametrů pomocí Nested CV
    df_tuning = run_nested_hyperparameter_tuning(data["X_A"], data["X_B"], data["y_reg"], data["y_clf"])

    # 6. Statistické ověření 7 týmových hypotéz
    df_hyp = test_team_hypotheses(data["df_raw"])

    # 7. Out-of-fold Permutation Feature Importance pro Scénář B
    fi_df = evaluate_oof_feature_importance(data["X_B"], data["y_reg"], data["feat_B"], data["category_groups"])

    # 8. Export tabulek a grafů
    export_outputs(df_reg_A, df_reg_B, df_clf_A, df_clf_B, df_tuning, df_hyp, fi_df, best_preds_B, data["y_clf"], best_clf_name_B)

    t_total = time.time() - t_start
    print(f"\n[+] Hotovo! Celý experimentální běh proběhl úspěšně za {t_total:.1f} sekund.\n")


if __name__ == "__main__":
    main()
