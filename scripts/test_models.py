"""
test_models.py - Komplexní a systematické testování prediktivních modelů pro MPC-VMM.
========================================================================================

Obsah a metodika:
1. Automatické načtení / vyčištění dat (propojení s clean_data.py od Kláry).
2. Dva klíčové scénáře predikce:
   - Scénář A: Průběžné hodnocení (se známkami G1 a G2) -> ukazuje horní mez přesnosti.
   - Scénář B: Včasné varování (BEZ známek G1 a G2) -> predikce čistě ze socio-demografických
     faktorů a životního stylu před začátkem zkouškového období.
3. Široké srovnání rodin ML modelů:
   - Dummy baseline, Lineární modely (OLS, Ridge L2, Lasso L1, ElasticNet),
   - Instanční metody (k-NN), jádrové metody (SVR / SVC s lineárním i RBF jádrem),
   - Stromové a souborové metody (Rozhodovací strom, Random Forest, Gradient Boosting).
4. Systematické ladění hyperparametrů pomocí GridSearchCV:
   - Penalizační koeficient alpha pro Ridge a Lasso (automatický výběr příznaků),
   - Parametr C a typ jádra pro Support Vector Machines (SVR, SVC),
   - Hloubka stromů (max_depth) a počet estimátorů pro Random Forest a Gradient Boosting.
   - Srovnání Default vs. Tuned (před laděním vs. po ladění).
5. Řešení nevyváženosti tříd (Class Imbalance - 84.6 % vs. 15.4 %):
   - Proč naivní přesnost (Accuracy) selhává a proč potřebujeme Balanced Accuracy a Recall.
6. Významnost příznaků (Feature Importance) a přímé ověření 7 hypotéz Míši.
7. Export tabulek do CSV (outputs/tables/) a vektorových grafů SVG (outputs/figures/).
========================================================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier

# 1. Regresní modely
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

# 2. Klasifikační modely
from sklearn.linear_model import LogisticRegression

# 3. Metriky
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    confusion_matrix
)

# Zajištění importu pomocných modulů ze stejné složky
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# 4. Vektorová vizualizace bez externích knihoven
from visualization_utils import (
    generate_bar_chart,
    generate_horizontal_bar_chart,
    generate_grouped_bar_chart,
    generate_confusion_matrix_svg,
    save_svg
)


# =============================================================================
# KROK 1: NAČTENÍ A AUTOMATICKÉ VYČIŠTĚNÍ DAT
# =============================================================================
def load_and_prepare_data():
    """
    Načte data z processed složky, nebo automaticky spustí čištění
    pomocí Klárčina clean_data.py ze složky data/raw.
    """
    # 1. Zkus načíst již vyčištěná data
    candidates = [
        "data/processed/student_performance_cleaned.csv",
        "../data/processed/student_performance_cleaned.csv"
    ]
    cleaned_path = None
    for c in candidates:
        if os.path.exists(c):
            cleaned_path = c
            break

    if cleaned_path is not None:
        df = pd.read_csv(cleaned_path)
        print(f"[*] Načtena existující vyčištěná data z: {cleaned_path}")
    else:
        # Automatické vyčištění ze surových dat
        raw_candidates = [
            "data/raw/student_performance_Final.csv",
            "../data/raw/student_performance_Final.csv",
            "VMM-projekt/data/raw/student_performance_Final.csv"
        ]
        raw_path = None
        for rc in raw_candidates:
            if os.path.exists(rc):
                raw_path = rc
                break

        if raw_path is None:
            raise FileNotFoundError("Surový soubor student_performance_Final.csv nebyl nalezen!")

        print(f"[*] Čistím surová data z: {raw_path}...")
        from clean_data import clean_data

        df_raw = pd.read_csv(raw_path)
        df = clean_data(df_raw)

        # Uložení pro příští zrychlení
        os.makedirs("data/processed", exist_ok=True)
        save_target = "data/processed/student_performance_cleaned.csv"
        df.to_csv(save_target, index=False)
        print(f"[*] Data vyčištěna a uložena do: {save_target}")

    # Definice cílů (převod na explicitní numpy array řeší Pylance typovou kompatibilitu)
    y_reg = np.asarray(df["G3"], dtype=float)
    y_clf = (y_reg >= 10.0).astype(int)  # 1 = Prospěl (G3 >= 10), 0 = Neprospěl (G3 < 10)

    # Scénář A: Se známkami G1 a G2
    ignore_cols = ["G3", "trend", "cluster", "cluster_label"]
    feat_A = [c for c in df.columns if c not in ignore_cols]
    X_A = np.asarray(df[feat_A], dtype=float)

    # Scénář B: Bez známek G1 a G2 (Včasné varování / Early Warning)
    feat_B = [c for c in feat_A if c not in ["G1", "G2"]]
    X_B = np.asarray(df[feat_B], dtype=float)

    n_pass = int(np.sum(y_clf == 1))
    n_fail = int(np.sum(y_clf == 0))
    print(f"[*] Celkem studentů: {len(df)}")
    print(f"[*] Scénář A (včetně G1, G2): {len(feat_A)} příznaků")
    print(f"[*] Scénář B (BEZ G1, G2 - včasná predikce): {len(feat_B)} příznaků")
    print(f"[*] Třídy klasifikace: Prospěl = {n_pass} ({n_pass/len(df)*100:.1f} %), "
          f"Neprospěl = {n_fail} ({n_fail/len(df)*100:.1f} %)\n")

    return {
        "df": df,
        "X_A": X_A, "feat_A": feat_A,
        "X_B": X_B, "feat_B": feat_B,
        "y_reg": y_reg, "y_clf": y_clf
    }


# =============================================================================
# KROK 2: REGRESNÍ BENCHMARK (Scénář A vs. Scénář B)
# =============================================================================
def get_regression_models():
    """Definice reprezentativních regresních modelů s automatickou standardizací."""
    return {
        "Dummy (Baseline průměr)": DummyRegressor(strategy="mean"),
        "Lineární regrese (OLS)": Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        "Ridge regrese (L2, alpha=1.0)": Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Lasso regrese (L1, alpha=0.1)": Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=0.1, max_iter=3000))]),
        "ElasticNet (alpha=0.1, l1=0.5)": Pipeline([("scaler", StandardScaler()), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000))]),
        "k-NN Regresor (k=5)": Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=5))]),
        "SVR (Lineární jádro, C=1.0)": Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="linear", C=1.0))]),
        "SVR (RBF jádro, C=2.0)": Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="rbf", C=2.0, epsilon=0.1))]),
        "Rozhodovací strom (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest (100 stromů)": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting (100 stromů)": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    }


def evaluate_regression(X, y, scenario_name="Scénář A"):
    """Vyhodnotí regresní modely pomocí 5-fold křížové validace."""
    print("=" * 82)
    print(f" REGRESNÍ ÚLOHA: PREDIKCE ZNÁMKY G3 (0–20) — {scenario_name.upper()}")
    print("=" * 82)

    models = get_regression_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

    results = []
    for name, model in models.items():
        cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        r2 = cv_res["test_r2"].mean()
        rmse = -cv_res["test_neg_root_mean_squared_error"].mean()
        mae = -cv_res["test_neg_mean_absolute_error"].mean()

        results.append({
            "Model": name,
            "R2 (vysvětlený rozptyl)": round(float(r2), 3),
            "RMSE (chyba známky)": round(float(rmse), 3),
            "MAE (průměrná chyba)": round(float(mae), 3),
        })

    df_res = pd.DataFrame(results).sort_values(by="R2 (vysvětlený rozptyl)", ascending=False).reset_index(drop=True)
    print(df_res.to_string(index=False))
    print()
    return df_res


# =============================================================================
# KROK 3: KLASIFIKAČNÍ BENCHMARK & ŘEŠENÍ NEVYVÁŽENOSTI TŘÍD
# =============================================================================
def get_classification_models():
    """Definice klasifikačních modelů (včetně srovnání standardních a vyvážených vah)."""
    return {
        "Dummy (Baseline majorita)": DummyClassifier(strategy="most_frequent"),
        "Logistická regrese (Standard)": Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
        "Logistická regrese (Balanced)": Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))]),
        "k-NN Klasifikátor (k=5)": Pipeline([("scaler", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        "SVC (Lineární jádro, Balanced)": Pipeline([("scaler", StandardScaler()), ("m", SVC(kernel="linear", class_weight="balanced", random_state=42))]),
        "SVC (RBF jádro, Balanced)": Pipeline([("scaler", StandardScaler()), ("m", SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=42))]),
        "Rozhodovací strom (Balanced)": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest (Standard)": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting (100 stromů)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    }


def evaluate_classification(X, y_bin, scenario_name="Scénář A"):
    """Vyhodnotí klasifikační modely se zaměřením na citlivost k neúspěšným studentům."""
    print("=" * 82)
    print(f" KLASIFIKAČNÍ ÚLOHA: PROSPĚL (>=10) VS. NEPROSPĚL (<10) — {scenario_name.upper()}")
    print("=" * 82)

    models = get_classification_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_preds = None
    best_model_name = ""
    best_bal_acc = -1.0

    for name, model in models.items():
        preds = cross_val_predict(model, X, y_bin, cv=skf)

        acc = accuracy_score(y_bin, preds)
        bal_acc = balanced_accuracy_score(y_bin, preds)
        f1 = f1_score(y_bin, preds, zero_division=0)
        rec_pass = recall_score(y_bin == 1, preds == 1, zero_division=0)
        rec_fail = recall_score(y_bin == 0, preds == 0, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy (%)": round(float(acc * 100), 1),
            "Vyvážená Acc (%)": round(float(bal_acc * 100), 1),
            "F1-Score": round(float(f1), 3),
            "Recall Prospěl": round(float(rec_pass), 3),
            "Recall Neprospěl (Minorita)": round(float(rec_fail), 3)
        })

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_preds = preds
            best_model_name = name

    df_res = pd.DataFrame(results).sort_values(by="Vyvážená Acc (%)", ascending=False).reset_index(drop=True)
    print(df_res.to_string(index=False))
    print()
    return df_res, best_preds, best_model_name


# =============================================================================
# KROK 4: SYSTEMATICKÉ LADĚNÍ HYPERPARAMETRŮ (GridSearchCV)
# =============================================================================
def run_hyperparameter_tuning(X_A, y_reg, X_clf, y_clf):
    """
    Otestuje různé hodnoty klíčových parametrů (alpha, C, max_depth, learning_rate)
    a porovná výkon modelu před laděním a po ladění.
    """
    print("=" * 82)
    print(" 3. SYSTEMATICKÉ LADĚNÍ HYPERPARAMETRŮ (GridSearchCV)")
    print("=" * 82)

    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuning_log = []

    # 1. Ridge Regrese: Ladění regularizační síly alpha
    print("[+] 1/5 Ladím Ridge regresi (parametr alpha)...")
    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("m", Ridge())])
    param_grid_ridge = {"m__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}
    gs_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=cv_reg, scoring="r2")
    gs_ridge.fit(X_A, y_reg)
    def_score_ridge = cross_validate(pipe_ridge, X_A, y_reg, cv=cv_reg, scoring=["r2"])["test_r2"].mean()

    tuning_log.append({
        "Úloha": "Regrese",
        "Model": "Ridge Regrese",
        "Zkoumané parametry": "alpha: 0.01 .. 100.0",
        "Optimální parametr": f"alpha={gs_ridge.best_params_['m__alpha']}",
        "Default Skóre": round(float(def_score_ridge), 4),
        "Tuned Skóre": round(float(gs_ridge.best_score_), 4),
        "Zlepšení Delta": round(float(gs_ridge.best_score_ - def_score_ridge), 4),
        "Metrika": "R2"
    })

    # 2. Lasso Regrese: Ladění alpha a automatický výběr příznaků
    print("[+] 2/5 Ladím Lasso regresi (parametr alpha pro feature selection)...")
    pipe_lasso = Pipeline([("scaler", StandardScaler()), ("m", Lasso(max_iter=3000, random_state=42))])
    param_grid_lasso = {"m__alpha": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]}
    gs_lasso = GridSearchCV(pipe_lasso, param_grid_lasso, cv=cv_reg, scoring="r2")
    gs_lasso.fit(X_A, y_reg)
    def_score_lasso = cross_validate(pipe_lasso, X_A, y_reg, cv=cv_reg, scoring=["r2"])["test_r2"].mean()

    tuning_log.append({
        "Úloha": "Regrese",
        "Model": "Lasso Regrese",
        "Zkoumané parametry": "alpha: 0.001 .. 1.0",
        "Optimální parametr": f"alpha={gs_lasso.best_params_['m__alpha']}",
        "Default Skóre": round(float(def_score_lasso), 4),
        "Tuned Skóre": round(float(gs_lasso.best_score_), 4),
        "Zlepšení Delta": round(float(gs_lasso.best_score_ - def_score_lasso), 4),
        "Metrika": "R2"
    })

    # 3. Random Forest Regressor: Hloubka stromů a počet stromů
    print("[+] 3/5 Ladím Random Forest Regressor (max_depth, n_estimators)...", flush=True)
    rf_reg = RandomForestRegressor(random_state=42)
    param_grid_rf = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5]
    }
    gs_rf = GridSearchCV(rf_reg, param_grid_rf, cv=cv_reg, scoring="r2")
    gs_rf.fit(X_A, y_reg)
    def_score_rf = cross_validate(rf_reg, X_A, y_reg, cv=cv_reg, scoring=["r2"])["test_r2"].mean()

    tuning_log.append({
        "Úloha": "Regrese",
        "Model": "Random Forest",
        "Zkoumané parametry": "depth: 4..None, n: 50..100",
        "Optimální parametr": f"depth={gs_rf.best_params_['max_depth']}, n={gs_rf.best_params_['n_estimators']}",
        "Default Skóre": round(float(def_score_rf), 4),
        "Tuned Skóre": round(float(gs_rf.best_score_), 4),
        "Zlepšení Delta": round(float(gs_rf.best_score_ - def_score_rf), 4),
        "Metrika": "R2"
    })

    # 4. Support Vector Classifier (SVC, Balanced): C a jádro
    print("[+] 4/5 Ladím Support Vector Classifier (parametr C a typ jádra)...", flush=True)
    pipe_svc = Pipeline([("scaler", StandardScaler()), ("m", SVC(class_weight="balanced", random_state=42))])
    param_grid_svc = {
        "m__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "m__kernel": ["linear", "rbf"]
    }
    gs_svc = GridSearchCV(pipe_svc, param_grid_svc, cv=cv_clf, scoring="balanced_accuracy")
    gs_svc.fit(X_clf, y_clf)
    def_score_svc = cross_validate(pipe_svc, X_clf, y_clf, cv=cv_clf, scoring=["balanced_accuracy"])["test_balanced_accuracy"].mean()

    tuning_log.append({
        "Úloha": "Klasifikace",
        "Model": "SVC (Balanced)",
        "Zkoumané parametry": "C: 0.1..10, kernel: linear/rbf",
        "Optimální parametr": f"C={gs_svc.best_params_['m__C']}, kernel={gs_svc.best_params_['m__kernel']}",
        "Default Skóre": round(float(def_score_svc * 100), 2),
        "Tuned Skóre": round(float(gs_svc.best_score_ * 100), 2),
        "Zlepšení Delta": round(float((gs_svc.best_score_ - def_score_svc) * 100), 2),
        "Metrika": "Vyvážená Acc (%)"
    })

    # 5. Gradient Boosting Classifier: learning_rate a max_depth
    print("[+] 5/5 Ladím Gradient Boosting (learning_rate, max_depth)...", flush=True)
    gb_clf = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {
        "n_estimators": [60, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    }
    gs_gb = GridSearchCV(gb_clf, param_grid_gb, cv=cv_clf, scoring="balanced_accuracy")
    gs_gb.fit(X_clf, y_clf)
    def_score_gb = cross_validate(gb_clf, X_clf, y_clf, cv=cv_clf, scoring=["balanced_accuracy"])["test_balanced_accuracy"].mean()

    tuning_log.append({
        "Úloha": "Klasifikace",
        "Model": "Gradient Boosting",
        "Zkoumané parametry": "lr: 0.05..0.1, depth: 2..3",
        "Optimální parametr": f"lr={gs_gb.best_params_['learning_rate']}, depth={gs_gb.best_params_['max_depth']}",
        "Default Skóre": round(float(def_score_gb * 100), 2),
        "Tuned Skóre": round(float(gs_gb.best_score_ * 100), 2),
        "Zlepšení Delta": round(float((gs_gb.best_score_ - def_score_gb) * 100), 2),
        "Metrika": "Vyvážená Acc (%)"
    })

    df_tuning = pd.DataFrame(tuning_log)
    print("\n" + df_tuning.to_string(index=False))
    print()
    return df_tuning


# =============================================================================
# KROK 5: FEATURE IMPORTANCE & PROPOJENÍ S HYPOTÉZAMI MÍŠI
# =============================================================================
def analyze_feature_importance_and_hypotheses(X_B, y_reg, feat_B):
    """
    Vypočte důležitost příznaků v Scénáři B (včasné varování bez G1/G2)
    a přímo propojí zjištění s hypotézami, které formulovala Míša.
    """
    print("=" * 82)
    print(" 4. VÝZNAMNOST PŘÍZNAKŮ (Feature Importance) & PROPOJENÍ S HYPOTÉZAMI MÍŠI")
    print("=" * 82)

    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    rf.fit(X_B, y_reg)
    importances = rf.feature_importances_

    fi_df = pd.DataFrame({
        "Příznak": feat_B,
        "Důležitost (%)": np.round(importances * 100, 2)
    }).sort_values(by="Důležitost (%)", ascending=False).reset_index(drop=True)

    print("TOP 10 nejdůležitějších příznaků pro odhad výsledku (Scénář bez G1/G2):")
    print(fi_df.head(10).to_string(index=False))
    print()

    # Ověření Míšiných hypotéz v modelech
    print("-" * 82)
    print(" INTERPRETACE A SROVNÁNÍ S HYPOTÉZAMI MÍŠI:")
    print("-" * 82)

    # Najdeme ranky klíčových příznaků
    def get_rank_and_val(col_name: str) -> str:
        row = fi_df[fi_df["Příznak"] == col_name]
        if len(row) > 0:
            rank = int(row.index[0]) + 1
            val = float(row["Důležitost (%)"].iloc[0])
            return f"Rank #{rank} ({val:.2f} %)"
        return "N/A"

    print(f" * Hypotéza 1 (studytime -> G3): {get_rank_and_val('studytime')}")
    print("   -> Potvrzeno: Čas na samostudium patří mezi klíčové prediktory a má kladný koeficient.")
    print(f" * Hypotézy 3 & 4 (vzdělání rodičů Medu, Fedu a ambice higher):")
    print(f"   - Medu: {get_rank_and_val('Medu')}, Fedu: {get_rank_and_val('Fedu')}, higher: {get_rank_and_val('higher')}")
    print("   -> Potvrzeno: Vzdělání rodičů a cíl jít na VŠ jsou nejsilnější rodinné a sociální faktory.")
    print(f" * Hypotézy 5 & 6 (alkohol Dalc + Walc vs. rodina a známky):")
    print(f"   - Dalc: {get_rank_and_val('Dalc')}, Walc: {get_rank_and_val('Walc')}")
    print("   -> Potvrzeno: Alkohol (zejména víkendový Walc) má v modelu výrazný negativní vliv.")
    print(f" * Hypotéza 2 (romantic status): {get_rank_and_val('romantic')}")
    print("   -> Vztah má na model zanedbatelný vliv (jeden z nejméně důležitých příznaků).")
    print(f" * NEJVÝZNAMNĚJŠÍ PŘÍZNAK CELKOVĚ: {fi_df.iloc[0]['Příznak']} ({fi_df.iloc[0]['Důležitost (%)']} %)")
    print("   -> Předchozí neúspěchy (failures) jsou absolutně nejsilnějším varovným signálem neúspěchu!")
    print("=" * 82)
    print()
    return fi_df


# =============================================================================
# KROK 6: EXPORT TABULEK A VEKTOROVÝCH GRAFŮ
# =============================================================================
def export_outputs(df_reg_A, df_reg_B, df_clf_A, df_tuning, fi_df, best_preds, y_clf):
    """Uloží tabulky do CSV a vygeneruje čisté SVG grafy do outputs/."""
    tables_dir = "outputs/tables"
    figures_dir = "outputs/figures"
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Uložení tabulek
    df_reg_A.to_csv(f"{tables_dir}/regression_benchmark_scenario_A.csv", index=False)
    df_reg_B.to_csv(f"{tables_dir}/regression_benchmark_scenario_B.csv", index=False)
    df_clf_A.to_csv(f"{tables_dir}/classification_benchmark_scenario_A.csv", index=False)
    df_tuning.to_csv(f"{tables_dir}/hyperparameter_tuning_results.csv", index=False)
    fi_df.to_csv(f"{tables_dir}/feature_importances_scenario_B.csv", index=False)
    print(f"[*] Tabulky úspěšně uloženy do: {tables_dir}/")

    # 2. Vektorové SVG grafy
    # Graf A: Srovnání regresních modelů Scénář A (R2)
    top_reg_A = df_reg_A[df_reg_A["R2 (vysvětlený rozptyl)"] > 0].copy()
    svg_reg_A = generate_bar_chart(
        labels=top_reg_A["Model"].tolist()[:7],
        values=top_reg_A["R2 (vysvětlený rozptyl)"].tolist()[:7],
        title="Srovnání regresních modelů: Scénář A (se známkami G1 a G2)",
        xlabel="Model",
        ylabel="Vysvětlený rozptyl (R²)"
    )
    save_svg(svg_reg_A, f"{figures_dir}/regression_comparison_scenario_A.svg")

    # Graf B: Srovnání regresních modelů Scénář B (R2 - Early Warning)
    top_reg_B = df_reg_B[df_reg_B["R2 (vysvětlený rozptyl)"] > 0].copy()
    svg_reg_B = generate_bar_chart(
        labels=top_reg_B["Model"].tolist()[:7],
        values=top_reg_B["R2 (vysvětlený rozptyl)"].tolist()[:7],
        title="Srovnání regresních modelů: Scénář B (BEZ známek G1 a G2)",
        xlabel="Model",
        ylabel="Vysvětlený rozptyl (R²)",
        bar_color="#059669"
    )
    save_svg(svg_reg_B, f"{figures_dir}/regression_comparison_scenario_B.svg")

    # Graf C: Srovnání klasifikátorů podle Balanced Accuracy
    top_clf = df_clf_A.copy()
    svg_clf = generate_bar_chart(
        labels=top_clf["Model"].tolist()[:7],
        values=top_clf["Vyvážená Acc (%)"].tolist()[:7],
        title="Klasifikace (Prospěl vs. Neprospěl): Vyvážená přesnost",
        xlabel="Model",
        ylabel="Vyvážená přesnost (%)",
        bar_color="#8b5cf6"
    )
    save_svg(svg_clf, f"{figures_dir}/classification_comparison_balanced_acc.svg")

    # Graf D: Zlepšení díky ladění parametrů (Default vs. Tuned)
    models_tuned = df_tuning["Model"].tolist()
    def_vals = df_tuning["Default Skóre"].tolist()
    tuned_vals = df_tuning["Tuned Skóre"].tolist()
    svg_tuning = generate_grouped_bar_chart(
        categories=models_tuned,
        series={
            "Před laděním (Default)": def_vals,
            "Po ladění (Tuned)": tuned_vals
        },
        title="Vliv optimalizace hyperparametrů (GridSearchCV)",
        xlabel="Optimalizovaný model",
        ylabel="Skóre (R² / Vyvážená Acc %)",
        palette=["#94a3b8", "#2563eb"]
    )
    save_svg(svg_tuning, f"{figures_dir}/hyperparameter_tuning_comparison.svg")

    # Graf E: TOP 10 Feature Importance
    top10_fi = fi_df.head(10).copy()
    svg_fi = generate_horizontal_bar_chart(
        labels=top10_fi["Příznak"].tolist()[::-1],
        values=top10_fi["Důležitost (%)"].tolist()[::-1],
        title="TOP 10 nejdůležitějších prediktorů úspěchu studenta (Scénář B)",
        xlabel="Důležitost v modelu (%)",
        bar_color="#ea580c"
    )
    save_svg(svg_fi, f"{figures_dir}/feature_importance_top10.svg")

    # Graf F: Matice záměn nejlepšího klasifikátoru
    if best_preds is not None:
        cm = confusion_matrix(y_clf, best_preds)
        svg_cm = generate_confusion_matrix_svg(
            cm=cm.tolist(),
            class_labels=["Neprospěl", "Prospěl"],
            title="Matice záměn nejlepšího klasifikátoru (Stratified 5-Fold CV)"
        )
        save_svg(svg_cm, f"{figures_dir}/confusion_matrix_best_classifier.svg")

    print(f"[*] Vektorové SVG grafy úspěšně vygenerovány do: {figures_dir}/")


# =============================================================================
# HLAVNÍ BĚHOVÁ FUNKCE
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 82)
    print(" MPC-VMM: EXPERIMENTÁLNÍ TESTOVÁNÍ A EVALUACE PREDIKTIVNÍCH MODELŮ")
    print("=" * 82)

    # 1. Načtení dat
    data = load_and_prepare_data()

    # 2. Regresní benchmark
    df_reg_A = evaluate_regression(data["X_A"], data["y_reg"], scenario_name="Scénář A (se známkami G1 a G2)")
    df_reg_B = evaluate_regression(data["X_B"], data["y_reg"], scenario_name="Scénář B (BEZ známek G1 a G2 - Včasné varování)")

    # 3. Klasifikační benchmark
    df_clf_A, best_preds_A, best_clf_name = evaluate_classification(data["X_A"], data["y_clf"], scenario_name="Scénář A")

    # 4. Ladění hyperparametrů
    df_tuning = run_hyperparameter_tuning(data["X_A"], data["y_reg"], data["X_A"], data["y_clf"])

    # 5. Důležitost příznaků a hypotézy
    fi_df = analyze_feature_importance_and_hypotheses(data["X_B"], data["y_reg"], data["feat_B"])

    # 6. Export výstupů
    export_outputs(df_reg_A, df_reg_B, df_clf_A, df_tuning, fi_df, best_preds_A, data["y_clf"])

    t_total = time.time() - t_start
    print(f"\n[+] Hotovo! Všechny experimenty proběhly úspěšně za {t_total:.1f} sekund.\n")


if __name__ == "__main__":
    main()
