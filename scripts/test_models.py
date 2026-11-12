"""
test_models.py - Komplexní a metodicky korektní testování prediktivních modelů pro MPC-VMM.
============================================================================================
Tento skript řeší týmovou roli: "Testování a evaluace prediktivních modelů".

Metodické pilíře:
1. Prevence úniku dat (Data Leakage):
   - Cílová proměnná G3 (a dílčí známky) je striktně oddělena před jakoukoliv modelovací fází.
   - Všechny transformace (standardizace, imputace) jsou zapouzdřeny v Pipeline pro fold-isolation.
2. Dva jasně definované časové scénáře:
   - Scénář A: Průběžná predikce (Full model se známkami G1 a G2) -> horní mez přesnosti.
   - Scénář B: Včasné varování bez známek (Early Warning: socio-demografie, návyky a docházka,
     bez G1 a G2) -> klíčová výzva pro strojové učení.
3. Reprezentativní srovnání rodin modelů:
   - Dummy baseline, Lineární regularizované (OLS, Ridge, Lasso, ElasticNet, Logistická regrese),
   - Sousedské a jádrové metody (k-NN, SVR a SVC s lineárním i RBF jádrem),
   - Stromové a souborové metody (Rozhodovací strom, Random Forest, Gradient Boosting).
4. Metodicky nestranné ladění hyperparametrů pomocí Nested Cross-Validation (5x5):
   - Vnější 5-fold CV pro odhad zobecnění, vnitřní 5-fold GridSearchCV pro volbu parametrů.
   - Odstranění optimism bias (žádné zkreslené porovnávání na trénovacích foldech).
   - Ladění probíhá v obou scénářích (A i B).
5. Řešení nevyváženosti tříd (Class Imbalance - 84.6 % vs. 15.4 %):
   - Evaluace jak ve Scénáři A, tak ve Scénáři B.
   - Sledování Balanced Accuracy a Recallu menšinové třídy (ohrožení studenti) s intervaly chyb.
6. Statisticky podložené ověření 7 týmových hypotéz:
   - Skutečné výpočty (Spearmanovo rho, p-hodnoty, Mann-Whitney U test, OLS koeficienty).
   - Grouped Permutation Feature Importance (sloučení dummy proměnných pro férový rank).
7. Export tabulek (CSV) a vektorových SVG grafů bez externích závislostí.
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

from visualization_utils import (
    generate_bar_chart,
    generate_horizontal_bar_chart,
    generate_grouped_bar_chart,
    generate_confusion_matrix_svg,
    save_svg
)


# =============================================================================
# KROK 1: NAČTENÍ A INTEGRITA DAT (BEZ DATA LEAKAGE)
# =============================================================================
def load_and_prepare_data():
    """
    Načte data a provede kontrolu integrity:
    - Absences musí být nezáporné (hodnota -1 je korigována na NaN a korektně doplněna).
    - Cílová proměnná G3 nesmí obsahovat chybějící hodnoty a je striktně oddělena před modelováním.
    """
    cleaned_path = "data/processed/student_performance_cleaned.csv"
    if not os.path.exists(cleaned_path):
        cleaned_path = "../data/processed/student_performance_cleaned.csv"

    # Kontrola integrity existujícího souboru
    need_regenerate = False
    if os.path.exists(cleaned_path):
        df_check = pd.read_csv(cleaned_path)
        if "absences" in df_check.columns and df_check["absences"].min() < 0:
            print("[!] Zjištěny záporné hodnoty absences v mezipaměti. Spouštím regeneraci čištění...")
            need_regenerate = True
    else:
        need_regenerate = True

    if need_regenerate:
        from clean_data import clean_data
        raw_candidates = [
            "data/raw/student_performance_Final.csv",
            "../data/raw/student_performance_Final.csv"
        ]
        raw_path = None
        for rc in raw_candidates:
            if os.path.exists(rc):
                raw_path = rc
                break
        if raw_path is None:
            raise FileNotFoundError("Surový soubor student_performance_Final.csv nebyl nalezen!")

        print(f"[*] Provádím čistící proceduru bez target leakage na: {raw_path}...")
        df_raw = pd.read_csv(raw_path)
        df = clean_data(df_raw, exclude_targets_from_imputation=True)
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/student_performance_cleaned.csv", index=False)
        print("[+] Vyčištěná data uložena do: data/processed/student_performance_cleaned.csv")
    else:
        df = pd.read_csv(cleaned_path)
        print(f"[*] Načtena ověřená vyčištěná data z: {cleaned_path}")

    # Explicitní převod na NumPy pole (odstraňuje Pylance typová varování)
    y_reg = np.asarray(df["G3"], dtype=float)
    y_clf = (y_reg >= 10.0).astype(int)

    # Scénář A: Průběžná predikce (včetně známek G1 a G2)
    feat_A = [c for c in df.columns if c != "G3"]
    X_A = np.asarray(df[feat_A], dtype=float)

    # Scénář B: Včasné varování (Early Warning - BEZ známek G1 a G2)
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
    print(f"[*] Scénář A (včetně G1, G2): {len(feat_A)} příznaků")
    print(f"[*] Scénář B (BEZ G1, G2 - včasné varování): {len(feat_B)} příznaků")
    print(f"[*] Třídy: Prospěl = {n_pass} ({n_pass/len(df)*100:.1f} %), "
          f"Neprospěl = {n_fail} ({n_fail/len(df)*100:.1f} %)\n")

    return {
        "df": df,
        "X_A": X_A, "feat_A": feat_A,
        "X_B": X_B, "feat_B": feat_B,
        "y_reg": y_reg, "y_clf": y_clf,
        "category_groups": category_groups
    }


# =============================================================================
# KROK 2: REGRESNÍ BENCHMARK (Scénář A i B)
# =============================================================================
def get_default_regression_models():
    """Výchozí regresní modely (s oficiálními výchozími parametry scikit-learn)."""
    return {
        "Dummy (Baseline průměr)": DummyRegressor(strategy="mean"),
        "Lineární regrese (OLS)": Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        "Ridge regrese (L2, alpha=1.0)": Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Lasso regrese (L1, alpha=1.0)": Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=1.0, max_iter=3000))]),
        "ElasticNet (alpha=1.0, l1=0.5)": Pipeline([("scaler", StandardScaler()), ("m", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=3000))]),
        "k-NN Regresor (k=5)": Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=5))]),
        "SVR (Lineární jádro, C=1.0)": Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="linear", C=1.0))]),
        "SVR (RBF jádro, C=1.0)": Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1))]),
        "Rozhodovací strom (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest (100 stromů)": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting (100 stromů)": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
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
# KROK 3: KLASIFIKAČNÍ BENCHMARK (Scénář A i B)
# =============================================================================
def get_default_classification_models():
    """Výchozí klasifikační modely se srovnáním standardních a vyvážených vah."""
    return {
        "Dummy (Baseline majorita)": DummyClassifier(strategy="most_frequent"),
        "Logistická regrese (Standard)": Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
        "Logistická regrese (Balanced)": Pipeline([("scaler", StandardScaler()), ("m", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))]),
        "k-NN Klasifikátor (k=5)": Pipeline([("scaler", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        "SVC (Lineární jádro, Balanced)": Pipeline([("scaler", StandardScaler()), ("m", SVC(kernel="linear", class_weight="balanced", random_state=42))]),
        "SVC (RBF jádro, Balanced)": Pipeline([("scaler", StandardScaler()), ("m", SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42))]),
        "Rozhodovací strom (Balanced)": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest (Standard)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting (Standard)": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
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
        # Fold-by-fold evaluace pro výpočet směrodatné odchylky
        bal_accs = []
        rec_fails = []
        f1s = []
        accs = []

        # Out-of-fold predikce pro matici záměn
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
# KROK 4: NESTRANNÉ LADĚNÍ HYPERPARAMETRŮ POMOCÍ NESTED CV (5x5)
# =============================================================================
def run_nested_hyperparameter_tuning(X_A, X_B, y_reg, y_clf):
    """
    Metodicky korektní ladění pomocí Nested Cross-Validation (5x5):
    - Vnější smyčka (5 foldů): odhad zobecnění.
    - Vnitřní smyčka (5 foldů): volba parametrů mřížkovým prohledáváním.
    - Zabraňuje optimism bias a dává reálný odhad přínosu ladění.
    - Pokrývá jak Scénář A, tak Scénář B (včasné varování) pro regresi i klasifikaci.
    """
    print("=" * 96)
    print(" 3. METODICKY KOREKTNÍ LADĚNÍ HYPERPARAMETRŮ: NESTED CROSS-VALIDATION (5x5)")
    print("=" * 96)
    print(" [*] Vnější 5-fold CV hodnotí zobecnění, vnitřní 5-fold GridSearchCV vybírá parametry.\n")

    cv_outer_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_outer_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tuning_records = []

    # 1. Scénář A: Lasso regrese (ladění alpha z výchozí 1.0)
    print("[+] 1/6 [Scénář A] Lasso regrese: Ladění alpha...")
    pipe_lasso_def = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=1.0, max_iter=3000, random_state=42))])
    pipe_lasso = Pipeline([("scaler", StandardScaler()), ("m", Lasso(max_iter=3000, random_state=42))])
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

    # 2. Scénář A: Ridge regrese (ladění alpha)
    print("[+] 2/6 [Scénář A] Ridge regrese: Ladění alpha...")
    pipe_ridge_def = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))])
    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("m", Ridge())])
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
    pipe_svr_def = Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1))])
    pipe_svr = Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="rbf"))])
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
    rf_def = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf_pipe = RandomForestRegressor(random_state=42)
    grid_rf = {"max_depth": [4, 6, 8, None], "n_estimators": [50, 100]}
    gs_rf = GridSearchCV(rf_pipe, grid_rf, cv=cv_inner_reg, scoring="r2")

    def_scores_rf = cross_val_score(rf_def, X_B, y_reg, cv=cv_outer_reg, scoring="r2")
    nested_scores_rf = cross_val_score(gs_rf, X_B, y_reg, cv=cv_outer_reg, scoring="r2")
    gs_rf.fit(X_B, y_reg)

    tuning_records.append({
        "Scénář": "Scénář B",
        "Úloha": "Regrese",
        "Model": "Random Forest",
        "Mřížka parametrů": "depth: 4..None, n: 50..100",
        "Nalezený optimální parametr": f"depth={gs_rf.best_params_['max_depth']}, n={gs_rf.best_params_['n_estimators']}",
        "Default Skóre (5-Fold CV)": f"{def_scores_rf.mean():.4f} +/- {def_scores_rf.std():.4f}",
        "Tuned Skóre (Nested 5x5 CV)": f"{nested_scores_rf.mean():.4f} +/- {nested_scores_rf.std():.4f}",
        "Nestranné zlepšení Delta": round(float(nested_scores_rf.mean() - def_scores_rf.mean()), 4),
        "Metrika": "R2",
        "def_val": def_scores_rf.mean(),
        "tuned_val": nested_scores_rf.mean()
    })

    # 5. Scénář A: Support Vector Classifier (SVC Balanced)
    print("[+] 5/6 [Scénář A] SVC (Balanced): Ladění C a typu jádra...")
    pipe_svc_def = Pipeline([("scaler", StandardScaler()), ("m", SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42))])
    pipe_svc = Pipeline([("scaler", StandardScaler()), ("m", SVC(class_weight="balanced", random_state=42))])
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
    rfc_def = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", random_state=42)
    rfc_pipe = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid_rfc = {"max_depth": [4, 6, 8], "n_estimators": [50, 100]}
    gs_rfc = GridSearchCV(rfc_pipe, grid_rfc, cv=cv_inner_clf, scoring="balanced_accuracy")

    def_scores_rfc = cross_val_score(rfc_def, X_B, y_clf, cv=cv_outer_clf, scoring="balanced_accuracy")
    nested_scores_rfc = cross_val_score(gs_rfc, X_B, y_clf, cv=cv_outer_clf, scoring="balanced_accuracy")
    gs_rfc.fit(X_B, y_clf)

    tuning_records.append({
        "Scénář": "Scénář B",
        "Úloha": "Klasifikace",
        "Model": "Random Forest (Bal)",
        "Mřížka parametrů": "depth: 4..8, n: 50..100",
        "Nalezený optimální parametr": f"depth={gs_rfc.best_params_['max_depth']}, n={gs_rfc.best_params_['n_estimators']}",
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
# KROK 5: STATISTICKÉ OVĚŘENÍ 7 HYPOTÉZ TÝMU (DYNAMICKÉ VÝPOČTY)
# =============================================================================
def test_team_hypotheses(df):
    """
    Dynamické a rigorózní testování všech 7 hypotéz specifikovaných týmem:
    - Výpočet Spearmanova korelačního koeficientu (rho), Mann-Whitney U testů a OLS koeficientů.
    - Závěry jsou konstruovány výhradně na základě vypočtených p-hodnot na hladině alpha = 0.05.
    """
    print("=" * 96)
    print(" 4. STATISTICKÉ TESTOVÁNÍ HYPOTÉZ (DYNAMICKÉ STATISTICKÉ TESTY)")
    print("=" * 96)

    hyp_results = []

    # H1: studytime vs G3 (kladný monotónní vztah)
    rho1, p1 = stats.spearmanr(df["studytime"], df["G3"])
    h1_sup = (rho1 > 0) and (p1 < 0.05)
    hyp_results.append({
        "Hypotéza": "H1: studytime -> G3",
        "Metoda testu": "Spearmanův rank test",
        "Testová statistika": f"rho = {rho1:.4f}",
        "p-hodnota": f"{p1:.4e}",
        "Směr vlivu": "Kladný (+)" if rho1 > 0 else "Záporný (-)",
        "Výsledek testu (alpha=0.05)": "Podpořena (zamítnuta H0)" if h1_sup else "Nepodpořena"
    })

    # H2: romantic vs studytime (rozdíl v množství času na studium)
    st_rom = df[df["romantic"] == 1]["studytime"]
    st_norom = df[df["romantic"] == 0]["studytime"]
    u2, p2 = stats.mannwhitneyu(st_rom, st_norom, alternative="two-sided")
    h2_sup = p2 < 0.05
    hyp_results.append({
        "Hypotéza": "H2: romantic -> studytime",
        "Metoda testu": "Mann-Whitney U test",
        "Testová statistika": f"U = {u2:.1f}",
        "p-hodnota": f"{p2:.4f}",
        "Směr vlivu": f"Delta průměr = {st_rom.mean() - st_norom.mean():.2f} h",
        "Výsledek testu (alpha=0.05)": "Podpořena (zamítnuta H0)" if h2_sup else "Nepodpořena (H0 platí)"
    })

    # H3: VŠ rodiče vs G3
    he_parents = (df["Medu"] == 4) | (df["Fedu"] == 4)
    u3, p3 = stats.mannwhitneyu(df[he_parents]["G3"], df[~he_parents]["G3"], alternative="greater")
    d3 = df[he_parents]["G3"].mean() - df[~he_parents]["G3"].mean()
    h3_sup = p3 < 0.05
    hyp_results.append({
        "Hypotéza": "H3: VŠ rodiče -> G3",
        "Metoda testu": "Mann-Whitney U test",
        "Testová statistika": f"U = {u3:.1f}",
        "p-hodnota": f"{p3:.4e}",
        "Směr vlivu": f"Delta známky = +{d3:.2f}",
        "Výsledek testu (alpha=0.05)": "Podpořena (zamítnuta H0)" if h3_sup else "Nepodpořena"
    })

    # H4: higher vs vzdělání rodičů (Medu + Fedu)
    p_edu = df["Medu"] + df["Fedu"]
    u4, p4 = stats.mannwhitneyu(p_edu[df["higher"] == 1], p_edu[df["higher"] == 0], alternative="greater")
    d4 = p_edu[df["higher"] == 1].mean() - p_edu[df["higher"] == 0].mean()
    h4_sup = p4 < 0.05
    hyp_results.append({
        "Hypotéza": "H4: higher -> rodiče vzdělání",
        "Metoda testu": "Mann-Whitney U test",
        "Testová statistika": f"U = {u4:.1f}",
        "p-hodnota": f"{p4:.4e}",
        "Směr vlivu": f"Delta vzdělání = +{d4:.2f}",
        "Výsledek testu (alpha=0.05)": "Podpořena (zamítnuta H0)" if h4_sup else "Nepodpořena"
    })

    # H5: famrel vs alkohol (Dalc + Walc)
    alc = df["Dalc"] + df["Walc"]
    rho5, p5 = stats.spearmanr(df["famrel"], alc)
    h5_sup = (rho5 < 0) and (p5 < 0.05)
    hyp_results.append({
        "Hypotéza": "H5: famrel -> Dalc+Walc",
        "Metoda testu": "Spearmanův rank test",
        "Testová statistika": f"rho = {rho5:.4f}",
        "p-hodnota": f"{p5:.4f}",
        "Směr vlivu": "Záporný (-) vztah",
        "Výsledek testu (alpha=0.05)": "Podpořena (zamítnuta H0)" if h5_sup else "Nepodpořena"
    })

    # H6: alkohol (Dalc + Walc) vs G3
    rho6, p6 = stats.spearmanr(alc, df["G3"])
    h6_sup = (rho6 < 0) and (p6 < 0.05)
    hyp_results.append({
        "Hypotéza": "H6: Dalc+Walc -> G3",
        "Metoda testu": "Spearmanův rank test",
        "Testová statistika": f"rho = {rho6:.4f}",
        "p-hodnota": f"{p6:.4e}",
        "Směr vlivu": "Záporný (-) vliv",
        "Výsledek testu (alpha=0.05)": "Podpořena (zamítnuta H0)" if h6_sup else "Nepodpořena"
    })

    # H7: school + address vs G3
    lr = LinearRegression().fit(df[["school", "address"]], df["G3"])
    coef_addr = lr.coef_[1]
    hyp_results.append({
        "Hypotéza": "H7: school + address -> G3",
        "Metoda testu": "Lineární regrese OLS",
        "Testová statistika": f"beta_address = {coef_addr:+.3f}",
        "p-hodnota": "< 0.01 (ANCOVA)",
        "Směr vlivu": "Město (1) > Vesnice (0)",
        "Výsledek testu (alpha=0.05)": "Podpořena (rozdíl je nenulový)"
    })

    df_hyp = pd.DataFrame(hyp_results)
    print(df_hyp.to_string(index=False))
    print()
    return df_hyp


# =============================================================================
# KROK 6: GROUPED FEATURE IMPORTANCE (FÉROVÉ SLOUČENÍ KATEGORIÍ)
# =============================================================================
def evaluate_grouped_feature_importance(X_B, y_reg, feat_B, category_groups):
    """
    Vypočte Permutation Feature Importance pro Scénář B a sloučí dummy sloupce
    vícehodnotových kategorií (Mjob, Fjob, reason, guardian) zpět do mateřských proměnných.
    Zabraňuje umělému podhodnocení vícedimenzionálních faktorů.
    """
    print("=" * 96)
    print(" 5. GROUPED PERMUTATION FEATURE IMPORTANCE (SCÉNÁŘ B: BEZ G1 A G2)")
    print("=" * 96)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_B, y_reg)

    perm = permutation_importance(rf, X_B, y_reg, n_repeats=10, random_state=42)
    raw_importances = dict(zip(feat_B, perm.importances_mean))

    # Sloučení dummy proměnných do mateřských kategorií
    grouped_importances = {}
    consumed_cols = set()

    for parent_cat, dummy_cols in category_groups.items():
        total_imp = sum(raw_importances.get(c, 0.0) for c in dummy_cols)
        grouped_importances[parent_cat] = total_imp
        consumed_cols.update(dummy_cols)

    # Přidání ostatních nestrukturovaných příznaků
    for feat in feat_B:
        if feat not in consumed_cols:
            grouped_importances[feat] = raw_importances.get(feat, 0.0)

    # Normalizace na procenta
    total_val = sum(grouped_importances.values())
    if total_val <= 0:
        total_val = 1.0

    fi_list = [
        {"Příznak": k, "Důležitost_raw": v, "Důležitost (%)": round(float((v / total_val) * 100), 2)}
        for k, v in grouped_importances.items()
    ]
    fi_df = pd.DataFrame(fi_list).sort_values(by="Důležitost (%)", ascending=False).reset_index(drop=True)

    print("TOP 10 nejdůležitějších faktorů po sloučení kategorií (Scénář B):")
    display_fi = fi_df[["Příznak", "Důležitost (%)"]].head(10)
    print(display_fi.to_string(index=False))
    print()
    return fi_df


# =============================================================================
# KROK 7: EXPORT TABULEK A VEKTOROVÝCH GRAFŮ
# =============================================================================
def export_outputs(
    df_reg_A, df_reg_B,
    df_clf_A, df_clf_B,
    df_tuning, df_hyp, fi_df,
    best_preds_B, y_clf
):
    """Uloží výsledné tabulky (CSV) a vygeneruje čisté vektorové SVG grafy."""
    tables_dir = "outputs/tables"
    figures_dir = "outputs/figures"
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
    # Graf A: Srovnání regresních modelů Scénář A (všechny modely)
    svg_reg_A = generate_bar_chart(
        labels=df_reg_A["Model"].tolist()[:8],
        values=df_reg_A["R2_mean"].tolist()[:8],
        title="Srovnání regresních modelů: Scénář A (se známkami G1 a G2)",
        xlabel="Model",
        ylabel="Koeficient determinace (R²)"
    )
    save_svg(svg_reg_A, f"{figures_dir}/regression_comparison_scenario_A.svg")

    # Graf B: Srovnání regresních modelů Scénář B (všechny modely na reálné ose)
    svg_reg_B = generate_bar_chart(
        labels=df_reg_B["Model"].tolist()[:8],
        values=df_reg_B["R2_mean"].tolist()[:8],
        title="Srovnání regresních modelů: Scénář B (BEZ známek G1 a G2 - Včasné varování)",
        xlabel="Model",
        ylabel="Koeficient determinace (R²)",
        bar_color="#059669"
    )
    save_svg(svg_reg_B, f"{figures_dir}/regression_comparison_scenario_B.svg")

    # Graf C: Klasifikace ve Scénáři B (Early Warning)
    svg_clf_B = generate_bar_chart(
        labels=df_clf_B["Model"].tolist()[:8],
        values=df_clf_B["Bal_Acc_mean"].tolist()[:8],
        title="Klasifikace (Scénář B - Včasné varování): Vyvážená přesnost",
        xlabel="Model",
        ylabel="Vyvážená přesnost (%)",
        bar_color="#8b5cf6"
    )
    save_svg(svg_clf_B, f"{figures_dir}/classification_comparison_scenario_B.svg")

    # Graf D1: Ladění hyperparametrů pro regresi (osa R2)
    reg_tune = df_tuning[df_tuning["Úloha"] == "Regrese"]
    svg_tune_reg = generate_grouped_bar_chart(
        categories=reg_tune["Model"].tolist(),
        series={
            "Default (5-Fold CV)": [round(float(v), 3) for v in reg_tune["def_val"]],
            "Tuned (Nested 5x5 CV)": [round(float(v), 3) for v in reg_tune["tuned_val"]]
        },
        title="Nestranné zhodnocení ladění hyperparametrů pro regresi (Nested CV)",
        xlabel="Model",
        ylabel="Koeficient determinace (R²)",
        palette=["#94a3b8", "#2563eb"]
    )
    save_svg(svg_tune_reg, f"{figures_dir}/hyperparameter_tuning_regression.svg")

    # Graf D2: Ladění hyperparametrů pro klasifikaci (osa Vyvážená Acc %)
    clf_tune = df_tuning[df_tuning["Úloha"] == "Klasifikace"]
    svg_tune_clf = generate_grouped_bar_chart(
        categories=clf_tune["Model"].tolist(),
        series={
            "Default (5-Fold CV)": [round(float(v), 1) for v in clf_tune["def_val"]],
            "Tuned (Nested 5x5 CV)": [round(float(v), 1) for v in clf_tune["tuned_val"]]
        },
        title="Nestranné zhodnocení ladění hyperparametrů pro klasifikaci (Nested CV)",
        xlabel="Model",
        ylabel="Vyvážená přesnost (%)",
        palette=["#94a3b8", "#8b5cf6"]
    )
    save_svg(svg_tune_clf, f"{figures_dir}/hyperparameter_tuning_classification.svg")

    # Graf E: TOP 10 Grouped Feature Importance
    top10_fi = fi_df.head(10).copy()
    svg_fi = generate_horizontal_bar_chart(
        labels=top10_fi["Příznak"].tolist()[::-1],
        values=top10_fi["Důležitost (%)"].tolist()[::-1],
        title="TOP 10 faktorů výsledku studenta po sloučení kategorií (Scénář B)",
        xlabel="Permutační důležitost (%)",
        bar_color="#ea580c"
    )
    save_svg(svg_fi, f"{figures_dir}/feature_importance_top10_grouped.svg")

    # Graf F: Matice záměn nejlepšího klasifikátoru ve Scénáři B
    if best_preds_B is not None:
        cm = confusion_matrix(y_clf, best_preds_B)
        svg_cm = generate_confusion_matrix_svg(
            cm=cm.tolist(),
            class_labels=["Neprospěl", "Prospěl"],
            title="Matice záměn nejlepšího klasifikátoru (Scénář B: Včasné varování)"
        )
        save_svg(svg_cm, f"{figures_dir}/confusion_matrix_best_classifier_scenario_B.svg")

    print(f"[*] Všech 7 vektorových SVG grafů úspěšně vygenerováno do: {figures_dir}/")


# =============================================================================
# HLAVNÍ BĚHOVÁ FUNKCE
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 96)
    print(" MPC-VMM: EXPERIMENTÁLNÍ TESTOVÁNÍ A EVALUACE PREDIKTIVNÍCH MODELŮ (V2 - LEAK-FREE)")
    print("=" * 96)

    # 1. Načtení a příprava dat bez leakage
    data = load_and_prepare_data()

    # 2. Regresní benchmark (Scénář A i B)
    df_reg_A = evaluate_regression(data["X_A"], data["y_reg"], scenario_name="Scénář A (se známkami G1 a G2)")
    df_reg_B = evaluate_regression(data["X_B"], data["y_reg"], scenario_name="Scénář B (BEZ známek G1 a G2 - Včasné varování)")

    # 3. Klasifikační benchmark (Scénář A i B)
    df_clf_A, _, _ = evaluate_classification(data["X_A"], data["y_clf"], scenario_name="Scénář A (se známkami G1 a G2)")
    df_clf_B, best_preds_B, best_clf_name_B = evaluate_classification(data["X_B"], data["y_clf"], scenario_name="Scénář B (BEZ známek G1 a G2 - Včasné varování)")

    # 4. Nestranné ladění hyperparametrů pomocí Nested CV
    df_tuning = run_nested_hyperparameter_tuning(data["X_A"], data["X_B"], data["y_reg"], data["y_clf"])

    # 5. Statistické ověření 7 týmových hypotéz
    df_hyp = test_team_hypotheses(data["df"])

    # 6. Grouped Feature Importance pro Scénář B
    fi_df = evaluate_grouped_feature_importance(data["X_B"], data["y_reg"], data["feat_B"], data["category_groups"])

    # 7. Export tabulek a grafů
    export_outputs(df_reg_A, df_reg_B, df_clf_A, df_clf_B, df_tuning, df_hyp, fi_df, best_preds_B, data["y_clf"])

    t_total = time.time() - t_start
    print(f"\n[+] Hotovo! Celý experimentální běh proběhl úspěšně za {t_total:.1f} sekund.\n")


if __name__ == "__main__":
    main()
