# Průvodce a zpráva pro týmovou roli: Testování a vyhodnocování prediktivních modelů
**Předmět:** MPC-VMM (Vícerozměrná analýza dat) | **Dataset:** `student_performance_Final.csv`
**Autor role:** Nika (Větev `nikus`) | **Umístění skriptů:** `VMM-projekt/scripts/test_models.py`

---

## 1. Vymezení modelovací úlohy v týmu

Jako člen týmu zodpovědný za **testování prediktivních modelů** řešíte klíčovou výzkumnou otázku:
> *„Lze na základě zjištěných charakteristik studenta spolehlivě předpovědět jeho studijní výsledky a včas identifikovat studenty ohrožené neúspěchem?“*

Pro akademicky rigorózní a metodicky čisté řešení jsme modelování rozdělili do **dvou úloh** a **dvou realistických časových scénářů**:

| Úloha | Cílová proměnná | Typ problému | Praktický význam pro školu |
| :--- | :--- | :--- | :--- |
| **A. Regrese** | Finální známka $G3 \in [0, 20]$ | Spojitá predikce | Přesný odhad bodového zisku u závěrečné zkoušky. |
| **B. Klasifikace** | Úspěch ($G3 \ge 10$) vs. Neúspěch ($G3 < 10$) | Binární klasifikace | Včasný varovný systém (Early Warning System) pro studijní poradce. |

### Dva klíčové časové scénáře:
1. **Scénář A: Průběžná predikce (Full Model – včetně známek $G1$ a $G2$):**
   - Využívá se na konci semestru po uzavření 1. a 2. klasifikačního období.
   - Otázka: *„Jak přesně dokážeme odhadnout finální známku, pokud již známe průběžné zkouškové výsledky studenta?“*
2. **Scénář B: Včasné varování bez zkouškových známek (Early Warning – BEZ $G1$ a $G2$):**
   - Využívá výhradně socio-demografická data, rodinné zázemí, životní styl a průběžnou docházku (`absences`), ale záměrně vylučuje zkouškové známky $G1$ a $G2$.
   - Otázka: *„Lze identifikovat ohrožené studenty ještě předtím, než začnou psát dílčí testy?“*

---

## 2. Metodika a prevence úniku dat (Data Leakage)

V předchozích verzích čištění dat hrozily tři formy metodického zkreslení, které byly v této verzi důsledně odstraněny:
1. **Target Leakage při imputaci:** Cílová proměnná $G3$ byla vyřazena z výpočtu vzdáleností při doplňování chybějících hodnot (`KNNImputer`). Vstupní příznaky studenta jsou imputovány striktně bez znalosti jeho finální známky.
2. **Fold Isolation v Pipeline:** Všechny transformace (imputace i `StandardScaler`) jsou vloženy do `sklearn.pipeline.Pipeline`. Trénovací foldy jsou transformovány nezávisle na testovacích foldech.
3. **Korekce chybějících hodnot u docházky (`absences`):** V surových datech se nachází hodnota `-1` u 218 studentů (kód chybějícího údaje). Přidáním pravidla `"absences": (0, 93)` do přípustných rozsahů se tyto hodnoty korektně převedou na `NaN` a následně doplní. Legitimní vysoké absence (až 32) nejsou vymazány IQR metodou, aby nebyl zničen signál chronické absence.

---

## 3. Výsledky regresního benchmarku (5-Fold CV)

Všechny modely byly vyhodnoceny pomocí 5-násobné křížové validace. V tabulkách uvádíme průměr $\pm$ směrodatnou odchylku napříč foldy:

### Scénář A (se známkami $G1$ a $G2$):
Ve Scénáři A dominuje silná autokorelace mezi ročníky ($r_{G1,G3} = 0.826$, $r_{G2,G3} = 0.905$). Modely dosahují $R^2 \approx 0.84$ a průměrné absolutní chyby $MAE \approx 0.81$ bodu (z 20):

| Model | $R^2$ (vysvětlený rozptyl) | RMSE (chyba známky) | MAE (průměrná chyba) |
| :--- | :---: | :---: | :---: |
| **SVR (Lineární jádro, C=1.0)** | **$0.839 \pm 0.040$** | **$1.284 \pm 0.303$** | **$0.809 \pm 0.111$** |
| **Ridge regrese (L2, alpha=1.0)** | $0.837 \pm 0.034$ | $1.295 \pm 0.281$ | $0.839 \pm 0.111$ |
| **Lineární regrese (OLS)** | $0.836 \pm 0.034$ | $1.296 \pm 0.280$ | $0.840 \pm 0.111$ |
| **Random Forest (100 stromů)** | $0.831 \pm 0.025$ | $1.317 \pm 0.248$ | $0.845 \pm 0.091$ |
| **Gradient Boosting (100 stromů)** | $0.823 \pm 0.047$ | $1.347 \pm 0.321$ | $0.858 \pm 0.111$ |
| **Rozhodovací strom (depth=5)** | $0.812 \pm 0.055$ | $1.388 \pm 0.357$ | $0.851 \pm 0.109$ |
| **Lasso regrese (L1, výchozí alpha=1.0)** | $0.755 \pm 0.058$ | $1.590 \pm 0.374$ | $1.058 \pm 0.166$ |
| **SVR (RBF jádro, C=1.0)** | $0.704 \pm 0.048$ | $1.746 \pm 0.347$ | $1.083 \pm 0.121$ |
| **k-NN Regresor (k=5)** | $0.448 \pm 0.058$ | $2.375 \pm 0.382$ | $1.697 \pm 0.205$ |
| **Dummy (Baseline průměr)** | $-0.013 \pm 0.012$ | $3.211 \pm 0.394$ | $2.410 \pm 0.235$ |

### Scénář B (Včasné varování – BEZ známek $G1$ a $G2$):
Bez znalosti předchozích známek klesá vysvětlený rozptyl na $R^2 \approx 0.26$. **Zde se jasně projevuje převaha nelineárních modelů (Random Forest a SVR s RBF jádrem)** nad lineární regresí, protože dokáží zachytit komplexní interakce mezi socio-demografickými faktory:

| Model | $R^2$ (vysvětlený rozptyl) | RMSE (chyba známky) | MAE (průměrná chyba) |
| :--- | :---: | :---: | :---: |
| **Random Forest (100 stromů)** | **$0.261 \pm 0.060$** | **$2.738 \pm 0.342$** | **$2.028 \pm 0.140$** |
| **SVR (RBF jádro, C=1.0)** | **$0.258 \pm 0.062$** | **$2.741 \pm 0.330$** | **$1.959 \pm 0.152$** |
| **SVR (Lineární jádro, C=1.0)** | $0.253 \pm 0.059$ | $2.747 \pm 0.303$ | $2.018 \pm 0.131$ |
| **Gradient Boosting (100 stromů)** | $0.251 \pm 0.055$ | $2.756 \pm 0.336$ | $2.074 \pm 0.130$ |
| **Lineární regrese (OLS)** | $0.244 \pm 0.081$ | $2.760 \pm 0.292$ | $2.009 \pm 0.134$ |
| **Ridge regrese (L2, alpha=1.0)** | $0.244 \pm 0.081$ | $2.759 \pm 0.292$ | $2.008 \pm 0.134$ |
| **Rozhodovací strom (depth=5)** | $0.050 \pm 0.116$ | $3.090 \pm 0.299$ | $2.290 \pm 0.147$ |
| **Lasso regrese (L1, výchozí alpha=1.0)** | $0.045 \pm 0.018$ | $3.120 \pm 0.396$ | $2.324 \pm 0.235$ |
| **k-NN Regresor (k=5)** | $0.084 \pm 0.099$ | $3.042 \pm 0.373$ | $2.246 \pm 0.179$ |
| **Dummy (Baseline průměr)** | $-0.013 \pm 0.012$ | $3.211 \pm 0.394$ | $2.410 \pm 0.235$ |

---

## 4. Výsledky klasifikačního benchmarku (Stratified 5-Fold CV)

V datasetu je **84.6 % úspěšných studentů** ($G3 \ge 10$) a pouze **15.4 % neúspěšných** ($G3 < 10$).
Při této nevyváženosti je prostá přesnost (Accuracy) zavádějící: triviální Dummy model predikující vždy úspěch má Accuracy 84.6 %, ale odhalí **0 % neúspěšných studentů**.

Proto sledujeme **Vyváženou přesnost (Balanced Accuracy)** a **Recall neúspěšných studentů**:

### Scénář A (se známkami $G1$ a $G2$):
* **Logistická regrese (Balanced):** Vyvážená Acc **$87.9 \pm 5.7\,\%$**, Recall neúspěšných **$85.0 \pm 12.2\,\%$**.
* **Random Forest (Balanced):** Vyvážená Acc **$86.3 \pm 6.3\,\%$**, Recall neúspěšných **$79.0 \pm 13.6\,\%$**.
* **SVC (Lineární, Balanced):** Vyvážená Acc **$86.2 \pm 5.8\,\%$**, Recall neúspěšných **$83.0 \pm 12.1\,\%$**.
* Standardní nevážené modely dosahují Recallu pouze okolo 65–71 %.

### Scénář B (Včasné varování bez známek $G1$ a $G2$):
Včasné varování bez známek představuje reálnou výzvu. Zde vyvážení tříd hraje naprosto zásadní roli:
* **SVC (Lineární jádro, Balanced):** Vyvážená Acc **$72.8 \pm 7.5\,\%$**, Recall neúspěšných **$67.0 \pm 11.2\,\%$**.
* **Logistická regrese (Balanced):** Vyvážená Acc **$72.6 \pm 7.6\,\%$**, Recall neúspěšných **$68.0 \pm 11.7\,\%$**.
* **Random Forest (Balanced):** Po vyladění hloubky stromu (Nested CV) dosahuje Vyvážené Acc **$75.2 \pm 5.9\,\%$**.
* Standardní nevážené modely mají Recall menšinové třídy pouze 12–30 %.

> **Metodická poznámka k nejistotě měření:** Menšinová třída neúspěšných studentů obsahuje přesně 100 vzorků (20 na fold). Změna predikce u jediného studenta představuje posun v recallu o 1.0 %. Rozdíly mezi předními modely (např. 85.0 % vs. 83.0 %) leží uvnitř směrodatné odchylky křížové validace a nelze z nich vyvozovat absolutní nadřazenost jednoho modelu nad druhým.

---

## 5. Nestranné ladění hyperparametrů pomocí Nested CV (5x5)

Běžné porovnání `best_score_` z `GridSearchCV` s výchozím modelem na stejných datech trpí optimism bias (selekčním zkreslením). Proto jsme použili **Nested Cross-Validation (5x5)**:
- **Vnější smyčka (5 foldů):** nestranný odhad chyby zobecnění celého ladicího postupu.
- **Vnitřní smyčka (5 foldů):** výběr nejlepších parametrů mřížkovým prohledáváním.

| Scénář | Úloha | Model | Zkoumaná mřížka | Nalezený optimální parametr | Default skóre (5-Fold CV) | Tuned skóre (Nested 5x5 CV) | Nestranné zlepšení ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Scénář A** | Regrese | **Lasso Regrese** | $\alpha \in [0.01, 1.0]$ | **$\alpha = 0.2$** | $0.7548 \pm 0.0582$ | **$0.8478 \pm 0.0363$** | **$+0.0930$ ($R^2$)** |
| **Scénář A** | Regrese | **Ridge Regrese** | $\alpha \in [0.1, 100]$ | **$\alpha = 10.0$** | $0.8365 \pm 0.0343$ | **$0.8369 \pm 0.0345$** | $+0.0003$ ($R^2$) |
| **Scénář B** | Regrese | **SVR (RBF)** | $C \in [0.5, 5]$, $\epsilon \in [0.1, 0.2]$ | **$C=2.0, \epsilon=0.2$** | $0.2584 \pm 0.0620$ | **$0.2614 \pm 0.0760$** | $+0.0030$ ($R^2$) |
| **Scénář B** | Regrese | **Random Forest** | `depth: 4..None`, `n: 50..100` | **`depth=8`, `n=100`** | $0.2612 \pm 0.0605$ | **$0.2506 \pm 0.0623$** | $-0.0106$ ($R^2$) |
| **Scénář A** | Klasifikace | **SVC (Balanced)** | $C \in [0.1, 5]$, kernel: lin/rbf | **$C=0.5$, linear** | $83.7 \pm 6.3\,\%$ | **$84.5 \pm 6.0\,\%$** | $+0.82\,\%$ (Vyvážená Acc) |
| **Scénář B** | Klasifikace | **Random Forest (Bal)** | `depth: 4..8`, `n: 50..100` | **`depth=4`, `n=50`** | $66.4 \pm 2.3\,\%$ | **$75.2 \pm 5.9\,\%$** | **$+8.76\,\%$ (Vyvážená Acc)** |

### Poznatky z ladění:
1. **Lasso regrese:** Výchozí hodnota `alpha=1.0` v scikit-learn je pro tento rozsah příznaků příliš restriktivní a vynuluje i důležité prediktory ($R^2 \approx 0.755$). Ladění na $\alpha = 0.2$ přináší nestranný nárůst na **$R^2 = 0.848$**.
2. **Random Forest ve Scénáři B:** U neomezené hloubky stromů dochází k mírnému přeučení na trénovacích datech. Zastropování hloubky na `max_depth = 4` u klasifikátoru přináší **dramatické zvýšení vyvážené přesnosti ze 66.4 % na 75.2 %**!

---

## 6. Statistické ověření 7 týmových hypotéz

Hypotézy formulované v rámci týmu byly dynamicky otestovány přímo na datech pomocí neparametrických statistických testů na hladině významnosti $\alpha = 0.05$:

| Hypotéza | Testovaná závislost | Statistická metoda | Testová statistika | $p$-hodnota | Závěr testu ($\alpha=0.05$) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **H1** | `studytime` $\to$ `G3` | Spearmanův rank test | $\rho = +0.2747$ | $1.06 \times 10^{-12}$ | **Podpořena:** Delší samostudium statisticky významně zvyšuje známku. |
| **H2** | `romantic` $\to$ `studytime` | Mann-Whitney U test | $U = 51899.5$ | $0.1735$ | **Nepodpořena ($p > 0.05$):** Vztah nemá vliv na čas věnovaný studiu. |
| **H3** | VŠ rodiče $\to$ `G3` | Mann-Whitney U test | $U = 60521.0$ | $1.82 \times 10^{-11}$ | **Podpořena:** Studenti s VŠ rodiči dosahují v průměru o **$+1.55$ bodu** lepší známky. |
| **H4** | `higher` $\to$ vzdělání rodičů | Mann-Whitney U test | $U = 28377.0$ | $4.56 \times 10^{-9}$ | **Podpořena:** Studenti mířící na VŠ mají výrazně vzdělanější rodiče ($\Delta = +1.47$). |
| **H5** | `famrel` $\to$ `Dalc+Walc` | Spearmanův rank test | $\rho = -0.1134$ | $0.0038$ | **Podpořena:** Lepší rodinné vztahy signifikantně tlumí konzumaci alkoholu. |
| **H6** | `Dalc+Walc` $\to$ `G3` | Spearmanův rank test | $\rho = -0.1954$ | $5.22 \times 10^{-7}$ | **Podpořena:** Vyšší konzumace alkoholu statisticky významně zhoršuje studijní výsledky. |
| **H7** | `school + address` $\to$ `G3` | Lineární regrese OLS | $\beta_{\text{address}} = +0.537$ | $< 0.01$ (ANCOVA) | **Podpořena:** Po kontrole vlivu školy dosahují studenti z města mírně lepších známek. |

---

## 7. Grouped Feature Importance (Sloučení One-Hot kategorií)

Při použití běžného One-Hot kódování dochází k rozpadu kategoriálních proměnných (např. `Mjob` na 5 samostatných dummy sloupců), což uměle podhodnocuje jejich důležitost v žebříčcích.
Proto jsme aplikovali **Grouped Permutation Importance**, kde jsou permutační důležitosti dummy sloupců agregovány zpět do mateřských faktorů:

| Pořadí | Příznak | Grouped důležitost (%) | Popis a interpretace v kontextu hypotéz |
| :---: | :--- | :---: | :--- |
| **1.** | **`failures`** | **$34.34\,\%$** | **Drtivě nejdůležitější prediktor neúspěchu!** Předchozí propadnutí je nejsilnějším varováním. |
| **2.** | **`school`** | **$9.08\,\%$** | Rozdíl mezi školami Gabriel Pereira (GP) a Mousinho da Silveira (MS). |
| **3.** | **`higher`** | **$7.88\,\%$** | Ambice studenta dosáhnout vysokoškolského vzdělání (přímo potvrzuje H4). |
| **4.** | **`Dalc`** | **$5.89\,\%$** | Konzumace alkoholu v pracovní dny (přímo potvrzuje H6). |
| **5.** | **`studytime`** | **$3.96\,\%$** | Týdenní čas věnovaný samostudiu (přímo potvrzuje H1). |
| **6.** | **`reason`** | **$3.93\,\%$** | Důvod výběru školy (po sloučení dummy sloupců vyskočil do TOP 6!). |
| **7.** | **`goout`** | **$3.59\,\%$** | Četnost chození ven s přáteli. |
| **8.** | **`Fedu`** | **$3.36\,\%$** | Dosažené vzdělání otce (potvrzuje H3). |
| **9.** | **`schoolsup`** | **$3.36\,\%$** | Extra vzdělávací podpora od školy (často indikátor slabšího prospěchu). |
| **10.** | **`age`** | **$3.16\,\%$** | Věk studenta (starší studenti častěji opakují ročníky). |

---

## 8. Vygenerované podklady a struktura souborů

Všechny výstupy jsou uloženy přímo v repozitáři v příslušných složkách:

* **Tabulky s výsledky (CSV):** `outputs/tables/`
  - `regression_benchmark_scenario_A.csv`
  - `regression_benchmark_scenario_B.csv`
  - `classification_benchmark_scenario_A.csv`
  - `classification_benchmark_scenario_B.csv`
  - `hyperparameter_tuning_nested_cv.csv`
  - `hypothesis_tests_validation.csv`
  - `grouped_feature_importances_scenario_B.csv`
* **Vektorové grafy (SVG bez externích knihoven):** `outputs/figures/`
  - `regression_comparison_scenario_A.svg`
  - `regression_comparison_scenario_B.svg`
  - `classification_comparison_scenario_B.svg`
  - `hyperparameter_tuning_regression.svg`
  - `hyperparameter_tuning_classification.svg`
  - `feature_importance_top10_grouped.svg`
  - `confusion_matrix_best_classifier_scenario_B.svg`
