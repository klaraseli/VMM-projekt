# Průvodce a technická zpráva: Testování a vyhodnocování prediktivních modelů
**Předmět:** MPC-VMM (Vícerozměrná analýza dat, FEKT VUT) | **Dataset:** `student_performance_Final.csv`  
**Autor role:** Nika (Větev `nikus`) | **Kódový skript:** `VMM-projekt/scripts/test_models.py`  
**Verze metodiky:** V3.1 (Striktní metodická integrita, StandardScaler -> KNNImputer v Pipeline, OOF validace)

---

## 1. Vymezení modelovací úlohy a výzkumný rámec

V rámci týmového projektu MPC-VMM zodpovídá tato část za **návrh, implementaci, ladění a rigorózní evaluaci prediktivních modelů strojového učení** pro predikci studijního výkonu studentů.

Výzkumný cíl vychází ze zadání:
> *„Lze na základě zjištěných charakteristik studenta spolehlivě předpovědět jeho studijní výsledky a včas identifikovat studenty ohrožené studijním neúspěchem?“*

Pro zajištění metodické čistoty a praktické relevance pro vedení škol jsme modelování rozdělili do **dvou úloh** a **dvou realistických časových scénářů**:

| Úloha | Cílová proměnná | Typ problému | Praktický význam pro školu |
| :--- | :--- | :--- | :--- |
| **A. Regrese** | Finální známka $G3 \in [0, 20]$ | Spojitá predikce | Přesný bodový odhad zisku u závěrečné zkoušky na konci roku. |
| **B. Klasifikace** | Úspěch ($G3 \ge 10$) vs. Neúspěch ($G3 < 10$) | Binární klasifikace | Včasný varovný systém (Early Warning System) pro intervenci studijních poradců. |

### Dva realistické časové scénáře:
1. **Scénář A: Průběžná predikce (Full Model – včetně dílčích známek $G1$ a $G2$):**
   - Situace na konci 2. klasifikačního období před finální zkouškou.
   - Reprezentuje horní mez predikční přesnosti, neboť zkouškové známky $G1$ a $G2$ jsou přirozeně silně autokorelované s finální známkou $G3$ ($r_{G1,G3} = 0.826$, $r_{G2,G3} = 0.905$).
2. **Scénář B: Včasné varování (Early Warning – BEZ zkouškových známek $G1$ a $G2$):**
   - Situace v průběhu semestru, kdy má škola k dispozici socio-demografické charakteristiky, rodinné zázemí, studijní návyky a průběžnou docházku (`absences`), ale studenti ještě nesložili dílčí zkoušky $G1$ a $G2$.
   - Z hlediska strojového učení jde o klíčovou výzvu: odhalit latentní rizikové vzorce chování dříve, než se projeví propadnutím v dílčích testech.

---

## 2. Metodická architektura a eliminace Data Leakage

Aby byly výsledky reprodukovatelné a odolné vůči metodické kritice, byly implementovány následující principy:

### 2.1 100% Fold-Isolated Preprocessing: Pořadí StandardScaler $\to$ KNNImputer
V předchozích fázích projektu byl `KNNImputer` spuštěn před škálováním, což vedlo k tomu, že euklidovské vzdálenosti byly zkresleny příznaky s velkým fyzickým rozsahem (např. `absences` s rozsahem 0 až 93 či věk studentů).

V aktuální verzi V3.1 je posloupnost transformací v `Pipeline` metodicky korigována:
$$\text{Pipeline} = \Big[ \big(\text{'scaler'}, \text{StandardScaler}()\big), \; \big(\text{'imputer'}, \text{KNNImputer}(n=5)\big), \; \big(\text{'m'}, \text{model}\big) \Big]$$
- `StandardScaler` v scikit-learn standardizuje data na nulový průměr a jednotkový rozptyl s **korektním ignorováním chybějících hodnot (`NaN`)**.
- `KNNImputer` následně počítá euklidovské vzdálenosti v **normalizovaném prostoru z-skóre**, kde žádný příznak nedominuje pouhou volbou fyzikální jednotky.
- Cílová proměnná $G3$ i dílčí známky jsou před modelováním striktně odděleny. Všechny transformace probíhají **výhradně na trénovacím foldu ($X_{\text{train}}$)** a testovací fold ($X_{\text{val}}$) pouze transformují, což eliminuje target leakage i test-fold leakage.

### 2.2 Transparentní ošetření chybějící docházky (`absences`) a citlivostní analýza
V surovém souboru `student_performance_Final.csv` má **218 z 649 studentů (33.6 %)** u proměnné `absences` hodnotu `-1` (kód chybějícího údaje, povolený rozsah je 0 až 93):
1. **Indikátor chybění:** Před konverzí `-1` na `NaN` byl vytvořen explicitní binární sloupec `absences_missing` ($1 = \text{údaj chyběl}$, $0 = \text{zaznamenán}$).
2. **Test mechanismu chybění:**
   - Průměrná známka studentů s chybějící docházkou: $G3 = 12.10 \pm 3.15$.
   - Průměrná známka studentů se zaznamenanou docházkou: $G3 = 11.81 \pm 3.27$.
   - Dvouvýběrový $t$-test: $t = 1.067, \; p = 0.2862$.
   - **Interpretace:** $t$-test nenalezl statisticky významný rozdíl ($p = 0.29$), což neodporuje předpokladu Missing at Random / Missing Completely at Random (MAR/MCAR). Neexistuje empirický důkaz o tom, že by studenti s chybějící docházkou systematicky prospívali hůře či lépe.
3. **Analýza citlivosti (Sensitivity Analysis) ve Scénáři B:**
   Ověřili jsme stabilitu predikce po úplném vyřazení proměnné `absences` i jejího indikátoru:
   - **Ridge regrese:** S absencemi $R^2 = 0.2533 \pm 0.0837$ vs. Zcela bez absencí $R^2 = 0.2534 \pm 0.0820$ ($\Delta R^2 = -0.0001$).
   - **Random Forest:** S absencemi $R^2 = 0.2564 \pm 0.0618$ vs. Zcela bez absencí $R^2 = 0.2549 \pm 0.0678$ ($\Delta R^2 = +0.0014$).
   - **Závěr:** Rozdíly v $R^2$ mezi modely s absencemi a zcela bez absencí se pohybují typicky v rozmezí $\pm 0.001$ až $\pm 0.008$ (v závislosti na náhodném seedu a dělení křížové validace). Vzhledem ke směrodatným odchylkám foldů ($\pm 0.06$ až $\pm 0.08$) jsou tyto rozdíly plně v rámci náhodné variability CV; nebyl nalezen důkaz, že by predikce modelů stála na artefaktu imputace docházky.

### 2.3 Rozdělení rolí skriptů v repozitáři
- **`scripts/clean_data.py`:** Slouží výhradně pro generování ad-hoc statického CSV souboru pro průzkumovou analýzu (EDA), vizualizaci a shlukování. Výsledky modelování z něj nečerpají. Zpracovaný CSV soubor byl zařazen do `.gitignore`.
- **`scripts/test_models.py`:** Načítá surová data a importuje ověřené konstanty (`RANGES`, `BINARY_MAPS`, `MULTI_CATEGORIES`) z `clean_data.py` (Single Source of Truth). Veškeré škálování a imputaci provádí za běhu uvnitř pipelines.

---

## 3. Výsledky regresního benchmarku (5-Fold CV)

Všechny modely byly vyhodnoceny pomocí 5-násobné křížové validace (`KFold(n_splits=5, shuffle=True, random_state=42)`) s Pipeline `StandardScaler` $\to$ `KNNImputer`.

### Scénář A: Průběžná predikce (včetně $G1$ a $G2$)
Vysvětlený rozptyl dosahuje $R^2 \approx 0.84$ a průměrná absolutní chyba činí cca 0.81 bodu:

| Model | $R^2$ (vysvětlený rozptyl) | RMSE (chyba známky) | MAE (průměrná chyba) |
| :--- | :---: | :---: | :---: |
| **SVR (Lineární jádro, C=1.0)** | **$0.841 \pm 0.039$** | **$1.277 \pm 0.304$** | **$0.809 \pm 0.113$** |
| **Ridge regrese (L2, $\alpha=1.0$)** | $0.837 \pm 0.033$ | $1.293 \pm 0.278$ | $0.841 \pm 0.109$ |
| **Lineární regrese (OLS)** | $0.837 \pm 0.033$ | $1.293 \pm 0.277$ | $0.842 \pm 0.108$ |
| **Random Forest (100 stromů)** | $0.837 \pm 0.026$ | $1.292 \pm 0.248$ | $0.839 \pm 0.091$ |
| **Gradient Boosting (100 stromů)** | $0.820 \pm 0.037$ | $1.354 \pm 0.278$ | $0.880 \pm 0.099$ |
| **Rozhodovací strom ($\text{depth}=5$)** | $0.794 \pm 0.054$ | $1.454 \pm 0.353$ | $0.877 \pm 0.096$ |
| **Lasso regrese (L1, výchozí $\alpha=1.0$)** | $0.755 \pm 0.058$ | $1.590 \pm 0.374$ | $1.058 \pm 0.166$ |
| **ElasticNet ($\alpha=1.0, l_1=0.5$)** | $0.735 \pm 0.059$ | $1.651 \pm 0.376$ | $1.088 \pm 0.158$ |
| **SVR (RBF jádro, C=1.0)** | $0.702 \pm 0.050$ | $1.750 \pm 0.349$ | $1.083 \pm 0.121$ |
| **k-NN Regresor ($k=5$)** | $0.465 \pm 0.044$ | $2.340 \pm 0.362$ | $1.662 \pm 0.183$ |
| **Dummy (Baseline průměr)** | $-0.013 \pm 0.012$ | $3.211 \pm 0.394$ | $2.410 \pm 0.235$ |

*Poznámka k Lasso:* Výchozí hodnota $\alpha=1.0$ v scikit-learn standardizované příznaky přepenalizuje ($R^2 \approx 0.755$). Po vyladění na $\alpha=0.1$ dosahuje Lasso $R^2 = 0.848$, čímž se plně vyrovnává Ridge a OLS.

---

### Scénář B: Včasné varování (BEZ známek $G1$ a $G2$)
Včasná predikce bez známek dosahuje stropu okolo $R^2 \approx 0.25 - 0.26$ s chybou MAE cca 2.0 bodu:

| Model | $R^2$ (vysvětlený rozptyl) | RMSE (chyba známky) | MAE (průměrná chyba) |
| :--- | :---: | :---: | :---: |
| **Random Forest (100 stromů)** | **$0.259 \pm 0.075$** | **$2.736 \pm 0.313$** | **$2.029 \pm 0.111$** |
| **SVR (RBF jádro, C=1.0)** | **$0.255 \pm 0.067$** | **$2.747 \pm 0.332$** | **$1.976 \pm 0.144$** |
| **Lineární regrese (OLS)** | $0.253 \pm 0.084$ | $2.742 \pm 0.282$ | $2.006 \pm 0.128$ |
| **Ridge regrese (L2, $\alpha=1.0$)** | $0.253 \pm 0.084$ | $2.741 \pm 0.282$ | $2.006 \pm 0.128$ |
| **SVR (Lineární jádro, C=1.0)** | $0.248 \pm 0.076$ | $2.753 \pm 0.288$ | $2.040 \pm 0.108$ |
| **Gradient Boosting (100 stromů)** | $0.222 \pm 0.092$ | $2.796 \pm 0.276$ | $2.095 \pm 0.060$ |
| **ElasticNet ($\alpha=1.0, l_1=0.5$)** | $0.158 \pm 0.027$ | $2.929 \pm 0.382$ | $2.155 \pm 0.191$ |
| **k-NN Regresor ($k=5$)** | $0.089 \pm 0.087$ | $3.039 \pm 0.402$ | $2.235 \pm 0.201$ |
| **Rozhodovací strom ($\text{depth}=5$)** | $0.065 \pm 0.130$ | $3.064 \pm 0.325$ | $2.268 \pm 0.159$ |
| **Lasso regrese (L1, $\alpha=1.0$)** | $0.045 \pm 0.018$ | $3.120 \pm 0.396$ | $2.324 \pm 0.235$ |
| **Dummy (Baseline průměr)** | $-0.013 \pm 0.012$ | $3.211 \pm 0.394$ | $2.410 \pm 0.235$ |

---

## 4. Výsledky klasifikačního benchmarku (Stratified 5-Fold CV)

### 4.1 Nevyváženost tříd a metodická interpretace
Dataset vykazuje výraznou nerovnováhu tříd:
- **Prospěl ($G3 \ge 10$):** 549 studentů (**84.6 %**)
- **Neprospěl ($G3 < 10$):** 100 studentů (**15.4 %**)

Triviální Dummy klasifikátor predikující majoritu sice dosahuje celkové Accuracy 84.6 %, ale jeho **Recall pro menšinovou třídu je 0 %**. Pro praktické nasazení ve škole je zásadní **Recall neprospívajících** a **Vyvážená přesnost (Balanced Accuracy)**.

> **Nejistota měření a ekvivalenční třídy:** Menšinová třída neprospívajících čítá přesně 100 studentů (20 na fold). Změna klasifikace u jediného studenta představuje posun v recallu o 1.0 %. **Rozdíly mezi předními modely leží uvnitř směrodatné odchylky křížové validace ($\pm 5 - 9\,\%$)**. Modely proto neklasifikujeme jako absolutně "nejlepší", ale dělíme je do ekvivalenčních výkonnostních tříd.

### Scénář A (se známkami $G1$ a $G2$):
| Model | Vyvážená Acc (%) | Recall Neprospěl (%) | Accuracy (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** | **$88.0 \pm 4.6$** | **$82.0 \pm 10.3$** | $92.1 \pm 1.0$ | $0.953 \pm 0.006$ |
| **SVC (Lineární jádro, Balanced)** | $86.3 \pm 3.0$ | $82.0 \pm 6.8$ | $89.2 \pm 0.5$ | $0.934 \pm 0.003$ |
| **Logistická regrese (Balanced)** | $86.0 \pm 5.3$ | $82.0 \pm 11.2$ | $88.7 \pm 1.5$ | $0.931 \pm 0.009$ |
| **Logistická regrese (Standard)** | $84.0 \pm 4.2$ | $72.0 \pm 10.3$ | $92.3 \pm 0.9$ | $0.955 \pm 0.005$ |
| **SVC (RBF jádro, Balanced)** | $82.9 \pm 6.9$ | $73.0 \pm 14.4$ | $89.8 \pm 2.0$ | $0.939 \pm 0.011$ |
| **Gradient Boosting (Standard)** | $82.8 \pm 4.2$ | $69.0 \pm 8.6$ | $92.3 \pm 1.5$ | $0.955 \pm 0.008$ |
| **Rozhodovací strom (Balanced)** | $81.2 \pm 3.1$ | $72.0 \pm 6.8$ | $87.5 \pm 1.6$ | $0.924 \pm 0.010$ |
| **Random Forest (Standard)** | $81.1 \pm 4.0$ | $64.0 \pm 9.2$ | $92.9 \pm 0.9$ | $0.959 \pm 0.005$ |
| **k-NN Klasifikátor ($k=5$)** | $59.4 \pm 5.5$ | $21.0 \pm 11.6$ | $86.0 \pm 1.3$ | $0.922 \pm 0.007$ |
| **Dummy (Baseline majorita)** | $50.0 \pm 0.0$ | $0.0 \pm 0.0$ | $84.6 \pm 0.0$ | $0.917 \pm 0.000$ |

---

### Scénář B: Včasné varování (BEZ známek $G1$ a $G2$):
Ve Scénáři B je kompenzace nevyváženosti tříd (`class_weight='balanced'`) klíčová:

| Model | Vyvážená Acc (%) | Recall Neprospěl (%) | Accuracy (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **SVC (Lineární jádro, Balanced)** | **$74.0 \pm 9.5$** | **$71.0 \pm 19.3$** | $76.1 \pm 2.9$ | $0.845 \pm 0.016$ |
| **SVC (RBF jádro, Balanced)** | $71.8 \pm 5.9$ | $53.0 \pm 12.5$ | $84.7 \pm 1.7$ | $0.909 \pm 0.010$ |
| **Logistická regrese (Balanced)** | $71.4 \pm 8.0$ | $65.0 \pm 15.8$ | $75.8 \pm 2.8$ | $0.845 \pm 0.016$ |
| **Rozhodovací strom (Balanced)** | $70.1 \pm 7.2$ | $60.0 \pm 13.0$ | $77.0 \pm 4.8$ | $0.855 \pm 0.033$ |
| **Random Forest (Balanced)** | $68.0 \pm 4.6$ | $43.0 \pm 10.3$ | $85.4 \pm 1.5$ | $0.915 \pm 0.009$ |
| **Gradient Boosting (Standard)** | $63.6 \pm 7.2$ | $31.0 \pm 13.6$ | $86.1 \pm 3.5$ | $0.921 \pm 0.020$ |
| **Logistická regrese (Standard)** | $63.5 \pm 4.7$ | $31.0 \pm 10.2$ | $86.0 \pm 1.2$ | $0.921 \pm 0.007$ |
| **Random Forest (Standard)** | $57.0 \pm 3.9$ | $15.0 \pm 8.4$ | $86.0 \pm 1.1$ | $0.923 \pm 0.005$ |
| **k-NN Klasifikátor ($k=5$)** | $56.1 \pm 4.5$ | $14.0 \pm 9.7$ | $85.2 \pm 1.0$ | $0.918 \pm 0.005$ |
| **Dummy (Baseline majorita)** | $50.0 \pm 0.0$ | $0.0 \pm 0.0$ | $84.6 \pm 0.0$ | $0.917 \pm 0.000$ |

---

## 5. Nestranné ladění hyperparametrů pomocí Nested CV (5x5)

Běžné porovnání `best_score_` z GridSearchCV na stejném CV rozdělení trpí selekčním optimismem. Proto jsme implementovali **Nested Cross-Validation (5x5)**:
- **Vnější smyčka (5 foldů):** odhaduje zobecnění na neviděných datech.
- **Vnitřní smyčka (5 foldů):** provádí mřížkové prohledávání a volbu hyperparametrů.
- **Pipeline:** Škálování `StandardScaler` i imputace `KNNImputer` probíhají striktně uvnitř jednotlivých foldů.

| Scénář | Úloha | Model | Prohledávaná mřížka | Nalezený parametr | Default skóre (5-Fold CV) | Tuned skóre (Nested 5x5 CV) | Nestranná změna ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Scénář A** | Regrese | **Lasso Regrese** | $\alpha \in [0.01, 1.0]$ | $\alpha = 0.1$ | $0.7548 \pm 0.0582$ | **$0.8480 \pm 0.0362$** | **$+0.0932$ ($R^2$)** |
| **Scénář A** | Regrese | **Ridge Regrese** | $\alpha \in [0.1, 100]$ | $\alpha = 10.0$ | $0.8374 \pm 0.0329$ | **$0.8376 \pm 0.0335$** | $+0.0003$ ($R^2$) |
| **Scénář B** | Regrese | **SVR (RBF)** | $C \in [0.5, 5]$, $\epsilon \in [0.1, 0.2]$ | $C=5.0, \epsilon=0.2$ | $0.2549 \pm 0.0665$ | **$0.2765 \pm 0.0721$** | $+0.0216$ ($R^2$) |
| **Scénář B** | Regrese | **Random Forest** | `depth: 4..None, n: 50..100` | `depth=None, n=100` | $0.2593 \pm 0.0751$ | **$0.2477 \pm 0.0645$** | $-0.0116$ ($R^2$) |
| **Scénář A** | Klasifikace | **SVC (Balanced)** | $C \in [0.1, 5]$, kernel: lin/rbf | $C=0.1$, linear | $82.9 \pm 6.9\,\%$ | **$82.9 \pm 3.5\,\%$** | $-0.00\,\%$ (Vyvážená Acc) |
| **Scénář B** | Klasifikace | **Random Forest (Bal)** | `depth: 4..8, n: 50..100` | `depth=4, n=100` | $68.0 \pm 4.6\,\%$ | **$76.2 \pm 6.1\,\%$** | **$+8.18\,\%$ (Vyvážená Acc)** |

### Objektivní interpretace přínosu ladění a reproducibilita:
1. **Lasso regrese (+0.0932):** Nárůst je dán **korekcí nevhodné výchozí penalizace $\alpha=1.0$**. Při $\alpha=0.1$ Lasso dosahuje $R^2 = 0.848$, což odpovídá Ridge i OLS.
2. **Random Forest Klasifikace ve Scénáři B:** Omezení hloubky na `max_depth=4` funguje jako účinná regularizace proti přeučení listů stromů. Zlepšení vyvážené přesnosti se typicky pohybuje **v rozmezí přibližně $+5$ až $+8$ procentních bodů** (v závislosti na verzi knihoven a dělení CV; v našem referenčním běhu $68.0\,\% \to 76.2\,\%$, tj. $+8.18\,\%$). Vzhledem k rozptylu foldů ($\pm 4.6\,\%$ vs. $\pm 6.1\,\%$) a velikosti vzorku menšiny (100 studentů) je třeba toto zlepšení interpretovat střízlivě jako stabilizaci modelu, nikoliv kvalitativní průlom.
3. **Marginální posuny u ostatních modelů:** Posun Ridge ($+0.0003$), SVR ($+0.0216$) i mírný pokles RF regrese ($-0.0116$) jsou v rámci směrodatné odchylky křížové validace ($\pm 0.03$ až $\pm 0.07$) a představují statistický šum.

---

## 6. Statistické ověření 7 týmových hypotéz

Všechny proměnné v hypotézách H1 až H7 mají v surovém datasetu nulový počet chybějících hodnot. Výpočty probíhají na skutečných datech pomocí neparametrických testů a OLS modelů.

### Korelační terminologie a Bonferroniho korekce:
Data jsou observační, nikoliv intervenční. Důsledně používáme **asociativní a korelační formulace** namísto kauzálních.

Při testování **7 hypotéz současně** aplikujeme konzervativní **Bonferroniho korekci**:
$$\alpha_{\text{adj}} = \frac{0.05}{7} \approx 0.00714$$

| Hypotéza | Testovaná závislost | Statistická metoda | Testová statistika & Velikost účinku | Přesná $p$-hodnota | Výsledek (Bonferroni $\alpha=0.0071$) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **H1** | `studytime` $\to$ `G3` | Spearmanův rank test | $\rho = +0.2747$ (slabá asociace) | $1.06 \times 10^{-12}$ | **Statisticky významná:** Delší studium slabě pozitivně koreluje se známkou. |
| **H2** | `romantic` $\to$ `studytime` | Mann-Whitney U test | $U = 51899.5$, $\Delta = -0.11$ h | $0.1735$ | **Neprokázána ($H_0$ platí):** Partnerský vztah nesouvisí s dobou studia. |
| **H3** | VŠ rodiče $\to$ `G3` | Mann-Whitney U test | $U = 60521.0$, $\Delta = +1.55$ bodu | $1.82 \times 10^{-11}$ | **Statisticky významná:** Děti vysokoškoláků dosahují v průměru o 1.55 bodu více. |
| **H4** | `higher` $\to$ vzdělání rodičů | Mann-Whitney U test | $U = 28377.0$, $\Delta = +1.47$ stupně | $4.56 \times 10^{-9}$ | **Statisticky významná:** Studenti mířící na VŠ mají vzdělanější rodiče. |
| **H5** | `famrel` $\to$ `Dalc+Walc` | Spearmanův rank test | $\rho = -0.1134$ (velmi slabá asociace) | $0.0038$ | **Statisticky významná ($p < 0.0071$):** Lepší vztahy slabě negativně korelují s alkoholem. |
| **H6** | `Dalc+Walc` $\to$ `G3` | Spearmanův rank test | $\rho = -0.1954$ (slabá negativní asociace) | $5.22 \times 10^{-7}$ | **Statisticky významná:** Vyšší konzumace mírně koreluje s horší známkou. |
| **H7** | `school + address` $\to$ `G3` | OLS s kontrolou školy | $\beta_{\text{addr}} = +0.537$ (univariátně $\Delta = +1.18$) | OLS $p = 0.0578$ (univariátně $p < 1\text{e-}4$) | **Marginální (po kontrole školy nevýznamná po Bonferroni).** |

### Podrobný rozbor hypotézy H7 (Bydliště vs. Škola):
Při univariátním testování mají studenti z města (`address = U`) významně vyšší známky než z venkova ($p < 1\text{e-}4$, rozdíl $+1.18$ bodu). Jakmile však do modelu zařadíme školu (`school = GP vs MS`, $\beta_{\text{school}} = -1.743, p < 10^{-9}$), koeficient bydliště klesne na $\beta = +0.537$ s $p = 0.0578$.
To dokazuje, že **rozdíl v prospěchu mezi městem a venkovem je z velké části vysvětlen volbou školy** (venkovští studenti častěji navštěvují školu MS, která má celkově nižší průměrné skóre).

---

## 7. Out-of-Fold (OOF) Permutation Feature Importance

Permutační důležitost je počítána na validačních foldech napříč 5-násobnou křížovou validací (StandardScaler $\to$ KNNImputer) s agregací dummy proměnných do mateřských kategorií:

| Pořadí | Příznak | OOF Důležitost (%) | Popis a význam |
| :---: | :--- | :---: | :--- |
| **1.** | **`failures`** | **$59.89\,\%$** | **Drtivě nejdůležitější prediktor!** Předchozí propadnutí je nejsilnějším varovným signálem. |
| **2.** | **`higher`** | **$10.76\,\%$** | Vnitřní ambice dosáhnout vysokoškolského vzdělání (silný protektivní faktor). |
| **3.** | **`schoolsup`** | **$5.06\,\%$** | Mimořádná školní podpora (indikátor dříve zachycených obtíží). |
| **4.** | **`school`** | **$4.49\,\%$** | Výběr školy (na validačních datech klesla z dřívějších 9.1 % na reálných 4.5 %). |
| **5.** | **`Dalc`** | **$3.60\,\%$** | Konzumace alkoholu v pracovní dny (konzistentní s H6). |
| **6.** | **`reason`** | **$2.82\,\%$** | Důvod výběru školy (reputace, preference kurzů). |
| **7.** | **`studytime`** | **$2.71\,\%$** | Týdenní čas věnovaný samostudiu (konzistentní s H1). |
| **8.** | **`Walc`** | **$2.64\,\%$** | Víkendová konzumace alkoholu. |
| **9.** | **`famrel`** | **$1.36\,\%$** | Kvalita rodinných vztahů. |
| **10.** | **`age`** | **$1.31\,\%$** | Věk studenta. |

---

## 8. Reprodukovatelnost a softwarové prostředí

Referenční číselné hodnoty uvedené v této zprávě byly generovány v následujícím prostředí:
- **Operační systém:** Windows 11 (PowerShell 7)
- **Python:** 3.12.10
- **scikit-learn:** 1.9.1
- **numpy:** 2.5.4 | **pandas:** 3.1.0 | **scipy:** 1.18.1

Při spuštění na jiných platformách (např. Linux), jiném počtu jader/vláken knihovny OpenMP/BLAS nebo jiných podverzích knihovny scikit-learn se mohou vyskytnout mírné odchylky v dělení foldů křížové validace u hraničních vzorků. Všechny závěry a trendy však zůstávají stabilní.
