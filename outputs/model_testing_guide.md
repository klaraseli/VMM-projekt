# Průvodce a technická zpráva: Testování a vyhodnocování prediktivních modelů
**Předmět:** MPC-VMM (Vícerozměrná analýza dat, FEKT VUT) | **Dataset:** `student_performance_Final.csv`  
**Autor role:** Nika (Větev `nikus`) | **Kódový skript:** `VMM-projekt/scripts/test_models.py`  
**Verze metodiky:** V3 (Striktní metodická integrita, Fold-Isolated Preprocessing a OOF evaluace)

---

## 1. Vymezení modelovací úlohy a výzkumný rámec

V rámci týmového projektu MPC-VMM zodpovídá tato část za **návrh, implementaci, ladění a rigorózní evaluaci prediktivních modelů strojového učení** pro predikci akademického výkonu studentů.

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

### 2.1 100% Fold-Isolated Preprocessing v Pipeline
V dřívějších implementacích hrozil drobný únik informací tím, že se imputace chybějících hodnot spouštěla na celém datasetu před křížovou validací. V této verzi je **jakýkoliv únik dat striktně vyloučen**:
- Cílová proměnná $G3$ a dílčí známky jsou před modelováním striktně odděleny.
- Původní matice příznaků obsahuje chybějící hodnoty (`NaN`).
- Všechny modely jsou zapouzdřeny v `sklearn.pipeline.Pipeline`:
  $$\text{Pipeline} = \Big[ \big(\text{'imputer'}, \text{KNNImputer}(n=5)\big), \; \big(\text{'scaler'}, \text{StandardScaler}()\big), \; \big(\text{'model'}, \mathcal{M}\big) \Big]$$
- Při každém kroku 5-násobné křížové validace i vnořené CV se `KNNImputer` a `StandardScaler` trénují **výhradně na trénovacím foldu ($X_{\text{train}}$)** a testovací fold ($X_{\text{val}}$) pouze transformují. Žádná informace z testovacích foldů neproniká do odhadu parametrů.

### 2.2 Transparentní ošetření chybějící docházky (`absences`)
V surovém souboru `student_performance_Final.csv` má **218 z 649 studentů (33.6 %)** u proměnné `absences` hodnotu `-1`. Jde o zjevný kód chybějícího údaje (docházka nemůže být záporná, povolený rozsah je 0 až 93):
1. **Přidání indikátoru chybění:** Před konverzí `-1` na `NaN` byl vytvořen explicitní binární sloupec `absences_missing` ($1 = \text{údaj chyběl}$, $0 = \text{zaznamenán}$). Modely tak mají možnost zachytit případnou nenáhodnost chybění.
2. **Ověření mechanismu chybění (MAR/MCAR):**
   - Průměrná známka studentů s chybějící docházkou: $G3 = 12.10 \pm 3.15$.
   - Průměrná známka studentů se zaznamenanou docházkou: $G3 = 11.81 \pm 3.27$.
   - Dvouvýběrový $t$-test: $t = 1.067, \; p = 0.2862$.
   - Rozdíl je statisticky nevýznamný ($p > 0.28$), což potvrzuje, že chybějící hodnota docházky nesouvisí s horším či lepším studijním výsledkem a splňuje předpoklad Missing at Random (MAR / MCAR).
3. **Analýza citlivosti (Sensitivity Analysis):**
   Ověřili jsme predikční stabilitu Scénáře B při kompletním vyřazení docházky z modelů:
   - **Ridge regrese:** S absencemi $R^2 = 0.2490 \pm 0.0846$ vs. Zcela bez absencí $R^2 = 0.2489 \pm 0.0819$ ($\Delta R^2 = +0.0001$).
   - **Random Forest:** S absencemi $R^2 = 0.2604 \pm 0.0672$ vs. Zcela bez absencí $R^2 = 0.2590 \pm 0.0697$ ($\Delta R^2 = +0.0014$).
   - **Závěr:** Změna výkonu modelů po úplném odstranění docházky nepřesahuje 0.0014, což jednoznačně dokazuje, že predikce Scénáře B není artefaktem imputace docházky, ale stojí na robustních socio-demografických příznacích.

---

## 3. Výsledky regresního benchmarku (5-Fold CV)

Všechny modely byly evaluovány pomocí 5-násobné křížové validace (`KFold(n_splits=5, shuffle=True, random_state=42)`). Níže uvedené tabulky obsahují průměr a směrodatnou odchylku napříč validačními foldy.

### Scénář A: Průběžná predikce (včetně $G1$ a $G2$)
V tomto scénáři vysvětlují průběžné známky drtivou většinu variability ($R^2 \approx 0.84$). Průměrná absolutní chyba činí pouhých cca 0.82 bodu na dvacetibodové škále.

| Model | $R^2$ (vysvětlený rozptyl) | RMSE (chyba známky) | MAE (průměrná chyba) |
| :--- | :---: | :---: | :---: |
| **SVR (Lineární jádro, C=1.0)** | **$0.840 \pm 0.039$** | **$1.283 \pm 0.300$** | **$0.817 \pm 0.114$** |
| **Ridge regrese (L2, $\alpha=1.0$)** | $0.837 \pm 0.033$ | $1.294 \pm 0.277$ | $0.845 \pm 0.109$ |
| **Lineární regrese (OLS)** | $0.837 \pm 0.033$ | $1.294 \pm 0.277$ | $0.846 \pm 0.109$ |
| **Random Forest (100 stromů)** | $0.836 \pm 0.028$ | $1.296 \pm 0.255$ | $0.843 \pm 0.092$ |
| **Gradient Boosting (100 stromů)** | $0.817 \pm 0.038$ | $1.367 \pm 0.289$ | $0.876 \pm 0.113$ |
| **Rozhodovací strom ($\text{depth}=5$)** | $0.799 \pm 0.073$ | $1.434 \pm 0.416$ | $0.873 \pm 0.127$ |
| **Lasso regrese (L1, výchozí $\alpha=1.0$)** | $0.755 \pm 0.058$ | $1.590 \pm 0.374$ | $1.058 \pm 0.166$ |
| **ElasticNet ($\alpha=1.0, l_1=0.5$)** | $0.735 \pm 0.059$ | $1.651 \pm 0.376$ | $1.088 \pm 0.158$ |
| **SVR (RBF jádro, C=1.0)** | $0.696 \pm 0.051$ | $1.768 \pm 0.353$ | $1.093 \pm 0.122$ |
| **k-NN Regresor ($k=5$)** | $0.473 \pm 0.059$ | $2.322 \pm 0.393$ | $1.646 \pm 0.204$ |
| **Dummy (Baseline průměr)** | $-0.013 \pm 0.012$ | $3.211 \pm 0.394$ | $2.410 \pm 0.235$ |

*Poznámka k výchozímu Lasso:* Nižší skóre výchozího Lasso ($R^2 = 0.755$) je způsobeno standardní knihovní hodnotou $\alpha=1.0$, která je pro škálované příznaky příliš penalizující a potlačuje i signifikantní prediktory. Po vyladění na $\alpha=0.1$ dosahuje Lasso $R^2 = 0.848$.

---

### Scénář B: Včasné varování (BEZ známek $G1$ a $G2$)
Včasná predikce bez známek dosahuje stropu okolo $R^2 \approx 0.25 - 0.26$ s chybou MAE cca 2.0 bodu. Zde se projevuje schopnost nelineárních modelů (Random Forest a SVR s RBF jádrem) modelovat interakce mezi prostředím a návyky studenta:

| Model | $R^2$ (vysvětlený rozptyl) | RMSE (chyba známky) | MAE (průměrná chyba) |
| :--- | :---: | :---: | :---: |
| **Random Forest (100 stromů)** | **$0.258 \pm 0.074$** | **$2.738 \pm 0.316$** | **$2.039 \pm 0.112$** |
| **SVR (RBF jádro, C=1.0)** | **$0.251 \pm 0.067$** | **$2.754 \pm 0.328$** | **$1.980 \pm 0.142$** |
| **Lineární regrese (OLS)** | $0.249 \pm 0.085$ | $2.748 \pm 0.273$ | $2.012 \pm 0.120$ |
| **SVR (Lineární jádro, C=1.0)** | $0.249 \pm 0.074$ | $2.750 \pm 0.279$ | $2.043 \pm 0.103$ |
| **Ridge regrese (L2, $\alpha=1.0$)** | $0.249 \pm 0.085$ | $2.748 \pm 0.273$ | $2.011 \pm 0.120$ |
| **Gradient Boosting (100 stromů)** | $0.226 \pm 0.065$ | $2.794 \pm 0.282$ | $2.078 \pm 0.071$ |
| **ElasticNet ($\alpha=1.0, l_1=0.5$)** | $0.158 \pm 0.027$ | $2.929 \pm 0.382$ | $2.155 \pm 0.191$ |
| **k-NN Regresor ($k=5$)** | $0.092 \pm 0.070$ | $3.035 \pm 0.384$ | $2.243 \pm 0.216$ |
| **Lasso regrese (L1, $\alpha=1.0$)** | $0.045 \pm 0.018$ | $3.120 \pm 0.396$ | $2.324 \pm 0.235$ |
| **Dummy (Baseline průměr)** | $-0.013 \pm 0.012$ | $3.211 \pm 0.394$ | $2.410 \pm 0.235$ |
| **Rozhodovací strom ($\text{depth}=5$)** | $-0.024 \pm 0.178$ | $3.205 \pm 0.395$ | $2.337 \pm 0.208$ |

---

## 4. Výsledky klasifikačního benchmarku (Stratified 5-Fold CV)

### 4.1 Nevyváženost tříd a metodická interpretace
Dataset vykazuje výraznou nerovnováhu tříd:
- **Prospěl ($G3 \ge 10$):** 549 studentů (**84.6 %**)
- **Neprospěl ($G3 < 10$):** 100 studentů (**15.4 %**)

Triviální Dummy klasifikátor sice dosahuje celkové Accuracy 84.6 %, ale jeho **Recall pro menšinovou třídu je přesně 0 %** (neidentifikuje ani jednoho ohroženého studenta). Pro praktické nasazení ve škole je zásadní **Recall neprospívajících** a **Vyvážená přesnost (Balanced Accuracy)**.

> **Upozornění k velikosti vzorku menšiny:** Menšinová třída čítá přesně 100 studentů, což znamená v průměru 20 studentů na jeden fold křížové validace. Změna klasifikace u jediného studenta představuje posun v recallu o 1.0 %. **Rozdíly mezi horními modely (např. 74.5 % vs. 73.8 %) leží hluboko uvnitř náhodného šumu křížové validace ($\pm 5 - 7\,\%$)**. Modely proto hodnotíme v rámci ekvivalenčních tříd, nikoliv jako absolutně "nejlepší" či "nejhorší".

### Scénář A (se známkami $G1$ a $G2$):
| Model | Vyvážená Acc (%) | Recall Neprospěl (%) | Accuracy (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** | **$87.8 \pm 3.5$** | **$81.0 \pm 8.0$** | $92.4 \pm 1.3$ | $0.955 \pm 0.008$ |
| **Logistická regrese (Balanced)** | $85.7 \pm 3.7$ | $82.0 \pm 8.1$ | $88.3 \pm 1.3$ | $0.928 \pm 0.008$ |
| **SVC (Lineární jádro, Balanced)** | $85.5 \pm 2.7$ | $80.0 \pm 5.5$ | $89.4 \pm 1.1$ | $0.935 \pm 0.006$ |
| **Logistická regrese (Standard)** | $84.3 \pm 3.6$ | $73.0 \pm 8.1$ | $92.1 \pm 1.3$ | $0.954 \pm 0.008$ |
| **Rozhodovací strom (Balanced)** | $83.6 \pm 2.2$ | $77.0 \pm 5.1$ | $88.1 \pm 1.4$ | $0.928 \pm 0.009$ |
| **Gradient Boosting (Standard)** | $83.4 \pm 3.7$ | $70.0 \pm 8.9$ | $92.6 \pm 0.9$ | $0.957 \pm 0.006$ |
| **SVC (RBF jádro, Balanced)** | $83.0 \pm 5.9$ | $73.0 \pm 12.1$ | $90.0 \pm 1.7$ | $0.940 \pm 0.009$ |
| **Random Forest (Standard)** | $82.0 \pm 4.1$ | $66.0 \pm 9.2$ | $93.1 \pm 1.1$ | $0.960 \pm 0.006$ |
| **k-NN Klasifikátor ($k=5$)** | $60.6 \pm 4.4$ | $23.0 \pm 9.3$ | $86.6 \pm 1.2$ | $0.925 \pm 0.006$ |
| **Dummy (Baseline majorita)** | $50.0 \pm 0.0$ | $0.0 \pm 0.0$ | $84.6 \pm 0.0$ | $0.917 \pm 0.000$ |

---

### Scénář B: Včasné varování (BEZ známek $G1$ a $G2$):
Ve Scénáři B je kompenzace nevyváženosti tříd (`class_weight='balanced'`) klíčová. Nevážené modely detekují pouze 11 až 31 % neprospívajících, zatímco vyvážené lineární a stromové modely odhalí **67 až 73 % ohrožených studentů**:

| Model | Vyvážená Acc (%) | Recall Neprospěl (%) | Accuracy (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **SVC (Lineární jádro, Balanced)** | **$74.5 \pm 7.1$** | **$73.0 \pm 15.7$** | $75.5 \pm 2.7$ | $0.840 \pm 0.020$ |
| **Rozhodovací strom (Balanced)** | $73.8 \pm 2.2$ | $67.0 \pm 5.1$ | $78.6 \pm 3.1$ | $0.864 \pm 0.023$ |
| **Logistická regrese (Balanced)** | $72.8 \pm 7.1$ | $68.0 \pm 15.0$ | $76.1 \pm 2.4$ | $0.846 \pm 0.016$ |
| **SVC (RBF jádro, Balanced)** | $71.7 \pm 5.4$ | $53.0 \pm 11.2$ | $84.6 \pm 1.8$ | $0.908 \pm 0.010$ |
| **Random Forest (Balanced)** | $66.9 \pm 4.5$ | $40.0 \pm 10.0$ | $85.5 \pm 2.2$ | $0.916 \pm 0.014$ |
| **Gradient Boosting (Standard)** | $63.4 \pm 4.6$ | $31.0 \pm 9.7$ | $85.8 \pm 2.8$ | $0.919 \pm 0.017$ |
| **Logistická regrese (Standard)** | $61.6 \pm 3.5$ | $28.0 \pm 8.1$ | $84.9 \pm 0.9$ | $0.914 \pm 0.005$ |
| **Random Forest (Standard)** | $55.7 \pm 3.1$ | $13.0 \pm 6.8$ | $85.2 \pm 0.9$ | $0.918 \pm 0.005$ |
| **k-NN Klasifikátor ($k=5$)** | $54.6 \pm 3.2$ | $11.0 \pm 6.6$ | $84.7 \pm 0.9$ | $0.916 \pm 0.005$ |
| **Dummy (Baseline majorita)** | $50.0 \pm 0.0$ | $0.0 \pm 0.0$ | $84.6 \pm 0.0$ | $0.917 \pm 0.000$ |

---

## 5. Nestranné ladění hyperparametrů pomocí Nested CV (5x5)

Běžné vyhodnocení mřížkového prohledávání (GridSearchCV), kdy se nejlepší parametr vybírá a současně vyhodnocuje na stejném CV rozdělení, trpí **optimism bias** (výsledky vypadají lépe, než jaké je skutečné zobecnění).

Abychom získali objektivní obraz, implementovali jsme **Nested Cross-Validation (5x5)**:
- **Vnější smyčka (5 foldů):** odhaduje zobecnění na neviděných datech.
- **Vnitřní smyčka (5 foldů):** provádí mřížkové prohledávání a volbu hyperparametrů.
- **Pipeline:** Imputace i škálování probíhají striktně uvnitř jednotlivých foldů.

| Scénář | Úloha | Model | Prohledávaná mřížka | Nalezený parametr | Default skóre (5-Fold CV) | Tuned skóre (Nested 5x5 CV) | Nestranná změna ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Scénář A** | Regrese | **Lasso Regrese** | $\alpha \in [0.01, 1.0]$ | $\alpha = 0.1$ | $0.7548 \pm 0.0582$ | **$0.8481 \pm 0.0360$** | **$+0.0933$ ($R^2$)** |
| **Scénář A** | Regrese | **Ridge Regrese** | $\alpha \in [0.1, 100]$ | $\alpha = 10.0$ | $0.8371 \pm 0.0329$ | **$0.8375 \pm 0.0332$** | $+0.0003$ ($R^2$) |
| **Scénář B** | Regrese | **SVR (RBF)** | $C \in [0.5, 5]$, $\epsilon \in [0.1, 0.2]$ | $C=5.0, \epsilon=0.2$ | $0.2505 \pm 0.0674$ | **$0.2696 \pm 0.0730$** | $+0.0191$ ($R^2$) |
| **Scénář B** | Regrese | **Random Forest** | `depth: 4..None, n: 50..100` | `depth=6, n=100` | $0.2584 \pm 0.0743$ | **$0.2487 \pm 0.0588$** | $-0.0097$ ($R^2$) |
| **Scénář A** | Klasifikace | **SVC (Balanced)** | $C \in [0.1, 5]$, kernel: lin/rbf | $C=0.1$, linear | $83.0 \pm 5.9\,\%$ | **$84.4 \pm 4.4\,\%$** | $+1.41\,\%$ (Vyvážená Acc) |
| **Scénář B** | Klasifikace | **Random Forest (Bal)** | `depth: 4..8, n: 50..100` | `depth=4, n=100` | $66.9 \pm 4.5\,\%$ | **$74.3 \pm 7.4\,\%$** | **$+7.35\,\%$ (Vyvážená Acc)** |

### Objektivní interpretace přínosu ladění:
1. **Lasso regrese (+0.0933):** Tento nárůst není zázrakem pokročilého algoritmu, nýbrž **korekcí nevhodné výchozí hodnoty $\alpha=1.0$** z knihovny scikit-learn. Při $\alpha=1.0$ Lasso na standardizovaných datech přehnaně vynulovalo i klíčové prediktory. Snížením na optimální $\alpha=0.1$ se Lasso vyrovnává Ridge regresi a OLS ($R^2 \approx 0.848$).
2. **Random Forest Klasifikace ve Scénáři B (+7.35 %):** Zde má ladění skutečný strukturní přínos. Výchozí neomezená hloubka stromů trpí přeučením na listech. Explicitní omezení na `max_depth=4` funguje jako silný regularizátor a zvedá vyváženou přesnost ze 66.9 % na **74.3 %**.
3. **Marginální změny u ostatních modelů:** Posuny u Ridge regrese ($+0.0003$), SVR ($+0.019$) i mírný pokles u RF regrese ($-0.0097$) jsou plně v mezích směrodatné odchylky křížové validace ($\pm 0.03$ až $\pm 0.07$). Reprezentují statistický šum, nikoliv kvalitativní skok.

---

## 6. Statistické ověření 7 týmových hypotéz

Všechny proměnné figurující v týmových hypotézách H1 až H7 jsou v datasetu kompletně zaznamenané (bez chybějících hodnot). Výpočty probíhají na skutečných datech pomocí neparametrických testů a OLS modelů.

### Přísná korelační terminologie a korekce na vícenásobné testování:
Data jsou observační, nikoliv intervenční. V souladu s akademickými standardy se vyhýbáme kauzálním formulacím typu „faktor má vliv/dopad“ a hovoříme o **statistické asociaci, korelaci a rozdílu průměrů**.

Jelikož testujeme **7 hypotéz současně**, hrozí kumulace chyby I. druhu. Aplikovali jsme proto konzervativní **Bonferroniho korekci**:
$$\alpha_{\text{adj}} = \frac{0.05}{7} \approx 0.00714$$

| Hypotéza | Testovaná závislost | Statistická metoda | Testová statistika & Velikost účinku | Přesná $p$-hodnota | Výsledek (Bonferroni $\alpha=0.0071$) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **H1** | `studytime` $\to$ `G3` | Spearmanův rank test | $\rho = +0.2747$ (středně slabá asociace) | $1.06 \times 10^{-12}$ | **Statisticky významná:** Delší studium pozitivně koreluje se známkou. |
| **H2** | `romantic` $\to$ `studytime` | Mann-Whitney U test | $U = 51899.5$, $\Delta = -0.11$ h | $0.1735$ | **Neprokázána ($H_0$ platí):** Partnerský vztah nesouvisí s dobou studia. |
| **H3** | VŠ rodiče $\to$ `G3` | Mann-Whitney U test | $U = 60521.0$, $\Delta = +1.55$ bodu | $1.82 \times 10^{-11}$ | **Statisticky významná:** Děti vysokoškoláků dosahují v průměru o 1.55 bodu více. |
| **H4** | `higher` $\to$ vzdělání rodičů | Mann-Whitney U test | $U = 28377.0$, $\Delta = +1.47$ stupně | $4.56 \times 10^{-9}$ | **Statisticky významná:** Studenti mířící na VŠ mají výrazně vzdělanější rodiče. |
| **H5** | `famrel` $\to$ `Dalc+Walc` | Spearmanův rank test | $\rho = -0.1134$ (velmi slabá asociace) | $0.0038$ | **Statisticky významná ($p < 0.0071$):** Lepší vztahy slabě tlumí konzumaci. |
| **H6** | `Dalc+Walc` $\to$ `G3` | Spearmanův rank test | $\rho = -0.1954$ (slabá negativní asociace) | $5.22 \times 10^{-7}$ | **Statisticky významná:** Vyšší konzumace mírně koreluje s horší známkou. |
| **H7** | `school + address` $\to$ `G3` | OLS s kontrolou školy | $\beta_{\text{addr}} = +0.537$ (univariátně $\Delta = +1.18$) | OLS $p = 0.0578$ (univariátně $p = 0.0022$) | **Marginální (po kontrole školy nevýznamná po Bonferroni).** |

### Důležitý poznatek k hypotéze H7 (Bydliště vs. Škola):
Při univariátním testování mají studenti z města (`address = U`) významně vyšší známky než z vesnice ($p = 0.0022$, rozdíl $+1.18$ bodu). Jakmile však do modelu zařadíme školu (`school = GP vs MS`, $\beta_{\text{school}} = -1.743, p < 10^{-9}$), koeficient bydliště klesne na $\beta = +0.537$ s $p = 0.0578$.
To znamená, že **rozdíl v prospěchu mezi městem a venkovem je z velké části vysvětlen volbou školy** (studenti z venkova častěji navštěvují školu MS, která má celkově nižší průměrné skóre).

---

## 7. Out-of-Fold (OOF) Permutation Feature Importance

Při výpočtu permutační důležitosti na celém trénovacím datasetu dochází k nadhodnocení příznaků, na které se model přeučil (v dřívější verzi takto dominovala škola s 9.1 % a failures s 34.3 %).

V této verzi počítáme **Out-of-Fold (OOF) Permutation Importance**:
1. Random Forest se natrénuje na trénovacím foldu $X_{\text{train}}$.
2. Permutace jednotlivých příznaků a měření propadu $R^2$ probíhá **výhradně na validačním foldu $X_{\text{val}}$**.
3. Výsledky se zprůměrují přes všech 5 validačních foldů.
4. Dummy sloupce vícehodnotových kategorií (`Mjob`, `Fjob`, `reason`, `guardian`) jsou sloučeny zpět do mateřských faktorů (Grouped Importance).

| Pořadí | Příznak | OOF Důležitost (%) | Popis a význam |
| :---: | :--- | :---: | :--- |
| **1.** | **`failures`** | **$58.70\,\%$** | **Drtivě nejdůležitější prediktor!** Předchozí propadnutí představuje nejsilnější varovný signál. |
| **2.** | **`higher`** | **$10.52\,\%$** | Vnitřní ambice dosáhnout vysokoškolského vzdělání (silný protektivní faktor). |
| **3.** | **`school`** | **$5.12\,\%$** | Výběr školy (na validačních datech klesla z dřívějších 9.1 % na reálných 5.1 %). |
| **4.** | **`schoolsup`** | **$5.04\,\%$** | Mimořádná školní podpora (indikátor dříve zachycených problémů). |
| **5.** | **`Dalc`** | **$3.84\,\%$** | Konzumace alkoholu v pracovní dny (konzistentní s H6). |
| **6.** | **`reason`** | **$3.30\,\%$** | Důvod výběru školy (reputace, zaměření kurzu). |
| **7.** | **`studytime`** | **$2.56\,\%$** | Týdenní čas věnovaný samostudiu (konzistentní s H1). |
| **8.** | **`Walc`** | **$2.48\,\%$** | Víkendová konzumace alkoholu. |
| **9.** | **`health`** | **$1.40\,\%$** | Zdravotní stav studenta. |
| **10.** | **`famrel`** | **$1.34\,\%$** | Kvalita rodinných vztahů. |

---

## 8. Souhrnné výstupy a doporučení pro obhajobu

Všechny výstupy jsou vygenerovány v repozitáři v čistých formátech:
- **CSV tabulky:** `outputs/tables/` (7 tabulek pokrývajících benchmarky, ladění, hypotézy a důležitost).
- **Vektorové SVG grafy:** `outputs/figures/` (7 grafů ve vysokém rozlišení bez externích knihoven).

### Doporučené formulace pro týmovou prezentaci:
1. *„Včasná predikce (Scénář B) dokáže spolehlivě odhalit cca 73 % neprospívajících studentů, pokud použijeme vážený klasifikátor kompenzující 85:15 nerovnováhu tříd.“*
2. *„Nejvýznamnějším varovným faktorem je počet dřívějších neúspěchů (`failures`, 58.7 % OOF důležitosti), následovaný chybějící ambicí studovat vysokou školu (`higher`).“*
3. *„Všechny metodické kroky (imputace i škálování) probíhají striktně uvnitř křížové validace, čímž je zaručen nulový únik dat a plná vědecká reprodukovatelnost.“*
