"""
Analýza příznaků studentů BEZ G1, G2, G3.
G1+G2+G3 se použije jen k obarvení grafů a k rozdělení studentů do skupin.

Potřebné balíčky: pip install pandas numpy matplotlib seaborn scikit-learn
Grafy se ukládají do složky "plots" (a volitelně zobrazují, viz SHOW).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA

from clean_data import clean_data
from load_data import load_data

# ------------------------------ nastavení ------------------------------
DATA_PATH = "C://Users//Acer Aspire//Desktop//7. semestr//VMM//projekt//student_performance_Final.csv"   # stejně jako ve vašem test.py
SHOW = True                               # False = jen uložit grafy, nezobrazovat
OUT_DIR = "plots"
GRADES = ["G1", "G2", "G3"]
GROUP_LABELS = ["nízké známky", "střední známky", "vysoké známky"]
GROUP_COLORS = {"nízké známky": "tomato", "střední známky": "gold", "vysoké známky": "seagreen"}

sns.set_theme(style="whitegrid")
os.makedirs(OUT_DIR, exist_ok=True)


def finish(name):
    """Uloží aktuální graf a případně zobrazí."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close()


# ------------------------------ příprava dat ------------------------------
data = load_data(DATA_PATH)
df = clean_data(data)

total = df[GRADES].sum(axis=1).rename("total_grade")       # součet 3 známek (0-60)
X = df.drop(columns=GRADES)                                 # příznaky bez známek
group = pd.qcut(total, 3, labels=GROUP_LABELS)              # tercily podle součtu

binary_cols = [c for c in X.columns if X[c].nunique() == 2]
other_cols = [c for c in X.columns if c not in binary_cols]  # ordinální / číselné

print(f"Příznaků: {X.shape[1]} (binárních {len(binary_cols)}, ostatních {len(other_cols)})")
print(f"Studentů: {X.shape[0]}")
print(f"Součet známek: min {total.min()}, max {total.max()}, průměr {total.mean():.1f}")
print(group.value_counts().sort_index(), "\n")

# ------------------------------ 1) ROZPTYL ------------------------------
var_raw = X.var().sort_values(ascending=False)
var_scaled = pd.Series(MinMaxScaler().fit_transform(X).var(axis=0), index=X.columns)
var_scaled = var_scaled.sort_values()

print("Rozptyl (původní měřítko), top 10:")
print(var_raw.head(10).round(2), "\n")
print("Nejnižší rozptyl po škálování 0-1 (téměř konstantní příznaky):")
print(var_scaled.head(10).round(4), "\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
var_raw.sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Rozptyl příznaků (původní měřítko)")
axes[0].set_xscale("log")
var_scaled.plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Rozptyl po škálování na 0-1 (srovnatelný)")
finish("01_rozptyl")

# ------------------------------ 2) ČETNOSTI ------------------------------
n = len(X.columns)
ncols = 7
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows))
for ax, c in zip(axes.ravel(), X.columns):
    if X[c].nunique() > 12:  # spojité (absences, ...)
        sns.histplot(X[c], bins=20, ax=ax, color="steelblue")
    else:
        vc = X[c].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color="steelblue")
    ax.set_title(c, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("Četnosti hodnot jednotlivých příznaků", y=1.0)
finish("02_cetnosti")

# ------------------------------ 3) BOXPLOTY podle skupin známek ------------------------------
fig, axes = plt.subplots(3, 5, figsize=(20, 11))
for ax, c in zip(axes.ravel(), other_cols):
    sns.boxplot(x=group, y=X[c], ax=ax, palette=GROUP_COLORS, order=GROUP_LABELS)
    ax.set_title(c)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
for ax in axes.ravel()[len(other_cols):]:
    ax.axis("off")
fig.suptitle("Číselné / ordinální příznaky podle skupiny studentů (dle součtu G1+G2+G3)", y=1.0)
finish("03_boxploty_podle_skupin")

# Binární příznaky: podíl "1" v každé skupině
share = X[binary_cols].groupby(group, observed=True).mean().T[GROUP_LABELS]
plt.figure(figsize=(7, 0.35 * len(binary_cols) + 2))
sns.heatmap(share, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Podíl hodnoty 1 u binárních příznaků podle skupiny")
finish("04_binarni_podily_podle_skupin")

# Standardizované průměry všech příznaků podle skupin
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
means_z = Xs.groupby(group, observed=True).mean().T[GROUP_LABELS]
means_z = means_z.loc[means_z["vysoké známky"].sub(means_z["nízké známky"]).sort_values().index]
plt.figure(figsize=(7, 0.3 * len(X.columns) + 2))
sns.heatmap(means_z, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Průměr příznaku (z-skóre) podle skupiny\nseřazeno podle rozdílu vysoké - nízké známky")
finish("05_z_prumery_podle_skupin")

# ------------------------------ 4) KORELACE ------------------------------
corr = X.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.6})
plt.title("Korelační matice příznaků (bez známek)")
finish("06_korelacni_matice")

corr_total = X.corrwith(total).sort_values()
plt.figure(figsize=(8, 10))
corr_total.plot(kind="barh", color=["tomato" if v < 0 else "steelblue" for v in corr_total])
plt.title("Korelace příznaků se součtem G1+G2+G3")
finish("07_korelace_se_souctem")

pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
         .stack().rename("corr").reset_index())
pairs["abs"] = pairs["corr"].abs()
print("Nejsilnější korelace mezi příznaky:")
print(pairs.sort_values("abs", ascending=False).head(15)[["level_0", "level_1", "corr"]]
      .round(2).to_string(index=False), "\n")
print("Příznaky nejvíc korelované se součtem známek:")
print(corr_total.reindex(corr_total.abs().sort_values(ascending=False).index).head(10).round(3), "\n")

# ------------------------------ 5) PCA ------------------------------
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
pcs = pca.transform(Z)
evr = pca.explained_variance_ratio_

print("PCA: vysvětlený rozptyl prvních 5 komponent:", np.round(evr[:5], 3))
print("Komponent pro 80 % rozptylu:", int(np.searchsorted(np.cumsum(evr), 0.80) + 1))
for i in range(3):
    print(f"Korelace PC{i+1} se součtem známek: {np.corrcoef(pcs[:, i], total)[0, 1]:.3f}")
print()

# scree plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(evr) + 1), evr, color="steelblue", label="jednotlivě")
ax.plot(range(1, len(evr) + 1), np.cumsum(evr), "o-", color="darkorange", label="kumulativně")
ax.axhline(0.8, ls="--", color="gray")
ax.set_xlabel("Komponenta")
ax.set_ylabel("Podíl vysvětleného rozptylu")
ax.set_title("PCA – scree plot")
ax.legend()
finish("08_pca_scree")

# PC1 vs PC2: spojité obarvení + skupiny
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sc = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=total, cmap="viridis", s=25, alpha=0.8)
fig.colorbar(sc, ax=axes[0], label="G1+G2+G3")
axes[0].set_title("PCA – obarveno součtem známek")
for g in GROUP_LABELS:
    m = (group == g).values
    axes[1].scatter(pcs[m, 0], pcs[m, 1], s=25, alpha=0.7, label=g, color=GROUP_COLORS[g])
axes[1].legend()
axes[1].set_title("PCA – obarveno skupinou")
for ax in axes:
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f} %)")
finish("09_pca_scatter")

# PC1 vs PC3 (někdy je vidět víc)
plt.figure(figsize=(7, 6))
sc = plt.scatter(pcs[:, 0], pcs[:, 2], c=total, cmap="viridis", s=25, alpha=0.8)
plt.colorbar(sc, label="G1+G2+G3")
plt.xlabel(f"PC1 ({evr[0]*100:.1f} %)")
plt.ylabel(f"PC3 ({evr[2]*100:.1f} %)")
plt.title("PCA – PC1 vs PC3")
finish("10_pca_pc1_pc3")

# loadings (co tvoří první komponenty)
loadings = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"])
fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=False)
for ax, pc in zip(axes, loadings.columns):
    s = loadings[pc].sort_values()
    s = pd.concat([s.head(8), s.tail(8)])
    s.plot(kind="barh", ax=ax, color=["tomato" if v < 0 else "steelblue" for v in s])
    ax.set_title(f"Nejsilnější váhy – {pc}")
finish("11_pca_loadings")

# biplot (šipky nejdůležitějších příznaků)
top = loadings[["PC1", "PC2"]].abs().sum(axis=1).sort_values(ascending=False).head(10).index
plt.figure(figsize=(9, 7))
plt.scatter(pcs[:, 0], pcs[:, 1], c=total, cmap="viridis", s=15, alpha=0.5)
scale = np.abs(pcs[:, :2]).max() * 0.8
for f in top:
    plt.arrow(0, 0, loadings.loc[f, "PC1"] * scale, loadings.loc[f, "PC2"] * scale,
              color="black", alpha=0.7, head_width=0.08)
    plt.text(loadings.loc[f, "PC1"] * scale * 1.08, loadings.loc[f, "PC2"] * scale * 1.08,
             f, fontsize=9)
plt.colorbar(label="G1+G2+G3")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA biplot (10 nejvlivnějších příznaků)")
finish("12_pca_biplot")

# ------------------------------ 6) KERNEL PCA ------------------------------
gamma = 1.0 / X.shape[1]
kernels = {
    "RBF": dict(kernel="rbf", gamma=gamma),
    "Polynomiální (stupeň 3)": dict(kernel="poly", degree=3, gamma=gamma),
    "Kosinový": dict(kernel="cosine"),
    "Sigmoid": dict(kernel="sigmoid", gamma=gamma),
}
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (name, params) in zip(axes.ravel(), kernels.items()):
    kp = KernelPCA(n_components=2, **params)
    emb = kp.fit_transform(Z)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=total, cmap="viridis", s=20, alpha=0.8)
    ax.set_title(f"Kernel PCA – {name}")
    ax.set_xlabel("KPC1")
    ax.set_ylabel("KPC2")
    print(f"Kernel PCA {name}: korelace KPC1 se součtem {np.corrcoef(emb[:, 0], total)[0, 1]:.3f}, "
          f"KPC2 {np.corrcoef(emb[:, 1], total)[0, 1]:.3f}")
fig.colorbar(sc, ax=axes, label="G1+G2+G3", shrink=0.6)
finish("13_kernel_pca")

# RBF s různým gamma (jak moc "lokální" struktura se projeví)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, mult in zip(axes, [0.25, 1, 4]):
    emb = KernelPCA(n_components=2, kernel="rbf", gamma=gamma * mult).fit_transform(Z)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=total, cmap="viridis", s=20, alpha=0.8)
    ax.set_title(f"RBF Kernel PCA, gamma = {gamma * mult:.4f}")
fig.colorbar(sc, ax=axes, label="G1+G2+G3", shrink=0.8)
finish("14_kernel_pca_rbf_gamma")

print(f"\nHotovo, grafy jsou ve složce '{OUT_DIR}'.")