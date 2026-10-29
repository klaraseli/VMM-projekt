import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from clean_data import *
from load_data import *

# pip install matplotlib seaborn
sns.set_theme(style="whitegrid")

data_path = "C://Users//Acer Aspire//Desktop//7. semestr//VMM//projekt//student_performance_Final.csv"
data = load_data(data_path)
data_clean = clean_data(data)

# 1) NaN vzniklé čištěním (stav těsně před KNN imputací)
tmp = replace_nonsense_numeric(data)
tmp = replace_outliers_iqr(tmp)
tmp = encode_categorical(tmp)
nan_counts = tmp.isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values()

plt.figure(figsize=(8, 5))
if len(nan_counts) > 0:
    nan_counts.plot(kind="barh", color="tomato")
    plt.xlabel("Počet NaN před imputací")
    plt.title("Hodnoty označené při čištění jako chybějící")
else:
    plt.text(0.5, 0.5, "Žádné NaN", ha="center")
plt.tight_layout()
plt.show()

# 2) absences před a po čištění
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
sns.boxplot(y=data["absences"], ax=axes[0], color="lightgray")
axes[0].set_title("absences – původní")
sns.boxplot(y=data_clean["absences"], ax=axes[1], color="steelblue")
axes[1].set_title("absences – vyčištěné")
plt.tight_layout()
plt.show()

# 3) histogramy všech příznaků
data_clean.hist(bins=15, figsize=(18, 14), color="steelblue", edgecolor="white")
plt.suptitle("Rozložení příznaků po vyčištění", y=1.00)
plt.tight_layout()
plt.show()

# 4) G1, G2, G3 dohromady
plt.figure(figsize=(8, 4))
for g in ["G1", "G2", "G3"]:
    sns.histplot(data_clean[g], bins=21, alpha=0.4, label=g)
plt.legend()
plt.title("Rozložení známek")
plt.tight_layout()
plt.show()

# 5) korelace s G3 + heatmapa
corr = data_clean.corr(numeric_only=True)
corr_g3 = corr["G3"].drop("G3").sort_values()

plt.figure(figsize=(8, 10))
corr_g3.plot(kind="barh", color=["tomato" if v < 0 else "steelblue" for v in corr_g3])
plt.title("Korelace příznaků s G3")
plt.tight_layout()
plt.show()

core = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
        "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
plt.figure(figsize=(10, 8))
sns.heatmap(data_clean[core].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Korelační matice")
plt.tight_layout()
plt.show()

# 6) G3 podle vybraných příznaků
cols = ["failures", "studytime", "higher", "school", "Medu", "sex"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), cols):
    sns.boxplot(data=data_clean, x=c, y="G3", ax=ax, color="lightsteelblue")
    ax.set_title(f"G3 podle {c}")
plt.tight_layout()
plt.show()

# 7) G2 vs G3
plt.figure(figsize=(6, 5))
sns.regplot(data=data_clean, x="G2", y="G3", x_jitter=0.2, y_jitter=0.2,
            scatter_kws={"alpha": 0.4})
plt.title("G2 vs G3")
plt.tight_layout()
plt.show()