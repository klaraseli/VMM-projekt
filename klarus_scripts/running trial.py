import pandas as pd
from plot_data import eda_plots
from clean_data import *
from load_data import *

data_path = "C://Users//Acer Aspire//Desktop//7. semestr//VMM//projekt//student_performance_Final.csv"
data = load_data(data_path)
df = clean_data(data)
eda_plots(df)                                   # jen číselné příznaky
eda_plots(df, hue="trida")                      # obarvení podle kategorie
eda_plots(df, n_cols=4, save_prefix="vystup")   # 4 sloupce v mřížce + uložení PNG