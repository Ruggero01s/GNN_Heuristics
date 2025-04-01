import pandas as pd
import numpy as np
from datetime import datetime
from params import *

    # PLOT DEI RISULTATI: Scatter plot per ogni encoding

import matplotlib.pyplot as plt

now = datetime.today()
date_time = now.strftime("%Y_%m_%d_%H%M%S_")

# Per ogni encoding, carica il CSV e crea uno scatter plot con tutte le previsioni dei modelli
for encoding in encodings_list:
    df = pd.read_csv(predictions_folder + f"{encoding}.csv", sep=";")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot per ogni modello sullo stesso grafico
    for model in models:
        ax.scatter(df["Label"], df[model], label=model, alpha=0.6)
    
    # Linea di riferimento y=x
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2)
    
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"Scatter plot - {encoding}")
    ax.legend()
    ax.grid(True, linestyle="--")
    
    # Salva il grafico per l'encoding corrente
    plt.savefig(result_analysis_folder + date_time + f"{encoding}_scatter.png")
    plt.close(fig)
    
    # Load the CSV with encoding size information
encoding_sizes = pd.read_csv("results_analysis/SOKOBAN/encoding_sizes_2025_03_31_20-49-49_.csv")

# Group data by encoding and compute aggregate statistics
stats = encoding_sizes.groupby("encoding").agg({
    "num_nodes": ["count", "mean", "median", "std", "min", "max"],
    "num_edges": ["mean", "median", "std", "min", "max"],
    "edge_features": ["mean", "median", "std", "min", "max"]
})

print("Statistics for each encoding:")
print(stats)

# Optionally, print overall descriptive statistics
print("\nOverall descriptive statistics:")
print(encoding_sizes.describe())