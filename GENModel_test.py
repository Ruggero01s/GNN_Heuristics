# encoding
from paper_code.parsing import get_datasets
from paper_code.encoding import (
    Object2ObjectGraph,
    Object2AtomGraph,
    Atom2AtomGraph,
    ObjectPair2ObjectPairGraph,
)
from paper_code.modelsTorch import get_compatible_model, get_tensor_dataset

# PyG
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import (
    GENConv,
    GINEConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import aggr
from torch_geometric.nn.models import MLP
from torch_geometric.data import Data
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

# other
import os
import shutil
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import optuna
from utils import (
    extract_labels_from_samples,
    normalize_between_zero_and_one,
    check_edge_features_dim,
    add_zeros_edge_attributes,
    train_model,
    evaluate_model,
)
from optuna_objectives import objective_GINEModel, objective_GENModel, objective_GAT
from models_architectures import GATModel, GENModel, GINEModel
from params import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
base_path = "studies_storage/GEN/"
encodings_list = ["o2o", "o2a", "a2a"]
domain = "sokoban"
domain_folder = f"data/{domain}/"
seed = 42
predictions_folder = f"GEN_test_predictions/{domain.upper()}/"
os.makedirs(predictions_folder, exist_ok=True)

for dir in os.listdir(base_path):
    os.makedirs(os.path.join(predictions_folder, dir), exist_ok=True)

only_graphs=False    
if (not only_graphs):
# Dataset loading
    instances = get_datasets(domain_folder, descending=False)

    for encoding in encodings_list:
        print(f"Processing encoding: {encoding}")
        
        if encoding == "o2o":
            encoding_name = "Object2ObjectGraph"
        elif encoding == "o2a":
            encoding_name = "Object2AtomGraph"
        elif encoding == "a2a":
            encoding_name = "Atom2AtomGraph"


        # Preprocessing
        data_list = []
        for instance in instances:
            samples = instance.get_samples(eval(encoding_name))
            labels = extract_labels_from_samples(samples)
            norm_labels = normalize_between_zero_and_one(labels)
            tensor_dataset = get_tensor_dataset(samples)

            # Add zeros edge attributes if there are no edges
            num_edge_features = check_edge_features_dim(tensor_dataset)
            tensor_dataset = add_zeros_edge_attributes(tensor_dataset, num_edge_features)

            min_label = np.min(labels)
            max_label = np.max(labels)
            for i, data in enumerate(tensor_dataset):
                data.custom_label = norm_labels[i]
                data.min_label = min_label
                data.max_label = max_label

            data_list.extend(tensor_dataset)

        
        
        # Split dataset into train and test sets
        train_list, test_list = train_test_split(data_list, random_state=seed, test_size=0.1)

        
        max_degree = -1

        for data in train_list:
            num_nodes = data.x.shape[0]
            edge_index = data.edge_index  # Assuming edge_index is of shape [2, num_edges]

            # Calculate in-degree and out-degree using scatter_add
            in_degree = torch.zeros(num_nodes, dtype=torch.long)
            out_degree = torch.zeros(num_nodes, dtype=torch.long)


            in_degree = in_degree.scatter_add(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
            out_degree = out_degree.scatter_add(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))

            # Find the maximum degree for the current graph
            max_degree_data = max(in_degree.max().item(), out_degree.max().item())

            # Update the overall max degree
            if max_degree_data > max_degree:
                max_degree = max_degree_data
        
        
        for data in train_list:
            data.to(device)
        for data in test_list:
            data.to(device)
        
        # Load best parameters for GENModel
        for dir in os.listdir(base_path):
            study_path = os.path.join(base_path+f"\{dir}", f"GENModel_study_{encoding}.pkl")
            try:
                study = joblib.load(study_path)
                best_trial = study.best_trial
            except FileNotFoundError:
                print(f"No study file found for {dir}, encoding: {encoding}")
                continue

            # Initialize GENModel with the best parameters
            train_loader = DataLoader(
                train_list, batch_size=best_trial.params["batch_size"], shuffle=True
            )
            test_loader = DataLoader(
                test_list, batch_size=best_trial.params["batch_size"], shuffle=True
            )

            node_in_dim = train_loader.dataset[0].num_node_features
            num_edge_features = train_loader.dataset[0].num_edge_features
            output_dim = 1

            model = GENModel(
                node_in_dim=node_in_dim,
                hidden_dim=best_trial.params["hidden_dim"],
                output_dim=output_dim,
                fc_hidden_dim=best_trial.params["fc_hidden_dim"],
                hidden_layers=best_trial.params["hidden_layers"],
                aggregation_function=dir,
                max_num_elements_mlp=max_degree,  # Adjust based on dataset
                hidden_channels_mlp=best_trial.params.get("hidden_channels_mlp", 0),
                num_layers_mlp=best_trial.params.get("num_layers_mlp", 0),
                num_edge_features=num_edge_features,
                readout=best_trial.params["readout"],
                dropout_rate=best_trial.params["dropout_rate"],
                training=False,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=best_trial.params["lr"])
            criterion = torch.nn.L1Loss().to(device)

            # Train and evaluate
            trained_model = train_model(model, train_loader, optimizer, criterion, epochs=50, verbose=0)
            mae = evaluate_model(trained_model, test_loader)

            # Save predictions
            predictions = []
            with torch.no_grad():
                for data in test_list:
                    out = trained_model(data)
                    predictions.append(out.item())

            df = pd.DataFrame(
                {
                    "Item": [i for i in range(len(test_list))],
                    "True_Label": [data.custom_label.item() for data in test_list],
                    "Predicted_Label": predictions,
                }
            )
            df.to_csv(f"{predictions_folder}/{dir}/{encoding}_predictions.csv", sep=";", index=False)
            print(f"Aggregation: {dir} || Encoding {encoding}: MAE = {mae}")


results = {}

# Load results for GENModel
for dir in os.listdir(base_path):
    for encoding in encodings_list:
        df = pd.read_csv(f"{predictions_folder}/{dir}/{encoding}_predictions.csv", sep=";")
        mae = mean_absolute_error(df["True_Label"], df["Predicted_Label"])
        results[encoding] = round(mae, 3)

    # Bar Plot for MAE with GENModel
    labels = encodings_list
    mae_values = [results[encoding] for encoding in labels]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(labels))
    width = 0.4
    bars = ax.bar(x, mae_values, width, label="GENModel", color="blue")

    ax.set_xlabel("Encoding Type")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title(f"{dir}: Mean Absolute Error for GENModel with Different Encodings")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--")
    ax.legend()

    # Add numerical values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Save bar plot
    result_analysis_folder = f"results_analysis/{domain.upper()}/GEN/{dir}"
    os.makedirs(result_analysis_folder, exist_ok=True)
    plt.savefig(result_analysis_folder + "/genmodel_mae_comparison.png")
    plt.close()

    # Histograms for Predictions
    print(f"{predictions_folder}/{dir}/{encoding}_predictions.csv")
    fig, ax = plt.subplots(len(encodings_list), 1, figsize=(12, len(encodings_list) * 4))
    for i, encoding in enumerate(encodings_list):
        df = pd.read_csv(f"{predictions_folder}/{dir}/{encoding}_predictions.csv", sep=";")
        ax[i].hist(df["Predicted_Label"], bins=50, range=(0, 1.001), color="blue", alpha=0.7)
        ax[i].set_title(f"{dir}: Histogram of Predictions for {encoding}")
        ax[i].set_xlabel("Normalized Prediction Values")
        ax[i].set_ylabel("Frequency")
    fig.tight_layout()
    plt.savefig(result_analysis_folder + "/genmodel_predictions_histograms.png")
    plt.close()

    # Scatter Plots of Predictions vs. True Labels
    fig, ax = plt.subplots(len(encodings_list), 1, figsize=(12, len(encodings_list) * 4))
    for i, encoding in enumerate(encodings_list):
        df = pd.read_csv(f"{predictions_folder}/{dir}/{encoding}_predictions.csv", sep=";")
        ax[i].scatter(df["True_Label"], df["Predicted_Label"], alpha=0.6, color="blue")
        ax[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
        ax[i].set_title(f"{dir}: Scatter Plot of Predictions vs. True Labels for {encoding}")
        ax[i].set_xlabel("True Labels")
        ax[i].set_ylabel("Predicted Labels")
    fig.tight_layout()
    plt.savefig(result_analysis_folder + "/genmodel_predictions_scatter.png")
    plt.close()