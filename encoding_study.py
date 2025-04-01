from paper_code.parsing import get_datasets
from paper_code.encoding import (
    Object2ObjectGraph,
    Object2ObjectMultiGraph,
    Object2AtomGraph,
    Atom2AtomGraph,
    ObjectPair2ObjectPairGraph,
    Atom2AtomHigherOrderGraph,
    Atom2AtomMultiGraph,
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
import pandas as pd
import numpy as np
encoding_results = {}


for root, dirs, files in os.walk(domain_folder):
    if ".ipynb_checkpoints" in dirs:
        shutil.rmtree(os.path.join(root, ".ipynb_checkpoints"))
        print(f"Removed: {os.path.join(root, '.ipynb_checkpoints')}")

    # Sfrutto la funzione get_datasets() del codice del paper per rappresentare il piano tramite un grafo
    # Il grafo è indipendente dalla tipologia di encoding.
    instances = get_datasets(domain_folder, descending=False)
    print(f"Number of instances: {len(instances)}")
    for encoding in encodings_list:
        print(
            f"==============================Starting {encoding} encoding ========================="
        )
        # PREPROCESSING
        data_list = []
        for i, instance in enumerate(instances):  # An instance is a PlanningDataset obj
            #print(f"Processing instance: {i}")
            samples = instance.get_samples(
                eval(encoding)
            )  # samples are the states of a PlanningDataset
            labels = extract_labels_from_samples(samples)
            norm_labels = normalize_between_zero_and_one(labels)
            # covert samples to PyG data format
            tensor_dataset = get_tensor_dataset(samples)
            # if there are no edge_attr add zeros edge_attributes in order to not have errors during training.
            # !!! N.B. !!! valutare di rimuovere questi pochi stati senza edge_Attr
            num_edge_features = check_edge_features_dim(tensor_dataset)

            # add zeros edge attributes if there are no edges
            tensor_dataset = add_zeros_edge_attributes(
                tensor_dataset, num_edge_features
            )

            # TODO leakeage of test data in train (min-max labels)??
            # add_custom label and min/max value of the label in order to be able to transform prediction to original label
            min_label = np.min(labels)
            max_label = np.max(labels)
            for i, data in enumerate(tensor_dataset):
                custom_label = norm_labels[i]
                data.custom_label = custom_label  # normalized label
                data.min_label = min_label  # Min original label (in order to be able to retrieve original label from normalized label)
                data.max_label = max_label  # Max original label (in order to be able to retrieve original label from normalized label)
            data_list.extend(tensor_dataset)
            

            rows = []
            for i, sample in enumerate(samples):
                row = {
                    "encoding": encoding,
                    "sample_index": i,
                    "label": labels[i],
                    "norm_label": norm_labels[i],
                    "data": tensor_dataset[i],
                    "x_size": list(tensor_dataset[i].x.size())
                    if tensor_dataset[i].x is not None
                    else None,
                    "edge_attr_size": list(tensor_dataset[i].edge_attr.size())
                    if tensor_dataset[i].edge_attr is not None
                    else None,
                    "edge_index_size": list(tensor_dataset[i].edge_index.size())
                    if tensor_dataset[i].edge_index is not None
                    else None,
                }
                rows.append(row)

                    # Create a DataFrame from the collected rows
            encoding_df = pd.DataFrame(rows)

            # Store the DataFrame in encoding_results for later use
            if "encoding_results" not in globals():
                encoding_results = {}
            encoding_results[encoding] = encoding_df
        
        
        print(f"--- Stats for encoding: {encoding} ---")
        print(f"Number of samples: {encoding_results[encoding].shape[0]}")
        print("\nDataFrame head:")
        print(encoding_results[encoding].head())
    
    
import matplotlib.pyplot as plt

# Compute average sizes for each encoding
encodings = list(encoding_results.keys())
avg_node_counts = []
avg_node_feature_dims = []
avg_edge_counts = []
avg_edge_feature_dims = []

for encoding in encodings:
    df = encoding_results[encoding]
    node_counts = []
    node_feature_dims = []
    edge_counts = []
    edge_feature_dims = []
    for x_size, edge_index_size, edge_attr_size in zip(
        df['x_size'], df['edge_index_size'], df['edge_attr_size']
    ):
        if x_size is not None:
            node_counts.append(x_size[0])
            node_feature_dims.append(x_size[1])
        if edge_index_size is not None:
            # edge_index shape is [2, num_edges]
            edge_counts.append(edge_index_size[1])
        if edge_attr_size is not None:
            # edge_attr shape is [num_edges, edge_feature_dim]
            edge_feature_dims.append(edge_attr_size[1])
    avg_node_counts.append(np.mean(node_counts))
    avg_node_feature_dims.append(np.mean(node_feature_dims))
    avg_edge_counts.append(np.mean(edge_counts))
    avg_edge_feature_dims.append(np.mean(edge_feature_dims))

# Create a bar chart comparing the average sizes for each encoding, including both node and edge features.
x_pos = np.arange(len(encodings))
bar_width = 0.20

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - 1.5 * bar_width, avg_node_counts, bar_width, label='Avg Nodes')
bars2 = ax.bar(x_pos - 0.5 * bar_width, avg_node_feature_dims, bar_width, label='Avg Node Feature Dim')
bars3 = ax.bar(x_pos + 0.5 * bar_width, avg_edge_counts, bar_width, label='Avg Edges')
bars4 = ax.bar(x_pos + 1.5 * bar_width, avg_edge_feature_dims, bar_width, label='Avg Edge Feature Dim')

ax.set_xlabel('Encodings')
ax.set_ylabel('Average Size')
ax.set_title('Comparison of Encoding Sizes')
ax.set_xticks(x_pos)
ax.set_xticklabels(encodings, rotation=45)
ax.legend()

# Annotate each bar with its height value.
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.1f}',
                ha='center', va='bottom')

plt.tight_layout()
plt.show()  # Display the graph

from datetime import datetime
now = datetime.today()
date_time = now.strftime("%Y_%m_%d_%H%M%S_")
plt.savefig(result_analysis_folder + date_time + "encoding_study.png")
plt.close(fig)
