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

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dizionari che specificano per ogni tipologia di encoding il relativo objective e il numero di trials di optuna
# (in questo modo posso dare n_trials diversi per tipologia di encoding)
objective_names = {
    "GINEModel": objective_GINEModel,
    "GENModel": objective_GENModel,
    "GATModel": objective_GAT,
}

encoding_results = {}

if __name__ == "__main__":
    """
    - Per prima cosa uso la funzione get_datasets() del paper per rappresentare il piano tramite un grafo indipendente dall'encoding.
    - Successivamente, per ogni tipologia di encoding specificata in encodings list:
        1. PREPROCESSING
            - Raccolta di tutti i piani in un'unica lista
            - Estrazione label e normalizzazione
            -Aggiunta di informazioni ad ogni elemento data in tensor_dataset
        2. CHECK CORRETTEZZA
            - controllo se ci sono Edge features con dimensione diversa
        3. TRAIN TEST SPLIT
            - split randomico
        4. OTTIMIZZAZIONE
            - per ogni tipologia di modello:
                - ottimizzazione degli iperparametri
                - salvataggio dello studio (sovrascritti i precedenti studi del dominio se presente)
        5. TEST
            - creazione di un df dove salvare le predizioni (data, original_label, custom_label)
            - per ogni tipologia di modello:
                - creazione del modello con i best_params
                - training
                - evaluate
                - store predictions
        6. SAVE PREDICTIONS DF
    - carico tutti i risultati delle diverse combinazioni encoding-model in un dizionario per fare i plot
    - bar plot per presentare i risultati delle combinazioni encoding-model nel dominio
    - istogrammi delle predizioni
    - scatter plot delle predizioni
    """
    # creo le cartelle che mi servono
    os.makedirs(result_analysis_folder, exist_ok=True)
    os.makedirs(predictions_folder, exist_ok=True)
    # I need to remove the autosave jupyter file from the folder
    for root, dirs, files in os.walk(domain_folder):
        if ".ipynb_checkpoints" in dirs:
            shutil.rmtree(os.path.join(root, ".ipynb_checkpoints"))
            print(f"Removed: {os.path.join(root, '.ipynb_checkpoints')}")

    # Sfrutto la funzione get_datasets() del codice del paper per rappresentare il piano tramite un grafo
    # Il grafo è indipendente dalla tipologia di encoding.
    instances = get_datasets(domain_folder, descending=False)
    
    for encoding in encodings_list:
        print(
            f"==============================Starting {encoding} encoding ========================="
        )
        # PREPROCESSING
        data_list = []
        for instance in instances:  # An instance is a PlanningDataset obj
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
            
            
            if encoding not in encoding_results:
                encoding_results[encoding] = {}
            
            for i, sample in enumerate(samples):
                if encoding not in encoding_results:
                    encoding_results[encoding] = {}
                if i not in encoding_results[encoding]:
                    encoding_results[encoding][i] = {}
                encoding_results[encoding][i]["string"] = str(sample)
                encoding_results[encoding][i]["label"] = labels[i]
                encoding_results[encoding][i]["norm_label"] = norm_labels[i]
                encoding_results[encoding][i]["data"] = tensor_dataset[i]

            # CHECK PREPROCESSING CORRECTNESS
        features = data_list[0].num_features
        process_correct = True
        for data in data_list:
            data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            if data.num_features != features:
                print(f"Different number of features! --> {data.num_features}")
                process_correct = False
                raise Exception("Edge features with different dimension!")

        # TRAIN TEST SPLIT
        train_list, test_list = train_test_split(
            data_list, random_state=seed, test_size=0.1
        )

        max_degree = -1

        for data in train_list:
            num_nodes = data.x.shape[0]
            edge_index = (
                data.edge_index
            )  # Assuming edge_index is of shape [2, num_edges]

            # Calculate in-degree and out-degree using scatter_add
            in_degree = torch.zeros(num_nodes, dtype=torch.long)
            out_degree = torch.zeros(num_nodes, dtype=torch.long)

            in_degree = in_degree.scatter_add(
                0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long)
            )
            out_degree = out_degree.scatter_add(
                0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long)
            )

            # Find the maximum degree for the current graph
            max_degree_data = max(in_degree.max().item(), out_degree.max().item())

            # Update the overall max degree
            if max_degree_data > max_degree:
                max_degree = max_degree_data

        # print(max_degree)

        for data in train_list:
            data.to(device)
        for data in test_list:
            data.to(device)

        # OPTIMIZE  "GENModel", "GINEModel", "GATModel"
        for model_type in models:
            optuna_model_path = f"./optuna/{encoding}/{domain.upper()}/"
            optuna_images_path = f"./optuna/{encoding}/{domain.upper()}/images/"
            os.makedirs(optuna_model_path, exist_ok=True)
            os.makedirs(optuna_images_path, exist_ok=True)

            optuna_file = f"./optuna/{encoding}/{domain.upper()}/{model_type}_study.pkl"

            # Optimization
            if os.path.exists(optuna_file):
                print(f"Loading existing study for {model_type}...")
                study = joblib.load(optuna_file)
            else:
                print(f"Starting new study for {model_type}...")
                study = optuna.create_study(
                    direction="minimize",
                    pruner=optuna.pruners.SuccessiveHalvingPruner(),
                )

            for cosa in train_list:
                cosa.to(device)

            study.optimize(
                lambda trial: objective_names[model_type](
                    trial, train_list, verbose=0, max_degree=max_degree
                ),
                n_trials=trials[model_type],
                n_jobs=2,
            )

            best_trial = study.best_trial
            print("-------------------")
            print(
                f"best trial has {best_trial.value} average accuracy on validation set (k-fold)"
            )
            # save study
            joblib.dump(study, optuna_file)
            # save optimization images (slice plot)
            optuna.visualization.matplotlib.plot_slice(study)
            plt.savefig(
                f"./optuna/{encoding}/{domain.upper()}/images/{model_type}_slice_plot.png",
                bbox_inches="tight",
            )
            # save optimization images (optimization history)
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(
                f"./optuna/{encoding}/{domain.upper()}/images/{model_type}_optimization_history.png",
                bbox_inches="tight",
            )
            # save optimization images (param importance)
            try:
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.savefig(
                    f"./optuna/{encoding}/{domain.upper()}/images/{model_type}_param_importances.png",
                    bbox_inches="tight",
                )
            except Exception as e:
                print(e)
                
            # TEST
        # creating a df to store predictions
        list_df = []
        for item in test_list:
            list_df.append({"item": item, "Label": item.custom_label.item()})
        df = pd.DataFrame(list_df)

        # test "GENModel", "GINEModel", "GATModel"
        for model_type in models:
            print(f"--> Starting {model_type} test")
            torch.manual_seed(seed)
            # load study and best trial
            optuna_file = f"./optuna/{encoding}/{domain.upper()}/{model_type}_study.pkl"
            study = joblib.load(optuna_file)
            best_trial = study.best_trial
            # get loaders
            train_loader = DataLoader(
                train_list, batch_size=best_trial.params["batch_size"], shuffle=True
            )
            test_loader = DataLoader(
                test_list, batch_size=best_trial.params["batch_size"], shuffle=True
            )

            for batch in train_loader:
                batch = batch.to(device)

            node_in_dim = train_loader.dataset[0].num_node_features
            num_edge_features = train_loader.dataset[0].num_edge_features
            output_dim = 1
            # istanziazione del modello
            if model_type == "GINEModel":
                model = GINEModel(
                    node_in_dim=node_in_dim,
                    hidden_dim=best_trial.params["hidden_dim"],
                    output_dim=output_dim,
                    fc_hidden_dim=best_trial.params["fc_hidden_dim"],
                    num_edge_features=num_edge_features,
                    readout=best_trial.params["readout"],
                    dropout_rate=best_trial.params["dropout_rate"],
                    # TOMOD
                    max_num_elements_mlp=max_degree,
                    hidden_channels_mlp=best_trial.params["hidden_channels_mlp"],
                    num_layers_mlp=best_trial.params["num_layers_mlp"],
                    #
                    training=False,
                ).to(device)
            if model_type == "GENModel":
                model = GENModel(
                    node_in_dim=node_in_dim,
                    hidden_dim=best_trial.params["hidden_dim"],
                    output_dim=output_dim,
                    fc_hidden_dim=best_trial.params["fc_hidden_dim"],
                    hidden_layers=best_trial.params["hidden_layers"],
                    aggregation_function=best_trial.params["aggregation_function"],
                    # TOMOD
                    max_num_elements_mlp=max_degree,
                    hidden_channels_mlp=best_trial.params["hidden_channels_mlp"],
                    num_layers_mlp=best_trial.params["num_layers_mlp"],
                    #
                    # TOMOD
                    # hidden_channels_mlp_readout=best_trial.params["hidden_channels_mlp_readout"],
                    # num_layers_mlp_readout=best_trial.params["num_layers_mlp_readout"],
                    #
                    num_edge_features=num_edge_features,
                    readout=best_trial.params["readout"],
                    dropout_rate=best_trial.params["dropout_rate"],
                    training=False,
                ).to(device)
            if model_type == "GATModel":
                model = GATModel(
                    node_in_dim=node_in_dim,
                    hidden_dim=best_trial.params["hidden_dim"],
                    output_dim=output_dim,
                    fc_hidden_dim=best_trial.params["fc_hidden_dim"],
                    num_edge_features=num_edge_features,
                    dropout_rate=best_trial.params["dropout_rate"],
                    readout=best_trial.params["readout"],
                    hidden_layers=best_trial.params["hidden_layers"],
                    activation_function=best_trial.params["activation_function"],
                    heads=best_trial.params["heads"],
                    training=False,
                    # TOMOD
                    # max_num_elements_mlp=max_degree,
                    # hidden_channels_mlp=best_trial.params["hidden_channels_mlp"],
                    # num_layers_mlp=best_trial.params["num_layers_mlp"],
                    #
                ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_trial.params["lr"])
            criterion = nn.L1Loss().to(device)
            # training del modello
            trained_model = train_model(
                model, train_loader, optimizer, criterion, epochs, verbose=1
            )
            # evaluation
            mae = evaluate_model(trained_model, test_loader)
            # memorizzo le prediction
            predictions = []
            with torch.no_grad():
                for data in test_list:
                    out = trained_model(data)
                    predictions.append(out.item())
            df[model_type] = predictions
            print(
                f" Best trial had {best_trial.value} mean MAE on validation set \n the {model_type} model created with the best values have {mae} MAE on the test set"
            )
            print("")

        # SAVE PREDICTIONS DF
        prediction_df = predictions_folder + f"{encoding}.csv"
        df.to_csv(prediction_df, sep=";")
        
    # PLOT DEI RISULTATI
    from datetime import datetime;
    now = datetime.today()
    date_time = now.strftime(f"%Y_%m_%d;%H:%M:%S_")
    # Load results in a dict from csv
    # dict_example --> {"encoding": {"model1":media_MAE,"model2":media_MAE, ...}, ...}
    results = {}
    for encoding in encodings_list:
        df = pd.read_csv(predictions_folder + f"{encoding}.csv", sep=";")
        results[encoding] = {}
        for model in models:
            MAE = mean_absolute_error(df["Label"], df[model])
            MAE = np.round(MAE, 3)
            results[encoding][model] = MAE

    # Code to plot different results achieved with differents encodings and models (grouped bar, with group=encoding)
    labels = encodings_list
    predictions_GEN = []
    predictions_GINE = []
    # populate predictions list
    for encoding in labels:
        predictions_GEN.append(results[encoding]["GENModel"])
        predictions_GINE.append(results[encoding]["GINEModel"])
    # plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(labels))
    width = 0.25
    bar2 = ax.bar(x, predictions_GEN, width, label="GEN")
    bar3 = ax.bar(x + width, predictions_GINE, width, label="GINE")
    ax.set_xlabel("Encoding Type")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Mean Absolute Error of different models with different Encodings")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--")
    ax.legend()
    fig.suptitle(f"{domain}", fontsize="20")

    # Function to add the numerical values on top of the bars
    def autolabel(bars):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Apply autolabel to each set of bars
    autolabel(bar2)
    autolabel(bar3)
    # save
    plt.savefig(result_analysis_folder + date_time +"_encodings_and_models_comparison.png")

    # Istogrammi relativi alle predizioni (normalizzate) di ogni coppia encoding-modello
    # Molto utile per capire se alcune coppie non funzionano come previsto
    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    row = 0
    for encoding in encodings_list:
        df = pd.read_csv(predictions_folder + f"{encoding}.csv", sep=";")
        for column, model in enumerate(models):
            ax[row, column].hist(df[model], bins=100, range=(0, 1.001))
            ax[row, column].set_title(f"{encoding} - {model}")
        row += 1
    fig.suptitle(
        "Histograms of Model Predictions with different encodings", fontsize="20"
    )
    plt.savefig(result_analysis_folder + date_time + "_predictions_histograms.png")

    # Scatter plot delle predizioni rispetto alle label (normalizzate).
    # utile per capire se il modello è buono (in questo caso avremo i punti distribuiti sulla retta y=x)
    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    row = 0
    for encoding in encodings_list:
        df = pd.read_csv(predictions_folder + f"{encoding}.csv", sep=";")
        for column, model in enumerate(models):
            ax[row, column].scatter(df["Label"], df[model])
            ax[row, column].plot([0, 1], [0, 1], color="red", linestyle="--")
            ax[row, column].set_title(f"{encoding} - {model}")
            ax[row, column].set_xlabel("True values")
            ax[row, column].set_ylabel("Predicted values")
        row += 1
    fig.suptitle("Scatter plot of Model Predictions w.r.t Labels", fontsize="20")
    plt.savefig(result_analysis_folder + date_time +"_predictions_scatter.png")





# Create a dataframe to store encoding and dimensions for x, edge_index, and edge_attr
encoding_dimensions = []

for encoding in encoding_results:
    for i in encoding_results[encoding]:
        data = encoding_results[encoding][i]["data"]
        # For nodes, get the number of nodes (first dimension of x)
        num_nodes = data.x.size(0) if data.x is not None else 0
        # For edges, take the number of edges from edge_index (should be on dimension 1)
        num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
        # For edge features, get the size of the second dimension of edge_attr if available
        edge_features = (
            data.edge_attr.size(1)
            if data.edge_attr is not None and len(data.edge_attr.size()) > 1
            else 0
        )
        encoding_dimensions.append({
            "encoding": encoding,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "edge_features": edge_features,
        })

# Convert the list to a pandas DataFrame
df_enc = pd.DataFrame(encoding_dimensions)
print(df_enc)

# Group by encoding and compute the average values
df_avg = df_enc.groupby("encoding").mean().reset_index()

# Create a grouped bar plot to compare the average numbers
labels = df_avg["encoding"]
x = range(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([p - width for p in x], df_avg["num_nodes"], width, label="Avg # Nodes")
ax.bar(x, df_avg["num_edges"], width, label="Avg # Edges")
ax.bar([p + width for p in x], df_avg["edge_features"], width, label="Avg Edge Features")

ax.set_xlabel("Encoding")
ax.set_ylabel("Average Count")
ax.set_title("Comparison of Graph Structural Sizes Across Encodings")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig(result_analysis_folder + date_time + "_encoding sizes.png")
