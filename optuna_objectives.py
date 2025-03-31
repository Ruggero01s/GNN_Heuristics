import optuna
from utils import extract_custom_labels_from_tensors, train_model, evaluate_model, GateAttentional
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.loader import DataLoader
from models_architectures import GINEModel, GENModel, GATModel
import torch
from torch_geometric.nn import aggr
from torch_geometric.nn.models import MLP
from torch_geometric.utils import sort_edge_index
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data
from params import *


def objective_GINEModel(trial, train_dataset, verbose=0, max_degree=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    HIDDEN_DIM = trial.suggest_int("hidden_dim",32,256, step=32)
    FC_HIDDEN_DIM = trial.suggest_int("fc_hidden_dim",32,256, step=32)
    READOUT = trial.suggest_categorical("readout", ['mean', 'max', 'add','mlp'])
    #READOUT = trial.suggest_categorical("readout", ['mlp'])
    LR = trial.suggest_float("lr",0.00001,0.1)
    DROPOUT_RATE = trial.suggest_float("dropout_rate",0.1,0.5)
    BATCH_SIZE = trial.suggest_int("batch_size",32,64, step=8)
    
    #TOMOD MLP/Attentional
    MAX_NUM_ELEMENTS_MLP = max_degree # max degree of node in dataset
    HIDDEN_CHANNELS_MLP = trial.suggest_int("hidden_channels_mlp", 16,64, step=8)
    NUM_LAYERS_MLP = trial.suggest_int("num_layers_mlp", 2,6, step=2)
    #

    
    # Extract labels for stratification
    labels = extract_custom_labels_from_tensors(train_dataset)
    # Set up KFold cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    fold_metrics = []
    # Perform stratified 3-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset, labels)):
        train_dataset_fold = [train_dataset[i] for i in train_idx]
        val_dataset_fold = [train_dataset[i] for i in val_idx]
        # Load the data for this fold
        train_loader = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        node_in_dim = train_loader.dataset[0].num_node_features
        num_edge_features = train_dataset[0].num_edge_features
        output_dim = 1      
        for batch in train_loader:
            batch = batch.to(device)
        # print()
        # print(node_in_dim)
        # print(HIDDEN_DIM)
        # print(output_dim)
        # print(FC_HIDDEN_DIM)
        # print(MAX_NUM_ELEMENTS_MLP)
        # print(HIDDEN_CHANNELS_MLP)
        # print(NUM_LAYERS_MLP)
        # print(num_edge_features)
        # Initialize the model for this fold
        model = GINEModel(node_in_dim=node_in_dim,
                         hidden_dim=HIDDEN_DIM, 
                         output_dim=output_dim,
                         fc_hidden_dim=FC_HIDDEN_DIM,
                         num_edge_features = num_edge_features, 
                         readout=READOUT,
                         dropout_rate=DROPOUT_RATE,
                         training=True,
                         #TOMOD
                         max_num_elements_mlp=MAX_NUM_ELEMENTS_MLP,
                         hidden_channels_mlp=HIDDEN_CHANNELS_MLP,
                         num_layers_mlp=NUM_LAYERS_MLP
                         #
                         ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss().to(device) #MAE
        # Train the model for this fold
        train_model(model, train_loader, optimizer, criterion, epochs, verbose)
        # Evaluate the model on the validation data
        mae = evaluate_model(model, val_loader)
        fold_metrics.append(mae)
        if verbose == 1:
            print(f"Fold {fold+1}, MAE: {rmse:.4f}")
    # Calculate the mean accuracy across all folds
    mean_metric = sum(fold_metrics) / len(fold_metrics)  
    if verbose == 1:
        print(f"Mean MAE over 3 folds: {mean_metric:.4f}") 
    return mean_metric


def objective_GENModel(trial, train_dataset, verbose=0, max_degree = -1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    HIDDEN_DIM = trial.suggest_int("hidden_dim",32,256, step=32)
    FC_HIDDEN_DIM = trial.suggest_int("fc_hidden_dim",32,256, step=32)
    HIDDEN_LAYERS = trial.suggest_int("hidden_layers",1,3)
    READOUT = trial.suggest_categorical("readout", ['mean', 'max', 'add'])
    #TOMOD
    AGGREGATION_FUNCTION = trial.suggest_categorical("aggregation_function", ['mean', 'max', 'sum', 'softmax', 'mlp' ]) #(powermean, mul) genera errori (input contains NaN), forse dovuto a sigmoid 
    #median gives cuda indexing error
    #AGGREGATION_FUNCTION = trial.suggest_categorical("aggregation_function", ['mlp'])
    
    #TOMOD MLP/Attentional
    MAX_NUM_ELEMENTS_MLP = max_degree # max degree of node in dataset
    HIDDEN_CHANNELS_MLP = trial.suggest_int("hidden_channels_mlp", 16,64, step=8)
    NUM_LAYERS_MLP = trial.suggest_int("num_layers_mlp", 2,6, step=2)
    #
    #TOMOD MLP/Attentional
    # MAX_NUM_ELEMENTS_MLP = max_degree # max degree of node in dataset
    # HIDDEN_CHANNELS_MLP_READOUT = trial.suggest_int("hidden_channels_mlp_readout", 16,64, step=8)
    # NUM_LAYERS_MLP_READOUT = trial.suggest_int("num_layers_mlp_readout", 2,6, step=2)
    #
    LR = trial.suggest_float("lr",0.00001,0.1)
    DROPOUT_RATE = trial.suggest_float("dropout_rate",0.1,0.5)
    BATCH_SIZE = trial.suggest_int("batch_size",32,64, step=8)

    # for data in train_dataset:
    #     assert (isinstance(data, Data))
    #     #print(f"edge_index optim1:\n{data.edge_index}")
    #     data.sort(sort_by_row = False)
    #     #data.edge_index = sort_edge_index(data.edge_index)
    #     #print(f"edge_index optim2:\n{data.edge_index}")
    # for i, data in enumerate(train_dataset[:3]):
    #     print(f"Graph {i} sorted edge_index:\n{data.edge_index}")
    # Extract labels for stratification
    labels = extract_custom_labels_from_tensors(train_dataset)
    # Set up KFold cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    fold_metrics = []
    # Perform stratified 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset, labels)):
        train_dataset_fold = [train_dataset[i] for i in train_idx]
        val_dataset_fold = [train_dataset[i] for i in val_idx]
            
        # Load the data for this fold
        train_loader = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        node_in_dim = train_loader.dataset[0].num_node_features
        output_dim = 1
        num_edge_features = train_dataset[0].num_edge_features
        for batch in train_loader:
            batch = batch.to(device)
        
        # print()
        # print(node_in_dim)
        # print(HIDDEN_DIM)
        # print(output_dim)
        # print(FC_HIDDEN_DIM)
        # print(HIDDEN_LAYERS)
        # print(num_edge_features)
        
        #Initialize the model for this fold
        model = GENModel(node_in_dim=node_in_dim,
                         hidden_dim=HIDDEN_DIM, 
                         output_dim=output_dim,
                         fc_hidden_dim=FC_HIDDEN_DIM,
                         hidden_layers=HIDDEN_LAYERS,
                         num_edge_features=num_edge_features,
                         aggregation_function=AGGREGATION_FUNCTION,
                         #TOMOD
                         max_num_elements_mlp=MAX_NUM_ELEMENTS_MLP,
                         hidden_channels_mlp=HIDDEN_CHANNELS_MLP,
                         num_layers_mlp=NUM_LAYERS_MLP,
                         #
                         #TOMOD
                        #  hidden_channels_mlp_readout=HIDDEN_CHANNELS_MLP_READOUT,
                        #  num_layers_mlp_readout=NUM_LAYERS_MLP_READOUT,
                         #
                         readout=READOUT,
                         dropout_rate=DROPOUT_RATE,
                         training=True
                         ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss().to(device) #MAE
        # Train the model for this fold
        train_model(model, train_loader, optimizer, criterion, epochs, verbose)
        # Evaluate the model on the validation data
        mae = evaluate_model(model, val_loader)
        fold_metrics.append(mae)
        if verbose == 1:
            print(f"Fold {fold+1}, MAE: {rmse:.4f}")
    # Calculate the mean accuracy across all folds
    mean_metric = sum(fold_metrics) / len(fold_metrics)  
    if verbose == 1:
        print(f"Mean MAE over 3 folds: {mean_metric:.4f}") 
    return mean_metric

def objective_GAT(trial, train_dataset, verbose=0, max_degree = -1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    HIDDEN_DIM = trial.suggest_int("hidden_dim",32,256, step=32)
    FC_HIDDEN_DIM = trial.suggest_int("fc_hidden_dim",32,256, step=32)
    HIDDEN_LAYERS = trial.suggest_int("hidden_layers",1,3)
    READOUT = trial.suggest_categorical("readout", ['mean', 'max', 'add'])
    #READOUT = trial.suggest_categorical("readout", ['mlp'])

    LR = trial.suggest_float("lr",0.00001,0.1)
    DROPOUT_RATE = trial.suggest_float("dropout_rate",0.1,0.5)
    BATCH_SIZE = trial.suggest_int("batch_size",32,64, step=8)
    ACTIVATION_FUNCTION = trial.suggest_categorical("activation_function",["relu","elu"])
    HEADS = trial.suggest_int("heads",2,6)

    #TOMOD MLP/Attentional
    #MAX_NUM_ELEMENTS_MLP = max_degree # max degree of node in dataset
    #HIDDEN_CHANNELS_MLP = trial.suggest_int("hidden_channels_mlp", 16,64, step=8)
    #NUM_LAYERS_MLP = trial.suggest_int("num_layers_mlp", 2,6, step=2)
    #
    
    # Set up KFold cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    fold_metrics = []
    # Perform stratified 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        # Create train and validation datasets for the current fold
        train_dataset_fold = [train_dataset[i] for i in train_idx]
        val_dataset_fold = [train_dataset[i] for i in val_idx]
        # Load the data for this fold
        train_loader = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        node_in_dim = train_loader.dataset[0].num_node_features
        output_dim = 1
        num_edge_features = train_dataset[0].num_edge_features
        for batch in train_loader:
            batch = batch.to(device)
        # Initialize the model for this fold
        model = GATModel(node_in_dim=node_in_dim,
                         hidden_dim=HIDDEN_DIM, 
                         output_dim=output_dim, 
                         fc_hidden_dim=FC_HIDDEN_DIM,
                         num_edge_features=num_edge_features,
                         dropout_rate=DROPOUT_RATE,
                         readout=READOUT, 
                         hidden_layers=HIDDEN_LAYERS,
                         activation_function=ACTIVATION_FUNCTION,
                         heads=HEADS,
                         training=True,
                         #TOMOD
                         #max_num_elements_mlp=MAX_NUM_ELEMENTS_MLP,
                         #hidden_channels_mlp=HIDDEN_CHANNELS_MLP,
                         #num_layers_mlp=NUM_LAYERS_MLP,
                         #
                         ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss().to(device) #MAE
        # Train the model for this fold
        train_model(model, train_loader, optimizer, criterion, epochs, verbose)
        # Evaluate the model on the validation data
        mae = evaluate_model(model, val_loader)
        fold_metrics.append(mae)
        if verbose == 1:
            print(f"Fold {fold+1}, MAE: {rmse:.4f}")
    # Calculate the mean accuracy across all folds
    mean_metric = sum(fold_metrics) / len(fold_metrics)  
    if verbose == 1:
        print(f"Mean MAE over 3 folds: {mean_metric:.4f}") 
    return mean_metric
