import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import torch_geometric
from params import seed, patience, epochs
from torch_geometric.nn import aggr
from torch_geometric.nn.models import MLP
from torch.nn import Linear, Sequential, ReLU


def normalize_between_zero_and_one(arr):
    '''Normalizza un array per ottenere un array con valori continui tra 0 e 1'''
    # Calcolare il minimo e il massimo del vettore
    min_val = np.min(arr)
    max_val = np.max(arr)
    # Applicare la normalizzazione
    norm_arr = (arr - min_val) / (max_val - min_val)
    return norm_arr.astype(np.float32)


def extract_labels_from_samples(samples):
    labels=[]
    for sample in samples:
        labels.append(sample.state.label)
    return labels


def extract_custom_labels_from_tensors(tensors):
    labels=[]
    for sample in tensors:
        labels.append(sample.custom_label)
    return labels


def train_model(model, train_loader, optimizer, criterion, epochs, verbose):
    '''Code used to train an existing model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model.train()
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            out = out.squeeze(-1).float()  # Removes the last dimension (converts [n, 1] to [n]) to match
            loss = criterion(out, torch.tensor(data.custom_label, dtype=torch.float32).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #Eearly stopping
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter == patience:
            if verbose == 1:
                print(f'Epoch: {epoch:02d}, '
                      f'Stopped for loss early_stopping')
            break
        #average loss per batch
        if verbose == 1:
            print(f'Epoch {epoch+1}, Loss (average per batch): {total_loss/len(train_loader)}')

    return model


def evaluate_model(model, test_loader):
    '''Code used to evaluate with MAE an existing model already trained'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)  # Model outputs the predicted values (regression output)
            out = out.squeeze(-1).float()  # Removes the last dimension (converts [n, 1] to [n]) to match
            custom_label_tensor = torch.tensor(data.custom_label, dtype=torch.float32).to(device)
            y_true.append(custom_label_tensor.cpu().numpy())  # Collect true values
            y_pred.append(out.cpu().numpy())  # Collect predicted values

    # Concatenate the lists into numpy arrays
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)

    return mae


def check_edge_features_dim(tensor_dataset):
    '''Se nel dataset sono presenti gli edge_Attr, controllo che abbiano tutti le stesse dim'''
    num_edge_features = set()
    for data in tensor_dataset:
        if 'edge_attr' in data:
            num_edge_features.add(data.num_edge_features)
    if len(num_edge_features)==1:
        num_edge_features = next(iter(num_edge_features))
        return num_edge_features
    else:
        raise Exception ("Found states with different number of edge features, with this models architectures it is impossible to handle this case.")


def add_zeros_edge_attributes(tensor_dataset, num_edge_features):
    '''
    Se nel dataset sono presenti gli edge_Attr, è possibile che in alcuni casi alcuni sample non abbiano edge_attr.
    In questo caso li aggiungo con valori zero e dimensione coerente con gli altri edge_attr
    '''
    for data in tensor_dataset:
        if 'edge_attr' not in data:
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.zeros((num_edges, num_edge_features))
    return tensor_dataset

class GateAttentional(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use torch_geometric.nn.models.MLP as the gate_nn
        self.gate_nn = MLP(
            in_channels=in_channels,  # Input dimension
            hidden_channels=16,       # Hidden layer dimension
            out_channels=1,           # Output a single attention score per node
            num_layers=2,             # Number of layers in the MLP
            act="relu",               # Activation function
            dropout=0.0,              # Dropout (optional)
            norm=None                 # No normalization
        )
        # Attentional aggregation using the MLP gate_nn
        self.aggregation = aggr.AttentionalAggregation(self.gate_nn, nn=None)
        # Final linear layer to reduce aggregated features to scalar
        self.out_layer = Linear(in_channels, 1)

    def forward(self, x, batch):
        """
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            batch: Batch vector [num_nodes], indicating graph assignment
        """
        # Aggregate features per graph
        aggregated = self.aggregation(x, batch)  # Shape: [num_graphs, in_channels]
        # Reduce to scalar output
        return self.out_layer(aggregated)  # Shape: [num_graphs, 1]
    
    def call_aggr(self, x, batch):
        # Aggregate features per graph
        aggregated = self.aggregation(x, batch)  # Shape: [num_graphs, in_channels]
        # Reduce to scalar output
        return self.out_layer(aggregated)  # Shape: [num_graphs, 1]
        