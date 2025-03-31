from torch_geometric.nn import GENConv, GINEConv, GATv2Conv, global_add_pool, global_mean_pool, global_max_pool, LayerNorm
from torch_geometric.nn.models import  MLP
from torch_geometric.nn.aggr import MLPAggregation, AttentionalAggregation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.utils import sort_edge_index


# class GINEModel(torch.nn.Module):
#     def __init__(self, node_in_dim, hidden_dim, output_dim, fc_hidden_dim,
#                  num_edge_features, readout, dropout_rate, training,
#                  max_num_elements_mlp=0, hidden_channels_mlp=0, num_layers_mlp=0):
#         from torch_geometric.nn.aggr import MLPAggregation

class GINEModel(torch.nn.Module):
    def __init__(self, node_in_dim, hidden_dim, output_dim, fc_hidden_dim, 
                 num_edge_features, readout, dropout_rate, training,
                 max_num_elements_mlp=0, hidden_channels_mlp=0, num_layers_mlp=0):
        super(GINEModel, self).__init__()
        self.readout = readout
        self.dropout_rate = dropout_rate
        self.fc_hidden_dim = fc_hidden_dim
        self.training = training
        self.readout_funcs = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }
        
        # Initialize MLPAggregation if readout is 'mlp'
        if self.readout == 'mlp':
            self.mlp_aggr = MLPAggregation(
                in_channels=hidden_dim,
                hidden_channels=hidden_channels_mlp,
                max_num_elements=max_num_elements_mlp,
                out_channels=hidden_dim,
                num_layers=num_layers_mlp,
            )
            self.readout_funcs['mlp'] = self.mlp_aggr
        
        self.conv1 = GINEConv(
            Sequential(Linear(node_in_dim, hidden_dim),
                       BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()),
            edge_dim=num_edge_features
        )
        self.conv2 = GINEConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()),
            edge_dim=num_edge_features
        )
        self.conv3 = GINEConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()),
            edge_dim=num_edge_features
        )
        self.lin1 = Linear(hidden_dim*3, fc_hidden_dim*3)
        self.lin2 = Linear(fc_hidden_dim*3, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.float()
        # print(edge_index)
        # print(edge_attr)
        # print(x)
        
        # Node embeddings 
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

        # print(h1)
        # print(h2)
        # print(h3)
        # Graph-level readout
        h1 = self.readout_funcs[self.readout](h1, batch)
        h2 = self.readout_funcs[self.readout](h2, batch)
        h3 = self.readout_funcs[self.readout](h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.lin2(h)
        
        return h.sigmoid()


class GENModel(torch.nn.Module):
    def __init__(self, node_in_dim, hidden_dim, output_dim, hidden_layers, fc_hidden_dim, 
                 num_edge_features, aggregation_function, readout, dropout_rate, training, 
                 max_num_elements_mlp=0, hidden_channels_mlp=0, num_layers_mlp=0, hidden_channels_mlp_readout=0, num_layers_mlp_readout=0):
        super(GENModel, self).__init__()
        # TOMOD attentional
        self.edge_transform = nn.Linear(num_edge_features, hidden_dim)
        #
        self.aggregation_function = aggregation_function
        self.readout = readout
        self.hidden_layers = hidden_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout_rate = dropout_rate
        self.training = training
        self.convs = nn.ModuleList()
        self.readout_funcs = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool
        }

        if self.readout == 'mlp':
            self.readout_funcs['mlp'] = MLPAggregation(
                in_channels=hidden_dim,
                hidden_channels=hidden_channels_mlp_readout,
                max_num_elements=max_num_elements_mlp,
                out_channels=hidden_dim,
                num_layers=num_layers_mlp_readout,
            )
        
        kwargs = {}
        if aggregation_function == "softmax":
            kwargs['learn_t'] = True
        elif aggregation_function == "powermean":
            kwargs['learn_p'] = True

        if aggregation_function == "mlp":
            #first layer
            self.convs.append(GENConv(
                        node_in_dim, 
                        hidden_dim, 
                        aggr=MLPAggregation(
                            in_channels=hidden_dim, 
                            out_channels=hidden_dim, 
                            max_num_elements=max_num_elements_mlp,
                            num_layers=num_layers_mlp, 
                            hidden_channels=hidden_channels_mlp,
                            #norm=LayerNorm(hidden_dim)
                            ), 
                        edge_dim=num_edge_features, 
                        **kwargs))
        elif aggregation_function == "attentional":
            #first layer
            gate_nn = MLP(in_channels=hidden_dim,
                          out_channels=hidden_dim, 
                          num_layers=num_layers_mlp, 
                          hidden_channels=hidden_channels_mlp
                          )
            # attention_nn = MLP(in_channels=hidden_dim,
            #               out_channels=hidden_dim, 
            #               num_layers=num_layers_mlp, 
            #               hidden_channels=hidden_channels_mlp
            #               )
            # Attentional aggregation using the MLP gate_nn
            aggregation = AttentionalAggregation(gate_nn=gate_nn, nn=None)
            self.convs.append(GENConv(node_in_dim, hidden_dim, aggr=aggregation, edge_dim=num_edge_features, **kwargs))
        else:
            #first layer
            self.convs.append(GENConv(node_in_dim, hidden_dim, aggr=aggregation_function, edge_dim=num_edge_features, **kwargs))
        #following layers
        for i in range(1, hidden_layers):
            next_hidden_dim = hidden_dim // 2
            if aggregation_function == "mlp":
                self.convs.append(GENConv(
                            hidden_dim, 
                            next_hidden_dim, 
                            aggr=MLPAggregation(
                                in_channels=next_hidden_dim, 
                                out_channels=next_hidden_dim, 
                                max_num_elements=max_num_elements_mlp, 
                                num_layers=num_layers_mlp, 
                                hidden_channels=hidden_channels_mlp,
                                #norm=LayerNorm(hidden_dim)
                                ), 
                            edge_dim=num_edge_features, 
                            **kwargs))
            elif aggregation_function == "attentional":
                gate_nn = MLP(in_channels=next_hidden_dim,
                          out_channels=next_hidden_dim, 
                          num_layers=num_layers_mlp, 
                          hidden_channels=hidden_channels_mlp
                          )
                # attention_nn = MLP(in_channels=next_hidden_dim,
                #           out_channels=next_hidden_dim, 
                #           num_layers=num_layers_mlp, 
                #           hidden_channels=hidden_channels_mlp
                #           )
                # Attentional aggregation using the MLP gate_nn
                aggregation = AttentionalAggregation(gate_nn=gate_nn, nn=None)
                self.convs.append(GENConv(hidden_dim, next_hidden_dim, aggr=aggregation, edge_dim=num_edge_features, **kwargs))
            else:
                self.convs.append(GENConv(hidden_dim, next_hidden_dim, aggr=aggregation_function, edge_dim=num_edge_features, **kwargs))

            hidden_dim = next_hidden_dim
        
        #fully connected layer
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Sort edge_index and corresponding edge_attr ONCE at the beginning
        sorted_edge_index, sorted_edge_attr = sort_edge_index(
            edge_index, 
            edge_attr,  # This automatically sorts edge_attr to match
            sort_by_row=False
        )
        
        x = x.float()
        sorted_edge_attr = sorted_edge_attr.float()
        
        # print(f"x in arc:{x.shape}")
        # print(f"sorted_edge_index in arc:{sorted_edge_index.shape}")
        # print(f"sorted_edge_attr in arc:{sorted_edge_attr.shape}")


        # Use sorted edges for all subsequent layers
        for conv in self.convs:
            x = conv(x, sorted_edge_index, sorted_edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Readout and FC layers remain unchanged
        x = self.readout_funcs[self.readout](x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return x.sigmoid()


class GATModel(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, node_in_dim, hidden_dim, output_dim, fc_hidden_dim, 
                 num_edge_features, dropout_rate, readout, hidden_layers, 
                 activation_function, heads, training,
                 max_num_elements_mlp=0, hidden_channels_mlp=0, num_layers_mlp=0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.readout = readout
        self.hidden_layers = hidden_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.convs = nn.ModuleList()
        self.activation_function = activation_function
        self.heads = heads
        self.training = training
        self.readout_funcs = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool
        }
        self.activation_funcs = {
            "relu": torch.nn.ReLU(),
            "elu": torch.nn.ELU()
        }

        if self.readout == "mlp":
            self.readout_funcs["mlp"] = MLPAggregation(
                in_channels=hidden_dim,
                hidden_channels=hidden_channels_mlp,
                max_num_elements=max_num_elements_mlp,
                out_channels=hidden_dim,
                num_layers=num_layers_mlp,
            )
            
        # embedding
        self.node_encoder = torch.nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(num_edge_features, hidden_dim)
        # first layer
        self.convs.append(GATv2Conv(node_in_dim, hidden_dim, edge_dim=num_edge_features, heads=heads))
        # following layers
        dim_memory = 0
        for i in range(1, hidden_layers):
            next_hidden_dim = hidden_dim // 2
            self.convs.append(GATv2Conv(hidden_dim*heads, next_hidden_dim, edge_dim=num_edge_features, heads=heads))
            hidden_dim = next_hidden_dim
            dim_memory = next_hidden_dim

        # fully connected layer
        self.fc1 = nn.Linear(hidden_dim*heads, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()
        # embeddings
        #x = self.node_encoder(x)
        #edge_attr = self.edge_encoder(edge_attr)
        # conv layers
        #print(f"edge ind{edge_index}")
        #print(f"edge attr{edge_attr}")
        #print(f"x1{x.shape}")
        for conv in self.convs:
            #print(f"x2{x.shape}")
            x = conv(x, edge_index, edge_attr)
            x = self.activation_funcs[self.activation_function](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Readout: Aggregating node features into a single graph representation
        #print(f"x3{x.shape}")
        x = self.readout_funcs[self.readout](x, batch)
        #print(f"x4{x.shape}")
        # fully connected layer
        x = self.fc1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate ,training=self.training)
        x = self.fc2(x)
        return x.sigmoid()
