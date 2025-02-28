import os
import sys
import time
import torch
import joblib
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from pyperplan import grounding, search
from pyperplan.pddl.parser import Parser
from pyperplan.heuristics.blind import BlindHeuristic
from pyperplan.heuristics.relaxation import hFFHeuristic
from unified_planning.io import PDDLReader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models_architectures import GENModel
from utils import (
    extract_labels_from_samples,
    normalize_between_zero_and_one,
    check_edge_features_dim,
    add_zeros_edge_attributes,
    train_model,
)
from paper_code.modelsTorch import get_tensor_dataset
from paper_code.parsing import get_dataset, get_datasets, get_dataset_from_str
from paper_code.encoding import Atom2AtomGraph
from params import *
from pddl_to_txt import translate_to_custom_encoding, append_state


class CustomHeuristic:
    def __init__(self, task, optuna_model_path="./optuna/"):
        """
        Initialization of the Heuristic GEN Model.
        :param optuna_model_path: The directory path for Optuna studies.
        """
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.domain = "blocks"
        self.encoding = "Atom2AtomGraph"
        # Load dataset and model
        self.train_list = self._prepare_dataset()
        #self.num_edge_features = check_edge_features_dim(self.train_list) ############
        #self.train_list = add_zeros_edge_attributes(self.train_list, self.num_edge_features) ############
        self.best_trial = self._load_optuna_best_trial(optuna_model_path)
        #print(self.best_trial)
        self.model = self._initialize_gen_model()
        self._train_model()
        
        
        """Generate a temporary file for PDDL encoding."""
        reader = PDDLReader()
        problem_instance = reader.parse_problem('solver/data_pddl/blocks/domain.pddl', 'solver/data_pddl/blocks/p01.pddl')
        self.name_file = os.path.basename('solver/data_pddl/blocks/p01.pddl')
        self.problem_partial , self.obj_map, self.pred_map = translate_to_custom_encoding(problem_instance)

    def _prepare_dataset(self):
        """Load and preprocess the dataset."""
        domain_folder = os.path.join(domain_folder_root, self.domain)
        instances = get_datasets(domain_folder, descending=False)
        data_list = []

        for instance in instances:
            samples = instance.get_samples(eval(self.encoding))
            labels = extract_labels_from_samples(samples)
            norm_labels = normalize_between_zero_and_one(labels)

            tensor_dataset = get_tensor_dataset(samples)
            num_edge_features = check_edge_features_dim(tensor_dataset)
            tensor_dataset = add_zeros_edge_attributes(tensor_dataset, num_edge_features)

            self.min_label, self.max_label = np.min(labels), np.max(labels)
            for i, data in enumerate(tensor_dataset):
                data.custom_label = norm_labels[i]
                data.min_label, data.max_label = self.min_label, self.max_label

            data_list.extend(tensor_dataset)

        self._validate_preprocessing(data_list)
        train_list, _ = train_test_split(data_list, random_state=seed, test_size=0.1)

        return train_list

    def _validate_preprocessing(self, data_list):
        """Ensure data preprocessing consistency."""
        features = data_list[0].num_features
        for data in data_list:
            if data.num_features != features:
                raise ValueError(f"Inconsistent number of features: {data.num_features}")

    def _load_optuna_best_trial(self, optuna_model_path):
        """Load the best trial for the GEN model from Optuna."""
        optuna_file = os.path.join(optuna_model_path, "Atom2AtomGraph/BLOCKS/GENModel_study.pkl")
        return joblib.load(optuna_file).best_trial

    def _initialize_gen_model(self):
        """Initialize the GEN model with the best hyperparameters."""
        best_params = self.best_trial.params
        node_in_dim = self.train_list[0].num_node_features
        num_edge_features = self.train_list[0].num_edge_features

        return GENModel(
            node_in_dim=node_in_dim,
            hidden_dim=best_params["hidden_dim"],
            output_dim=1,
            fc_hidden_dim=best_params["fc_hidden_dim"],
            hidden_layers=best_params["hidden_layers"],
            aggregation_function=best_params["aggregation_function"],
            num_edge_features=num_edge_features,
            readout=best_params["readout"],
            dropout_rate=best_params["dropout_rate"],
            max_num_elements_mlp=self._get_max_degree(),
            hidden_channels_mlp=best_params["hidden_channels_mlp"],
            num_layers_mlp=best_params["num_layers_mlp"],
            training=False
        ).to(self.device)

    def _get_max_degree(self):
        """Calculate the maximum node degree in the dataset."""
        return max(
            max(
                torch.bincount(data.edge_index[0]).max().item(),
                torch.bincount(data.edge_index[1]).max().item()
            ) for data in self.train_list
        )

    def _train_model(self):
        """Train the GEN model using the training dataset."""
        best_params = self.best_trial.params
        optimizer = torch.optim.Adam(self.model.parameters(), lr=best_params["lr"])
        criterion = nn.L1Loss().to(self.device)
        #for data in self.train_list:
        #    print(data.x)
        train_loader = DataLoader(self.train_list, batch_size=best_params["batch_size"], shuffle=True)

        self.trained_model = train_model(self.model, train_loader, optimizer, criterion, epochs=epochs, verbose=1)

    def __call__(self, node):
        """Evaluate a given state using the trained model."""
        # temp_path = os.path.join(os.path.dirname(__file__), "./temp/temp.txt")
        # self._create_temp_file(node, temp_path)

        temp_encoding = self._create_encoding(node)
        
        #samples = get_dataset(temp_path).get_samples(eval(self.encoding))
        samples = get_dataset_from_str(self.name_file, temp_encoding).get_samples(eval(self.encoding))

        tensor_dataset = get_tensor_dataset(samples)

        #tensor_dataset = add_zeros_edge_attributes(tensor_dataset, self.num_edge_features)

        self.model.eval()
        data = tensor_dataset[0].to(self.device)
        # print(data.x.shape)
        # print(data.edge_index.shape)
        # print(data.edge_attr.shape)
        # print(data.y)

        with torch.no_grad():
            value = self.trained_model(data).item()
            scaled_value = value*(self.max_label-self.min_label) + self.min_label 
            # The scaling implies that the inferred heuristic can at maximum reach the max euristic label found in training 
            # print(f"Valore norm:{value}")
            # print(f"Valore scalato:{value}")
            return scaled_value

    def _create_temp_file(self, node, file_path):

        temp_encoding = append_state(node.state, self.obj_map, self.pred_map)
        
        full_encoding = self.problem_partial + "\n" + "\n".join(temp_encoding)
        with open(file_path, "w") as f:
            f.write(full_encoding)
            
    def _create_encoding(self, node):

        temp_encoding = append_state(node.state, self.obj_map, self.pred_map)
        
        full_encoding = self.problem_partial + "\n" + "\n".join(temp_encoding)
        return full_encoding


def solve_problem(domain_file: str, problem_file: str, heuristic=None):
    """Solve a PDDL problem using A* search with a heuristic."""
    parser = Parser(domain_file, problem_file)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    task = grounding.ground(problem)

    heuristic = heuristic(task) if heuristic else BlindHeuristic(task)
    #heuristic = hFFHeuristic(task)
    start_time = time.time()
    solution = search.astar_search(task, heuristic)
    end_time = time.time()
    
    print(end_time-start_time)
    if solution:
        print("Plan found:")
        for action in solution:
            print(action)
    else:
        print("No plan found.")


if __name__ == "__main__":
    
    solve_problem(
        domain_file='solver/data_pddl/blocks/domain.pddl',
        problem_file='solver/data_pddl/blocks/p01.pddl',
        heuristic=CustomHeuristic
    )
