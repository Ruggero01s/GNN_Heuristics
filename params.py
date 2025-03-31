seed = 42
epochs = 70
patience = 5  # Patience for early stopping
encodings_list = [
     "Object2ObjectGraph",
     "Object2AtomGraph",
     "Atom2AtomGraph",
     "Atom2AtomMultiGraph",
     "Atom2AtomHigherOrderGraph",
    #"ObjectPair2ObjectPairGraph",
    #"HierarchicalGridGraph",

]  # List of different Encodings to use
# encodings_list = ["ObjectPair2ObjectPairGraph"]

domain_folder_root = "./data/"  # Path to the folder where are stored the subfolders for the data of each domain
predictions_folder_root = "./predictions/"  # Path to the folder where will be saved the predictions of each domain

domain = "sokoban"  # Dominio di cui fare l'encoding
domain_folder = (
    domain_folder_root + domain + "/"
)  # Percorso in cui sono memroizzati i plan.txt del dominio

# Dizionari che specificano per ogni tipologia di encoding il relativo objective e il numero di trials di optuna
# (in questo modo posso dare n_trials diversi per tipologia di encoding)
trials = {"GENModel": 50, "GINEModel": 50, "GATModel": 100}
models = ["GENModel", "GINEModel", "GATModel"]  # Models to use for the encoding
result_analysis_folder = f"./results_analysis/{domain.upper()}/"
predictions_folder = predictions_folder_root + domain.upper() + "/"
