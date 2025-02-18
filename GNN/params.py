seed = 1234
epochs = 70
patience = 5  # Patience for early stopping
encodings_list = [
    "Object2ObjectGraph",
    "Object2AtomGraph",
    "Atom2AtomGraph",
]  # List of different Encodings to use
# encodings_list = ["ObjectPair2ObjectPairGraph"]
domain_folder_root = "./data/"  # Path to the folder where are stored the subfolders for the data of each domain
predictions_folder_root = "./predictions/"  # Path to the folder where will be saved the predictions of each domain

