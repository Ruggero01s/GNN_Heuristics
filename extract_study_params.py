import os
import optuna
import joblib

base_path = "studies_storage\\GEN\\"
encodings_list = [
    "o2o",
    "o2a",
    "a2a",
]
dirs = os.listdir(base_path)

for dir in dirs:
    if (dir == "max"):
        print(f"====================={dir}========================")
        for encoding in encodings_list:
            try:
                study_path = base_path + dir + f"\\GENModel_study_{encoding}.pkl"
                study = joblib.load(study_path)
                best_value = study.best_value
                best_trial = study.best_trial
                print(f"\n========={encoding}==========") 

                # List of parameters you want to extract
                parameters_to_extract = [
                    "hidden_dim",
                    "fc_hidden_dim",
                    "hidden_layers",
                    "aggregation_function",
                    "hidden_channels_mlp",
                    "num_layers_mlp",
                    "readout",
                    "dropout_rate",
                ]

                for param in parameters_to_extract:
                    # Check if the parameter exists in the trial
                    if param in best_trial.params:
                        print(f"{param}: {best_trial.params[param]}")
                    else:
                        print(f"{param}: Not found")

                print(f"Value:{best_value}")
                print(f"==============================")
            except Exception as e:
                print(f"Error processing {encoding} in {dir}: {e}")
                continue
