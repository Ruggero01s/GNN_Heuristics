import os
import optuna
import joblib

base_path = "optuna/"
dirs = os.listdir(base_path)

for dir in dirs:
    domains = os.listdir(os.path.join(base_path, dir))
    for domain in domains:
        if domain == "SOKOBAN":
            try:
                study_path = f"optuna/{dir}/{domain}/GENModel_study.pkl"
                study = joblib.load(study_path)
                best_value = study.best_value
                best_trial = study.best_trial

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
                print(f"============={dir}================")
                for param in parameters_to_extract:
                    # Check if the parameter exists in the trial
                    if param in best_trial.params:
                        print(f"{param}: {best_trial.params[param]}")
                    else:
                        print(f"{param}: Not found")

                print(f"Value:{best_value}")
                print(f"==============================")
            except Exception as e:
                continue
