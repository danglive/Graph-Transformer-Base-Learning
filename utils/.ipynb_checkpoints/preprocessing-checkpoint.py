# utils/preprocessing.py

import ast
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

def convert_data_grid(data):
    """
    Converts raw data into node features, edge indices, edge attributes, and labels.

    Args:
        data (list): A list of dictionaries containing raw data entries.

    Returns:
        tuple: A tuple containing:
            - node_features (list): A list of node feature lists. Example values extracted from a dictionary:
                {
                    'p': 71.5158,
                    'q': -16.7881,
                    'v': 142.10000610351562,
                    'sub_id': 0,
                    'theta': 0.0,
                    'cooldown': 0,
                    'local_bus_id': 1,
                    'global_bus_id': 0
                }
            - edge_indices (list): A list of edge index lists. Example:
                [[0, 1], [0, 2], ..., [i, j]]
            - edge_attr (list): A list of edge attribute lists. Example values extracted from a dictionary:
                {
                    'p': 38.38889694213867,
                    'p_or': 38.68954,
                    'p_ex': -38.388897,
                    'q_or': -14.963034,
                    'q_ex': 10.282533,
                    'a_or': 168.54158,
                    'a_ex': 161.47176,
                    'theta_or': 0.0,
                    'theta_ex': -1.364565,
                    'v_or': 142.1,
                    'v_ex': 142.1,
                    'rho': 0.31153712,
                    'cooldown': 0,
                    'thermal_limit': 541.0,
                    'time_next_maintenance': -1,
                    'duration_next_maintenance': 0,
                    'nb_connected': 1,
                    'timestep_overflow': 0,
                    'sub_id_or': 0,
                    'sub_id_ex': 1,
                    'node_id_or': 0,
                    'node_id_ex': 1,
                    'bus_or': 1,
                    'bus_ex': 1,
                    'global_bus_or': 0,
                    'global_bus_ex': 1
                }
            - labels (list): A list of labels.
    """
    labels = [entry["label"] for entry in data]

    # Process node features to include p, q, v, and theta
    node_features = [
        [
            [nf[0], nf[1], nf[2], nf[4]]
            for nf in ast.literal_eval(entry["node_features"])
        ]
        for entry in tqdm(data, desc="Processing node features")
    ]

    # Process edge attributes to include specified features
    edge_attr = [
        [
            [
                etr[0], etr[1], etr[2], etr[3], etr[4], etr[5], etr[6],
                etr[7], etr[8], etr[9], etr[10], etr[11], etr[13]
            ]
            for etr in ast.literal_eval(entry["edge_attr"])
        ]
        for entry in tqdm(data, desc="Processing edge attributes")
    ]

    edge_indices = [
        ast.literal_eval(entry["edge_index"])
        for entry in tqdm(data, desc="Processing edge indices")
    ]

    return node_features, edge_indices, edge_attr, labels

def load_data_huggingface(config):
    """
    Loads and processes the training and validation datasets based on the configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        train_data: Processed training dataset.
        valid_data: Processed validation dataset.
    """
    split = config['data']['split']
    dataset_train_name = config['data']['dataset_train_name']
    print("dataset_train_name:", dataset_train_name)
    if split is None:
        # No split specified, load dataset directly
        train_data_raw = load_dataset(dataset_train_name, split="train") # dataset['train']
        valid_data_raw = load_dataset(dataset_train_name, split="validation")
    else:
        # Splits are specified
        split_indices = [int(n) for n in list(split.split("_")[-1])]

        # Load datasets
        train_datasets = []
        valid_datasets = []

        for i in split_indices:
            split_name = f"split_{i}"
            dataset = load_dataset(
                dataset_train_name,
                data_files={
                    "train": f"{split_name}/train.csv",
                    "validation": f"{split_name}/validation.csv",
                }
            )
            train_datasets.append(dataset["train"])
            valid_datasets.append(dataset["validation"])

        # Concatenate datasets
        train_data_raw = concatenate_datasets(train_datasets).shuffle(seed=42)
        valid_data_raw = concatenate_datasets(valid_datasets)
        print(train_data_raw)
        print(valid_data_raw)

    # Process data
    train_ds = convert_data_grid(train_data_raw)
    valid_ds = convert_data_grid(valid_data_raw)

    return train_ds, valid_ds