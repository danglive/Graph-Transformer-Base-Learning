
# Graph-Transformer-Base-Learning

This repository provides an implementation of graph-based Transformer models for graph classification and self-supervised learning (GraphMAE) tasks. The code is designed to handle datasets representing power grids (e.g., IEEE_14, IEEE_118), and integrates seamlessly with Hugging Face Datasets. The provided code allows training a Graph Transformer model for both supervised classification and self-supervised masked autoencoding (GraphMAE).

## Key Features

- **Graph Classification**: Train a Graph Transformer to classify graphs into one of multiple classes or perform multi-label classification.
- **GraphMAE (Self-Supervised)**: Implement a masked graph autoencoder, masking node features or entire nodes to learn meaningful graph representations without explicit labels.
- **Flexible Configuration**: Easily modify model dimensions, number of layers, training tasks, dataset sources, etc., via a YAML config file.
- **Support for Power Grid Data**: Preprocessing utilities and dataset loading functions tailored for power grid-based graph data, including node and edge features extracted from power systems.
- **Pretrained Encoder**: Optionally load and fine-tune a pretrained encoder for downstream tasks.

## Repository Structure

```
Graph-Transformer-Base-Learning
├─ main.py
├─ config.yaml
├─ model/
│  ├─ __init__.py
│  ├─ attention.py
│  ├─ graph_transformer.py
│  ├─ model.py
│  ├─ rotary_embedding.py
└─ utils/
   ├─ __init__.py
   ├─ dataset.py
   ├─ loss.py
   ├─ preprocessing.py
```

### `main.py`
- The entry point for training, evaluation, or inference.
- Parses command-line arguments, loads the YAML configuration, sets up the training environment, and runs the specified task.
- Depending on `training_method` in `config.yaml`:
  - **`mae`**: Runs self-supervised GraphMAE training.
  - **`multi-class` / `multi-label`**: Runs supervised graph classification training.
- Handles loading datasets (via Hugging Face Datasets), building data loaders, initializing models, and saving checkpoints.

### `config.yaml`
- Centralized configuration for the entire project.
- **`type_data`**: Specifies the dataset type (e.g., IEEE_14 for power grid).
- **`task`**: Task to perform (`train`, `evaluate`, `inference`).
- **`device`**: Compute device, e.g., `cuda:0`, `cuda:1`, or `cpu`.
- **`training_method`**: Choice between `mae` (GraphMAE), `multi-class`, or `multi-label`.
- **`model`**: Model hyperparameters such as `dim`, `depth`, `num_classes`, `activation`, `mask_ratio`, etc.
- **`training`**: Training hyperparameters including `num_epochs`, `learning_rate`, `batch_size`, `early_stop_threshold`, and checkpoint paths.
- **`data`**: Dataset configuration, e.g., Hugging Face dataset names, splits, and whether to evaluate on unseen data.

### `utils/`
This directory contains helper functions for data handling, preprocessing, and loss computation.

- **`utils/dataset.py`**:  
  Defines `CustomDataset`, a PyTorch-compatible dataset class that converts raw lists of node features, edge indices, and labels into `torch_geometric.data.Data` objects.

- **`utils/loss.py`**:  
  Provides various loss functions:
  - `multi_classification_loss`: Cross-entropy for multi-class tasks.
  - `mse_loss`: Mean Squared Error loss.
  - `multi_label_classification_loss`: Binary cross-entropy for multi-label tasks.

- **`utils/preprocessing.py`**:  
  Functions to process raw power grid data into a suitable format (node features, edge attributes, and labels):
  - `convert_data_grid(data)`: Converts raw dataset entries into node features, edge indices, edge attributes, and labels.
  - `load_data_huggingface(config)`: Loads and processes training/validation data from Hugging Face Datasets according to `config.yaml`.

### `model/`
This directory contains the core models and building blocks:

- **`model/attention.py`**:  
  Implements multi-head self-attention for graphs.  
  - `MultiHeadAttention`: Incorporates node features, edge features, and optional positional embeddings.  
  Uses rotary embeddings if configured.

- **`model/graph_transformer.py`**:  
  Defines the `GraphTransformer` class, a Transformer encoder adapted for graph data.  
  - Supports optional gated residual connections and relative positional embeddings.
  - `PreNorm`, `Residual`, `GatedResidual`, and `feed_forward` layers provide Transformer-like building blocks.
  
- **`model/model.py`**:  
  Contains models that wrap the `GraphTransformer`:
  - `GraphTransformerClassifier`: For supervised graph classification, includes MLP heads for prediction.
  - `GraphTransformerMAE`: A masked autoencoder variant that masks node features or entire nodes and learns to reconstruct them.
  - `count_parameters(model)`: Helper to count trainable parameters.

- **`model/rotary_embedding.py`**:  
  Implements Rotary Embeddings for positional encoding, originally developed for language models but adapted here for graph nodes.

## Training Methods

### GraphMAE (Self-Supervised)
- Set `training_method: mae` in `config.yaml`.
- The model `GraphTransformerMAE` masks a portion of node features (either entire nodes or selected features) and aims to reconstruct them.
- Useful for pretraining a Graph Transformer encoder on unlabeled graph data.

### Multi-Class / Multi-Label Classification (Supervised)
- Set `training_method: multi-class` or `multi-label` in `config.yaml`.
- `GraphTransformerClassifier` takes the node and edge features, encodes them, and aggregates node embeddings to produce a class probability distribution.
- Can be applied to power grid datasets to predict certain grid states or conditions.

## Data Handling for Power Grid
- The provided dataset preprocessing code (`utils/preprocessing.py`) is tailored to handle power grid data:
  - Node features include electrical parameters (e.g., active/reactive power, voltage, etc.).
  - Edge features represent line parameters (e.g., power flow, thermal limits).
- The configuration in `config.yaml` specifies the number of node/edge features and the dataset name (e.g., `dangvantuan/IEEE_14_dataset` on Hugging Face).

## How to Run

1. **Install Dependencies**:  
   Ensure you have PyTorch, PyTorch Geometric, Hugging Face Datasets, and other dependencies installed.

2. **Set Configuration**:  
   Modify `config.yaml` to point to the desired dataset and adjust model/training parameters.

3. **Run Training**:
   ```bash
   python main.py --config config.yaml
   ```
   This will start the training process according to the specified `task` and `training_method`.

4. **Checkpoints and Logs**:  
   Check the output directory (e.g., `output_IEEE_14_multi-class/` or `output_mae_IEEE_14/`) for checkpoints and logs.

5. **Evaluation on Unseen Data**:  
   If `evaluate_unseen` is set to `true` in `config.yaml` and a test dataset is provided, the script will evaluate on unseen test data after training completes.

## Customization

- **Change the Dataset**:  
  Update `data.dataset_train_name` and `data.dataset_test_name` in `config.yaml`.
- **Adjust Model Size**:  
  Modify `dim`, `depth`, `heads`, `activation`, etc., in the `model` section of `config.yaml`.
- **Switch Between Training Methods**:  
  Change `training_method` to `mae`, `multi-class`, or `multi-label` for different use cases.
- **Pretrained Encoder**:  
  Set `use_pretrained: true` and provide `pretrained_path` to load a previously saved encoder before training a new task.

## License
[MIT License](LICENSE)

---

This Code aims to simplify building and experimenting with Transformer-based models on graph-structured power grid data. For any questions or contributions, please open an issue or submit a pull request.