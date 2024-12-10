# main.py
import os
import yaml
import torch
import argparse
from datasets import load_dataset
from torch_geometric.loader import DataLoader
from utils.preprocessing import load_data_huggingface, convert_data_grid
from utils.dataset import CustomDataset
from model.model import GraphTransformerClassifier, GraphTransformerMAE, count_parameters
from tools.train import train, train_mae, fix_random_seed  # Added train_mae
from tools.eval import evaluate

def main():
    """Main function to run the training and evaluation."""
    parser = argparse.ArgumentParser(description='Graph Transformer Model Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device configuration
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    task = config.get('task', 'train')
    training_method = config.get('training_method', 'multi-class')
    use_pretrained = config.get('use_pretrained', False)
    pretrained_path = config.get('pretrained_path', '')

    if task == 'train':

        # Model initialization
        model_config = config['model']
        num_classes = model_config.get('num_classes', 100)
        num_feature_node = model_config['num_feature_node']
        num_feature_edge = model_config['num_feature_edge']
        activation_func = model_config.get('activation', 'silu')

        fix_random_seed(int(config['training']['seed']))

        # Load and process data
        train_ds, valid_ds = load_data_huggingface(config)

        # Load data to Graph format for training
        train_data = CustomDataset(*train_ds)
        valid_data = CustomDataset(*valid_ds) 

        # Data loaders
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        if valid_data is not None:
            valid_loader = DataLoader(
                valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
        else:
            valid_loader = None

        if training_method == 'mae':
            # Initialize MAE model
            model = GraphTransformerMAE(
                dim=model_config['dim'],
                depth=model_config['depth'],
                num_feature_node=num_feature_node,
                num_feature_edge=num_feature_edge,
                mask_ratio=model_config.get('mask_ratio', 0.15),
                mask_feature_per_node=model_config.get('mask_feature_per_node', 0),
                device=device,
                activation=activation_func,
            )
        
            # Training parameters
            training_config = config['training']
            num_epochs = int(training_config['num_epochs'])
            learning_rate = float(training_config['learning_rate'])
            weight_decay = float(training_config['weight_decay'])
            early_stop_threshold = int(training_config.get('early_stop_threshold', 10))  # Default to 10 if not provided
            save_best_model = training_config.get('save_best_model', True)
        
            # Save path for checkpoints
            save_path_checkpoint = f"output_mae_{config['type_data']}/"
            if not os.path.exists(save_path_checkpoint):
                os.makedirs(save_path_checkpoint)
            model_path = os.path.join(save_path_checkpoint, training_config['model_path'])
            print("Output of MAE checkpoints:", save_path_checkpoint)

            print(model)
            print("Number of parameter:", count_parameters(model))
            if config['training']['load_checkpoint']:
                model.load_state_dict(torch.load(config['training']['load_checkpoint']))
                
            # Start MAE training
            print("Starting MAE training...")
        
            # Ensure that you have validation data
            if valid_data is not None:
                train_losses, validation_losses = train_mae(
                    model,
                    train_loader,
                    valid_loader,
                    save_best_model,
                    model_path,
                    num_epochs,
                    learning_rate,
                    weight_decay,
                    early_stop_threshold,
                )
            else:
                print("Validation data is required for MAE training with validation loss computation.")

        elif training_method in ['multi-class', 'multi-label']:
            # Initialize classifier model
            model = GraphTransformerClassifier(
                dim=model_config['dim'],
                depth=model_config['depth'],
                num_classes=num_classes,
                num_feature_node=num_feature_node,
                num_feature_edge=num_feature_edge,
                device=device,
                dropout_rate=model_config.get('dropout_rate', 0.5),
                activation=activation_func,
                pretrained_path=pretrained_path if use_pretrained else None,
            )

            # Training parameters
            training_config = config['training']
            num_epochs = int(training_config['num_epochs'])
            early_stop_threshold = int(training_config['early_stop_threshold'])
            save_best_model = training_config['save_best_model']
            step_lr = int(training_config['step_lr'])
            weight_decay = float(training_config['weight_decay'])
            learning_rate = float(training_config['learning_rate'])
            
            # Save path for checkpoints
            save_path_checkpoint = f"output_{config['type_data']}_{training_method}/"
            if not os.path.exists(save_path_checkpoint):
                os.makedirs(save_path_checkpoint)
            model_path = os.path.join(save_path_checkpoint, training_config['model_path'])
            print("Output of classifier checkpoints:", save_path_checkpoint)

            # Start supervised training
            print(f"Starting supervised training ({training_method})...")
            print(model)
            print("Number of parameter:", count_parameters(model))
            if config['training']['load_checkpoint']:
                model.load_state_dict(torch.load(config['training']['load_checkpoint']))
                
            train(
                model,
                train_loader,
                valid_loader,
                save_best_model,
                model_path,
                num_epochs,
                learning_rate,
                weight_decay,
                step_lr,
                early_stop_threshold,
                training_method
            )

            # Optionally, evaluate on unseen test set
            if config.get('evaluate_unseen', False):
                print("Evaluation on unseen test set:")
                dataset_test_name = config['data']['dataset_test_name']

                # Load the test dataset
                unseen_dataset = load_dataset(dataset_test_name)
                test_data_raw = unseen_dataset['test'] if 'test' in unseen_dataset else unseen_dataset['train']

                # Process the test data
                test_ds = convert_data_grid(test_data_raw)
                test_data = CustomDataset(*test_ds)
                test_loader = DataLoader(
                    test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
                )

                # Load the best model
                model.load_state_dict(torch.load(model_path+".pth", map_location=device))
                model.eval()
                if training_method == 'multi-class':
                    loss_fn = torch.nn.CrossEntropyLoss()
                elif training_method == 'multi-label':
                    loss_fn = torch.nn.BCEWithLogitsLoss()

                test_loss, acc_test = evaluate(
                    test_loader,
                    model,
                    loss_fn,
                    calc_loss=True,
                    topk=(1, 5, 10),
                    training_method=training_method
                )
                print(
                    f"Test Loss: {test_loss:.4f}, Test Acc Top-1: {acc_test[1]*100:.2f}%, "
                    f"Top-5: {acc_test[5]*100:.2f}%, Top-10: {acc_test[10]*100:.2f}%"
                )

        else:
            print(f"Unknown training method: {training_method}")

    elif task == 'evaluate':
        # Load model and evaluate
        print("Evaluation task is not yet implemented.")
    elif task == 'inference':
        # Inference code
        print("Inference task is not yet implemented.")
    else:
        print(f"Unknown task: {task}")

if __name__ == "__main__":
    main()