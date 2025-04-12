import argparse
import yaml
import numpy as np
from .train import train
from .test import test
from .param_search import param_search
from .data_loader import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--param_search', action='store_true')
    parser.add_argument('--config', type=str, default='./config/base_config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            exit(1)
    
    X_train, y_train, X_val, y_val = load_dataset(
        train_batch_names=[f"data_batch_{i}" for i in range(1, 6)],
        test_batch_name="test_batch"
    )

    
    if args.train:
        train_params = config.get('train', {})
        train(
            X_train, y_train, X_val, y_val,
            hidden_size=train_params.get('hidden_size', 1024),
            learning_rate=train_params.get('learning_rate', 0.1),
            activation=train_params.get('activation', 'relu'),
            reg=train_params.get('reg', 0.01),
            dropout_rate=train_params.get('dropout_rate', 0.2),
            lr_decay=train_params.get('lr_decay', 0.9),
            momentum = train_params.get('momentum', 0.9),
            augment_train = train_params.get('data_augment', True),
            warmup_step = train_params.get('warmup_step', 0),
            early_stop_step = train_params.get('early_stop_step', 20),
        )
    elif args.test:
        test_params = config.get('test', {})
        test(
            test_params.get('model_path', './ckp/best_model.npz'),
            X_val, y_val
        )
    elif args.param_search:
        search_params = config.get('param_search', {})
        print(search_params)
        param_search(X_train, y_train, X_val, y_val, search_params)