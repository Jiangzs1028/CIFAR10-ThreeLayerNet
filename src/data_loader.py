import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

def load_cifar_batches(
    batch_names: List[str], 
    data_dir: Path = DEFAULT_DATA_DIR
) -> Tuple[np.ndarray, np.ndarray]:
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X_list, y_list = [], []
    for batch_name in tqdm(batch_names, desc="Loading batches"):
        batch_path = data_dir / batch_name
        if not batch_path.exists():
            raise FileNotFoundError(f"CIFAR batch not found: {batch_path}")
            
        with batch_path.open('rb') as f:
            batch = pickle.load(f, encoding='bytes')
            
        X_list.append(batch[b'data'].astype(np.float32))
        y_list.append(np.array(batch[b'labels'], dtype=np.int64))

    return np.concatenate(X_list), np.concatenate(y_list)


def load_dataset(
    train_batch_names: List[str],
    test_batch_name: str = "test_batch",
    data_dir: Path = DEFAULT_DATA_DIR
) -> Dict[str, np.ndarray]:
    
    # 加载训练集
    train_X, train_y = load_cifar_batches(train_batch_names, data_dir)
    print(f"成功加载训练集: 总样本数={len(train_X):,}, 类别数={len(np.unique(train_y))}")
    
    # 加载测试集
    test_X, test_y = load_cifar_batches([test_batch_name], data_dir)
    print(f"成功加载测试集: 总样本数={len(test_X):,}, 类别数={len(np.unique(test_y))}")
    
    
    return train_X, train_y, test_X, test_y
