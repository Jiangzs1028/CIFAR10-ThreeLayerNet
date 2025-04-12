import itertools
from .train import train
import numpy as np

def param_search(X_train, y_train, X_val, y_val, params):
    
    best_acc = 0
    best_params = {}
    
    # 生成所有参数组合的键和值
    param_names = params.keys()
    param_values = params.values()
    
    # 遍历所有参数组合
    for combination in itertools.product(*param_values):
        current_params = dict(zip(param_names, combination))
        print(f"Training with {current_params}")
        
        # 解包参数并训练模型
        model_params, val_acc = train(X_train, y_train, X_val, y_val, **current_params)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = current_params.copy()  # 保存深拷贝避免引用问题
            best_model_params = model_params
            np.savez('cv_best_model.npz', 
                    W1=best_model_params['W1'],
                    b1=best_model_params['b1'],
                    W2=best_model_params['W2'],
                    b2=best_model_params['b2'],
                    hidden_size=best_params['hidden_size'])
    
    print("Best parameters:", best_params)
    print("Best validation accuracy:", best_acc)