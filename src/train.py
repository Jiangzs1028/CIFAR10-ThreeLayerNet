import numpy as np
import matplotlib.pyplot as plt
from .model import ThreeLayerNet

def augment_data(X: np.ndarray, seed: int = None) -> np.ndarray:
    """修正后的数据增强函数"""
    if seed is not None:
        np.random.seed(seed)
    
    X = X.reshape(-1, 3, 32, 32)
    
    # 随机水平翻转
    flip_mask = np.random.rand(X.shape[0]) > 0.5
    X[flip_mask] = X[flip_mask, :, :, ::-1]
    
    # 恢复随机裁剪（关键修改）
    pad_width = ((0, 0), (0, 0), (4, 4), (4, 4))
    padded = np.pad(X, pad_width=pad_width, mode='reflect')
    crops = np.zeros_like(X)
    for i in range(X.shape[0]):
        top = np.random.randint(0, 8)
        left = np.random.randint(0, 8)
        crops[i] = padded[i, :, top:top+32, left:left+32]
    
    # 调整颜色抖动参数（适合标准化前数据）
    color_scale = np.random.normal(loc=1.0, scale=0.05, size=(X.shape[0], 3, 1, 1))
    color_shift = np.random.normal(loc=0.0, scale=10.0, size=(X.shape[0], 3, 1, 1))  # 调整scale参数
    crops = crops * color_scale + color_shift
    return np.clip(crops, 0, 255).reshape(-1, 3072)

def standardize(X: np.ndarray) -> np.ndarray:
    """标准化CIFAR-10图像数据"""
    CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    CIFAR_STD = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    
    X = X.reshape(-1, 3, 32, 32)  # 转换为NCHW格式
    X = (X / 255.0 - CIFAR_MEAN) / CIFAR_STD
    return X.reshape(-1, 3072)  # 恢复为展平格式

def evaluate(model, X, y):
    model.eval()  # 设置为评估模式(不应用dropout)
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    model.train()  # 恢复训练模式
    return np.mean(predictions == y)

def compute_loss(model, X, y, reg):
    model.eval()  # 设置为评估模式(不应用dropout)
    probs = model.forward(X)
    data_loss = -np.log(probs[range(len(y)), y]).mean()
    reg_loss = 0.5 * reg * (np.sum(model.params['W1']**2) + np.sum(model.params['W2']**2))
    model.train()  # 恢复训练模式
    return data_loss + reg_loss

def train(X_train, y_train, X_val, y_val,
          hidden_size=100, activation='relu',
          reg=0.01, learning_rate=1e-3,
          epochs=1000, batch_size=200,
          lr_decay=0.9, lr_decay_every=5, 
          early_stop_step=20, dropout_rate=0.2,
          augment_train=True, momentum=0.9, warmup_step=0):
    
    input_dim = X_train.shape[1]
    model = ThreeLayerNet(input_dim, hidden_size, 10, activation, dropout_rate)
    model.train()  # 设置为训练模式
    best_val_acc = 0.0
    no_improvement_count = 0
    
    # 初始化记录变量
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    recorded_epochs = []  # 记录实际epoch数的列表

    # 初始化动量速度变量
    velocity = {}
    for param in model.params:
        velocity[param] = np.zeros_like(model.params[param])  # 与参数相同形状的零矩阵
    # 保存原始训练数据用于评估(不增强)
    X_train_original = X_train.copy()
    X_val = standardize(X_val)


    current_lr = learning_rate
    for epoch in range(epochs):
        # 学习率预热与衰减
        if epoch < warmup_step:
            current_lr = learning_rate * ((epoch + 1) / 20)
        else:
            # 衰减
            if (epoch - warmup_step) % lr_decay_every == 0 and (epoch - warmup_step) >= 0:
                current_lr *= lr_decay

        
        # 对训练数据进行增强(每个epoch不同)
        if augment_train:
            # 使用epoch作为随机种子，确保每个epoch增强不同但可复现
            X_augmented = augment_data(X_train_original, seed=epoch)
        else:
            X_augmented = X_train

        X_augmented = standardize(X_augmented)
        
        # 随机打乱数据
        shuffle_idx = np.random.permutation(X_augmented.shape[0])
        X_shuffled = X_augmented[shuffle_idx]
        y_shuffled = y_train[shuffle_idx]
        
        # 小批量训练
        for i in range(0, X_augmented.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 前向传播(自动应用dropout)
            model.forward(X_batch)
            
            # 反向传播
            grads = model.backward(X_batch, y_batch, reg)
            
            # 参数更新（添加动量计算）
            for param in model.params:
                # 更新速度：动量项 * 历史速度 + 当前梯度
                velocity[param] = momentum * velocity[param] + grads[param]
                # 参数更新：使用带有动量的速度
                model.params[param] -= current_lr * velocity[param]
        
        val_acc = evaluate(model, X_val, y_val)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

        # 每10个epoch记录一次数据
        if (epoch+1) % 10 == 0 or epoch == 0:
            # 使用原始训练数据(不增强)进行评估
            X_train_eval = standardize(X_train_original.copy())
            train_loss = compute_loss(model, X_train_eval, y_train, reg)
            val_loss = compute_loss(model, X_val, y_val, reg)
            train_acc = evaluate(model, X_train_eval, y_train)
            
            # 记录历史数据
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            recorded_epochs.append(epoch + 1)  # 记录实际epoch数
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            no_improvement_count = 0
            best_val_acc = val_acc
            best_params = {
                'W1': model.params['W1'].copy(),
                'b1': model.params['b1'].copy(),
                'W2': model.params['W2'].copy(),
                'b2': model.params['b2'].copy()
            }
            np.savez('best_model.npz', 
                    W1=best_params['W1'],
                    b1=best_params['b1'],
                    W2=best_params['W2'],
                    b2=best_params['b2'],
                    hidden_size=hidden_size,
                    activation=activation,
                    dropout_rate=dropout_rate)
        else:
            no_improvement_count += 1
            if no_improvement_count > early_stop_step:
                print(f"\nEarly stopping triggered after {epoch+1} epochs without improvement.")
                break
    
    # 绘制loss曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(recorded_epochs, train_loss_history, 'o-', label='Train Loss')
    plt.plot(recorded_epochs, val_loss_history,'o-',  label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recorded_epochs, train_acc_history,'o-',  label='Train Accuracy')
    plt.plot(recorded_epochs, val_acc_history,'o-',  label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return best_params, best_val_acc