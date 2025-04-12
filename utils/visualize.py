import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建保存图像的目录（如果不存在）
os.makedirs('figure', exist_ok=True)

# 加载模型参数
model_params = np.load('./ckp/best_model.npz')
W1, b1 = model_params['W1'], model_params['b1']
W2, b2 = model_params['W2'], model_params['b2']

# 定义自动对称x轴范围的函数
def auto_symmetric_x(data):
    max_abs = max(abs(data.min()), abs(data.max()))
    return (-max_abs, max_abs) if max_abs != 0 else (-1, 1)

# 第一部分：参数分布柱状图（以0为中心）
plt.figure(figsize=(12, 8))

# W1分布
plt.subplot(2, 2, 1)
w1_data = W1.flatten()
plt.hist(w1_data, bins=50, edgecolor='black')
plt.xlim(auto_symmetric_x(w1_data))
plt.title('W1 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# b1分布
plt.subplot(2, 2, 2)
b1_data = b1.flatten()
plt.hist(b1_data, bins=50, edgecolor='black')
plt.xlim(auto_symmetric_x(b1_data))
plt.title('b1 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# W2分布
plt.subplot(2, 2, 3)
w2_data = W2.flatten()
plt.hist(w2_data, bins=50, edgecolor='black')
plt.xlim(auto_symmetric_x(w2_data))
plt.title('W2 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# b2分布
plt.subplot(2, 2, 4)
b2_data = b2.flatten()
plt.hist(b2_data, bins=50, edgecolor='black')
plt.xlim(auto_symmetric_x(b2_data))
plt.title('b2 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Model Parameters Distribution (0-Centered)', y=1.02)

# 保存分布图
dist_path = 'figure/parameter_distributions.png'
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"参数分布图已保存至: {dist_path}")
plt.close()

# 第二部分：权重矩阵热力图（保持原代码）
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(W1, cmap='coolwarm', center=0)
plt.title('W1 Heatmap')

plt.subplot(1, 2, 2)
sns.heatmap(W2, cmap='coolwarm', center=0)
plt.title('W2 Heatmap')

plt.tight_layout()
plt.suptitle('Weight Matrices Visualization', y=1.02)

heatmap_path = 'figure/weight_heatmaps.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"权重热力图已保存至: {heatmap_path}")
plt.close()

print("所有可视化已完成并保存至 figure/ 目录")